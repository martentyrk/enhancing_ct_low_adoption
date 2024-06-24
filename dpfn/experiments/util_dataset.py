"""Utility functions for a dump of dataset for GNNs."""
from dpfn import logger
import dpfn_util
import json
import numpy as np
import os
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import add_self_loops, to_undirected
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as PygDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from constants import GRAPH_MODELS
import numba
from sklearn.preprocessing import OneHotEncoder
import time

import os

def dump_features_graph(
        contacts_now: np.ndarray,
        observations_now: np.ndarray,
        z_states_inferred: np.ndarray,
        user_free: np.ndarray,
        z_states_sim: np.ndarray,
        users_age: np.ndarray,
        app_users: np.ndarray,
        trace_dir: str,
        num_users: int,
        num_time_steps: int,
        t_now: int,
        rng_seed: int,
        infection_prior: float = -1.,
        infection_prior_now: float = -1.,
        infection_rate_prev: float = -1.
        ) -> None:
    """Dump graphs for GNNs."""
    datadump = dpfn_util.fn_features_dump(
        num_workers=1,
        num_users=num_users,
        num_time_steps=num_time_steps,
        q_marginal=z_states_inferred[:, :, 2],
        contacts=contacts_now,
        users_age=users_age)

    # Process observations
    observations_json = create_observations_json(
        num_users, observations_now)

    # Try to dump the dataset
    dirname = os.path.join(trace_dir, f'test_with_obs_{rng_seed}')
    os.makedirs(dirname, exist_ok=True)
    fname_pos = os.path.join(dirname, f'positive_{t_now:05d}.jl')
    fname_neg = os.path.join(dirname, f'negative_{t_now:05d}.jl')

    user_positive = np.logical_or(
        z_states_sim == 1, z_states_sim == 2)

    # TODO(rob) This dump depends on JSON library. We can make this way faster
    # with something like protobuf or TFRecord.
    num_ignored = 0
    degrees = [0] * num_users
    with open(fname_pos, 'w') as fpos:
        with open(fname_neg, 'w') as fneg:
            for user in range(num_users):
                if user_free[user] == 0 or app_users[user] == 0:
                    num_ignored += 1
                    continue
                degree_counter = 0
                
                for row in datadump[user]:
                    if row[0] < 0:
                        break
                    assert row[3] <= 1024
                    degree_counter += 1
                degrees[user] = degree_counter
                
            for user in range(num_users):
                # Don't collect features on quarantined users
                if user_free[user] == 0 or app_users[user] == 0:
                    num_ignored += 1
                    continue
                # Each row is a json file
                output = {
                    "fn_pred": float(z_states_inferred[user][-1][2]),
                    "user": int(user),
                    "sim_state": int(z_states_sim[user]),  # in constants.py
                    "user_age": int(users_age[user]),
                    "observations": observations_json[user],
                    "contacts": [],
                    "infection_prior": float(infection_prior),
                    "infection_prior_now": float(infection_prior_now),
                    't_now': int(t_now),
                    'degree': degrees[user],
                    'infection_rate': float(infection_rate_prev), 
                }

                for row in datadump[user]:
                    if row[0] < 0:
                        break
                    assert row[3] <= 1024
                    output['contacts'].append([
                        int(row[0]),  # timestep
                        int(row[1]),  # sender
                        int(row[2]),  # age (age groups)
                        int(row[3]),  # pinf
                        int(row[4]),  # interaction type
                        int(app_users[row[1]] == 1),  # app user status
                        int(degrees[row[1]])
                    ])
                # TODO: REMOVE AFTER RUN
                # if len(output['contacts']) > 0:
                #     output['contacts'] = np.array(output['contacts'])
                #     non_app_user_mask = output['contacts'][:, -2] == 0
                #     non_app_contacts = output['contacts'][non_app_user_mask]
                #     app_contacts = output['contacts'][~non_app_user_mask]
                    
                #     non_app_contacts[:, [1, 2, 3, 6]] = -1.
                #     unique_contacts = np.unique(non_app_contacts, axis=0)
                #     output['contacts'] = np.vstack((unique_contacts, app_contacts))
                #     np.random.shuffle(output['contacts'])
                #     output['contacts'] = output['contacts'].tolist()
                
                # in pytorch its json.loads
                if user_positive[user]:
                    fpos.write(json.dumps(output) + "\n")
                else:
                    fneg.write(json.dumps(output) + "\n")
    print(f"Ignored {num_ignored} out of {num_users} users")

def dump_preds(
    fn_preds: np.ndarray,
    dl_preds: np.ndarray,
    incorporated_users:np.ndarray,
    t_now: int,
    dirname: str,
    app_user_ids: np.ndarray,
    users_age: np.ndarray,
    ):
    
    fn_preds = fn_preds.astype(float)
    dl_preds = dl_preds.astype(float)
    incorporated_users = incorporated_users.astype(int)
    fname_dump = os.path.join(dirname, f'pred_dump_{t_now:05d}.jl')
    fname_extras = os.path.join(dirname, 'extras.jl')
    if t_now == 1:
        extra_output = {
            'app_users': app_user_ids.tolist(),
            'users_age': users_age.tolist(),
        }
        with open(fname_extras, 'w') as f_extra:
            f_extra.write(json.dumps(extra_output))
        
    output = {
        'fn_preds': fn_preds.tolist(),
        'dl_preds': dl_preds.tolist(),
        'incorporated_users': incorporated_users.tolist(),
        't_now': t_now,
    }
    
    with open(fname_dump, 'w') as f_dump:
        f_dump.write(json.dumps(output) + "\n")
    

@numba.njit
def get_user_contacts(contacts_now, user, temp_contacts):
    count = 0
    for i in numba.prange(contacts_now.shape[0]):
        if contacts_now[i, 0] == user:
            temp_contacts[count] = contacts_now[i]
            count += 1

    return temp_contacts[:count]

def inplace_features_data_creation(
    contacts_now: np.ndarray,
    observations_now: np.ndarray,
    z_states_inferred: np.ndarray,
    user_free: np.ndarray,
    users_age: np.ndarray,
    app_users: np.ndarray,
    num_users: int,
    num_time_steps: int,
    app_user_ids: np.ndarray,
    infection_rate: float,
):  

    model_data = []

    observations_json = create_observations_json(num_users, observations_now)

    datadump = dpfn_util.fn_features_dump(
        num_workers=1,
        num_users=num_users,
        num_time_steps=num_time_steps,
        q_marginal=z_states_inferred[:, :, 2],
        contacts=contacts_now,
        users_age=users_age,)
    
    degrees = [0] * num_users

    for user in app_user_ids:
        if user_free[user] == 0:
            continue

        output = {
            'fn_pred': float(z_states_inferred[user][-1][2]),
            "user": int(user),
            "user_age": int(users_age[user]),
            "observations": observations_json[user],
            "contacts": [],
            "infection_rate": float(infection_rate),
        }

        for row in datadump[user]:
            if row[0] < 0:
                break
            output['contacts'].append([
                int(row[0]),  # timestep
                int(row[1]),  # sender
                int(row[2]),  # age (age groups)
                int(row[3]),  # pinf
                int(row[4]),  # interaction type
                int(app_users[row[1]] == 1),  # app user status
                int(degrees[row[1]])
            ])
            
        model_data.append(output)
    
    return model_data


def create_observations_json(num_users: int, observations_now: np.ndarray):
    observations_json = [[] for _ in range(num_users)]
    for observation in observations_now:
        user = int(observation[0])
        timestep = int(observation[1])
        outcome = int(observation[2])
        observations_json[user].append([timestep, outcome])

    return observations_json


class NoContacts(Exception):
     """Custom exception for no contacts in a data row."""
    
class DeepSet_Dataset(Dataset):
    def __init__(self, data):
        self.features = data
        
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return self.features[idx]
    
def create_dataset_setmodel(data, model_type, infection_prior:float = None, add_weights=False):
    data_list = []
    
    for single_data in data:
        try:
            single_user_data = make_features_set(single_data, infection_prior)
        except NoContacts:
            continue
        
        data_list.append(single_user_data)

    
    set_dataset = DeepSet_Dataset(data_list)
        
    return TorchDataLoader(set_dataset, batch_size=1024, shuffle=False)

def make_features_set(data, infection_prior):
    """Converts the JSON to the graph features."""
    #Contacts object: [timestep, sender, age (age groups), pinf, interaction type, app_user (1 or 0)]
    contacts = np.array(data['contacts'], dtype=np.int64)
    # Observations object: [timestep, result]
    observations = np.array(data['observations'], dtype=np.int64)
    
    #Normalize user data
    data['user_age'] /= 10

    if len(contacts) == 0:
        contacts = -1 * torch.ones(size=(900, 5), dtype=torch.float32)
        
    else:
        # Remove sender information
        contacts = np.concatenate((contacts[:, 0:1], contacts[:, 2:]), axis=1)
        #Remove degree
        contacts = contacts[:, :-1]
        contacts = torch.tensor(contacts, dtype=torch.float32)
    
    
    if len(observations) == 0:
        observations = -1 * torch.ones(size=(14, 5), dtype=torch.float32)
    else:
        observations = torch.tensor(observations, dtype=torch.float32)

        observations = torch.nn.functional.pad(
          observations, [0, 3, 0, 14-len(observations)],
          mode='constant', value=-1.)
        
    # Column 0 is the timestep
    
    contacts[:, 1] /= 10  # Column 1 is the age
    contacts[:, 2] /= 1024  # Column 2 is the pinf according to FN
    
    # We know timestep and interaction type for non users, other values will be set to -1.
    app_users_mask = contacts[:, -1] == 1
    contacts[~app_users_mask, 1] = -1.
    contacts[~app_users_mask, 2] = infection_prior

    contacts = torch.nn.functional.pad(
      contacts, [0, 0, 0, 900-len(contacts)],
      mode='constant', value=-1.)
    contacts = torch.cat((contacts, observations), dim=0)
    
    return contacts

def create_dataset(data, model_type, cfg, infection_prior:float = None, add_weights=False, local_mean_base:bool=False):
    data_list = []
    one_hot_encoding = cfg.get('one_hot')
    simulator_type = cfg.get('simulator')
    interaction_encoder = OneHotEncoder(sparse_output=False)
    interaction_types = [[0], [1], [2], [3]] if simulator_type == 'covasim' else [[0], [1], [2]]
    interaction_encoder.fit(interaction_types)
    added_user_ids = []
    
    for single_data in data:
        try:    
            if model_type != 'hetero_gnn':    
                node_features, single_contact_edges, fn_averages, infection_rates  = create_graph_features(
                    single_data, interaction_encoder,interaction_types, infection_prior, local_mean_base, one_hot_encoding)
                added_user_ids.append(single_data['user'])
        except NoContacts:
            continue
        
        if model_type in GRAPH_MODELS:
            if add_weights:
                unk_mask, known_mask, obs_mask = generate_node_masks(node_features, single_contact_edges)
                
                # single_contact_edges, _ = add_self_loops(single_contact_edges)
                
                data_single = Data(
                    x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=single_contact_edges.long(),
                    known_mask=torch.tensor(known_mask, dtype=torch.int),
                    unk_mask=torch.tensor(unk_mask, dtype=torch.int),
                    obs_mask = torch.tensor(obs_mask, dtype=torch.int),
                    fn_averages=torch.tensor(fn_averages, dtype=torch.float),
                    infection_rates=torch.tensor(infection_rates, dtype=torch.float),
                )
            elif model_type=='hetero_gnn':
                try:
                    data_single = create_graph_heterogeneous(single_data, interaction_encoder,interaction_types, infection_prior, local_mean_base, one_hot_encoding)
                    added_user_ids.append(single_data['user'])
                except NoContacts:
                    continue

            else:
                # single_contact_edges, _ = add_self_loops(single_contact_edges)
                # single_contact_edges = to_undirected(single_contact_edges)
                
                data_single = Data(
                    x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=single_contact_edges.long(),
                )

            data_list.append(data_single)
        
    
    return PygDataLoader(data_list, batch_size=2048, shuffle=False), np.array(added_user_ids)
    
MAX_DAYS = 14.

def create_graph_w_edge_features(data, interaction_encoder,interaction_types: np.ndarray, infection_prior: float= None, local_mean_base:bool=False, one_hot_encoding:bool=False):
    '''
    Function to create graphical data, where timestep and interaction types are modelled as edge features
    '''
    contacts = np.array(data['contacts'], dtype=np.float32)
    contact_dict = {}
    #Normalize user data
    data['user_age'] /= 10
    single_contact_edges = torch.tensor([[],[]])
    
    # Observations object: [timestep, result]
    observations = np.array(data['observations'], dtype=np.float32)
    obs = {}

    if len(observations) > 0:
        observations, obs = generate_observations(observations, one_hot_encoding, interaction_types)

    
    if len(contacts) > 0:
        # Remove sender information
        contacts = np.delete(contacts, 1, axis=1)
        contacts = torch.tensor(contacts, dtype=torch.float32)
    
        if one_hot_encoding:
            interaction_types_categorical = interaction_encoder.transform(contacts[:, 3].reshape(-1, 1))
            contacts = np.concatenate((contacts[:,0:3], interaction_types_categorical, contacts[:,4:]), axis = 1)

        normalize_contact_features(contacts)

    
        # We know timestep and interaction type, other values will be set to -1.
        app_users_mask = contacts[:, -1] == 1
        
        #if data['infection_prior'] > 0:
        # infection_prior_contacts = torch.mean(contacts[app_users_mask, 2])
        # if infection_prior_contacts.isnan().any():
        
        if infection_prior is not None:
            contacts[~app_users_mask, 2] = torch.tensor(infection_prior, dtype=torch.float)
        elif local_mean_base:
            if sum(app_users_mask) > 0:
                local_mean_prior = torch.mean(contacts[app_users_mask, 2])
                contacts[~app_users_mask, 2] = local_mean_prior
            else:
                contacts[~app_users_mask, 2] = 0.
        else:
            contacts[~app_users_mask, 2] = -1.
        # interval_val = 0.1
        # age_prior_contacts = torch.round(torch.mean(contacts[app_users_mask, 1] / interval_val)) * interval_val
        # if age_prior_contacts.isnan().any():
        #     contacts[~app_users_mask, 1] = data['user_age']
        # else:
        #     contacts[~app_users_mask, 1] = age_prior_contacts
        # else:
        
        contacts[~app_users_mask, 1] = -1. #Age
        #     contacts[~app_users_mask, 2] = -1.

        num_contacts = len(contacts)
        num_observations = len(observations)
        single_contact_edges = generate_contact_edges(num_contacts, num_observations)
        
        contact_interactions = interaction_types_categorical if one_hot_encoding else contacts[:, 3][:, np.newaxis]
        
        contact_dict = {
            'fn': contacts[:, 2][:, np.newaxis],
            'age': contacts[:, 1][:, np.newaxis],
            'time': contacts[:, 0][:, np.newaxis],
            'interaction_type': contact_interactions,
            'app_user_ind': contacts[:, -1][:, np.newaxis],
        }
        
    node_features = np.array([
      data['fn_pred'],
      data['user_age'],
      1.
    ]).reshape(1, -1)
  
    attr_dim = len(interaction_types) + 1 if one_hot_encoding else 2
    edge_attr = np.empty((0, attr_dim))
    
    if len(contact_dict.keys()) > 0:
        contact_features = np.hstack((
            contact_dict['fn'],
            contact_dict['age'],
            contact_dict['app_user_ind']
        ))
        
        contact_edge_attr = np.hstack((
        contact_dict['time'],
        contact_dict['interaction_type'],
        ))
    
        node_features = np.concatenate((node_features, contact_features), axis=0)
        edge_attr = np.concatenate((edge_attr, contact_edge_attr))
    
    if len(obs.keys()) > 0:
        obs_features = np.hstack((
        obs['outcome'],
        obs['age'],
        obs['app_user_ind']
        ))
        
        obs_edge_attr = np.hstack((
        obs['time'],
        obs['interaction_type'] 
        ))
        
        node_features = np.concatenate((node_features, obs_features), axis=0)
        edge_attr = np.concatenate((edge_attr, obs_edge_attr))
    
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    return node_features, single_contact_edges, edge_attr
    
def create_graph_features(data, interaction_encoder,interaction_types: np.ndarray, infection_prior: float= None, local_mean_base:bool=False, one_hot_encoding:bool=False):
    """Converts the JSON to the graph features."""
    #Contacts object: [timestep, sender, age (age groups), pinf, interaction type, app_user (1 or 0)]

    contacts = np.array(data['contacts'], dtype=np.float32)
    contact_dict = {}
    #Normalize user data
    data['user_age'] /= 10
    single_contact_edges = torch.tensor([[],[]])
    
    # Observations object: [timestep, result]
    observations = np.array(data['observations'], dtype=np.float32)
    obs = {}
    
    local_mean_prior = 0.0

    if len(observations) > 0:
        observations, obs = generate_observations(observations, one_hot_encoding, interaction_types)

    
    if len(contacts) > 0:
        # Remove sender information
        contacts = np.concatenate((contacts[:, 0:1], contacts[:, 2:]), axis=1)
        #contacts = torch.tensor(contacts, dtype=torch.float32)
    
        if one_hot_encoding:
            interaction_types_categorical = interaction_encoder.transform(contacts[:, 3].reshape(-1, 1))
            contacts = np.concatenate((contacts[:,0:3], interaction_types_categorical, contacts[:,4:]), axis = 1)

        normalize_contact_features(contacts)

    
        # We know timestep and interaction type, other values will be set to -1.
        app_users_mask = contacts[:, -2] == 1
        
        #if data['infection_prior'] > 0:
        # infection_prior_contacts = torch.mean(contacts[app_users_mask, 2])
        # if infection_prior_contacts.isnan().any():
        
        if infection_prior is not None:
            contacts[~app_users_mask, 2] = infection_prior
        elif local_mean_base:
            if sum(app_users_mask) > 0:
                local_mean_prior = contacts[app_users_mask, 2].mean()
                contacts[~app_users_mask, 2] = local_mean_prior
            else:
                contacts[~app_users_mask, 2] = 0.0
        else:
            contacts[~app_users_mask, 2] = -1.
        # interval_val = 0.1
        # age_prior_contacts = torch.round(torch.mean(contacts[app_users_mask, 1] / interval_val)) * interval_val
        # if age_prior_contacts.isnan().any():
        #     contacts[~app_users_mask, 1] = data['user_age']
        # else:
        #     contacts[~app_users_mask, 1] = age_prior_contacts
        # else:
        
        contacts[~app_users_mask, 1] = -1. #Age
        #     contacts[~app_users_mask, 2] = -1.
        if sum(app_users_mask) > 0:
            local_mean_prior = contacts[app_users_mask, 2].mean()
        else:
            local_mean_prior = 0.0
                
        num_contacts = len(contacts)
        num_observations = len(observations)
        single_contact_edges = generate_contact_edges(num_contacts, num_observations)
        
        contact_interactions = interaction_types_categorical if one_hot_encoding else contacts[:, 3][:, np.newaxis]
        
        contact_dict = {
            'fn': contacts[:, 2][:, np.newaxis],
            'age': contacts[:, 1][:, np.newaxis],
            'time': contacts[:, 0][:, np.newaxis],
            'interaction_type': contact_interactions,
            'app_user_ind': contacts[:, -2][:, np.newaxis],
        }
        
    _, node_features = process_node_features(data, contact_dict, obs, one_hot_encoding, interaction_types)
    fn_averages = np.array([infection_prior, local_mean_prior])
    infection_rates = np.array(data['infection_rate'])
    
    return node_features, single_contact_edges, fn_averages, infection_rates    

def create_graph_heterogeneous(data, interaction_encoder, interaction_types: np.ndarray, infection_prior: float=None, local_base: bool=False, one_hot_encoding:bool = False):
    #Contacts object: [timestep, sender, age (age groups), pinf, interaction type, app_user (1 or 0)]
    contacts = np.array(data['contacts'], dtype=np.float32)
    contact_dict = {}
    #Normalize user data
    data['user_age'] /= 10
    single_contact_edges = torch.tensor([[],[]])

    # Observations object: [timestep, result]
    observations = np.array(data['observations'], dtype=np.float32)
    obs = {}

    hetero_data = HeteroData()
    if len(observations) > 0:
        observations, obs = generate_observations(observations, one_hot_encoding, interaction_types)
        
    if len(contacts) > 0:
        # Remove sender information
        contacts = np.concatenate((contacts[:, 0:1], contacts[:, 2:]), axis=1)
        contacts = torch.tensor(contacts, dtype=torch.float32)
                
        if one_hot_encoding:
            interaction_types_categorical = interaction_encoder.transform(contacts[:, 3].reshape(-1, 1))
            contacts = np.concatenate((contacts[:,0:3], interaction_types_categorical, contacts[:,4:]), axis = 1)
            contacts = torch.tensor(contacts, dtype=torch.float32)
            
        normalize_contact_features(contacts)
        # We know timestep and interaction type, other values will be set to -1.
        app_users_mask = contacts[:, -2] == 1
        
        if infection_prior is not None:
            contacts[~app_users_mask, 2] = torch.tensor(infection_prior, dtype=torch.float)
        elif local_base:
            if sum(app_users_mask) > 0:
                local_mean_prior = torch.mean(contacts[app_users_mask, 2])
                contacts[~app_users_mask, 2] = local_mean_prior
            else:
                contacts[~app_users_mask, 2] = 0.
        else:
            contacts[~app_users_mask, 2] = -1.
            
        #Age
        contacts[~app_users_mask, 1] = -1.
    
        num_contacts = len(contacts)
        num_observations = len(observations)
        # From node index 1 to number of contacts + 1. 
        single_contact_edges = generate_contact_edges(num_contacts, num_observations)
        
        interactions = interaction_types_categorical if one_hot_encoding else contacts[:, 3][:, np.newaxis]
    
        contact_dict = {
            'fn': contacts[:, 2][:, np.newaxis],
            'age': contacts[:, 1][:, np.newaxis],
            'time': contacts[:, 0][:, np.newaxis],
            'interaction_type': interactions,
            'app_user_ind': contacts[:, -2][:, np.newaxis],
            'degree': contacts[:, -1][:, np.newaxis]
        }
    
    target_features, node_features = process_node_features(data, contact_dict, obs, one_hot_encoding, interaction_types)
    
    unk_mask, known_mask, obs_mask = generate_node_masks(node_features, single_contact_edges, add_root_to_known=False)
    # Create dummy elements if no known, unknown or observation nodes are present
    imputable_array = np.zeros(3 + len(interaction_types), dtype=np.int32) if one_hot_encoding else np.zeros(4, dtype=np.int32)
    
    if len(node_features[known_mask]) < 1:
        impute_values = np.concatenate((imputable_array, np.array([1])))  
        node_features = np.concatenate((node_features, impute_values.reshape(1,-1)), axis=0)
        
    if len(node_features[unk_mask]) < 1:
        impute_values = np.concatenate((imputable_array, np.array([0])))  
        node_features = np.concatenate((node_features, impute_values.reshape(1,-1)), axis=0)
        
    if len(node_features[obs_mask]) < 1:
        impute_values = np.concatenate((imputable_array, np.array([-1])))
        node_features = np.concatenate((node_features, impute_values.reshape(1,-1)), axis=0)
    
    
    if len(known_mask) < 1:
        known_mask = np.array([0])
        
    if len(unk_mask) < 1:
        unk_mask = np.array([0])
    
    if len(obs_mask) < 1:
        obs_mask = np.array([0])
        
    app_user_edge_index = generate_homogeneous_edges(len(known_mask))
    non_app_user_edge_index = generate_homogeneous_edges(len(unk_mask))
    obs_edge_index = generate_homogeneous_edges(len(obs_mask))
    
    # Features
    hetero_data['app_user'].x = torch.tensor(node_features[known_mask], dtype=torch.float)
    hetero_data['non_app_user'].x = torch.tensor(node_features[unk_mask], dtype=torch.float)
    hetero_data['observation'].x = torch.tensor(node_features[obs_mask], dtype=torch.float)
    hetero_data['target'].x = torch.tensor(target_features, dtype=torch.float)
    
    hetero_data['app_user', 'connect', 'target'].edge_index = app_user_edge_index
    hetero_data['non_app_user', 'connect', 'target'].edge_index = non_app_user_edge_index
    hetero_data['observation', 'connect', 'target'].edge_index = obs_edge_index
        
    return hetero_data


def generate_observations(observations, int_categorical, interaction_types):
  observations[:, 0] /= MAX_DAYS
        
  if int_categorical:
      observations = np.concatenate(
      (observations, -1. * np.ones((observations.shape[0], 2))), axis=1)
      observations = np.concatenate(
          (observations, np.zeros((observations.shape[0], len(interaction_types)))), axis = 1)
      #observations currently = [timestep, result, -1, -1, 0, 0,0]
      
  else:
      observations = np.concatenate(
      (observations, -1. * np.ones((observations.shape[0], 3))), axis=1)
  #observations currently = [timestep, result, -1, -1, -1]
  
  int_obs = observations[:, 4:4 + len(interaction_types) + 1] if int_categorical else observations[:, 2][:, np.newaxis]
  obs = {
          'time': observations[:, 0][:, np.newaxis],
          'outcome': observations[:, 1][:, np.newaxis],
          'age': observations[:, 3][:, np.newaxis],
          'interaction_type': int_obs,
          'app_user_ind': observations[:, 2][:, np.newaxis],
      }
  
  return observations, obs


def normalize_contact_features(data):
  # Normalize timestep, age and pinf
  data[:, 0] /= MAX_DAYS
  data[:, 1] /= 10
  data[:, 2] /= 1024 
  
  
def process_node_features(data, contact_dict, obs, int_categorical, interaction_types):
    if int_categorical:
        initial_features = np.array([
        data['fn_pred'],
        data['user_age'],
        -1.
        ])
        interaction_feature = np.zeros((len(interaction_types)))
        app_user_indicator = np.array([1.])
        target_features = np.concatenate((initial_features, interaction_feature, app_user_indicator))
        target_features = target_features.reshape(1, -1)
        
    else:
        target_features = np.array([
        data['fn_pred'],
        data['user_age'],
        -1.,
        -1.,
        1.
        ])
        target_features = target_features.reshape(1, -1)
    
    node_features = target_features

    if len(contact_dict.keys()) > 0:
        contact_features = np.hstack((
        contact_dict['fn'],
        contact_dict['age'],
        contact_dict['time'],
        contact_dict['interaction_type'],
        contact_dict['app_user_ind']
        ))
        node_features = np.concatenate((node_features, contact_features), axis=0)
        
    if len(obs.keys()) > 0:
        obs_features = np.hstack((
        obs['outcome'],
        obs['age'],
        obs['time'],
        obs['interaction_type'],
        obs['app_user_ind']
        ))
        node_features = np.concatenate((node_features, obs_features), axis=0)
  
    return target_features, node_features


def generate_contact_edges(num_contacts, num_observations):
  # From node index 1 to number of contacts + 1. 
  edges_source = np.arange(1, num_contacts + 1 + num_observations)
  edges_target = [0] * (num_contacts + num_observations)
  
  single_contact_edges = np.vstack([edges_source, edges_target])
  single_contact_edges = torch.tensor(single_contact_edges)
  
  return single_contact_edges


def generate_node_masks(node_features, single_contact_edges, add_root_to_known=True):
  incorporated_nodes = np.unique(single_contact_edges[0]).astype(np.int32)
  # Add 0th node which is the target user as well
  if add_root_to_known:
    incorporated_nodes = np.insert(incorporated_nodes, 0, 0)
  
  unk_mask = np.where((node_features[incorporated_nodes, -1] == 0))[0]
  known_mask = np.where(node_features[incorporated_nodes, -1] == 1)[0]
  obs_mask = np.where(node_features[incorporated_nodes, -1] == -1)[0]
  
  return unk_mask, known_mask, obs_mask


def generate_homogeneous_edges(num_nodes):
  # From node index 1 to number of contacts + 1. 
  edges_source = np.arange(0, num_nodes)
  edges_target = [0] * (num_nodes)
  
  single_contact_edges = np.vstack([edges_source, edges_target])
  single_contact_edges = torch.tensor(single_contact_edges).long()
  
  return single_contact_edges