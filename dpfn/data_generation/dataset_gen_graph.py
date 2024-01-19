import json
import numpy as np
from dpfn import logger
import torch
import argparse
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import add_self_loops
from pathlib import Path


NUM_FEATURES_PER_CONTACT = 5
  

class NoContacts(Exception):
  """Custom exception for no contacts in a data row."""

class ABMInMemoryDataset(InMemoryDataset):
    def __init__(self, root):
        super(ABMInMemoryDataset, self).__init__(root)
        self.load(self.processed_paths[0])
        # self.num_obs_features = self._get_num_obs_features()

    @property
    def raw_file_names(self):
        # Define the raw data file names (if any)
        return ['all.jl']

    @property
    def processed_file_names(self):
        return ['data.pt']
      
    def download(self):
        # Implement the code to download the raw data (if needed)
        pass

    def process(self):
        data_list=[]

        jl_pths = [pth for pth in Path(self.root).iterdir()
                    if pth.suffix == '.jl']
        
        for pth in jl_pths:    
          logger.info(f'Processing the file: {pth}')
                
          with open(pth) as f:
              for line in f:
                    line_data = json.loads(line.rstrip('\n'))
                    
                    try:
                        single_user, single_edge_index, observations = make_features_graph(line_data)
                    except NoContacts:
                        continue
                    
                    #Labels
                    y = single_user['outcome']
                    
                    # Initialize node_features with the target user who is affected.
                    # Features for a single node [fn_pred, age, time, interaction type, target user or not, observation or not
                    node_features = np.array([single_user['fn_pred'], single_user['user_age'], -1, -1, 1, 0])
                    node_features = node_features.reshape(1, -1)
                    
                    if len(single_user['contacts'] > 0):
                        contact_features = np.concatenate((single_user['contacts'][:, [2, 1, 0, 3]], np.zeros((single_user['contacts'].shape[0], 2))),axis = 1)
                        node_features = np.concatenate((node_features, contact_features), axis=0)
                    
                    if len(observations) > 0:
                        # Reorder observations to match other node features
                        # new orderding = [outcome, -1 (age), time, -1 (interaction type), 0 (not user), 1 (observation type)]
                        obs_features = observations[:, [1, 4, 0, 5, 2, 3]]
                        node_features = np.concatenate((node_features, obs_features), axis=0)
                    

                    data_single = Data(
                    x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=single_edge_index.long(),
                    y=torch.tensor(y).long(),
                    )

                    data_list.append(data_single)

        logger.info('Processing complete, saving file.')
        self.save(data_list, self.processed_paths[0])
        

        
        

def make_features_graph(data):
    """Converts the JSON to the graph features."""
    #Contacts object: [timestep, sender, age (age groups), pinf, interaction type]
    contacts = np.array(data['contacts'], dtype=np.int64)
    
    #Normalize user data
    data['user_age'] /= 10

    # Observations object: [timestep, result]
    observations = np.array(data['observations'], dtype=np.int64)
    if len(observations) == 0:
        observations = np.array([])
    else:
        # observations = torch.tensor(observations, dtype=torch.float32)
        observations = np.concatenate((observations, np.zeros((observations.shape[0], 1))), axis=1)
        observations = np.concatenate((observations, np.ones((observations.shape[0], 1))), axis=1)
        observations = np.concatenate((observations, -1. * np.ones((observations.shape[0], 2))), axis=1)
        
  
    if len(contacts) == 0:
        # contacts = -1 * torch.ones(size=(constants.CTC, NUM_FEATURES_PER_CONTACT), dtype=torch.float32)
        return ({
        'fn_pred': torch.tensor(data['fn_pred'], dtype=torch.float32),
        'user_age': torch.tensor(data['user_age'], dtype=torch.float32),
        'contacts': contacts,
        'outcome': torch.tensor(data['sim_state'] == 2 or data['sim_state'] == 1, dtype=torch.float32)
    }, torch.tensor(np.array([[],[]])), observations)
    else:
        # Remove sender information
        contacts = np.concatenate((contacts[:, 0:1], contacts[:, 2:]), axis=1)    
        contacts = torch.tensor(contacts, dtype=torch.float32)


    # Column 0 is the timestep
    
    contacts[:, 1] /= 10  # Column 1 is the age
    contacts[:, 2] /= 1024  # Column 2 is the pinf according to FN
    # edge_attributes = []
    num_contacts = len(contacts)
    num_observations = len(observations)
    edges_source = np.arange(1, num_contacts + 1 + num_observations)
    edges_target = [0] * (num_contacts + num_observations)
    
    single_contact_edges = np.vstack([edges_source, edges_target])
    single_contact_edges = torch.tensor(single_contact_edges)
    single_contact_edges, _ = add_self_loops(single_contact_edges)
    
    return ({
        'fn_pred': torch.tensor(data['fn_pred'], dtype=torch.float32),
        'user_age': torch.tensor(data['user_age'], dtype=torch.float32),
        'contacts': contacts,
        'outcome': int(data['sim_state'] == 2 or data['sim_state'] == 1)
    }, single_contact_edges, observations)
  
  
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Compare statistics acrosss inference methods')
    parser.add_argument('--path', type=str, default="dpfn/data/train_app_users/partial")

    args = parser.parse_args()
    logger.info('Initializing ABMInMemoryDataset with path: %s', str(args.path))
    dataset = ABMInMemoryDataset(args.path)
    
    logger.info('File saved!')