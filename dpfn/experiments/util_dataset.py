"""Utility functions for a dump of dataset for GNNs."""
from dpfn import logger
import dpfn_util
import json
import numpy as np
import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import add_self_loops
import shutil
import os


def dump_features_flat(
        contacts_now: np.ndarray,
        observations_now: np.ndarray,
        z_states_inferred: np.ndarray,
        z_states_sim: np.ndarray,
        contacts_age: np.ndarray,
        users_age: np.ndarray,
        trace_dir: str,
        num_users: int,
        t_now: int) -> None:
    """Dump graphs for GNNs.

    Args:
      contacts_now: Contacts at current time.
      observations_now: Observations at current time.
      z_states_inferred: Inferred latent states.
      z_states_sim: States according to the simulator.
      contacts_age: Contacts age.
      users_age: Users age.
      trace_dir: Directory to dump graphs.
      num_users: Number of users.
      t_now: Current time.
    """
    # Assertions to ensure datatypes
    assert len(contacts_now.shape) == 2
    assert len(observations_now.shape) == 2

    assert len(z_states_inferred.shape) == 3
    assert len(z_states_sim.shape) == 1

    assert len(contacts_age.shape) == 2
    assert len(users_age.shape) == 1

    # Filenames
    fname_train = os.path.join(trace_dir, "train.jl")
    fname_val = os.path.join(trace_dir, "val.jl")
    fname_test = os.path.join(trace_dir, "test.jl")

    if t_now < 14:
        return

    if t_now == 14:
        # Clear dataset
        with open(fname_train, "w") as f:
            f.write("")
        with open(fname_val, "w") as f:
            f.write("")
        with open(fname_test, "w") as f:
            f.write("")
    else:
        logger.info(f"Dump graph dataset at day {t_now}")

    # Initialize data
    dataset = [
        {'contacts': [], 'observations': [], 't_now': t_now}
        for _ in range(num_users)]

    for user in range(num_users):
        dataset[user]["contacts_age_50"] = int(contacts_age[user][0])
        dataset[user]["contacts_age_80"] = int(contacts_age[user][1])
        dataset[user]["users_age"] = int(users_age[user])
        dataset[user]["fn_pred"] = float(z_states_inferred[user, -1, 2])
        dataset[user]["sim_state"] = float(z_states_sim[user])

    # Figure out riskscore of contacts
    for contact in contacts_now:
        user_u = int(contact[0])
        user_v = int(contact[1])
        timestep = int(contact[2])
        dataset[user_v]["contacts"].append(
            float(z_states_inferred[user_u, timestep, 2]))

    for user in range(num_users):
        if len(dataset[user]["contacts"]) == 0:
            dataset[user]["riskscore_contact_max"] = 0.0
            dataset[user]["riskscore_contact_median"] = 0.0
            continue

        score_max = np.max(dataset[user]["contacts"])
        score_median = np.median(dataset[user]["contacts"])
        dataset[user]["riskscore_contact_max"] = float(score_max)
        dataset[user]["riskscore_contact_median"] = float(score_median)

    # Dump dataset to file
    with open(fname_train, 'a') as f_train:
        with open(fname_val, 'a') as f_val:
            with open(fname_test, 'a') as f_test:
                for user in range(num_users):
                    if user < int(0.8*num_users):
                        f = f_train
                    elif user < int(0.9*num_users):
                        f = f_val
                    else:
                        f = f_test

                    f.write(json.dumps(dataset[user]) + "\n")


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
        rng_seed: int) -> None:
    """Dump graphs for GNNs."""
    datadump = dpfn_util.fn_features_dump(
        num_workers=1,
        num_users=num_users,
        num_time_steps=num_time_steps,
        q_marginal=z_states_inferred[:, :, 2],
        contacts=contacts_now,
        users_age=users_age)

    # Process observations
    observations_json = observations_json = create_observations_json(
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
    with open(fname_pos, 'w') as fpos:
        with open(fname_neg, 'w') as fneg:
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
                    ])
                # in pytorch its json.loads
                if user_positive[user]:
                    fpos.write(json.dumps(output) + "\n")
                else:
                    fneg.write(json.dumps(output) + "\n")
    print(f"Ignored {num_ignored} out of {num_users} users")


def inplace_features_graph_creation(
    contacts_now: np.ndarray,
    observations_now: np.ndarray,
    z_states_inferred: np.ndarray,
    user_free: np.ndarray,
    users_age: np.ndarray,
    app_users: np.ndarray,
    num_users: int,
    num_time_steps: int,
):

    model_data = []

    observations_json = create_observations_json(num_users, observations_now)

    datadump = dpfn_util.fn_features_dump(
        num_workers=1,
        num_users=num_users,
        num_time_steps=num_time_steps,
        q_marginal=z_states_inferred[:, :, 2],
        contacts=contacts_now,
        users_age=users_age)

    for user in range(num_users):
        if user_free[user] == 0 or app_users[user] == 0:
            continue

        output = {
            'fn_pred': float(z_states_inferred[user][-1][2]),
            "user": int(user),
            "user_age": int(users_age[user]),
            "observations": observations_json[user],
            "contacts": []
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


def create_dataset(data):
    data_list = []

    for single_data in data:
        try:
            single_user, single_edge_index, observations = make_features_graph(
                single_data, True)
        except NoContacts:
            continue

        # Initialize node_features with the target user who is affected.
        # Features for a single node [fn_pred, age, time, interaction type, target user or not, observation or not, app user
        node_features = np.array(
            [single_user['fn_pred'], single_user['user_age'], -1, -1, 1, 0, 1])
        node_features = node_features.reshape(1, -1)

        if len(single_user['contacts'] > 0):
            contact_features = np.concatenate((single_user['contacts'][:, [2, 1, 0, 3]], np.zeros(
                (single_user['contacts'].shape[0], 2)), single_user['contacts'][:, [4]]), axis=1)
            node_features = np.concatenate(
                (node_features, contact_features), axis=0)

        if len(observations) > 0:
            # Reorder observations to match other node features
            # new orderding = [outcome, -1 (age), time, -1 (interaction type), 0 (not target user), 1 (observation type), -1 (app users)]

            obs_features = observations[:, [1, 4, 0, 5, 2, 3, 6]]
            node_features = np.concatenate(
                (node_features, obs_features), axis=0)

        data_single = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=single_edge_index.long(),
        )

        data_list.append(data_single)

    return data_list


def make_features_graph(data, include_non_users: bool):
    """Converts the JSON to the graph features."""
    # Contacts object: [timestep, sender, age (age groups), pinf, interaction type, app_user (1 or 0)]
    contacts = np.array(data['contacts'], dtype=np.int64)

    # Normalize user data
    # TODO: check if this turns into float
    data['user_age'] /= 10

    # Observations object: [timestep, result]
    observations = np.array(data['observations'], dtype=np.int64)
    if len(observations) == 0:
        observations = np.array([])
    else:
        # observations = torch.tensor(observations, dtype=torch.float32)
        observations = np.concatenate(
            (observations, np.zeros((observations.shape[0], 1))), axis=1)
        observations = np.concatenate(
            (observations, np.ones((observations.shape[0], 1))), axis=1)
        observations = np.concatenate(
            (observations, -1. * np.ones((observations.shape[0], 3))), axis=1)

    if len(contacts) == 0:
        # contacts = -1 * torch.ones(size=(constants.CTC, NUM_FEATURES_PER_CONTACT), dtype=torch.float32)
        return ({
            'fn_pred': torch.tensor(data['fn_pred'], dtype=torch.float32),
            'user_age': torch.tensor(data['user_age'], dtype=torch.float32),
            'contacts': contacts,
        }, torch.tensor(np.array([[], []])), observations)
    else:
        # Remove sender information
        contacts = np.concatenate((contacts[:, 0:1], contacts[:, 2:]), axis=1)
        contacts = torch.tensor(contacts, dtype=torch.float32)

    # Column 0 is the timestep

    contacts[:, 1] /= 10  # Column 1 is the age
    contacts[:, 2] /= 1024  # Column 2 is the pinf according to FN

    if include_non_users:
        # We know timestep and interaction type, other values will be set to -1.
        app_users_mask = contacts[:, -1] == 1
        contacts[~app_users_mask, 1] = -1.
        contacts[~app_users_mask, 2] = -1.

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
    }, single_contact_edges, observations)
