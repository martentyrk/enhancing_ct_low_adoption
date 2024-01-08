import argparse
import numpy as np
import os
from dpfn import logger
import json
import torch
import constants
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_dense_adj, add_self_loops

NUM_FEATURES_PER_CONTACT = 4

class NoContacts(Exception):
  """Custom exception for no contacts in a data row."""
  
  
class ABMDataset(Dataset):
    def __init__(self, data_list):
        super(ABMDataset, self).__init__()

        self.data_list = data_list

        self.dim_node_features = 2
        self.dim_edge_features = 2


        self.num_labels = 2
        self.labels = torch.tensor([0.0, 1.0])

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def make_features_graph(data):
  """Converts the JSON to the graph features."""
  # TODO(rob): make flag to indicate use of age feature!
  
  contacts = np.array(data['contacts'], dtype=np.int64)
  #Contacts object: [timestep, sender, age (age groups), pinf, interaction type]
  observations = np.array(data['observations'], dtype=np.int64)
  
  if len(contacts) == 0:
    contacts = -1 * torch.ones(size=(constants.CTC, NUM_FEATURES_PER_CONTACT), dtype=torch.float32)

  else:
    # Remove sender information
    contacts = np.concatenate((contacts[:, 0:1], contacts[:, 2:]), axis=1)
    contacts = torch.tensor(contacts, dtype=torch.float32)

    contacts = torch.nn.functional.pad(
      contacts, [0, 0, 0, constants.CTC-len(contacts)],
      mode='constant', value=-1.)


  if len(observations) == 0:
    observations = -1 * torch.ones(size=(14, 2), dtype=torch.float32)
  else:
    observations = torch.tensor(observations, dtype=torch.float32)

    observations = torch.nn.functional.pad(
    observations, [0, 2, 0, 14-len(observations)],
    mode='constant', value=-1.)

  # Concatenate the contacts and observations
  #TODO: what to do here?
  # contacts = torch.cat((contacts, observations), dim=0)


  # Column 0 is the timestep
  data['user_age'] /= 10
  contacts[:, 1] /= 10  # Column 1 is the age
  contacts[:, 2] /= 1024  # Column 2 is the pinf according to FN
  edge_attributes = []
  edges_source = np.arange(1, contacts.shape[0] + 1)
  edges_target = [0] * contacts.shape[0]
  
  
  #Edge attributes = timestep, interaction type
  edge_attributes = contacts[:, [0, 3]]
      
  single_contact_edges = np.vstack([edges_source, edges_target])

  
  return ({
    'fn_pred': torch.tensor(data['fn_pred'], dtype=torch.float32),
    'user_age': torch.tensor(data['user_age'], dtype=torch.float32),
    'contacts': contacts,
    'outcome': torch.tensor(data['sim_state'] == 2 or data['sim_state'] == 1, dtype=torch.float32)
  }, single_contact_edges, edge_attributes)


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    description='Compare statistics acrosss inference methods')
  parser.add_argument('--path', type=str, default="dpfn/data/val_app_users/partial")
  parser.add_argument('--out_name', type=str, default="val_dataset.pt")
  
  args = parser.parse_args()
  
  data_list = []
  for subdir, dirs, files in os.walk(args.path):
    for file in files:
      filename = str(subdir + '/' + file)
      logger.info(f'Processing the file: {filename}')
      with open(filename) as f:
        for line in f:
          line_data = json.loads(line.rstrip('\n'))
          try:
            single_user, single_edge_index, single_edge_attr = make_features_graph(line_data)

          except NoContacts:
            continue
          
          #Labels
          y = single_user['outcome']

          # Initialize node_features with the target user who is affected.
          node_features = np.array([single_user['fn_pred'], single_user['user_age']])

          for contact in single_user['contacts']:
            # Add all contacts, ordering does not matter since these contacts in the current
            # graph are permutation invariant. Only node that has a fixed position
            # is the first one and should be at index = 0.
            node_features = np.vstack([node_features, np.array([contact[2], contact[1]])])

          single_edge_index = torch.tensor(single_edge_index, dtype=torch.long).contiguous()
          
          data_single = Data(x=node_features, edge_index=single_edge_index, y=y, edge_attr=single_edge_attr)
          del single_edge_index # Free up some memory.

          data_list.append(data_single)
  
  logger.info('Processing complete, saving file.')
  #Save file
  dataset = ABMDataset(data_list)
  torch.save(dataset, str(args.path + '/' + args.out_name))
  logger.info('File saved!')
  
  