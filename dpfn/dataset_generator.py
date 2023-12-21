import argparse
import numpy as np
import os
from dpfn import logger
import json
import torch
import constants
from torch_geometric.data import Data
from torch_geometric.data.in_memory_dataset import collate

NUM_FEATURES_PER_CONTACT = 4

class NoContacts(Exception):
  """Custom exception for no contacts in a data row."""


def make_features_graph(data):
  """Converts the JSON to the graph features."""
  # TODO(rob): make flag to indicate use of age feature!
  
  contacts = np.array(data['contacts'], dtype=np.int64)
  #Contacts object: [timestep, sender, age (age groups), pinf, interaction type]
  
  if len(contacts) == 0:
    contacts = -1 * torch.ones(size=(constants.CTC, NUM_FEATURES_PER_CONTACT), dtype=torch.float32)

  else:
    # Remove sender information
    contacts = np.concatenate((contacts[:, 0:1], contacts[:, 2:]), axis=1)
    contacts = torch.tensor(contacts, dtype=torch.float32)


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
    'outcome': np.float32(data['sim_state'] == 2 or data['sim_state'] == 1)
  }, single_contact_edges, edge_attributes)


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    description='Compare statistics acrosss inference methods')
  parser.add_argument('--path', type=str, default="results_temp/train_data/final")
  
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
  torch.save(collate(data_list), str(args.path + '/only_app_users_data.pt'))
  logger.info('File saved!')
  
  