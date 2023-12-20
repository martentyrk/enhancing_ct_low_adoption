import argparse
import numpy as np
import os
import random
import pandas as pd
from experiments import (util_experiments)
from dpfn import util
import json
import torch
import constants
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data.in_memory_dataset import collate

class NoContacts(Exception):
  """Custom exception for no contacts in a data row."""


def make_features_graph(data):
  """Converts the JSON to the graph features."""
  # TODO(rob): make flag to indicate use of age feature!
  
  contacts = np.array(data['contacts'], dtype=np.int64)
  #Contacts object: [timestep, sender, age (age groups), pinf, interaction type]
  
  if len(contacts) == 0:
    contacts = -1 * torch.ones(size=(constants.CTC, 3), dtype=torch.float32)

  else:
    # Remove sender information
    contacts = np.concatenate((contacts[:, 0:1], contacts[:, 2:]), axis=1)
    contacts = torch.tensor(contacts, dtype=torch.float32)


  # Column 0 is the timestep
  contacts[:, 2] /= 10  # Column 2 is the age
  contacts[:, 3] /= 1024  # Column 3 is the pinf according to FN
  edge_attributes = []
  edges_source = np.arange(1, contacts.shape[0] + 1)
  edges_target = [0] * len(contacts.shape[0])
  
  
  #Edge attributes = timestep, interaction type
  edge_attributes = np.concatenate((contacts[:, 0] ,contacts[:, 4]), axis=1)
      
  single_contact_edges = np.vstack([edges_source, edges_target])

  
  return ({
    'fn_pred': torch.tensor(data['fn_pred'], dtype=torch.float32),
    'user': torch.tensor(data['user'], dtype=torch.int32),
    'user_age': torch.tensor(data['user_age'], dtype=torch.int32),
    'contacts': contacts,
    'outcome': np.float32(data['sim_state'] == 2 or data['sim_state'] == 1)
  }, single_contact_edges, edge_attributes)


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    description='Compare statistics acrosss inference methods')
  parser.add_argument('--path', type=str, default="data_results/")
  
  args = parser.parse_args()

  
  data_list = []
  for subdir, dirs, files in os.walk(args.path):
    for file in files:
      if file == 'all.jl':
        with open(file) as f:
          for line in f:
            line_data = json.loads(line.rstrip('\n'))
            try:
              single_user, single_edge_index, single_edge_attr = make_features_graph(line_data)

            except NoContacts:
              continue
            
            single_dataframe = pd.DataFrame(single_user)
            
            labels = single_dataframe[['outcome']]
            y = labels.to_numpy()
            
            node_features = single_dataframe[['pinf', 'user_age']]
            for contact in single_user['contacts']:
              temp_contact_dataframe = pd.DataFrame({'user': contact[1],'pinf': contact[3], 'user_age': contact[2]})
              node_features = pd.concat([node_features, temp_contact_dataframe])
                
            single_edge_index = torch.tensor(single_edge_index, dtype=torch.long).t().contiguous()
            data_single = Data(x=node_features.to_numpy(), edge_index=single_edge_index, y=y, edge_attr=single_edge_attr)

            data_list.append(data_single)
              
  #Save file
  torch.save(collate(data_list), str(args.path + 'only_app_users_data.pt'))
  
  