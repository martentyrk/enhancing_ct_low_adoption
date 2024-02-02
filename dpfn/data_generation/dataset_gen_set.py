import json
import numpy as np
from dpfn import logger
import torch
import argparse
import glob
from pathlib import Path


NUM_FEATURES_PER_CONTACT = 5
  

class NoContacts(Exception):
  """Custom exception for no contacts in a data row."""

class ABMIterableDataset(torch.utils.data.IterableDataset):
  """An iterable dataset that reads from a file."""

  def __init__(self, fname, num_workers=1, verbose=False):
    super(ABMIterableDataset).__init__()

    assert num_workers in [1, 2, 4], "Only 1, 2, or 4 workers supported"
    self.verbose = verbose

    self.fname = fname

    self._num_samples = None
    self._num_workers = num_workers

  def _get_filenames(self):
    """Return the filenames to read from."""
    worker_id = torch.utils.data.get_worker_info().id

    fnames = glob.glob(self.fname)
    if worker_id is not None:
      fnames = fnames[worker_id::self._num_workers]

    return fnames

  def __len__(self):  # pylint: disable=invalid-length-returned
    """Return the number of samples in the dataset."""
    assert False, "Not implemented yet"

  def __getitem__(self, index):
    """Return the item at the given index."""
    del index
    raise NotImplementedError("Not implemented yet")

  def __iter__(self):
    """Return an iterator over the dataset."""
    fnames_worker = self._get_filenames()
    worker_id = torch.utils.data.get_worker_info().id

    if self.verbose:
      logger.info(f"Worker {worker_id} reading from {fnames_worker}")

    def iterable():
      for fname in fnames_worker:
        with open(fname) as f:
          for line in f:
            data = json.loads(line.rstrip('\n'))

            try:
                data = make_features_set(data)
            except NoContacts:
              continue

            yield data

    return iterable()

        
        

def make_features_set(data):
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
    contacts[~app_users_mask, 2] = -1.

    contacts = torch.nn.functional.pad(
      contacts, [0, 0, 0, 900-len(contacts)],
      mode='constant', value=-1.)
    contacts = torch.cat((contacts, observations), dim=0)
    
    return {
        'fn_pred': torch.tensor(data['fn_pred'], dtype=torch.float32),
        'features': contacts,
        'outcome': int(data['sim_state'] == 2 or data['sim_state'] == 1)
    }
  
  
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Compare statistics acrosss inference methods')
    parser.add_argument('--path', type=str, default="dpfn/data/data_all_users/frac_0.4/val/all_35_0.4.jl")
    parser.add_argument('--include_non_users', action='store_true')
    
    args = parser.parse_args()
    logger.info(f'Initializing ABMInMemoryDataset with path: {str(args.path)}')
    dataset = ABMIterableDataset(args.path)
    dataloader_train = torch.utils.data.DataLoader(
        dataset, num_workers=1, batch_size=64)
    
    for features in dataloader_train:
        X = features['features']
        y = features['outcome']
        logger.info(f"Shape of X [N, D]: {X.shape}")
        logger.info(f"Shape of y: {y.shape} {y.dtype}")

        assert X.shape[-1] == 5
        break
