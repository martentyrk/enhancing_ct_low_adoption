"""Utility functions for a dump of dataset for GNNs."""
from dpfn import logger
import json
import numpy as np
import os


def dump_graphs(
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
