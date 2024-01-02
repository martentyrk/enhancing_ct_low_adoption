"""Utility functions for a dump of dataset for GNNs."""
from dpfn import logger
import dpfn_util
import json
import numpy as np
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
  observations_json = [[] for _ in range(num_users)]
  for observation in observations_now:
    user = int(observation[0])
    timestep = int(observation[1])
    outcome = int(observation[2])
    observations_json[user].append([timestep, outcome])

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
        #Each row is a json file
        output = {
          "fn_pred": float(z_states_inferred[user][-1][2]),
          "user": int(user),
          "sim_state": int(z_states_sim[user]), #in constants.py
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
          ])
        #in pytorch its json.loads
        if user_positive[user]:
          fpos.write(json.dumps(output) + "\n")
        else:
          fneg.write(json.dumps(output) + "\n")
  print(f"Ignored {num_ignored} out of {num_users} users")