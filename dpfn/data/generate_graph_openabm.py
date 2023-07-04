"""Dump a graph of the OpenABM simulator."""
# pylint: disable=redefined-outer-name
import argparse
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from dpfn.config import config
from dpfn import constants, logger, util, simulator
import os
import random
import seaborn as sns
from typing import Any, Iterable, List, Dict, Tuple

sns.set_style('whitegrid')


def plot_traces(states: np.ndarray):
  """Plots the simulated traces for easy inspection."""
  # Convert to traces for plotting
  traj_mean = np.zeros((num_time_steps, 4))
  for user in range(num_users):
    # one hot expand
    traj_mean += np.take(np.eye(4), states[user], axis=0)

  sns.set(font_scale=1.5)
  _, axarr = plt.subplots(
    nrows=1, ncols=1, squeeze=False, sharey=True, figsize=(12, 6))
  ax = axarr[0, 0]

  trace_average = traj_mean / num_users
  for seir_state in range(4):
    ax.plot(
      trace_average[:, seir_state], f'{constants.colors[seir_state]}.-',
      label=constants.state_names[seir_state])
    ax.set_title("Simulated forward pass")

  ax.set_xlabel('time')
  axarr[0, 0].set_ylabel('ave p state (over users)')
  axarr[0, 0].legend()
  fname_plot = os.path.join(data_dir, "traces_forward_pass.png")
  plt.savefig(fname_plot, dpi=400)


def analyse_contacts(
    contacts: constants.ContactList,
    num_users: int,
    num_time_steps: int) -> np.ndarray:
  """Returns a matrix of contact counts."""
  contact_count = np.zeros((num_users, num_time_steps), dtype=np.int32)
  for c in contacts:
    contact_count[c[0], c[2]] += 1

  return contact_count


def advance_simulator(
    cfg) -> Tuple[constants.ContactList, constants.ObservationList, np.ndarray]:
  """Advances the simulator all time steps."""
  num_users = cfg.get_value("num_users")
  fraction_test = cfg.get_value("fraction_test")

  # Set conditional distributions for observations
  p_obs_infected = np.array(
    [cfg["model"]["alpha"], 1-float(cfg["model"]["alpha"])], dtype=np.float32)
  p_obs_not_infected = np.array(
    [1-float(cfg["model"]["beta"]), cfg["model"]["beta"]], dtype=np.float32)

  sim = simulator.ABMSimulator(num_time_steps, num_users)
  sim.init_day0(contacts=[])

  states = np.zeros((num_users, num_time_steps), dtype=np.int32)
  obs_rng = np.random.default_rng(seed=2345)

  for t in range(1, num_time_steps):
    users_to_test = np.random.choice(
      num_users, size=int(fraction_test*num_users))
    sim.get_observations_today(
      users_to_test,
      p_obs_infected,
      p_obs_not_infected,
      obs_rng)

    states[:, t] = sim.get_states_today()
    sim.step()

  return sim.get_contacts(), sim.get_observations_all(), states


def contacts_to_json(
    contacts: constants.ContactList) -> Iterable[Dict[str, Any]]:
  for c in contacts:
    yield {
      'u': int(c[0]),
      'v': int(c[1]),
      'time': int(c[2]),
      'features': [int(c[3])]}


def observations_to_json(
    observations: constants.ObservationList) -> Iterable[Dict[str, Any]]:
  for o in observations:
    yield {'u': int(o[0]), 'time': int(o[1]), 'outcome': int(o[2])}


def dump_disk(
    dir_data: str,
    observations: List[Any],
    contacts: List[Any],
    states: np.ndarray):
  """Dumps the dataset to disk in JSON format."""
  fname_obs = os.path.join(dir_data, "observations.json")
  fname_contacts = os.path.join(dir_data, "contacts.json")
  fname_states = os.path.join(dir_data, "states.json")

  logger.info((
    f"Dumping to: \n\t{fname_obs} \n\t{fname_contacts} \n\t{fname_states}"
  ))

  with open(fname_obs, 'w') as fp:
    json.dump(observations, fp)
  with open(fname_contacts, 'w') as fp:
    json.dump(contacts, fp)

  with open(fname_states, 'wb') as fp:
    np.savez(fp, states=states)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Generate graph of contacts')
  parser.add_argument('--config', type=str, default='large_graph_02',
                      help='Name of the config file')
  parser.add_argument('--sample_contact', action='store_true')

  args = parser.parse_args()

  logger.info("Hello world")

  configname = args.config
  fname_config = f"dpfn/config/{configname}.ini"
  data_dir = f"dpfn/data/{configname}/"
  fnames = os.path.join(data_dir, "data*.json")

  if not os.path.exists(fname_config):
    raise FileNotFoundError((
      f"{fname_config} not found. Current wd: {os.getcwd()}"))

  util.maybe_make_dir(data_dir)
  if len(list(glob.glob(os.path.join(data_dir, 'data*.json')))) > 0:
    logger.warning(f"Overwriting datafiles in {data_dir}")

  np.random.seed(2345)
  random.seed(234)

  cfg = config.ConfigBase(fname_config)

  num_time_steps = cfg.get_value("num_time_steps")
  num_users = cfg.get_value("num_users")

  contacts, observations, states = advance_simulator(cfg)

  num_contacts = analyse_contacts(contacts, num_users, num_time_steps)
  num_contacts_mean = np.mean(num_contacts)
  num_contacts_total = np.sum(num_contacts)
  logger.info(f"Average number of overall contacts per day {num_contacts_mean}")

  R0_estimate = (num_contacts_mean/cfg.get_value("num_users")
                 * 1/cfg.get_value("prob_h")
                 * cfg.get_value("p1"))
  logger.info(f"Estimate for R0: {R0_estimate:5.3f}")

  logger.info((
    f"Total contacts {num_contacts_total} over {num_users} users in "
    f"{num_time_steps} steps"))

  plot_traces(states)

  pir = np.max(np.mean(states == 2, axis=0))
  logger.info(f"Peak infection rate: {pir:5.3f}")

  dump_disk(
    data_dir,
    list(observations_to_json(observations)),
    list(contacts_to_json(contacts)),
    states,
  )
