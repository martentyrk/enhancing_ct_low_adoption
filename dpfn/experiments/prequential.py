"""Experiments related to sequential predicton and simulation."""
import datetime
import json
import numpy as np
import os
import socket
from typing import Any, Dict, Tuple


def dump_results(
    datadir: str, **kwargs):
  fname = os.path.join(datadir, "prec_recall_ir.npz")
  with open(fname, 'wb') as fp:
    np.savez(fp, **kwargs)


def dump_results_json(
    datadir: str,
    cfg: Dict[str, Any],
    **kwargs):
  """Dumps the results of an experiment to JSONlines."""
  kwargs["time"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  kwargs["time_day"] = datetime.datetime.now().strftime("%Y%m%d")
  kwargs["slurm_id"] = str(os.getenv('SLURM_JOB_ID'))  # Defaults to 'none'
  kwargs["slurm_name"] = str(os.getenv('SLURM_JOB_NAME'))  # Defaults to 'none'
  kwargs["sweep_id"] = str(os.getenv('SWEEPID'))  # Defaults to 'none'
  kwargs["hostname"] = socket.gethostname()

  for key in cfg["model"].keys():
    kwargs[f"model.{key}"] = cfg["model"].get(key, -1.)

  for key in cfg["data"].keys():
    kwargs[f"data.{key}"] = cfg["data"].get(key, -1.)

  fname = os.path.join(datadir, "results.jl")
  with open(fname, 'a') as fp:
    fp.write(json.dumps(kwargs) + "\n\r")


def get_observations_one_day(
    states: np.ndarray,
    users_to_observe: np.ndarray,
    num_obs: int,
    timestep: int,
    p_obs_infected: np.ndarray,
    p_obs_not_infected: np.ndarray,
    obs_rng: np.random._generator.Generator) -> np.ndarray:
  """Makes observations for tests on one day.

  Args:
    states: The states of the users, should be in values {0, 1, 2, 3},
      array of length num_users.
    users_to_observe: The users to observe, array of length num_obs.
    num_obs: The number of observations to make.
    timestep: The timestep of the observations.
    p_obs_infected: The probability of a positive test for an infected user.
    p_obs_not_infected: The probability of a positive test for a not infected.
    obs_rng: Random number generator to ensure reproducibility for a fixed seed.

  Returns:
    The observations, array of shape (num_obs, 3), where the columns are (user,
      timestep, outcome)
  """
  if num_obs < 1:
    return np.zeros((0, 3), dtype=np.int32)

  assert len(states.shape) == 1

  observations = np.zeros((num_obs, 3), dtype=np.int32)

  assert np.abs(p_obs_infected[0] + p_obs_infected[1] - 1.) < 0.001
  assert np.abs(p_obs_not_infected[0] + p_obs_not_infected[1] - 1.) < 0.001

  states_user = states[users_to_observe]
  positive = np.logical_or(states_user == 2, states_user == 1)

  sample_prob = np.where(positive, p_obs_infected[1], p_obs_not_infected[1])

  assert sample_prob.shape == (num_obs, )

  observations[:, 0] = users_to_observe
  observations[:, 1] = timestep
  observations[:, 2] = sample_prob >= obs_rng.random(num_obs)

  return observations.astype(np.int32)


def calc_prec_recall(
    states: np.ndarray, users_quarantine: np.ndarray) -> Tuple[float, float]:
  """Calculates precision and recall for quarantine assignments.

  Note that when no users are in state E or I, the recall is 1.0 (achieved by
  seting epsilon to a small number).
  """
  assert len(states.shape) == 1
  eps = 1E-9  # Small number to avoid division by zero

  states_e_i = np.logical_or(
    states == 1,
    states == 2,
  )

  true_positives = np.sum(
    np.logical_and(states_e_i, users_quarantine)) + eps

  precision = true_positives / (np.sum(users_quarantine) + eps)
  recall = true_positives / (np.sum(states_e_i) + eps)
  return precision, recall


def get_evidence_obs(
    observations: np.ndarray,
    z_states: np.ndarray,
    alpha: float,
    beta: float) -> float:
  """Calculates evidence for the observations, integrating out the states."""
  p_obs_infected = [alpha, 1-alpha]
  p_obs_not_infected = [1-beta, beta]

  log_like = 0.
  for obs in observations:
    user, timestep, outcome = obs[0], obs[1], obs[2]
    p_inf = z_states[user, timestep, 2]

    log_like += np.log(
      p_inf*p_obs_infected[outcome] + (1-p_inf)*p_obs_not_infected[outcome]
      + 1E-9)
  return log_like


def decide_tests(
    scores_infect: np.ndarray,
    num_tests: int,
    user_ids: np.ndarray) -> np.ndarray:
  
  assert num_tests < len(scores_infect)
  sort_indeces = np.argsort(scores_infect)
  # Sort the list of user_ids based on scores, then assign tests to the num tests highest
  users_to_test = sort_indeces[user_ids][-num_tests:]
  return users_to_test.astype(np.int32)


def generate_app_users(num_users: int, users_ages: np.ndarray, app_users_fraction: np.float32):
  '''
  From the population randomly pick a subset of people to be app users from each age group
  
  Returns a binary array where 1 denotes that at this index individual is an app user and 0 denotes
  that they are not using the app.
  '''
  if app_users_fraction == 1.0:
    return np.ones((num_users), dtype=np.int32)
  
  population_array = np.arange(0, num_users)
  app_user_ids_binary = np.zeros((num_users), dtype=np.int32)
  user_age_groups = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

  app_user_ids = np.array([], dtype=np.int32)
  for age in user_age_groups:
      age_group_filtered = population_array[np.argwhere(users_ages == age)].flatten()          
      amount_per_group = int(np.ceil(app_users_fraction * age_group_filtered.shape[0]))
      choose_age_users = np.random.choice(age_group_filtered, amount_per_group, replace=False)
      app_user_ids = np.append(app_user_ids, choose_age_users)
    
  app_user_ids = np.sort(app_user_ids)
  
  
  app_user_ids_binary[app_user_ids] = 1
  return app_user_ids_binary
