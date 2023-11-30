"""Utility functions for inference in CRISP-like models."""

import contextlib
import functools
import itertools
import time  # pylint: disable=unused-import
import math
from dpfn import logger
import numba
import numpy as np
import os
from typing import Any, Iterable, List, Tuple, Union


@numba.njit
def get_past_contacts_fast(
    user_interval: Tuple[int, int],
    contacts: np.ndarray) -> Tuple[np.ndarray, int]:
  """Returns past contacts as a NumPy array, for easy pickling."""
  num_users_int = user_interval[1] - user_interval[0]

  if len(contacts) == 0:
    return -1 * np.ones((num_users_int, 1, 2), dtype=np.int32), 0
  if contacts.shape[1] == 0:
    return -1 * np.ones((num_users_int, 1, 2), dtype=np.int32), 0

  contacts_past = [[(-1, -1)] for _ in range(num_users_int)]

  # First find all contacts that are in the interval
  for contact in contacts:
    user_v = contact[1]
    if user_interval[0] <= user_v < user_interval[1]:
      datum = (contact[2], contact[0])
      contacts_past[user_v - user_interval[0]].append(datum)

  # Then construct Numpy array
  max_messages = max(map(len, contacts_past)) - 1  # Subtract 1 to make clear!
  pc_tensor = -1 * np.ones((num_users_int, max_messages + 1, 2), dtype=np.int32)
  for i in range(num_users_int):
    num_contacts = len(contacts_past[i])
    if num_contacts > 1:
      pc_array = np.array(contacts_past[i][1:], dtype=np.int32)
      pc_tensor[i][:num_contacts-1] = pc_array

  return pc_tensor, max_messages


@numba.njit(
  'Tuple((int32[:, :, :], int64))(int64[:], int32[:, :], int64)')
def get_past_contacts_static(
    user_ids: np.ndarray,
    contacts: np.ndarray,
    num_msg: int) -> Tuple[np.ndarray, int]:
  """Returns past contacts as a NumPy array, for easy pickling."""
  num_users_int = user_ids.shape[0]

  if len(contacts) == 0:
    return (-1 * np.ones((num_users_int, 1, 2))).astype(np.int32), 0
  if contacts.shape[1] == 0:
    return (-1 * np.ones((num_users_int, 1, 2))).astype(np.int32), 0

  contacts_past = -1 * np.ones((num_users_int, num_msg, 2), dtype=np.int32)

  contacts_counts = np.zeros(num_users_int, dtype=np.int32)

  for contact in contacts:
    user_v = contact[1]
    if user_v in user_ids:
      contact_rel = np.where(user_ids == user_v)[0][0]
      contact_count = contacts_counts[contact_rel] % (num_msg - 1)
      contacts_past[contact_rel, contact_count] = np.array(
        (contact[2], contact[0]), dtype=np.int32)

      contacts_counts[contact_rel] += 1

  return contacts_past.astype(np.int32), int(np.max(contacts_counts))


def state_at_time(days_array, timestamp):
  """Calculates the SEIR state at timestamp given the Markov state.

  Note that this function is slower than 'state_at_time_cache' when evaluating
  only one data point.
  """
  if isinstance(days_array, list):
    days_array = np.array(days_array, ndmin=2)
  elif len(days_array.shape) == 1:
    days_array = np.expand_dims(days_array, axis=0)

  days_cumsum = np.cumsum(days_array, axis=1)
  days_binary = days_cumsum <= timestamp

  # Append vector of 1's such that argmax defaults to 3
  days_binary = np.concatenate(
    (days_binary, np.zeros((len(days_binary), 1))), axis=1)
  return np.argmin(days_binary, axis=1).astype(np.int32)


# @functools.lru_cache()  # Using cache gives no observable speed up for now
def state_at_time_cache(t0: int, de: int, di: int, t: int) -> int:
  if t < t0:
    return 0
  if t < t0+de:
    return 1
  if t < t0+de+di:
    return 2
  return 3


@numba.njit('float32[:, :](UniTuple(int64, 2), float32[:, :, :], int32[:, :])')
def calc_c_z_u(
    user_interval: Tuple[int, int],
    obs_array: np.ndarray,
    observations: np.ndarray) -> np.ndarray:
  """Precompute the Cz terms.

  Args:
    user_interval: Tuple of (start, end) user indices.
    obs_array: Array in [num_time_steps, num_sequences, 2] of observations.
      obs_array[t, :, i] is about the log-likelihood of the observation being
      i=0 or i=1, at time step t.
    observations: Array in [num_observations, 3] of observations.

  Notation follows the original CRISP paper.
  """
  interval_num_users = user_interval[1] - user_interval[0]
  log_prob_obs = np.zeros(
    (interval_num_users, obs_array.shape[1]), dtype=np.float32)

  num_days = obs_array.shape[0]

  if observations.shape[1] > 1:
    # Only run if there are observations
    for obs in observations:
      user_u = obs[0]

      if user_interval[0] <= user_u < user_interval[1]:
        assert obs[1] < num_days
        #obs[1] == time obs[2] == test result, 0 = neg; 1 = pos
        # log_prob_obs gets assigned all possible options that this user might be
        # in and then the probability for that result being true(?)
        log_prob_obs[user_u - user_interval[0]] += obs_array[obs[1], :, obs[2]]

  return log_prob_obs


def calc_log_a_start(
    seq_array: np.ndarray,
    probab_0: float,
    g: float,
    h: float) -> np.ndarray:
  """Calculate the basic A terms.

  This assumes no contacts happen. Thus the A terms are simple Geometric
  distributions. When contacts do occur, subsequent code would additional log
  terms.
  """
  if isinstance(seq_array, list):
    seq_array = np.stack(seq_array, axis=0)

  num_sequences = seq_array.shape[0]
  log_A_start = np.zeros((num_sequences))

  time_total = np.max(np.sum(seq_array, axis=1))

  # Due to time in S
  # Equation 17
  term_t = (seq_array[:, 0] >= time_total).astype(np.float32)
  log_A_start += (seq_array[:, 0]-1) * np.log(1-probab_0)

  # Due to time in E
  term_e = ((seq_array[:, 0] + seq_array[:, 1]) >= time_total).astype(np.int32)
  log_A_start += (1-term_t) * (
    (seq_array[:, 1]-1)*np.log(1-g) + (1-term_e)*np.log(g)
  )

  # Due to time in I
  term_i = (seq_array[:, 0] + seq_array[:, 1] + seq_array[:, 2]) >= time_total
  log_A_start += (1-term_e) * (
    (seq_array[:, 2]-1)*np.log(1-h) + (1-term_i.astype(np.int32))*np.log(h))
  return log_A_start


def iter_state(ts, te, ti, tt):
  yield from itertools.repeat(0, ts)
  yield from itertools.repeat(1, te)
  yield from itertools.repeat(2, ti)
  yield from itertools.repeat(3, tt-ts-te-ti)


def state_seq_to_time_seq(
    state_seqs: Union[np.ndarray, List[List[int]]],
    time_total: int) -> np.ndarray:
  """Unfolds trace tuples to full traces of SEIR.

  Args:
    state_seqs: np.ndarray in [num_sequences, 3] with columns for t_S, t_E, t_I
    time_total: total amount of days. In other words, each row sum is
      expected to be less than time_total, and including t_R should equal
      time_total.

  Returns:
    np.ndarray of [num_sequences, time_total], with values in {0,1,2,3}
  """
  iter_state_partial = functools.partial(iter_state, tt=time_total)
  iter_time_seq = map(list, itertools.starmap(iter_state_partial, state_seqs))
  return np.array(list(iter_time_seq))


def state_seq_to_hot_time_seq(
    state_seqs: Union[np.ndarray, List[List[int]]],
    time_total: int) -> np.ndarray:
  """Unfolds trace tuples to one-hot traces of SEIR.

  Args:
    state_seqs: np.ndarray in [num_sequences, 3] with columns for t_S, t_E, t_I
    time_total: total amount of days. In other words, each row sum is
      expected to be less than time_total, and including t_R should equal
      time_total.

  Returns:
    np.ndarray of [num_sequences, time_total, 4], with values in {0,1}
  """
  iter_state_partial = functools.partial(iter_state, tt=time_total)
  iter_time_seq = map(list, itertools.starmap(iter_state_partial, state_seqs))

  states = np.zeros((len(state_seqs), time_total, 4))
  for i, time_seq in enumerate(iter_time_seq):
    states[i] = np.take(np.eye(4), np.array(time_seq), axis=0)
  return states


def iter_sequences(time_total: int, start_se=True):
  """Iterate possible sequences.

  Assumes that first time step can be either S or E.
  """
  for t0 in range(time_total+1):
    if t0 == time_total:
      yield (t0, 0, 0)
    else:
      e_start = 1 if (t0 > 0 or start_se) else 0
      for de in range(e_start, time_total-t0+1):
        if t0+de == time_total:
          yield (t0, de, 0)
        else:
          #TODO: Marten: should de not be > 0, otherwise you cant
          # move to state I?
          i_start = 1 if (t0 > 0 or de > 0 or start_se) else 0
          for di in range(i_start, time_total-t0-de+1):
            if t0+de+di == time_total:
              yield (t0, de, di)
            else:
              yield (t0, de, di)


def generate_sequence_days(time_total: int):
  """Iterate possible sequences.

  Assumes that first time step must be S.
  """
  # t0 ranges in {T,T-1,...,1}
  for t0 in range(time_total, 0, -1):
    # de ranges in {T-t0,T-t0-1,...,1}
    # de can only be 0 when time_total was already spent
    de_start = min((time_total-t0, 1))
    non_t0 = time_total - t0
    for de in range(de_start, non_t0+1):
      # di ranges in {T-t0-de,T-t0-de-1,...,1}
      # di can only be 0 when time_total was already spent
      di_start = min((time_total-t0-de, 1))
      non_t0_de = time_total - t0 - de
      for di in range(di_start, non_t0_de+1):
        yield (t0, de, di)


@functools.lru_cache(maxsize=1)
def make_inf_obs_array(
    num_time_steps: int, alpha: float, beta: float) -> np.ndarray:
  """Makes an array with observation log-terms per day.

  Obs_array is of shape [num_time_steps, num_sequences, 2], where the last
  dimension is about the log-likelihood of the observation being 0 or 1.
  """
  pot_seqs = np.stack(list(
    iter_sequences(time_total=num_time_steps, start_se=False)))
  time_seqs = state_seq_to_time_seq(pot_seqs, num_time_steps)

  out_array = np.zeros((num_time_steps, len(pot_seqs), 2), dtype=np.float32)
  for t in range(num_time_steps):
    out_array[t, :, 0] = np.log(
      np.where(time_seqs[:, t] == 2, alpha, 1-beta))
    out_array[t, :, 1] = np.log(
      np.where(time_seqs[:, t] == 2, 1-alpha, beta))
  return out_array


def enumerate_log_prior_values(
    params_start: Union[np.ndarray, List[float]],
    params: Union[np.ndarray, List[float]],
    sequences: np.ndarray,
    time_total: int) -> np.ndarray:
  """Enumerate values of log prior."""
  # TODO: drop the option to start in I or R state
  np.testing.assert_almost_equal(np.sum(params_start), 1.)
  # TODO: Marten, need explanation on this part of code.
  b0, b1, b2 = params[0], params[1], params[2]

  #These are all boolean arrays denoting when something happened
  # Are these for a single user? which one?
  start_s = reach_s = sequences[:, 0] > 0
  start_e = (1-start_s) * (sequences[:, 1] > 0)
  start_i = (1-start_s) * (1-start_e) * (sequences[:, 2] > 0)
  start_r = (1-start_s) * (1-start_e) * (1-start_i)

  reach_e = (sequences[:, 1] > 0)
  reach_i = (sequences[:, 2] > 0)
  reach_r = (sequences[:, 0] + sequences[:, 1] + sequences[:, 2]) < time_total

  log_q_z = np.zeros((len(sequences)), dtype=np.float32)

  # Terms due to start state
  log_q_z += start_s * np.log(params_start[0] + 1E-12)
  log_q_z += start_e * np.log(params_start[1] + 1E-12)
  log_q_z += start_i * np.log(params_start[2] + 1E-12)
  log_q_z += start_r * np.log(params_start[3] + 1E-12)

  # Terms due to days spent in S
  log_q_z += np.maximum(sequences[:, 0]-1, 0.) * np.log(b0)
  log_q_z += reach_s * reach_e * np.log(1-b0)  # Only when transit to E is made

  # Terms due to days spent in E
  log_q_z += np.maximum(sequences[:, 1] - 1, 0.) * np.log(b1)
  log_q_z += reach_e * reach_i * np.log(1-b1)  # Only when transit to I is made

  # Terms due to days spent in I
  log_q_z += np.maximum(sequences[:, 2] - 1, 0.) * np.log(b2)
  log_q_z += reach_i * reach_r * np.log(1-b2)  # Only when transit to R is made

  return log_q_z


def enumerate_log_q_values(
    params: np.ndarray,
    sequences: np.ndarray) -> np.ndarray:
  """Enumerate values of log_q for variational parameters."""
  a0, b0, b1, b2 = params[0], params[1], params[2], params[3]
  time_total = np.max(sequences)

  log_q_z = np.zeros((len(sequences)))

  term_s = sequences[:, 0] >= time_total
  term_i = (sequences[:, 0] + sequences[:, 1] + sequences[:, 2]) >= time_total

  start_s = sequences[:, 0] > 0
  reach_i = (sequences[:, 2] > 0)

  # Terms due to s
  log_q_z += start_s * np.log(a0) + (1-start_s) * np.log(1-a0)
  log_q_z += start_s * (sequences[:, 0]-1)*np.log(b0)
  log_q_z += start_s * (1-term_s) * np.log(1-b0)

  # Terms due to e
  log_q_z += (1-term_s) * (sequences[:, 1] - 1) * np.log(b1)
  log_q_z += reach_i * np.log(1-b1)

  # Terms due to I
  log_q_z += reach_i * (sequences[:, 2]-1) * np.log(b2)
  log_q_z += reach_i * (1-term_i) * np.log(1-b2)

  return log_q_z


def sigmoid(x):
  return 1/(1+np.exp(-x))


def softmax(x):
  y = x - np.max(x)
  return np.exp(y)/np.sum(np.exp(y))


@numba.njit((
  'float32[:](float32[:], int64, float64, float64, float64)'))
def add_lognormal_noise_rdp(
    log_means: np.ndarray,
    num_contacts: int,
    a_rdp: float,
    epsilon_dp: float,
    sensitivity: float) -> np.ndarray:
  """Adds noise to log_means using RDP."""
  # Add RDP noise
  assert np.abs(sensitivity) > 1E-5, "Sensitivity must be defined"
  assert len(log_means.shape) == 1, "Only implemented for arrays"

  # For 0 contacts the d_no_term will be 0 anyway
  num_contacts = np.maximum(num_contacts, 1)

  sigma_squared_lognormal = a_rdp / (2 * num_contacts * epsilon_dp)
  sigma_squared_lognormal *= sensitivity**2

  # Mu parameter is the mean in the log-domain
  mu_lognormal = log_means - 0.5 * sigma_squared_lognormal

  num_vars = len(log_means)
  log_values = (mu_lognormal
                + np.sqrt(sigma_squared_lognormal) * np.random.randn(num_vars))
  return log_values.astype(np.float32)


@numba.njit((
  'UniTuple(float32[:], 2)(float32[:, :], float64, float64, int32[:, :], '
  'int64)'))
def precompute_d_penalty_terms_fn(
    q_marginal_infected: np.ndarray,
    p0: float,
    p1: float,
    past_contacts: np.ndarray,
    num_time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
  """Precompute penalty terms for inference with Factorised Neighbors.

  Consider similarity to 'precompute_d_penalty_terms_vi' and how the log-term is
  applied.
  """
  # Make num_time_steps+1 longs, such that penalties are 0 when t0==0
  d_term = np.zeros((num_time_steps+1), dtype=np.float32)
  d_no_term = np.zeros((num_time_steps+1), dtype=np.float32)

  if len(past_contacts) == 0:
    return d_term, d_no_term

  assert past_contacts[-1][0] < 0

  # Scales with O(T)
  t_contact = past_contacts[0][0]
  contacts = [np.int32(x) for x in range(0)]
  for row in past_contacts:
    if row[0] == t_contact:
      contacts.append(row[1])
    else:
      # Calculate in log domain to prevent underflow
      log_expectation = 0.

      # Scales with O(num_contacts)
      for user_contact in contacts:
        prob_infected = q_marginal_infected[user_contact][t_contact]
        log_expectation += np.log(prob_infected*(1-p1) + (1-prob_infected))

      d_no_term[t_contact+1] = log_expectation
      d_term[t_contact+1] = (
        np.log(1 - (1-p0)*np.exp(log_expectation)) - np.log(p0))

      # Reset loop stuff
      t_contact = row[0]
      contacts = [np.int32(x) for x in range(0)]

      if t_contact < 0:
        break

  # No termination (when t0 == num_time_steps) should not incur penalty
  # because the prior doesn't contribute the p0 factor either
  d_term[num_time_steps] = 0.
  return d_term, d_no_term


@numba.njit((
  'UniTuple(float32[:], 2)(int64[:], float32[:, :], float64, float64, int32[:, :], '
  'int64)'))
def precompute_d_penalty_terms_fn2(
    user_ids: np.ndarray,
    q_marginal_infected: np.ndarray,
    p0: float,
    p1: float,
    past_contacts: np.ndarray,
    num_time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
  """Precompute penalty terms for inference with Factorised Neighbors.

  Consider similarity to 'precompute_d_penalty_terms_vi' and how the log-term is
  applied.
  """
  # Make num_time_steps+1 longs, such that penalties are 0 when t0==0
  d_term = np.zeros((num_time_steps+1), dtype=np.float32)
  d_no_term = np.zeros((num_time_steps+1), dtype=np.float32)

  if len(past_contacts) == 0:
    return d_term, d_no_term

  # past_contacts is padded with -1, so break when contact time is negative
  assert past_contacts[-1][0] < 0

  # Scales with O(T)
  log_expectations = np.zeros((num_time_steps+1), dtype=np.float32)
  happened = np.zeros((num_time_steps+1), dtype=np.float32)

  # t_contact = past_contacts[0][0]
  # contacts = [np.int32(x) for x in range(0)]
  for row in past_contacts:
    time_inc = int(row[0])
    user_id = int(row[1])

    if time_inc < 0:
      # past_contacts is padded with -1, so break when contact time is negative
      break

    happened[time_inc+1] = 1

    q_marginal_index = np.where(user_ids == user_id)  
    assert q_marginal_index[0].shape[0] == 1
    
    p_inf_inc = q_marginal_infected[q_marginal_index[0][0]][time_inc]
    log_expectations[time_inc+1] += np.log(p_inf_inc*(1-p1) + (1-p_inf_inc))

  # Additional penalty term for not terminating, negative by definition
  d_no_term = log_expectations
  # Additional penalty term for not terminating, usually positive
  d_term = (np.log(1 - (1-p0)*np.exp(log_expectations)) - np.log(p0))

  # Prevent numerical imprecision error
  d_term *= happened

  # No termination (when t0 == num_time_steps) should not incur penalty
  # because the prior doesn't contribute the p0 factor either
  d_term[num_time_steps] = 0.
  return d_term.astype(np.float32), d_no_term.astype(np.float32)


@numba.njit((
  'UniTuple(float32[:], 2)(float32[:, :], float64, float64, float64, float64,'
  ' float64, float64, int32[:, :], int64)'))
def precompute_d_penalty_terms_rdp(
    q_marginal_infected: np.ndarray,
    p0: float,
    p1: float,
    clip_lower: float,
    clip_upper: float,
    a_rdp: float,
    epsilon_rdp: float,
    past_contacts: np.ndarray,
    num_time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
  """Precompute penalty terms for inference with Factorised Neighbors.

  Consider similarity to 'precompute_d_penalty_terms_vi' and how the log-term is
  applied.
  """
  # Make num_time_steps+1 longs, such that penalties are 0 when t0==0
  d_term = np.zeros((num_time_steps+1), dtype=np.float32)
  d_no_term = np.zeros((num_time_steps+1), dtype=np.float32)

  if len(past_contacts) == 0:
    return d_term, d_no_term

  # past_contacts is padded with -1, so break when contact time is negative
  assert past_contacts[-1][0] < 0

  # Scales with O(T)
  log_expectations = np.zeros((num_time_steps+1), dtype=np.float32)
  num_contacts = 0
  happened = np.zeros((num_time_steps+1), dtype=np.float32)

  # t_contact = past_contacts[0][0]
  # contacts = [np.int32(x) for x in range(0)]
  for row in past_contacts:
    time_inc = int(row[0])
    if time_inc < 0:
      # past_contacts is padded with -1, so break when contact time is negative
      break

    happened[time_inc+1] = 1
    p_inf_inc = q_marginal_infected[int(row[1])][time_inc]
    log_expectations[time_inc+1] += np.log(p_inf_inc*(1-p1) + (1-p_inf_inc))

    num_contacts += 1

  if a_rdp > 0:
    # Set clip values to [0, 1], because default values are [-1, 10000]
    clip_upper = np.minimum(clip_upper, 0.99999)
    clip_lower = np.maximum(clip_lower, 0.00001)
    sensitivity = np.abs(np.log(1-clip_upper*p1) - np.log(1-clip_lower*p1))

    log_expectations_noised = add_lognormal_noise_rdp(
      log_expectations, num_contacts, a_rdp, epsilon_rdp, sensitivity)

    # Everything hereafter is post-processing
    # Clip to [0, 1], equals clip to [\infty, 0] in logdomain
    log_expectations = np.minimum(
      log_expectations_noised,
      num_contacts * np.log(1 - clip_lower*p1)).astype(np.float32)

    # Public knowledge: No expectation can be lower than (1-\gamma*p_1)**num_c
    log_expectations = np.maximum(
      log_expectations,
      num_contacts * np.log(1 - clip_upper*p1)).astype(np.float32)

  # Additional penalty term for not terminating, negative by definition
  d_no_term = log_expectations
  # Additional penalty term for not terminating, usually positive
  d_term = (np.log(1 - (1-p0)*np.exp(log_expectations)) - np.log(p0))

  # Prevent numerical imprecision error
  d_term *= happened
  d_no_term *= happened

  # No termination (when t0 == num_time_steps) should not incur penalty
  # because the prior doesn't contribute the p0 factor either
  d_term[num_time_steps] = 0.
  return d_term.astype(np.float32), d_no_term.astype(np.float32)


@numba.njit((
  'UniTuple(float32[:], 2)(float32[:, :], float64, float64,'
  ' float64, float64, int32[:, :], int64)'))
def precompute_d_penalty_terms_dp_gaussian(
    q_marginal_infected: np.ndarray,
    p0: float,
    p1: float,
    epsilon_dp: float,
    delta_dp: float,
    past_contacts: np.ndarray,
    num_time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
  """Precompute penalty terms for inference with Factorised Neighbors.

  This method is similar to precompute_d_penalty_terms_rdp, but does not apply
  bias correction and thus uses straightforward (e,d)-DP
  """
  # Make num_time_steps+1 longs, such that penalties are 0 when t0==0
  d_term = np.zeros((num_time_steps+1), dtype=np.float32)
  d_no_term = np.zeros((num_time_steps+1), dtype=np.float32)

  if len(past_contacts) == 0:
    return d_term, d_no_term

  # past_contacts is padded with -1, so break when contact time is negative
  assert past_contacts[-1][0] < 0

  # Scales with O(T)
  log_expectations = np.zeros((num_time_steps+1), dtype=np.float32)
  num_contacts = np.zeros((num_time_steps+1), dtype=np.int32)
  happened = np.zeros((num_time_steps+1), dtype=np.float32)

  # t_contact = past_contacts[0][0]
  # contacts = [np.int32(x) for x in range(0)]
  for row in past_contacts:
    time_inc = int(row[0])
    if time_inc < 0:
      # past_contacts is padded with -1, so break when contact time is negative
      break

    happened[time_inc+1] = 1
    p_inf_inc = q_marginal_infected[int(row[1])][time_inc]
    log_expectations[time_inc+1] += np.log(p_inf_inc*(1-p1) + (1-p_inf_inc))

    num_contacts[time_inc+1] += 1

  if epsilon_dp > 0:
    assert delta_dp > 0

    sensitivity = np.log(1-p1)
    sigma = sensitivity / epsilon_dp * np.sqrt(2 * np.log(1.25 / delta_dp))

    # For 0 contacts the d_no_term will be 0 anyway
    num_contacts = np.maximum(num_contacts, 1)

    log_expectations_noised = (
      log_expectations+(sigma/num_contacts)*np.random.randn(num_time_steps+1))

    # Everything hereafter is post-processing
    # Clip to [0, 1], equals clip to [\infty, 0] in logdomain
    log_expectations = np.minimum(
      log_expectations_noised, 0.)

    log_expectations = np.maximum(
      log_expectations, num_contacts * np.log(1 - p1)).astype(np.float32)

  # Additional penalty term for not terminating, negative by definition
  d_no_term = log_expectations
  # Additional penalty term for not terminating, usually positive
  d_term = (np.log(1 - (1-p0)*np.exp(log_expectations)) - np.log(p0))

  # Prevent numerical imprecision error
  d_term *= happened
  d_no_term *= happened

  # No termination (when t0 == num_time_steps) should not incur penalty
  # because the prior doesn't contribute the p0 factor either
  d_term[num_time_steps] = 0.
  return d_term.astype(np.float32), d_no_term.astype(np.float32)


def it_num_infected_probs(probs: List[float]) -> Iterable[Tuple[int, float]]:
  """Iterates over the number of infected neighbors and its probabilities.

  NOTE: this function scales exponential in the number of neighbrs,
  O(2**len(probs))

  Args:
    probs: List of floats, each being the probability of a neighbor being
    infected.

  Returns:
    iterator with tuples of the number of infected neighbors and its probability
  """
  for is_infected in itertools.product([0, 1], repeat=len(probs)):
    yield sum(is_infected), math.prod(
      abs(is_infected[i] - 1 + probs[i]) for i in range(len(probs)))


def maybe_make_dir(dirname: str):
  if not os.path.exists(dirname):
    logger.info(os.getcwd())
    logger.info(f"Making data_dir {dirname}")
    os.makedirs(dirname)


@numba.njit([
  'float32[:](float32[:], int64)',
  'float32[:, :](float32[:, :], int64)',
  'float32[:, :, :](float32[:, :, :], int64)'])
def quantize(message: np.ndarray, num_levels: int) -> np.ndarray:
  """Quantizes a message based on rounding.

  Numerical will be mid-bucket.

  TODO: implement quantization with coding scheme.
  """
  if num_levels < 0:
    return message

  message = np.minimum(message, np.float32(1.-1E-9))
  message_at_floor = np.floor(message * num_levels) / num_levels
  return message_at_floor + np.float32(.5 / num_levels)


@numba.njit([
  'float32[:](float32[:], int64)',
  'float32[:, :](float32[:, :], int64)',
  'float32[:, :, :](float32[:, :, :], int64)'])
def quantize_floor(message: np.ndarray, num_levels: int) -> np.ndarray:
  """Quantizes a message based on rounding.

  Numerical will be at the floor of the bucket.

  TODO: implement quantization with coding scheme.
  """
  if num_levels < 0:
    return message

  return np.floor(message * num_levels) / num_levels


def get_cpu_count() -> int:
  # Divide cpu_count among tasks when running multiple tasks via SLURM
  num_tasks = 1
  if (slurm_ntasks := os.getenv("SLURM_NTASKS")):
    num_tasks = int(slurm_ntasks)
  return int(os.cpu_count() // num_tasks)


@numba.njit(['float32[:](float32[:])', 'float64[:](float64[:])'])
def normalize(x: np.ndarray) -> np.ndarray:
  return x / np.sum(x)


def check_exists(filename: str):
  if not os.path.exists(filename):
    logger.warning(f"File does not exist {filename}, current wd {os.getcwd()}")


def make_plain_observations(obs):
  return [(o['u'], o['time'], o['outcome']) for o in obs]


def make_plain_contacts(contacts) -> List[Any]:
  return [
    (c['u'], c['v'], c['time'], int(c['features'][0])) for c in contacts]


def spread_buckets(num_samples: int, num_buckets: int) -> np.ndarray:
  assert num_samples >= num_buckets
  num_samples_per_bucket = (int(np.floor(num_samples / num_buckets))
                            * np.ones((num_buckets)))
  num_remaining = int(num_samples - np.sum(num_samples_per_bucket))
  num_samples_per_bucket[:num_remaining] += 1
  return num_samples_per_bucket


# @functools.lru_cache(maxsize=1)
def spread_buckets_interval(num_samples: int, num_buckets: int) -> np.ndarray:
  num_users_per_bucket = spread_buckets(num_samples, num_buckets)
  return np.concatenate(([0], np.cumsum(num_users_per_bucket)))


@contextlib.contextmanager
def timeit(message: str):
  tstart = time.time()
  yield
  logger.info(f"{message} took {time.time() - tstart:.3f} seconds")


def root_find_a_rdp(
    eps: float,
    delta: float):
  """Finds a root of the RDP lagrangian."""
  assert eps > 0
  assert 0 < delta < 1

  # Fourth order polynomial, see paper for derivation of the coefficients
  delta_mod = np.log(1/delta)
  roots = np.roots([eps**2, -2*delta_mod*eps, delta_mod**2, 0, -delta_mod**2])

  # Only consider real roots
  roots = np.abs(roots[np.imag(roots) < 1E-3])

  # Primal feasibility of 'x = a-1>0'
  roots = roots[roots > 0]

  rho_values = eps - delta_mod / roots

  # Primal feasibility of 'rho>0'
  rho_values = rho_values[rho_values > 0]

  # Prevent numerical error
  rho_values = np.maximum(rho_values, 1E-6)
  a_values = 1 + delta_mod / (eps - rho_values)

  mult_values = a_values / rho_values
  if len(mult_values) == 0:
    raise ValueError(f"No solutions {eps:.2e}, {delta:.2e}")

  # Find lowest multiplier
  idx = np.argmin(mult_values)
  return a_values[idx], rho_values[idx]
