"""Inference methods for contact-graphs."""
from dpfn import constants, logger, util
from mpi4py import MPI  # pytype: disable=import-error
import numba
import numpy as np
import os
import time
from typing import Any, Optional, Tuple

comm_world = MPI.COMM_WORLD
mpi_rank = comm_world.Get_rank()
num_proc = comm_world.Get_size()


@numba.njit
def softmax(x):
  y = x - np.max(x)
  return np.exp(y)/np.sum(np.exp(y))


@numba.njit(parallel=True)
def fn_step_wrapped(
    user_interval: Tuple[int, int],
    seq_array_hot: np.ndarray,
    log_c_z_u: np.ndarray,
    log_A_start: np.ndarray,
    p_infected_matrix: np.ndarray,
    num_time_steps: int,
    probab0: float,
    probab1: float,
    clip_margin: float,
    past_contacts_array: np.ndarray,
    start_belief: np.ndarray,
    quantization: int = -1,):
  """Wraps one step of Factorised Neighbors over a subset of users.

  Args:
    user_interval: tuple of (user_start, user_end)
    seq_array_hot: array in [num_time_steps, 4, num_sequences]
    log_c_z_u: array in [num_users_int, num_sequences], C-terms according to
      CRISP paper
    log_A_start: array in [num_sequences], A-terms according to CRISP paper
    p_infected_matrix: array in [num_users, num_time_steps]
    num_time_steps: number of time steps
    probab0: probability of transitioning S->E
    probab1: probability of transmission given contact
    clip_margin: margin for clipping in preparation for DP calculations
    past_contacts_array: iterator with elements (timestep, user_u, features)
    start_belief: matrix in [num_users_int, 4], i-th row is assumed to be the
      start_belief of user user_slice[i]
    quantization: number of quantization levels
  """
  with numba.objmode(t0='f8'):
    t0 = time.time()

  # Apply quantization
  if quantization > 0:
    p_infected_matrix = util.quantize_floor(
      p_infected_matrix, num_levels=quantization)

  p_infected_matrix = p_infected_matrix.astype(np.float32)
  if clip_margin > 0:
    # Apply clipping
    p_infected_matrix = np.maximum(
      np.minimum(p_infected_matrix, 1.-clip_margin), clip_margin)
    p_infected_matrix = p_infected_matrix.astype(np.float32)

  interval_num_users = user_interval[1] - user_interval[0]

  post_exps = np.zeros((interval_num_users, num_time_steps, 4))
  num_days_s = np.sum(seq_array_hot[:, 0], axis=0).astype(np.int32)

  assert np.all(np.sum(seq_array_hot, axis=1) == 1), (
    "seq_array_hot is expected as one-hot array")

  seq_array_hot = seq_array_hot.astype(np.single)
  num_sequences = seq_array_hot.shape[2]

  # Array in [4, num_sequences]
  state_start_hot = seq_array_hot[0]
  # Array in [num_users, num_sequences]
  start_belief_all = np.log(start_belief.dot(state_start_hot) + 1E-12)

  for i in numba.prange(interval_num_users):  # pylint: disable=not-an-iterable

    d_term, d_no_term = util.precompute_d_penalty_terms_fn2(
      p_infected_matrix,
      p0=probab0,
      p1=probab1,
      past_contacts=past_contacts_array[i],
      num_time_steps=num_time_steps)
    d_noterm_cumsum = np.cumsum(d_no_term)

    d_penalties = (
      np.take(d_noterm_cumsum, np.maximum(num_days_s-1, 0))
      + np.take(d_term, num_days_s))

    # Numba only does matmul with 2D-arrays, so do reshaping below
    log_joint = log_c_z_u[i] + log_A_start + d_penalties + start_belief_all[i]
    joint_distr = softmax(log_joint).astype(np.single)
    post_exps[i] = np.reshape(np.dot(
      seq_array_hot.reshape(num_time_steps*4, num_sequences), joint_distr),
      (num_time_steps, 4))

  with numba.objmode(t1='f8'):
    t1 = time.time()

  return post_exps, t0, t1


def fact_neigh(
    num_users: int,
    num_time_steps: int,
    observations_all: constants.ObservationList,
    contacts_all: constants.ContactList,
    probab_0: float,
    probab_1: float,
    g_param: float,
    h_param: float,
    dp_noise: float = -1.,
    clip_margin: float = -1.,
    start_belief: Optional[np.ndarray] = None,
    alpha: float = 0.001,
    beta: float = 0.01,
    quantization: int = -1,
    users_stale: Optional[np.ndarray] = None,
    num_updates: int = 1000,
    verbose: bool = False,
    trace_dir: Optional[str] = None,
    diagnostic: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray]:
  """Inferes latent states using Factorised Neighbor method.

  Uses Factorised Neighbor approach from
  'The dlr hierarchy of approximate inference, Rosen-Zvi, Jordan, Yuille, 2012'

  Args:
    num_users: Number of users to infer latent states
    num_time_steps: Number of time steps to infer latent states
    observations_all: List of all observations
    contacts_all: List of all contacts
    probab_0: Probability to be infected spontaneously
    probab_1: Probability of transmission given contact
    g_param: float, dynamics parameter, p(E->I|E)
    h_param: float, dynamics parameter, p(I->R|I)
    start_belief: array in [num_users, 4], which are the beliefs for the start
      state
    alpha: False positive rate of observations, (1 minus specificity)
    beta: False negative rate of observations, (1 minus sensitivity)
    quantization: number of levels for quantization. Negative number indicates
      no use of quantization.
    num_updates: Number of rounds to update using Factorised Neighbor algorithm
    verbose: set to true to get more verbose output

  Returns:
    array in [num_users, num_timesteps, 4] being probability of over
    health states {S, E, I, R} for each user at each time step
  """
  del diagnostic
  t_start_preamble = time.time()

  user_ids_bucket = util.spread_buckets_interval(num_users, num_proc)
  user_interval = (
    int(user_ids_bucket[mpi_rank]), int(user_ids_bucket[mpi_rank+1]))
  num_users_interval = user_interval[1] - user_interval[0]

  # Slice out stale_users
  users_stale_binary = util.get_stale_users_binary(
    users_stale, num_users)

  seq_array = np.stack(list(
    util.iter_sequences(time_total=num_time_steps, start_se=False)))
  seq_array_hot = np.transpose(util.state_seq_to_hot_time_seq(
    seq_array, time_total=num_time_steps), [1, 2, 0]).astype(np.int8)

  # If 'start_belief' is provided, the prior will be applied per user, later
  if start_belief is None:
    prior = [1-probab_0, probab_0, 0., 0.]
  else:
    prior = [.25, .25, .25, .25]

  log_A_start = util.enumerate_log_prior_values(
    prior, [1-probab_0, 1-g_param, 1-h_param],
    seq_array, num_time_steps)

  obs_array = util.make_inf_obs_array(int(num_time_steps), alpha, beta)
  # Precompute log(C) terms, relating to observations
  log_c_z_u = util.calc_c_z_u(
    user_interval,
    obs_array,
    observations_all)

  q_marginal_infected = np.zeros((num_users, num_time_steps), dtype=np.single)
  post_exp = np.zeros((num_users, num_time_steps, 4), dtype=np.single)

  t_preamble1 = time.time() - t_start_preamble
  t_start_preamble = time.time()

  num_max_msg = int(100*num_time_steps)
  past_contacts, max_num_contacts = util.get_past_contacts_static(
    user_interval, contacts_all, num_msg=num_max_msg)

  if max_num_contacts >= num_max_msg:
    logger.warning(
      f"Max number of contacts {max_num_contacts} >= {num_max_msg}")

  if trace_dir:
    fname = os.path.join(trace_dir, f"fact_neigh_{mpi_rank}.txt")
    with open(fname, 'a') as fp:
      fp.write(f"{max_num_contacts:.0f}\n")

  start_belief_matrix = np.ones((num_users_interval, 4), dtype=np.single)
  if start_belief is not None:
    assert len(start_belief) == num_users
    start_belief_matrix = start_belief[user_interval[0]:user_interval[1]]

  if mpi_rank == 0:
    t_preamble2 = time.time() - t_start_preamble
    logger.info(
      f"Time spent on preamble1/preamble2 {t_preamble1:.1f}/{t_preamble2:.1f}")

  for num_update in range(num_updates):
    if verbose:
      if mpi_rank == 0:
        logger.info(f"Num update {num_update}")

    if users_stale is not None:
      q_marginal_infected *= users_stale_binary

    post_exp, tstart, t_end = fn_step_wrapped(
      user_interval,
      seq_array_hot,
      log_c_z_u,  # already depends in mpi_rank
      log_A_start,
      q_marginal_infected,
      num_time_steps,
      probab_0,
      probab_1,
      clip_margin=clip_margin,
      past_contacts_array=past_contacts,
      start_belief=start_belief_matrix,
      quantization=quantization)

    # if np.any(np.isinf(post_exp)):
    #   logger.info(f"post_exp has inf {post_exp}")
    # if np.any(np.isnan(post_exp)):
    #   logger.info(f"post_exp has nan {post_exp}")
    #   users_nan = np.where(
    #     np.sum(np.sum(np.isnan(post_exp), axis=-1), axis=-1))[0]
    #   logger.info(f"At users {repr(users_nan)}")

    if verbose:
      if mpi_rank == 0:
        logger.info(f"Time for fn_step: {t_end - tstart:.1f} seconds")

    # Prepare buffer for Allgatherv
    memory_bucket = user_ids_bucket*num_time_steps
    offsets = memory_bucket[:-1].tolist()
    sizes_memory = (memory_bucket[1:] - memory_bucket[:-1]).tolist()

    q_send = np.ascontiguousarray(post_exp[:, :, 2], dtype=np.single)
    comm_world.Allgatherv(
      q_send,
      recvbuf=[q_marginal_infected, sizes_memory, offsets, MPI.FLOAT])

  # Prepare buffer for Allgatherv
  post_exp_collect = np.empty((num_users, num_time_steps, 4), dtype=np.single)

  memory_bucket = user_ids_bucket*num_time_steps*4
  offsets = memory_bucket[:-1].tolist()
  sizes_memory = (memory_bucket[1:] - memory_bucket[:-1]).tolist()

  comm_world.Gatherv(
    post_exp.astype(np.single),
    recvbuf=[post_exp_collect, sizes_memory, offsets, MPI.FLOAT])

  if dp_noise > 0:
    # Add noise for Differential Privacy
    sensitivity = (1-clip_margin) * (1-probab_1) * dp_noise
    noise = np.random.randn(num_users, num_time_steps)

    post_exp_collect[:, :, 2] += noise * sensitivity

    # Post-process to ensure that the probabilities are still valid
    post_exp_collect = np.clip(post_exp_collect, 0., 1.)
    post_exp_collect /= np.sum(post_exp_collect, axis=-1, keepdims=True)
  return post_exp_collect
