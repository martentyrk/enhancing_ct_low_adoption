"""Inference methods for contact-graphs."""
from dpfn import constants, logger, util
from mpi4py import MPI
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


@numba.njit
def fn_step_wrapped(
    user_interval: Tuple[int, int],
    seq_array_hot: np.ndarray,
    log_c_z_u: np.ndarray,
    log_A_start: np.ndarray,
    p_infected_matrix: np.ndarray,
    num_time_steps: int,
    probab0: float,
    probab1: float,
    past_contacts_array: np.ndarray,
    start_belief: Optional[np.ndarray] = None,
    quantization: int = -1,):
  """Wraps one step of Factorised Neighbors over a subset of users.

  Args:
    user_slice: list of user id for this step
    seq_array_hot: array in [num_time_steps, 4, num_sequences]
    log_c_z_u: array in [num_users_int, num_sequences], C-terms according to
      CRISP paper
    log_A_start: array in [num_sequences], A-terms according to CRISP paper
    p_infected_matrix: array in [num_users, num_time_steps]
    num_time_steps: number of time steps
    probab0: probability of transitioning S->E
    probab1: probability of transmission given contact
    past_contacts: iterator with elements (timestep, user_u, features)
    start_belief: matrix in [num_users_int, 4], i-th row is assumed to be the
      start_belief of user user_slice[i]
  """
  with numba.objmode(t0='f8'):
    t0 = time.time()

  # Apply quantization
  if quantization > 0:
    p_infected_matrix = util.quantize_floor(
      p_infected_matrix, num_levels=quantization)

  interval_num_users = user_interval[1] - user_interval[0]

  post_exps = np.zeros((interval_num_users, num_time_steps, 4))
  num_days_s = np.sum(seq_array_hot[:, 0], axis=0).astype(np.int64)

  assert np.all(np.sum(seq_array_hot, axis=1) == 1), (
    "seq_array_hot is expected as one-hot array")

  seq_array_hot = seq_array_hot.astype(np.float64)
  num_sequences = seq_array_hot.shape[2]

  # Numba dot only works on float arrays
  states = np.arange(4, dtype=np.float64)
  state_start = seq_array_hot[0].T.dot(states).astype(np.int16)

  for i in range(interval_num_users):

    d_term, d_no_term = util.precompute_d_penalty_terms_fn(
      p_infected_matrix,
      p0=probab0,
      p1=probab1,
      past_contacts=past_contacts_array[i],
      num_time_steps=num_time_steps)
    d_noterm_cumsum = np.cumsum(d_no_term)

    d_penalties = (
      np.take(d_noterm_cumsum, np.maximum(num_days_s-1, 0))
      + np.take(d_term, num_days_s))

    # Apply local start_belief
    start_belief_enum = np.zeros((num_sequences))
    if start_belief is not None:
      start_belief_enum = np.take(start_belief[i], state_start)
      start_belief_enum = np.log(start_belief_enum + 1E-12)
      assert start_belief_enum.shape == log_A_start.shape

    # Numba only does matmul with 2D-arrays, so do reshaping below
    log_joint = softmax(
      log_c_z_u[i] + log_A_start + d_penalties + start_belief_enum)
    post_exps[i] = np.reshape(np.dot(
      seq_array_hot.reshape(num_time_steps*4, num_sequences), log_joint),
      (num_time_steps, 4))

  with numba.objmode(t1='f8'):
    t1 = time.time()

  p_infected_matrix[user_interval[0]:user_interval[1]] = post_exps[:, :, 2]
  return post_exps, t0, t1, p_infected_matrix


def fact_neigh(
    num_users: int,
    num_time_steps: int,
    observations_all: constants.ObservationList,
    contacts_all: constants.ContactList,
    probab_0: float,
    probab_1: float,
    g_param: float,
    h_param: float,
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

  # Precompute log(C) terms, relating to observations
  log_c_z_u = util.calc_c_z_u(
    user_interval,
    seq_array,
    observations_all,
    alpha=alpha,
    beta=beta)

  q_marginal_infected = np.zeros((num_users, num_time_steps)).astype(np.double)
  post_exp = np.zeros((num_users, num_time_steps, 4))

  infect_counter = util.InfectiousContactCount(
    contacts=contacts_all,
    samples=None,
    num_users=num_users,
    num_time_steps=num_time_steps,
  )
  past_contacts = infect_counter.get_past_contacts_slice(
    list(range(user_interval[0], user_interval[1])))

  if start_belief is not None:
    assert len(start_belief) == num_users
    start_belief = start_belief[user_interval[0]:user_interval[1]]

  logger.info(f"{time.time() - t_start_preamble:.1f} seconds on preamble")

  for num_update in range(num_updates):
    if verbose:
      if mpi_rank == 0:
        logger.info(f"Num update {num_update}")
    # Sample stale users
    # TODO(rob) implement stale users
    if users_stale is not None:
      assert False, "Not implemented yet"
      # users_stale_now = util.sample_stale_users(users_stale)

    post_exp, tstart, t_end, q_marginal_infected = fn_step_wrapped(
      user_interval,
      seq_array_hot,
      log_c_z_u,  # already depends in mpi_rank
      log_A_start,
      q_marginal_infected,
      num_time_steps,
      probab_0,
      probab_1,
      past_contacts,
      start_belief,
      quantization=quantization)

    if verbose:
      if mpi_rank == 0:
        logger.info(f"Time for fn_step: {t_end - tstart:.1f} seconds")

    # Prepare buffer for Allgatherv
    q_collect = np.empty((num_users, num_time_steps), dtype=np.double)

    memory_bucket = user_ids_bucket*num_time_steps
    offsets = memory_bucket[:-1].tolist()
    sizes_memory = (memory_bucket[1:] - memory_bucket[:-1]).tolist()

    comm_world.Allgatherv(
      q_marginal_infected[user_interval[0]:user_interval[1]],
      recvbuf=[q_collect, sizes_memory, offsets, MPI.DOUBLE])
    q_marginal_infected = q_collect

    # # TODO(rob) update start belief per process
    #   post_exp = util.update_beliefs(
    #     post_exp, post_exp_users, user_slice, users_stale_now)

    if trace_dir:
      fname = os.path.join(
        trace_dir, f"trace_{num_update:05d}_rank{mpi_rank}.npy")
      with open(fname, 'wb') as fp:
        np.save(fp, post_exp)

  # Prepare buffer for Allgatherv
  post_exp_collect = np.empty((num_users, num_time_steps, 4), dtype=np.double)

  memory_bucket = user_ids_bucket*num_time_steps*4
  offsets = memory_bucket[:-1].tolist()
  sizes_memory = (memory_bucket[1:] - memory_bucket[:-1]).tolist()

  comm_world.Gatherv(
    post_exp,
    recvbuf=[post_exp_collect, sizes_memory, offsets, MPI.DOUBLE])
  return post_exp_collect
