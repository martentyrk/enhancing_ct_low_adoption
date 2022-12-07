"""Inference methods for contact-graphs."""
import datetime
import joblib
from dpfn import constants, logger, util
import numba
import numpy as np
import os
import time
from typing import Any, List, Optional, Tuple


@numba.njit
def softmax(x):
  y = x - np.max(x)
  return np.exp(y)/np.sum(np.exp(y))


@numba.njit
def fn_step_wrapped(
    user_slice: np.ndarray,
    seq_array_hot: np.ndarray,
    log_c_z_u: np.ndarray,
    log_A_start: np.ndarray,
    p_infected_matrix: np.ndarray,
    num_time_steps: int,
    probab0: float,
    probab1: float,
    past_contacts_array: np.ndarray,
    start_belief: Optional[np.ndarray] = None):
  """Wraps one step of Factorised Neighbors over a subset of users.

  Args:
    user_slice: list of user id for this step
    seq_array_hot: array in [num_time_steps, 4, num_sequences]
    log_c_z_u: array in [len(user_slice), num_sequences], C-terms according to
      CRISP paper
    log_A_start: array in [num_sequences], A-terms according to CRISP paper
    p_infected_matrix: array in [num_users, num_time_steps]
    num_time_steps: number of time steps
    probab0: probability of transitioning S->E
    probab1: probability of transmission given contact
    past_contacts: iterator with elements (timestep, user_u, features)
    start_belief: matrix in [len(user_slice), 4], i-th row is assumed to be the
      start_belief of user user_slice[i]
  """
  with numba.objmode(t0='f8'):
    t0 = time.time()

  post_exps = np.zeros((len(user_slice), num_time_steps, 4))
  num_days_s = np.sum(seq_array_hot[:, 0], axis=0).astype(np.int64)

  assert np.all(np.sum(seq_array_hot, axis=1) == 1), (
    "seq_array_hot is expected as one-hot array")

  seq_array_hot = seq_array_hot.astype(np.float64)
  num_sequences = seq_array_hot.shape[2]

  # Numba dot only works on float arrays
  states = np.arange(4, dtype=np.float64)
  state_start = seq_array_hot[0].T.dot(states).astype(np.int16)

  for i in range(len(user_slice)):

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
  return user_slice, post_exps, t0, t1


def fact_neigh(
    num_users: int,
    num_time_steps: int,
    observations_all: List[constants.Observation],
    contacts_all: List[constants.Contact],
    probab_0: float,
    probab_1: float,
    g_param: float,
    h_param: float,
    start_belief: Optional[np.ndarray] = None,
    alpha: float = 0.001,
    beta: float = 0.01,
    damping: float = 0.0,
    quantization: int = -1,
    users_stale: Optional[np.ndarray] = None,
    num_updates: int = 1000,
    verbose: bool = False,
    num_jobs: int = 8,
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
    damping: number between 0 and 1 to damp messages. 0 corresponds to no
      damping, number close to 1 correspond to high damping.
    quantization: number of levels for quantization. Negative number indicates
      no use of quantization.
    num_updates: Number of rounds to update using Factorised Neighbor algorithm
    verbose: set to true to get more verbose output
    num_jobs: Number of jobs to use for parallelisation. Recommended to set to
      number of cores on your machine.

  Returns:
    array in [num_users, num_timesteps, 4] being probability of over
    health states {S, E, I, R} for each user at each time step
  """
  t_start_preamble = time.time()
  assert num_jobs <= num_users, "Cannot run more parallel jobs than users"

  t_start1 = time.time()
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
    seq_array, observations_all, num_users=num_users, alpha=alpha, beta=beta)

  q_marginal_infected = np.zeros((num_users, num_time_steps))
  q_marginal_acc = np.zeros((num_updates+1, num_users, num_time_steps, 4))
  post_exp = np.zeros((num_users, num_time_steps, 4))

  infect_counter = util.InfectiousContactCount(
    contacts=contacts_all,
    samples=None,
    num_users=num_users,
    num_time_steps=num_time_steps,
  )
  print(f"Preamble1: {time.time() - t_start1}")
  t_start1 = time.time()

  # Parellelise one inference step over users.
  # Split users among number of jobs
  num_users_per_job = util.spread_buckets(num_users, num_jobs)
  slices = np.concatenate(([0], np.cumsum(num_users_per_job))).astype(np.int64)
  assert slices[-1] == num_users
  if verbose:
    print("Slices", slices)

  user_slices = [
    list(range(slices[n_job], slices[n_job+1])) for n_job in range(num_jobs)]
  log_c_z_u_s = [
    np.stack([log_c_z_u[user] for user in user_slice], axis=0)
    for user_slice in user_slices]

  start_belief_slices = [None for _ in range(num_jobs)]
  if start_belief is not None:
    start_belief_slices = [
      start_belief[user_slice] for user_slice in user_slices]

  print(f"Preamble2: {time.time() - t_start1}")
  logger.info(
    f"Parallelise FN with {num_jobs} jobs "
    f"after {time.time() - t_start_preamble:.1f} seconds on preamble")
  backend = util.get_joblib_backend()
  with joblib.Parallel(n_jobs=num_jobs, backend=backend) as parallel:
    for num_update in range(num_updates):
      # logger.info(f"Update {num_update}")

      # Sample stale users
      users_stale_now = util.sample_stale_users(users_stale)

      results = parallel(joblib.delayed(fn_step_wrapped)(
        np.array(user_slice),
        seq_array_hot,
        log_c_z_u_s[num_slice],
        log_A_start,
        q_marginal_infected,
        num_time_steps,
        probab_0,
        probab_1,
        infect_counter.get_past_contacts_slice(user_slice),
        start_belief_slices[num_slice],
      ) for num_slice, user_slice in enumerate(user_slices))

      for (user_slice, post_exp_users, tstart, tend) in results:
        if verbose:
          tstart_fmt = datetime.datetime.fromtimestamp(tstart).strftime(
            "%Y.%m.%d_%H:%M:%S")
          logger.info(f'Started on {tstart_fmt}, for {tend-tstart:12.1f}')

        post_exp = util.update_beliefs(
          post_exp, post_exp_users, user_slice, users_stale_now)

      # Collect statistics
      damping_use = damping if num_update > 0 else 0.0
      q_marginal_infected = (damping_use * q_marginal_infected
                             + (1-damping_use) * post_exp[:, :, 2])
      q_marginal_acc[num_update+1] = post_exp

      # Quantization
      if quantization > 0:
        q_marginal_infected = util.quantize_floor(
          q_marginal_infected, num_levels=quantization)

      if trace_dir:
        fname = os.path.join(trace_dir, f"trace_{num_update:05d}.npy")
        with open(fname, 'wb') as fp:
          np.save(fp, post_exp)

      if verbose:
        with np.printoptions(precision=2, suppress=True):
          print(q_marginal_infected[0])
          if num_users > 2:
            print(q_marginal_infected[2])
          print()
      if diagnostic:
        diagnostic.log({'user0': post_exp[0][:, 2].tolist()}, commit=False)
  return post_exp, q_marginal_acc
