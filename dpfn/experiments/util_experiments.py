"""Utility functions for running experiments."""
import numba
import numpy as np
from mpi4py import MPI  # pytype: disable=import-error
from dpfn import constants, inference, logger, belief_propagation, util, util_bp
from dpfn.experiments import util_sib
import sib
import subprocess
import time
from typing import Any, Dict, Optional

comm_world = MPI.COMM_WORLD
mpi_rank = comm_world.Get_rank()
num_proc = comm_world.Get_size()


def wrap_fact_neigh_inference(
    num_users: int,
    alpha: float,
    beta: float,
    p0: float,
    p1: float,
    g_param: float,
    h_param: float,
    clip_margin: float,
    dp_method: int = -1,
    epsilon_dp: float = -1.,
    delta_dp: float = -1.,
    quantization: int = -1,
    trace_dir: Optional[str] = None,
    ):
  """Wraps the inference function that runs Factorised Neighbors."""

  def fact_neigh_wrapped(
      observations_list: constants.ObservationList,
      contacts_list: constants.ContactList,
      num_updates: int,
      num_time_steps: int,
      start_belief: Optional[np.ndarray] = None,
      users_stale: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None):

    traces_per_user_fn = inference.fact_neigh(
      num_users=num_users,
      num_time_steps=num_time_steps,
      observations_all=observations_list,
      contacts_all=contacts_list,
      alpha=alpha,
      beta=beta,
      probab_0=p0,
      probab_1=p1,
      g_param=g_param,
      h_param=h_param,
      dp_method=dp_method,
      epsilon_dp=epsilon_dp,
      delta_dp=delta_dp,
      clip_margin=clip_margin,
      start_belief=start_belief,
      quantization=quantization,
      users_stale=users_stale,
      num_updates=num_updates,
      verbose=False,
      trace_dir=trace_dir,
      diagnostic=diagnostic)
    return traces_per_user_fn
  return fact_neigh_wrapped


def wrap_dummy_inference(
    num_users: int,):
  """Wraps the inference function for dummy inference."""

  def dummy_wrapped(
      observations_list: constants.ObservationList,
      contacts_list: constants.ContactList,
      num_updates: int,
      num_time_steps: int,
      start_belief: Optional[np.ndarray] = None,
      users_stale: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None) -> np.ndarray:
    del diagnostic, start_belief, num_updates, contacts_list, observations_list
    del users_stale

    predictions = np.random.randn(num_users, num_time_steps, 4)
    predictions /= np.sum(predictions, axis=-1, keepdims=True)

    return predictions

  return dummy_wrapped


def wrap_dct_inference(
    num_users: int,):
  """Wraps the DCT function for dummy inference.

  Mimicked after
  https://github.com/...
    sibyl-team/epidemic_mitigation/blob/master/src/rankers/dct_rank.py#L24
  """

  @numba.njit
  def dct_wrapped(
      observations_list: constants.ObservationList,
      contacts_list: constants.ContactList,
      num_updates: int,  # pylint: disable=unused-argument
      num_time_steps: int,
      start_belief: Optional[np.ndarray] = None,    # pylint: disable=unused-argument
      users_stale: Optional[np.ndarray] = None,    # pylint: disable=unused-argument
      diagnostic: Optional[Any] = None) -> np.ndarray:    # pylint: disable=unused-argument
    # del num_updates, start_belief, users_stale, diagnostic

    score = 0.25 * np.ones((num_users, num_time_steps, 4))
    score += 0.001 * np.random.rand(num_users, num_time_steps, 4)
    positive_tests = np.zeros((num_users))

    for row in observations_list:
      if row[2] > 0:
        user_u = int(row[0])
        positive_tests[user_u] += 1

    for row in contacts_list:
      user_u = int(row[0])
      user_v = int(row[1])
      if positive_tests[user_u] > 0:
        score[user_v, :, 2] = 5.0  # 20x bigger than noise floor
        score[user_u, :, 2] = 10.0  # 40x bigger than noise floor

    score /= np.expand_dims(np.sum(score, axis=-1), axis=-1)
    return score

  return dct_wrapped


def wrap_dpct_inference(
    num_users: int,
    epsilon_dp: float,
    delta_dp: float = 1 / constants.CTC):
  """Wraps the DPCT function for dummy inference.

  Differentially Private version of DCT.

  Args:
    num_users: Number of users.
    epsilon_dp: Privacy parameter epsilon.
    delta_dp: Privacy parameter delta, defaults to 1/CTC, as that is the number
      of contacts we registery at most anyway.
  """
  noise_sigma = np.sqrt(2 * np.log(1.25 / delta_dp)) / epsilon_dp

  @numba.njit
  def dpct_wrapped(
      observations_list: constants.ObservationList,
      contacts_list: constants.ContactList,
      num_updates: int,  # pylint: disable=unused-argument
      num_time_steps: int,
      start_belief: Optional[np.ndarray] = None,    # pylint: disable=unused-argument
      users_stale: Optional[np.ndarray] = None,    # pylint: disable=unused-argument
      diagnostic: Optional[Any] = None) -> np.ndarray:    # pylint: disable=unused-argument
    # del num_updates, start_belief, users_stale, diagnostic

    score = 0.25 * np.ones((num_users, num_time_steps, 4))
    score += 0.001 * np.random.rand(num_users, num_time_steps, 4)
    has_positive_test = np.zeros((num_users), dtype=np.float32)
    num_positive_neighbors = np.zeros((num_users), dtype=np.float32)

    for obs in observations_list:
      if obs[2] > 0:
        user_u = int(obs[0])
        has_positive_test[user_u] = 1.

    for row in contacts_list:
      user_u = int(row[0])
      user_v = int(row[1])
      num_positive_neighbors[user_v] += has_positive_test[user_u]

    num_positive_neighbors += (
      noise_sigma * np.random.randn(num_users)).astype(np.float32)
    num_positive_neighbors = np.maximum(num_positive_neighbors, 0.)

    score[:, -1, 2] = 6*has_positive_test + 3 * num_positive_neighbors
    score /= np.expand_dims(np.sum(score, axis=-1), axis=-1)
    return score

  return dpct_wrapped


def wrap_belief_propagation(
    num_users: int,
    param_g: float,
    param_h: float,
    alpha: float,
    beta: float,
    p0: float,
    p1: float,
    quantization: int = -1,
    freeze_backwards: bool = False,
    trace_dir: Optional[str] = None,
    ):
  """Wraps the inference function runs Belief Propagation."""
  del freeze_backwards, trace_dir

  A_matrix = np.array([
    [1-p0, p0, 0, 0],
    [0, 1-param_g, param_g, 0],
    [0, 0, 1-param_h, param_h],
    [0, 0, 0, 1]
  ])

  obs_distro = {
    0: np.array([1-beta, 1-beta, alpha, 1-beta]),
    1: np.array([beta, beta, 1-alpha, beta]),
  }

  def bp_wrapped(
      observations_list: np.ndarray,
      contacts_list: np.ndarray,
      num_updates: int,
      num_time_steps: int,
      start_belief: Optional[np.ndarray] = None,
      users_stale: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None):
    del users_stale, diagnostic

    # Set up MPI constants
    user_ids_bucket = util.spread_buckets_interval(num_users, num_proc)
    user_interval = (
      int(user_ids_bucket[mpi_rank]), int(user_ids_bucket[mpi_rank+1]))
    num_users_interval = user_interval[1] - user_interval[0]
    max_num_contacts = num_time_steps * constants.CTC

    # Contacts on last day are not of influence
    def filter_fn(contact):
      return (contact[2] + 1) < num_time_steps
    contacts_list = list(filter(filter_fn, contacts_list))

    # Collect observations, allows for multiple observations per user per day
    obs_messages = np.ones((num_users, num_time_steps, 4))
    for obs in observations_list:
      if obs[1] < num_time_steps:
        obs_messages[obs[0]][obs[1]] *= obs_distro[obs[2]]

    # Slice up obs_messages and start_belief
    # TODO make this slice within a function for obs_messages
    obs_messages = obs_messages[user_interval[0]:user_interval[1]]
    if start_belief is not None:
      start_belief = start_belief[user_interval[0]:user_interval[1]]

    map_forward_message, map_backward_message = (
      belief_propagation.init_message_maps(
        contacts_list, user_interval, num_time_steps))

    t_inference, t_quant, t_comm = 0., 0., 0.
    for _ in range(num_updates):

      (bp_beliefs, map_backward_message, map_forward_message, timing) = (
        belief_propagation.do_backward_forward_and_message(
          A_matrix, p0, p1, num_time_steps, obs_messages, num_users,
          map_backward_message, map_forward_message, user_interval,
          start_belief=start_belief,
          quantization=quantization))
      if num_time_steps > 5:
        # Only check after a few burnin days
        assert np.max(map_forward_message[:, -1, :]) < 0
        assert np.max(map_backward_message[:, -1, :]) < 0

      t_inference += timing[1] - timing[0]
      t_quant += timing[2] - timing[1]

      t_start = time.time()
      if num_proc > 1:
        messages_fwd_received = -10 * np.ones(
          (num_proc, num_users_interval, max_num_contacts, 4), dtype=np.single)
        messages_bwd_received = -10 * np.ones(
          (num_proc, num_users_interval, max_num_contacts, 7), dtype=np.single)
        for i in range(num_proc):
          # Forward messages
          memory = user_ids_bucket*max_num_contacts*4
          offsets = memory[:-1].astype(np.int32).tolist()
          sizes_memory = (memory[1:] - memory[:-1]).astype(np.int32).tolist()
          comm_world.Scatterv(
            [map_forward_message.astype(np.single), sizes_memory, offsets,
             MPI.FLOAT],
            [messages_fwd_received[i], MPI.FLOAT], root=i)

          # Backward messages
          memory = user_ids_bucket*max_num_contacts*7
          offsets = memory[:-1].astype(np.int32).tolist()
          sizes_memory = (memory[1:] - memory[:-1]).astype(np.int32).tolist()
          comm_world.Scatterv(
            [map_backward_message.astype(np.single), sizes_memory, offsets,
             MPI.FLOAT],
            [messages_bwd_received[i], MPI.FLOAT], root=i)

        # Some assertions, uncomment for debugging
        # assert np.all(messages_fwd_received > -2)
        # assert np.all(messages_bwd_received > -2)
        # if num_time_steps > 5:
        #   # Only check after a few burnin days
        #   assert np.max(messages_fwd_received[:, :, -1, :]) < 0
        #   assert np.max(messages_bwd_received[:, :, -1, :]) < 0

        map_forward_message = util_bp.collapse_null_messages(
          messages_fwd_received, num_proc, num_users_interval,
          max_num_contacts, 4)

        map_backward_message = util_bp.collapse_null_messages(
          messages_bwd_received, num_proc, num_users_interval,
          max_num_contacts, 7)
      t_comm += time.time() - t_start

    logger.info((
      f"Time for {num_updates} bp passes: {t_inference:.2f}s, "
      f"{t_quant:.2f}s, {t_comm:.2f}s"))

    # Collect beliefs
    bp_collect = np.empty((num_users, num_time_steps, 4), dtype=np.single)

    memory_bucket = user_ids_bucket*num_time_steps*4
    offsets = memory_bucket[:-1].tolist()
    sizes_memory = (memory_bucket[1:] - memory_bucket[:-1]).tolist()

    comm_world.Gatherv(
      bp_beliefs.astype(np.single),
      recvbuf=[bp_collect, sizes_memory, offsets, MPI.FLOAT])

    bp_collect /= np.sum(bp_collect, axis=-1, keepdims=True)
    return bp_collect
  return bp_wrapped


def wrap_sib(
    num_users: int,
    recovery_days: float,
    p0: float,
    p1: float,
    damping: float):
  """Wraps the inference function for the SIB library.

  https://github.com/sibyl-team/sib
  """
  sib.set_num_threads(util.get_cpu_count())

  def sib_wrapped(
      observations_list: constants.ObservationList,
      contacts_list: constants.ContactList,
      num_updates: int,
      num_time_steps: int,
      start_belief: Optional[np.ndarray] = None,
      users_stale: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None):
    del start_belief

    if users_stale is not None:
      raise ValueError('Not implemented stale users for SIB')

    # Prepare contacts
    contacts_sib = []
    for contact in contacts_list:
      if contact[0] < 0:
        break

      ctc = (int(contact[0]), int(contact[1]), int(contact[2]), p1)
      contacts_sib.append(ctc)

    # Prepare observations
    obs_sib = []
    for obs in observations_list:
      if obs[0] < 0:
        break

      if obs[2] == 1:
        test = sib.Test(0, 1, 0)
      else:
        test = sib.Test(.5, 0, .5)
      obs_sib.append((obs[0], test, obs[1]))
    # Add dummy observations to query marginals later
    obs_sib += [(i, sib.Test(1, 1, 1), timestep)
                for i in range(num_users) for timestep in range(num_time_steps)]

    # Sort observations and contacts (required for SIB library)
    obs_sib = list(sorted(obs_sib, key=lambda x: x[2]))
    contacts_sib = list(sorted(contacts_sib, key=lambda x: x[2]))

    sib_params = util_sib.make_sib_params(num_time_steps, p0, recovery_days)

    # Run inference
    f = sib.FactorGraph(
      params=sib_params,
      contacts=contacts_sib,
      tests=obs_sib)

    if diagnostic:
      diagnostic.log({"damping": damping}, commit=False)
    sib.iterate(f, maxit=num_updates, damping=damping)

    nodes_all = {node.index: node for node in f.nodes}

    # Collect marginals
    marginals_sib = np.zeros((num_users, num_time_steps, 3))
    for user in range(num_users):
      for timestep in range(num_time_steps):
        message = sib.marginal_t(nodes_all[user], timestep)
        marginals_sib[user, timestep] = np.array(list(message))

    logger.info(f"Marginals SIB contain NaN {np.any(np.isnan(marginals_sib))}")

    # Insert a slice with all zeros for E state (SIB does only SIR)
    marginals_sib = np.concatenate(
      (marginals_sib[:, :, :1],
       np.zeros((num_users, num_time_steps, 1)),
       marginals_sib[:, :, 1:]),
      axis=-1)
    return marginals_sib
  return sib_wrapped


def set_noisy_test_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
  """Sets the noise parameters of the observational model."""
  noise_level = cfg["model"]["noisy_test"]
  assert 0 <= noise_level <= 3

  if noise_level == 0:
    return cfg

  alpha_betas = [(), (.001, 0.01), (.01, .1), (.03, .25)]

  # Set model parameters
  cfg["model"]["alpha"] = alpha_betas[noise_level][0]
  cfg["model"]["beta"] = alpha_betas[noise_level][1]

  # Don't assume model misspecification, set data parameters the same
  cfg["data"]["alpha"] = alpha_betas[noise_level][0]
  cfg["data"]["beta"] = alpha_betas[noise_level][1]

  return cfg


def make_git_log():
  """Logs the git diff and git show.

  Note that this function has a general try/except clause and will except most
  errors produced by the git commands.
  """
  try:
    result = subprocess.run(
      ['git', 'show', '--summary'], stdout=subprocess.PIPE, check=True)
    logger.info(f"Git show \n{result.stdout.decode('utf-8')}")

    result = subprocess.run(['git', 'diff'], stdout=subprocess.PIPE, check=True)
    logger.info(f"Git diff \n{result.stdout.decode('utf-8')}")
  except Exception as e:  # pylint: disable=broad-except
    logger.info(f"Git log not printed due to {e}")
