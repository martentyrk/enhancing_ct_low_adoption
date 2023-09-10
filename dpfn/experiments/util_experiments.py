"""Utility functions for running experiments."""
import crisp
import numba
import numpy as np
from dpfn import inference, logger, belief_propagation, util
import dpfn_util
import subprocess
from typing import Any, Dict, Optional


def wrap_fact_neigh_inference(
    num_users: int,
    alpha: float,
    beta: float,
    probab0: float,
    probab1: float,
    g_param: float,
    h_param: float,
    clip_lower: float,
    clip_upper: float,
    dp_method: int = -1,
    epsilon_dp: float = -1.,
    delta_dp: float = -1.,
    a_rdp: float = -1.,
    quantization: int = -1,
    trace_dir: Optional[str] = None,
    ):
  """Wraps the inference function that runs Factorised Neighbors."""

  def fact_neigh_wrapped(
      observations_list: np.ndarray,
      contacts_list: np.ndarray,
      num_updates: int,
      num_time_steps: int,
      users_stale: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None) -> np.ndarray:

    traces_per_user_fn = inference.fact_neigh(
      num_users=num_users,
      num_time_steps=num_time_steps,
      observations_all=observations_list,
      contacts_all=contacts_list,
      alpha=alpha,
      beta=beta,
      probab_0=probab0,  # Probability of spontaneous infection
      probab_1=probab1,  # Probability of transmission given a contact
      g_param=g_param,  # Dynamics parameter for E -> I transition
      h_param=h_param,  # Dynamics parameter for I -> R transition
      dp_method=dp_method,  # Integer to choose an experimental dp method
      epsilon_dp=epsilon_dp,  # epsilon parameter for Differential Privacy
      delta_dp=delta_dp,  # delta parameter for Differential Privacy
      a_rdp=a_rdp,  # alpha parameter for Renyi Differential Privacy
      clip_lower=clip_lower,  # Lower bound for clipping, depends on method
      clip_upper=clip_upper,  # Upper bound for clipping, depends on method
      quantization=quantization,
      users_stale=users_stale,
      num_updates=num_updates,
      verbose=False,
      trace_dir=trace_dir,
      diagnostic=diagnostic)
    return traces_per_user_fn
  return fact_neigh_wrapped


def wrap_dummy_inference(
    num_users: int,
    trace_dir: Optional[str] = None,):
  """Wraps the inference function for dummy inference."""
  del trace_dir

  def dummy_wrapped(
      observations_list: np.ndarray,
      contacts_list: np.ndarray,
      num_updates: int,
      num_time_steps: int,
      users_stale: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None) -> np.ndarray:
    del diagnostic, num_updates, contacts_list, observations_list
    del users_stale

    predictions = np.random.rand(num_users, num_time_steps, 4)
    predictions /= np.sum(predictions, axis=-1, keepdims=True)

    return predictions

  return dummy_wrapped


def wrap_fact_neigh_cpp(
    num_users: int,
    alpha: float,
    beta: float,
    probab0: float,
    probab1: float,
    g_param: float,
    h_param: float,
    dp_method: int = -1,
    epsilon_dp: float = -1.,
    delta_dp: float = -1.,
    a_rdp: float = -1.,
    clip_lower: float = -1.,
    clip_upper: float = 10.,
    quantization: int = -1,
    trace_dir: Optional[str] = None,
    ):
  """Wraps the inference function that runs FN from pybind."""
  assert (dp_method < 0) or (dp_method == 5) or (dp_method == 2), (
    "Not implemented for dp_method > 0")
  del trace_dir

  num_workers = max((util.get_cpu_count()-1, 1))

  # Heuristically cap the number of workers. If the number of workers is too
  # large, then the overhead of creating a single thread is too large.
  max_num_workers = 8 if num_users < 200000 else 16
  num_workers = min((num_workers, max_num_workers))
  logger.info(f"Using {num_workers} workers for FN inference")

  epsilon_dp_use = epsilon_dp if dp_method != 2 else -1.

  def fact_neigh_cpp(
      observations_list: np.ndarray,
      contacts_list: np.ndarray,
      num_updates: int,
      num_time_steps: int,
      users_stale: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None) -> np.ndarray:
    del diagnostic, users_stale

    post_exp = dpfn_util.fn_full_func(
      num_workers=num_workers,
      num_rounds=num_updates,
      num_users=num_users,
      num_time_steps=num_time_steps,
      probab0=probab0,
      probab1=probab1,
      g_param=g_param,
      h_param=h_param,
      alpha=alpha,
      beta=beta,
      rho_rdp=epsilon_dp_use,
      a_rdp=a_rdp,
      clip_lower=clip_lower,
      clip_upper=clip_upper,
      quantization=quantization,
      observations=observations_list,
      contacts=contacts_list)
    assert post_exp.shape == (num_users, num_time_steps, 4)

    if dp_method == 2:
      assert epsilon_dp > 0.
      assert delta_dp > 0.
      assert a_rdp < 0
      covidscore = post_exp[:, -1, 2]

      c_upper = np.min((clip_upper, 1.))
      c_lower = np.max((clip_lower, 0.))

      sensitivity = probab1*(c_upper - c_lower)
      sigma = (sensitivity / epsilon_dp) * np.sqrt(2 * np.log(1.25 / delta_dp))

      covidscore += sigma*np.random.randn(num_users)

      post_exp = np.zeros((num_users, num_time_steps, 4), dtype=np.float32)
      post_exp[:, -1, 2] = np.clip(covidscore, c_lower, c_upper)
    return post_exp
  return fact_neigh_cpp


def wrap_bp_cpp(
    num_users: int,
    alpha: float,
    beta: float,
    probab0: float,
    probab1: float,
    g_param: float,
    h_param: float,
    dp_method: int = -1,
    rho_rdp: float = -1.,
    delta_dp: float = -1.,
    a_rdp: float = -1.,
    clip_lower: float = -1.,
    clip_upper: float = 10.,
    quantization: int = -1,
    trace_dir: Optional[str] = None,
    ):
  """Wraps the inference function that runs BP from pybind."""
  assert ((dp_method < 0) or (dp_method == 5)), (
    "Not implemented for dp_method > 0")
  del trace_dir

  if dp_method == 5:
    assert delta_dp < 0
    assert rho_rdp > 0
    assert a_rdp > 1

  num_workers = max((util.get_cpu_count()-1, 1))

  # Heuristically cap the number of workers. If the number of workers is too
  # large, then the overhead of creating a single thread is too large.
  max_num_workers = 8 if num_users < 200000 else 16
  num_workers = min((num_workers, max_num_workers))
  logger.info(f"Using {num_workers} workers for FN inference")

  def bp_cpp(
      observations_list: np.ndarray,
      contacts_list: np.ndarray,
      num_updates: int,
      num_time_steps: int,
      users_stale: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None) -> np.ndarray:
    del diagnostic, users_stale

    post_exp_out = dpfn_util.bp_full_func(
      num_workers=num_workers,
      num_rounds=num_updates,
      num_users=num_users,
      num_time_steps=num_time_steps,
      probab0=probab0,
      probab1=probab1,
      g_param=g_param,
      h_param=h_param,
      alpha=alpha,
      beta=beta,
      rho_rdp=rho_rdp,
      a_rdp=a_rdp,
      clip_lower=clip_lower,
      clip_upper=clip_upper,
      quantization=quantization,
      observations=observations_list,
      contacts=contacts_list)
    assert post_exp_out.shape == (num_users, num_time_steps, 4)
    return post_exp_out
  return bp_cpp


def wrap_dpct_inference(
    num_users: int,
    epsilon_dp: float,
    delta_dp: float):
  """Wraps the DPCT function for dummy inference.

  Differentially Private version of Traditional contact tracing.

  The methods checks for the number of contacts that tested positively. This
  method has global sensitivity of 1, according to which Gaussian noise is added
  following the Gaussian mechanism.

  Args:
    num_users: Number of users.
    epsilon_dp: Privacy parameter epsilon.
    delta_dp: Privacy parameter delta, defaults to 1/CTC, as that is the number
      of contacts we registery at most anyway.
  """
  noise_sigma = np.sqrt(2 * np.log(1.25 / delta_dp)) / epsilon_dp

  @numba.njit
  def dpct_wrapped(
      observations_list: np.ndarray,
      contacts_list: np.ndarray,
      num_updates: int,  # pylint: disable=unused-argument
      num_time_steps: int,
      users_stale: Optional[np.ndarray] = None,    # pylint: disable=unused-argument
      diagnostic: Optional[Any] = None) -> np.ndarray:    # pylint: disable=unused-argument
    # del num_updates, users_stale, diagnostic
    score_small = 0.0001  # Small number to prevent division by zero

    # Break symmetry
    score = score_small * np.random.rand(num_users, num_time_steps, 4)
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

    if epsilon_dp > 0:
      num_positive_neighbors += (
        noise_sigma * np.random.randn(num_users)).astype(np.float32)
      num_positive_neighbors = np.maximum(num_positive_neighbors, score_small)

    score[:, -1, 2] = num_positive_neighbors
    score /= np.expand_dims(np.sum(score, axis=-1), axis=-1)
    score = score.astype(np.float32)
    return score

  return dpct_wrapped


def wrap_belief_propagation(
    num_users: int,
    param_g: float,
    param_h: float,
    alpha: float,
    beta: float,
    probab0: float,
    probab1: float,
    clip_lower: float,
    clip_upper: float,
    epsilon_dp: float,
    a_rdp: float,
    quantization: int = -1,
    freeze_backwards: bool = False,
    trace_dir: Optional[str] = None,
    ):
  """Wraps the inference function runs Belief Propagation."""
  del freeze_backwards, trace_dir

  A_matrix = np.array([
    [1-probab0, probab0, 0, 0],
    [0, 1-param_g, param_g, 0],
    [0, 0, 1-param_h, param_h],
    [0, 0, 0, 1]
  ], dtype=np.float32)

  obs_distro = {
    0: np.array([1-beta, 1-beta, alpha, 1-beta], dtype=np.float32),
    1: np.array([beta, beta, 1-alpha, beta], dtype=np.float32),
  }

  def bp_wrapped(
      observations_list: np.ndarray,
      contacts_list: np.ndarray,
      num_updates: int,
      num_time_steps: int,
      users_stale: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None) -> np.ndarray:
    del users_stale, diagnostic
    # Collect observations, allows for multiple observations per user per day
    obs_messages = np.ones((num_users, num_time_steps, 4), dtype=np.float32)
    for obs in observations_list:
      if obs[1] < num_time_steps:
        obs_messages[obs[0]][obs[1]] *= obs_distro[obs[2]]

    map_forward_message, map_backward_message = (
      belief_propagation.init_message_maps(
        contacts_list, (0, num_users)))

    t_inference, t_quant, t_comm = 0., 0., 0.
    for _ in range(num_updates):

      (bp_beliefs, map_backward_message, map_forward_message, timing) = (
        belief_propagation.do_backward_forward_and_message(
          A_matrix, probab0, probab1, num_time_steps, obs_messages, num_users,
          map_backward_message, map_forward_message, (0, num_users),
          clip_lower, clip_upper, epsilon_dp, a_rdp,
          quantization=quantization))

      if np.any(np.isnan(bp_beliefs)):
        logger.info(f"Max message fwd {np.max(map_forward_message[:, :, 3])}")
        logger.info(f"Min message fwd {np.min(map_forward_message[:, :, 3])}")
        logger.info(f"Max message bwd {np.max(map_backward_message[:, :, 3:])}")
        logger.info(f"Min message bwd {np.min(map_backward_message[:, :, 3:])}")
        raise ValueError('bp_beliefs is NaN')
      if np.any(np.isnan(map_forward_message)):
        raise ValueError('Forward message is NaN')
      if np.any(np.isinf(map_forward_message)):
        raise ValueError('Forward message is infinite')

      if np.any(np.isnan(map_backward_message)):
        raise ValueError('Backward message is NaN')
      if np.any(np.isinf(map_backward_message)):
        raise ValueError('Backward message is infinite')

      if np.any(np.isinf(bp_beliefs)):
        raise ValueError('bp_beliefs is infinite')
      if np.any(bp_beliefs < 0):
        raise ValueError('bp_beliefs is negative ??')
      if np.any(bp_beliefs > 1):
        raise ValueError('bp_beliefs is bigger than 1 ??')

      t_inference += timing[1] - timing[0]
      t_quant += timing[2] - timing[1]

    logger.info((
      f"Time for {num_updates} bp passes: {t_inference:.2f}s, "
      f"{t_quant:.2f}s, {t_comm:.2f}s"))

    # Collect beliefs
    bp_collect = bp_beliefs
    bp_collect /= np.sum(bp_collect, axis=-1, keepdims=True)
    return bp_collect
  return bp_wrapped


def wrap_gibbs_inference(
    num_users: int,
    g_param: float,
    h_param: float,
    clip_lower: float,
    epsilon_dp: float,
    alpha: float,
    beta: float,
    probab_0: float,
    probab_1: float):
  """Wraps the inference function that runs Gibbs sampling."""
  # Construct Geometric distro's for E and I states
  q_e_vec = [0] + [
    g_param*(1-g_param)**(i-1) for i in range(1, 100*100+1)]
  q_i_vec = [0] + [
    h_param*(1-h_param)**(i-1) for i in range(1, 100*100+1)]

  pmf_e = np.array(q_e_vec) / np.sum(q_e_vec)
  pmf_i = np.array(q_i_vec) / np.sum(q_i_vec)

  qE = crisp.Distribution(pmf_e.tolist())
  qI = crisp.Distribution(pmf_i.tolist())

  def gibbs_wrapped(
      observations_list: np.ndarray,
      contacts_list: np.ndarray,
      num_updates: int,
      num_time_steps: int,
      users_stale: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None) -> np.ndarray:
    del diagnostic

    if users_stale is not None:
      raise ValueError('Not implemented stale users for Gibbs')

    num_burnin = min((num_updates, 10))
    skip = 10

    if clip_lower > 0:
      assert epsilon_dp > 0, "If clipping is enabled, then epsilon_dp must be"
      # When doing private inference, do a lot of skip steps, because each
      # released sample incurs an (epsilon, delta)-DP cost.
      skip = 10
      num_burnin = skip*num_updates

    result = crisp.GibbsPIS(
      num_users,
      num_time_steps,
      contacts_list.astype(np.int64),
      observations_list.astype(np.int64),
      qE, qI,
      alpha, beta,
      probab_0, probab_1,
      clip_lower, epsilon_dp, False)
    marginals = result.get_marginals(num_updates, burnin=num_burnin, skip=skip)
    return marginals

  return gibbs_wrapped


def set_noisy_test_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
  """Sets the noise parameters of the observational model."""
  noise_level = cfg["model"]["noisy_test"]
  assert noise_level <= 4

  if noise_level < 0:
    return cfg

  alpha_betas = [
    (1E-9, 1E-9), (.001, 0.01), (.01, .1), (.03, .25), (.5, .5)]

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
