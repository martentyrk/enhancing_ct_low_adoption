"""Utility functions for running experiments."""
import numba
import numpy as np
from dpfn import inference, logger, util
import dpfn_util
import subprocess
from typing import Any, Dict, Optional, Tuple

def make_inference_func(
    inference_method: str,
    num_users: int,
    cfg: Dict[str, Any],
    trace_dir: Optional[str] = None
    ):
  """Pulls together the inference function with parameters.

  Args:
    inference_method: string describing the inference method
    num_users: number of users in this simulation
    num_time_steps: number of time steps
    cfg: the configuration dict generated upon init of the experiment

  Returns:
    the inference function (input: data; output: marginals over SEIR per user)
  """
  p0 = cfg["model"]["p0"]
  p1 = cfg["model"]["p1"]
  g = cfg["model"]["prob_g"]
  h = cfg["model"]["prob_h"]
  alpha = cfg["model"]["alpha"]
  beta = cfg["model"]["beta"]
  quantization = cfg["model"]["quantization"]
  epsilon_dp = cfg["model"]["epsilon_dp"]
  delta_dp = cfg["model"]["delta_dp"]
  a_rdp = cfg["model"]["a_rdp"]
  clip_lower = cfg["model"]["clip_lower"]
  clip_upper = cfg["model"]["clip_upper"]

  dedup_contacts = cfg["model"]["dedup_contacts"]

  # DP method to use, explanation in constants.py, value of -1 means no DP
  dp_method = cfg["model"]["dp_method"]

  # Construct dynamics
  # Construct Geometric distro's for E and I states

  do_random_quarantine = False
  if inference_method == "fn":
    inference_func = wrap_fact_neigh_inference(
      num_users=num_users,
      alpha=alpha,
      beta=beta,
      probab0=p0,
      probab1=p1,
      g_param=g,
      h_param=h,
      clip_lower=clip_lower,
      clip_upper=clip_upper,
      quantization=quantization,
      trace_dir=trace_dir,
      )
  elif inference_method == "fncpp":
    inference_func = wrap_fact_neigh_cpp(
      num_users=num_users,
      alpha=alpha,
      beta=beta,
      probab0=p0,
      probab1=p1,
      g_param=g,
      h_param=h,
      dp_method=dp_method,
      epsilon_dp=epsilon_dp,
      delta_dp=delta_dp,
      a_rdp=a_rdp,
      clip_lower=clip_lower,
      clip_upper=clip_upper,
      quantization=quantization,
      trace_dir=trace_dir,
      dedup_contacts=dedup_contacts)

  elif inference_method == "random":
    inference_func = None
    do_random_quarantine = True
  elif inference_method == "dummy":
    inference_func = wrap_dummy_inference(
      num_users=num_users, trace_dir=trace_dir)
  else:
    raise ValueError((
      f"Not recognised inference method {inference_method}. Should be one of"
      f"['random', 'fn', 'dummy', 'dpct']"
    ))
  return inference_func, do_random_quarantine


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
    quantization: int = -1,
    trace_dir: Optional[str] = None,
    ):
  """Wraps the inference function that runs Factorised Neighbors."""

  def fact_neigh_wrapped(
      observations_list: np.ndarray,
      contacts_list: np.ndarray,
      app_user_ids:np.ndarray,
      app_users:np.ndarray,
      non_app_user_ids:np.ndarray,
      num_updates: int,
      num_time_steps: int,
      infection_prior: float,
      user_age_pinf_mean:np.ndarray,
      linear_feature_imputation: Any,
      neural_feature_imputation: Any,
      infection_rate: float,
      local_mean_baseline:bool,
      prev_z_states:np.ndarray = None,
      mse_states:np.ndarray=None,
      non_app_users_age: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None
      ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

    traces_per_user_fn, mse_loss = inference.fact_neigh(
      num_users=num_users,
      app_user_ids=app_user_ids,
      app_users=app_users,
      non_app_user_ids=non_app_user_ids,
      num_time_steps=num_time_steps,
      observations_all=observations_list,
      contacts_all=contacts_list,
      alpha=alpha,
      beta=beta,
      probab_0=probab0,  # Probability of spontaneous infection
      probab_1=probab1,  # Probability of transmission given a contact
      g_param=g_param,  # Dynamics parameter for E -> I transition
      h_param=h_param,  # Dynamics parameter for I -> R transition
      infection_prior=infection_prior,
      user_age_pinf_mean=user_age_pinf_mean,
      non_app_users_age=non_app_users_age,
      linear_feature_imputation=linear_feature_imputation,
      neural_feature_imputation=neural_feature_imputation,
      infection_rate=infection_rate,
      prev_z_states=prev_z_states,
      mse_states=mse_states,
      local_mean_baseline=local_mean_baseline,
      clip_lower=clip_lower,  # Lower bound for clipping, depends on method
      clip_upper=clip_upper,  # Upper bound for clipping, depends on method
      quantization=quantization,
      users_stale=None,
      num_updates=num_updates,
      verbose=False,
      trace_dir=trace_dir,
      diagnostic=diagnostic,
      )
    return traces_per_user_fn, None, mse_loss

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
      users_age: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None
      ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    del diagnostic, num_updates, contacts_list, observations_list
    del users_age

    predictions = np.random.rand(num_users, num_time_steps, 4)
    predictions /= np.sum(predictions, axis=-1, keepdims=True)

    return predictions, None

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
    dedup_contacts: int = 0,
    ):
  """Wraps the inference function that runs FN from pybind."""
  assert (dp_method < 0) or (dp_method == 5) or (dp_method == 2), (
    "Not implemented for dp_method > 0")
  del trace_dir

  num_workers = util.get_cpu_count()

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
      users_age: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None
      ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    del diagnostic

    if users_age is None:
      users_age = -1*np.ones((num_users), dtype=np.int32)

    post_exp, contacts_age = dpfn_util.fn_full_func(
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
      contacts=contacts_list,
      users_age=users_age,
      dedup_contacts=dedup_contacts)
    assert post_exp.shape == (num_users, num_time_steps, 4)
    contacts_age = np.reshape(contacts_age, (2, num_users)).T

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
    return post_exp, contacts_age
  return fact_neigh_cpp


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
      users_age: Optional[np.ndarray] = None,    # pylint: disable=unused-argument
      diagnostic: Optional[Any] = None  # pylint: disable=unused-argument
      ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    # del num_updates, users_age, diagnostic
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
    return score, None

  return dpct_wrapped



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


def convert_log_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
  """Converts any parameters that are defined in the log-domain.

  When doing sweeps or Bayesian optimization, it is convenient to define some
  parameters in the log-domain. In the sweep.yaml file, one can prepend any
  parameter with 'convertlog_' to indicate that the parameter is in the log10
  domain. This function will convert those parameters to the original domain.
  """
  for key in cfg["model"].keys():
    if key.startswith("convertlog_"):
      key_original = key.replace("convertlog_", "")
      value_new = np.power(10., cfg["model"][key])

      logger.info(f"Converting {key_original} from log-domain to {value_new}")

      cfg["model"][key_original] = value_new

  return cfg


def make_git_log():
  """Logs the git diff and git show.

  Note that this function has a general try/except clause and will except most
  errors produced by the git commands.
  """
  try:
    result = subprocess.run(
      ['git', 'status'], stdout=subprocess.PIPE, check=True)
    logger.info(f"Git status \n{result.stdout.decode('utf-8')}")

    result = subprocess.run(
      ['git', 'show', '--summary'], stdout=subprocess.PIPE, check=True)
    logger.info(f"Git show \n{result.stdout.decode('utf-8')}")

    result = subprocess.run(['git', 'diff'], stdout=subprocess.PIPE, check=True)
    logger.info(f"Git diff \n{result.stdout.decode('utf-8')}")
  except Exception as e:  # pylint: disable=broad-except
    logger.info(f"Git log not printed due to {e}")
