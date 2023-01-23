"""Compare inference methods on likelihood and AUROC"""
import argparse
import copy
from mpi4py import MPI  # pytype: disable=import-error
import numpy as np
from dpfn.config import config
from dpfn.data import data_load
from dpfn.experiments import prequential, util_experiments
from dpfn import constants
from dpfn import LOGGER_FILENAME, logger
from dpfn import simulator
from dpfn import util
from dpfn import util_wandb
import numba
import os
import random
from sklearn import metrics
import sys
import time
import tqdm
import traceback
import tracemalloc
from typing import Any, Dict, List, Optional
import wandb

TRACEMALLOC = False

comm_world = MPI.COMM_WORLD
mpi_rank = comm_world.Get_rank()
num_proc = comm_world.Get_size()
print(f"num_proc {num_proc} mpi_rank {mpi_rank}")


def make_inference_func(
    inference_method: str,
    num_users: int,
    cfg: Dict[str, Any],
    trace_dir: Optional[str] = None,
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

  # Construct dynamics
  # Construct Geometric distro's for E and I states

  do_random_quarantine = False
  if inference_method == "bp":
    inference_func = util_experiments.wrap_belief_propagation(
      num_users=num_users,
      alpha=alpha,
      beta=beta,
      p0=p0,
      p1=p1,
      param_g=g,
      param_h=h,
      quantization=quantization,
      trace_dir=trace_dir)
  elif inference_method == "fn":
    inference_func = util_experiments.wrap_fact_neigh_inference(
      num_users=num_users,
      alpha=alpha,
      beta=beta,
      p0=p0,
      p1=p1,
      g_param=g,
      h_param=h,
      quantization=quantization,
      trace_dir=trace_dir)
  elif inference_method == "sib":
    # Belief Propagation
    sib_mult = cfg["model"]["sib_mult"]
    recovery_days = 1/h + sib_mult*1/g

    inference_func = util_experiments.wrap_sib(
      num_users=num_users,
      recovery_days=recovery_days,
      p0=p0,
      p1=p1,
      damping=0.0)
  elif inference_method == "random":
    inference_func = None
    do_random_quarantine = True
  elif inference_method == "dummy":
    inference_func = util_experiments.wrap_dummy_inference(num_users=num_users)
  elif inference_method == "dct":
    inference_func = util_experiments.wrap_dct_inference(
      num_users=num_users)
  else:
    raise ValueError((
      f"Not recognised inference method {inference_method}. Should be one of"
      f"['random', 'fn', 'dummy', 'dct', 'bp', 'sib']"
    ))
  return inference_func, do_random_quarantine


def compare_prequential_quarantine(
    inference_method: str,
    num_users: int,
    num_time_steps: int,
    observations: List[constants.Observation],
    contacts: List[constants.Contact],
    states: np.ndarray,
    cfg: Dict[str, Any],
    runner,
    results_dir: str,
    trace_dir: Optional[str] = None,
    quick: bool = False,
    do_diagnosis: bool = False,
    use_abm_simulator: bool = False):
  """Compares different inference algorithms on the supplied contact graph."""
  del states

  if TRACEMALLOC:
    tracemalloc.start()

  num_users = int(num_users)
  del observations

  num_days_window = cfg["model"]["num_days_window"]
  quantization = cfg["model"]["quantization"]
  threshold_quarantine = cfg["model"]["threshold_quarantine"]

  num_rounds = cfg["model"]["num_rounds"]
  rng_seed = cfg.get("seed", 123)

  fraction_test = cfg["data"]["fraction_test"]
  do_conditional_testing = bool(cfg["data"]["do_conditional_testing"])
  do_conditional_quarantine = bool(cfg["data"]["do_conditional_quarantine"])

  # Data and simulator params
  fraction_stale = cfg["data"]["fraction_stale"]
  fraction_quarantine = cfg["data"]["fraction_quarantine"]
  num_days_quarantine = cfg["data"]["num_days_quarantine"]

  probab_0 = cfg["model"]["p0"]
  params_dynamics = {
    "p0": cfg["data"]["p0"],
    "p1": cfg["data"]["p1"],
    "g": cfg["data"]["prob_g"],
    "h": cfg["data"]["prob_h"],
  }

  logger.info((
    f"Settings at experiment: {quantization:.0f} quant, "
    f"{threshold_quarantine:.3f} threshold, "
    f"conditional testing {do_conditional_testing} at {fraction_test}%"))

  # Manual parameters for quarantining
  t_start_quarantine = 3
  num_quarantine = int(fraction_quarantine * num_users)

  users_stale = None
  if fraction_stale > 0:
    users_stale = np.random.choice(
      num_users, replace=False, size=(int(fraction_stale*num_users)))

  diagnostic = runner if do_diagnosis else None

  inference_func, do_random_quarantine = make_inference_func(
    inference_method, num_users, cfg, trace_dir=trace_dir)

  # Set conditional distributions for observations
  p_obs_infected = [cfg["model"]["alpha"], 1-cfg["model"]["alpha"]]
  p_obs_not_infected = [1-cfg["model"]["beta"], cfg["model"]["beta"]]

  start_belief_global = (
    np.ones((num_users, 4)) * np.array([1. - probab_0, probab_0, 0., 0.]))

  if quick:
    num_rounds = 2

  # Arrays to accumulate statistics
  pir_running = 0.
  precisions = np.zeros((num_time_steps))
  recalls = np.zeros((num_time_steps))
  infection_rates = np.zeros((num_time_steps))
  exposed_rates = np.zeros((num_time_steps))
  likelihoods_state = np.zeros((num_time_steps))
  ave_prob_inf = np.zeros((num_time_steps))
  ave_prob_inf_at_inf = np.zeros((num_time_steps))
  ave_precision = np.zeros((num_time_steps))
  num_quarantined = np.zeros((num_time_steps), dtype=np.int32)
  num_tested = np.zeros((num_time_steps), dtype=np.int32)

  # Placeholder for tests on first day
  z_states_inferred = np.zeros((num_users, 1, 4))
  test_include = np.ones((num_users))
  users_not_quarantined = np.ones((num_users), dtype=np.bool)

  logger.info(f"Do random quarantine? {do_random_quarantine}")
  t0 = time.time()

  if mpi_rank == 0:
    if use_abm_simulator:
      sim_factory = simulator.ABMSimulator
      contacts = []
    else:
      sim_factory = simulator.CRISPSimulator

    sim = sim_factory(num_time_steps, num_users, params_dynamics, rng_seed)
    sim.init_day0(copy.deepcopy(contacts))

    logger.info((
      f"Start simulation with {num_rounds} updates"))
  else:
    sim = simulator.DummySimulator(num_time_steps, num_users, params_dynamics)

  for t_now in tqdm.trange(1, num_time_steps):
    t_start_loop = time.time()
    num_days = min((t_now + 1, num_days_window))

    # For each value of t_now, only receive observations up to 't_now-1'
    assert sim.get_current_day() == t_now - 1

    rank_score = np.random.randn(num_users)
    if do_conditional_testing:
      # TODO double check -1 here!
      rank_score = z_states_inferred[:, -1, 2]
      if do_conditional_quarantine:
        rank_score *= users_not_quarantined

    if mpi_rank == 0:
      # Grab tests on the main process
      users_to_test = prequential.decide_tests(
        scores_infect=rank_score,
        test_include=test_include,
        num_tests=int(fraction_test * num_users))

      obs_today = sim.get_observations_today(
        users_to_test,
        p_obs_infected,
        p_obs_not_infected
      )

      test_include = prequential.remove_positive_users(
        obs_today, test_include)
    else:
      users_to_test = []
      obs_today = []

    # When num_days exceeds t_now, then offset should start counting at 0
    days_offset = t_now + 1 - num_days
    assert 0 <= days_offset <= num_time_steps

    sim.set_window(days_offset)

    if TRACEMALLOC and (t_now > 7):
      snapshot = tracemalloc.take_snapshot()
      util.display_top(snapshot)

    if not do_random_quarantine:
      t_start = time.time()

      # Make inference over SEIR states
      start_belief = start_belief_global
      if num_days <= t_now:
        if mpi_rank == 0:
          logger.info("Use window!")
        start_belief = z_states_inferred[:, 1]
      start_belief = np.ascontiguousarray(start_belief, dtype=np.single)
      comm_world.Bcast([start_belief, MPI.FLOAT], root=0)

      contacts_now = util.make_default_array(
        sim.get_contacts(), dtype=np.int32, rowlength=4)
      observations_now = util.make_default_array(
        sim.get_observations_all(), dtype=np.int32, rowlength=3)

      num_contacts = np.array(contacts_now.shape[0], dtype=np.int64)
      num_obs = np.array(observations_now.shape[0], dtype=np.int64)

      comm_world.Bcast([num_contacts, MPI.INT], root=0)
      comm_world.Bcast([num_obs, MPI.INT], root=0)

      if mpi_rank == 0:
        logger.info(f"Day {t_now}: {num_contacts} contacts, {num_obs} obs")

      if mpi_rank > 0:
        # Make receiving buffers on all but head process
        contacts_now = np.zeros((num_contacts, 4), dtype=np.int32)
        observations_now = np.zeros((num_obs, 3), dtype=np.int32)

      # TODO: send np.int32
      contacts_now = np.ascontiguousarray(contacts_now, dtype=np.int32)
      observations_now = np.ascontiguousarray(observations_now, dtype=np.int32)

      comm_world.Bcast([contacts_now, MPI.INT32_T], root=0)
      comm_world.Bcast([observations_now, MPI.INT32_T], root=0)

      # if t_now == 7:
      #   fname = f"results/tmp/state_{mpi_rank}_{t_now}.npy"
      #   with open(fname, 'wb') as f:
      #     np.savez(
      #       f, contacts_now=contacts_now, observations_now=observations_now,
      #       num_rounds=num_rounds, num_days=num_days,
      #       start_belief=start_belief)

      z_states_inferred = inference_func(
        observations_now,
        contacts_now,
        num_rounds,
        num_days,
        start_belief,
        users_stale=users_stale,
        diagnostic=diagnostic)

      np.testing.assert_array_almost_equal(
        z_states_inferred.shape, [num_users, num_days, 4])
      if mpi_rank == 0:
        logger.info(f"Time spent on inference_func {time.time() - t_start:.0f}")

      # Decide who to quarantine and subtract contacts
      if threshold_quarantine > 0:
        users_to_quarantine = prequential.select_quarantine_users(
          z_states_inferred, threshold=threshold_quarantine)
      else:
        users_to_quarantine = prequential.select_quarantine_users_max(
          z_states_inferred, num_quarantine=num_quarantine)
    else:
      z_states_inferred = np.zeros((num_users, num_days, 4))
      users_to_quarantine = np.random.choice(
        num_users, size=(num_quarantine)).tolist()

    if do_conditional_quarantine:
      users_to_quarantine = [
        obs[0] for obs in obs_today if obs[2] > 0]

    if t_now < t_start_quarantine:
      users_to_quarantine = np.array([], dtype=np.int32)

    users_not_quarantined[users_to_quarantine] = False

    # This function will remove the contacts that happen TODAY (and which may
    # spread the virus and cause people to shift to E-state tomorrow).
    sim.quarantine_users(users_to_quarantine, num_days_quarantine)
    assert sim.get_current_day() == t_now - 1
    sim.step()

    if mpi_rank == 0:
      # NOTE: fpr is only defined as long as num_users_quarantine is fixed.
      # else switch to precision and recall
      states_today = sim.get_states_today()

      precision, recall = prequential.calc_prec_recall(
        states_today, users_to_quarantine)
      infection_rate = np.mean(states_today == 2)
      exposed_rate = np.mean(
        np.logical_or(states_today == 1, states_today == 2))
      pir_running = max((pir_running, infection_rate))
      logger.info((f"precision: {precision:5.2f}, recall: {recall: 5.2f}, "
                   f"infection rate: {infection_rate:5.3f}({pir_running:5.3f}),"
                   f"{exposed_rate:5.3f}, tests: {len(users_to_test):5.0f}"))

      precisions[t_now] = precision
      recalls[t_now] = recall
      infection_rates[t_now] = infection_rate
      exposed_rates[t_now] = np.mean(
        np.logical_or(states_today == 1, states_today == 2))
      num_quarantined[t_now] = len(users_to_quarantine)
      num_tested[t_now] = len(users_to_test)

      # Inspect using sampled states
      p_at_state = z_states_inferred[range(num_users), num_days-1, states_today]
      likelihoods_state[t_now] = np.mean(np.log(p_at_state + 1E-9))
      ave_prob_inf_at_inf[t_now] = np.mean(
        p_at_state[states_today == 2])
      ave_prob_inf[t_now] = np.mean(z_states_inferred[:, num_days-1, 2])

      if infection_rate > 0:
        ave_precision[t_now] = metrics.average_precision_score(
          y_true=(states_today == 2),
          y_score=z_states_inferred[:, num_days-1, 2])
      else:
        ave_precision[t_now] = 0.

      time_full_loop = time.time() - t_start_loop
      logger.info(f"Time spent on full_loop {time_full_loop:.0f}")

      loadavg1, loadavg5, _ = os.getloadavg()
      runner.log({
        "time_step": time_full_loop,
        "infection_rate": infection_rate,
        "load1": loadavg1,
        "load5": loadavg5,
        "recall": recall,
        })

  if mpi_rank == 0:
    time_pir, pir = np.argmax(infection_rates), np.max(infection_rates)
    logger.info(f"At day {time_pir} peak infection rate is {pir}")

    prequential.dump_results_json(
      datadir=results_dir,
      cfg=cfg,
      ave_prob_inf=ave_prob_inf.tolist(),
      ave_prob_inf_at_inf=ave_prob_inf_at_inf.tolist(),
      ave_precision=ave_precision.tolist(),
      exposed_rates=exposed_rates.tolist(),
      inference_method=inference_method,
      infection_rates=infection_rates.tolist(),
      likelihoods_state=likelihoods_state.tolist(),
      name=runner.name,
      num_quarantined=num_quarantined.tolist(),
      num_tested=num_tested.tolist(),
      pir=float(pir),
      precisions=precisions.tolist(),
      quantization=quantization,
      recalls=recalls.tolist(),
      seed=cfg.get("seed", -1),
    )

    time_spent = time.time() - t0
    logger.info(f"With {num_rounds} rounds, PIR {pir:5.2f}")
    runner.log({
      "time_spent": time_spent,
      "pir_mean": pir})

    # Overwrite every experiment, such that code could be pre-empted
    prequential.dump_results(
      results_dir, precisions=precisions, recalls=recalls,
      infection_rates=infection_rates)


def compare_inference_algorithms(
    inference_method: str,
    num_users: int,
    num_time_steps: int,
    observations: List[constants.Observation],
    contacts: List[constants.Contact],
    states: np.ndarray,
    cfg: Dict[str, Any],
    runner,
    results_dir: str,
    trace_dir: Optional[str] = None,
    quick: bool = False,
    do_diagnosis: bool = False,
    use_abm_simulator: bool = False):
  """Compares different inference algorithms on the supplied contact graph."""
  del results_dir, use_abm_simulator

  # Contacts on last day are not of influence
  def filter_fn(datum):
    return datum[2] < (num_time_steps - 1)
  contacts = list(filter(filter_fn, contacts))

  alpha = cfg["model"]["alpha"]
  beta = cfg["model"]["beta"]

  num_rounds = cfg["model"]["num_rounds"]
  num_users = int(num_users)

  # Data and simulator params
  fraction_stale = cfg["data"]["fraction_stale"]

  users_stale = None
  if fraction_stale > 0:
    users_stale = np.random.choice(
      num_users, replace=False, size=(int(fraction_stale*num_users)))

  inference_func, _ = make_inference_func(
    inference_method, num_users, cfg, trace_dir)

  if quick:
    num_rounds = 2

  diagnostic = runner if do_diagnosis else None

  if mpi_rank == 0:
    logger.info(f"Start inference method {inference_method}")

  time_start = time.time()
  z_states_inferred = inference_func(
    np.array(observations),
    np.array(contacts),
    num_rounds,
    num_time_steps,
    users_stale=users_stale,
    diagnostic=diagnostic)
  np.testing.assert_array_almost_equal(
    z_states_inferred.shape, [num_users, num_time_steps, 4])
  time_spent = time.time() - time_start

  z_states_reshaped = z_states_inferred.reshape((num_users*num_time_steps, 4))
  like = z_states_reshaped[
    range(num_users*num_time_steps), states.flatten()].reshape(states.shape)

  # Calculate AUPR
  score_pos = np.array(z_states_inferred[:, :, 2][states == 2]).flatten()
  score_neg = np.array(z_states_inferred[:, :, 2][states != 2]).flatten()
  scores = np.concatenate((score_pos, score_neg))
  labels = np.concatenate((np.ones_like(score_pos), np.zeros_like(score_neg)))
  auroc = metrics.roc_auc_score(labels, scores)
  av_precision = metrics.average_precision_score(labels, scores)

  log_like = np.mean(np.log(like+1E-9))

  log_like_obs = prequential.get_evidence_obs(
    observations, z_states_inferred, alpha, beta)

  if mpi_rank == 0:
    logger.info((
      f"{num_rounds:5} rounds for {num_users:10} users in {time_spent:10.2f} "
      f"seconds with log-like {log_like:10.2f}/{log_like_obs:10.2f} nats "
      f"and AUROC {auroc:5.3f} and AP {av_precision:5.3f}"))
  sys.stdout.flush()

  runner.log({
    "num_rounds": num_rounds,
    "time_spent": time_spent,
    "log_likelihood": log_like,
    "log_like_obs": log_like_obs,
    "AUROC": auroc,
    "AP": av_precision})


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    description='Compare statistics acrosss inference methods')
  parser.add_argument('--inference_method', type=str, default='fn',
                      choices=['fn', 'dummy', 'random', 'bp', 'dct', 'sib'],
                      help='Name of the inference method')
  parser.add_argument('--experiment_setup', type=str, default='single',
                      choices=['single', 'prequential'],
                      help='Name of the experiment_setup')
  parser.add_argument('--config_data', type=str, default='large_graph_02',
                      help='Name of the config file for the data')
  parser.add_argument('--config_model', type=str, default='model_02',
                      help='Name of the config file for the model')
  parser.add_argument('--name', type=str, default=None,
                      help=('Name of the experiments. WandB will set a random'
                            ' when left undefined'))
  parser.add_argument('--do_diagnosis', action='store_true')
  parser.add_argument('--dump_traces', action='store_true')
  parser.add_argument('--quick', action='store_true',
                      help=('include flag --quick to run a minimal version of'
                            'the code quickly, usually for debugging purpose'))

  # TODO make a better heuristic for this:
  numba.set_num_threads(2)

  args = parser.parse_args()

  configname_data = args.config_data
  configname_model = args.config_model
  fname_config_data = f"dpfn/config/{configname_data}.ini"
  fname_config_model = f"dpfn/config/{configname_model}.ini"
  data_dir = f"dpfn/data/{configname_data}/"
  do_abm = '_abm' in configname_data

  inf_method = args.inference_method
  # Set up locations to store results
  experiment_name = args.experiment_setup
  if args.quick:
    experiment_name += "_quick"
  results_dir_global = (
    f'results/{experiment_name}/{configname_data}__{configname_model}/')

  if mpi_rank == 0:
    util.maybe_make_dir(results_dir_global)
  if args.dump_traces:
    trace_dir_global = (
      f'results/trace_{experiment_name}/{configname_data}__{configname_model}/')
    util.maybe_make_dir(trace_dir_global)
    logger.info(f"Dump traces to results_dir_global {trace_dir_global}")
  else:
    trace_dir_global = None

  config_data = config.ConfigBase(fname_config_data)
  config_model = config.ConfigBase(fname_config_model)

  # Start WandB
  config_wandb = {
    "config_data_name": configname_data,
    "config_model_name": configname_model,
    "cpu_count": util.get_cpu_count(),
    "data": config_data.to_dict(),
    "model": config_model.to_dict(),
  }

  # WandB tags
  tags = [
    args.experiment_setup, inf_method, f"cpu{util.get_cpu_count()}"]
  tags.append("quick" if args.quick else "noquick")
  tags.append("local" if (os.getenv('SLURM_JOB_ID') is None) else "slurm")

  if mpi_rank == 0:
    runner_global = wandb.init(
      project="dpfn",
      notes=" ",
      name=args.name,
      tags=tags,
      config=config_wandb,
    )

    config_wandb = config.clean_hierarchy(dict(runner_global.config))
    config_wandb = util_experiments.set_noisy_test_params(config_wandb)
    logger.info(config_wandb)
  else:
    runner_global = util_wandb.WandbDummy()
    config_wandb = None

  config_wandb = comm_world.bcast(config_wandb, root=0)
  logger.info((
    f"Process {mpi_rank} has data_fraction_test "
    f"{config_wandb['data']['fraction_test']}"))

  logger.info(f"Logger filename {LOGGER_FILENAME}")
  logger.info(f"Saving to results_dir_global {results_dir_global}")
  logger.info(f"sweep_id: {os.getenv('SWEEPID')}")
  logger.info(f"slurm_id: {os.getenv('SLURM_JOB_ID')}")
  logger.info(f"slurm_name: {os.getenv('SLURM_JOB_NAME')}")
  logger.info(f"slurm_ntasks: {os.getenv('SLURM_NTASKS')}")

  if mpi_rank == 0:
    util_experiments.make_git_log()

  # Set random seed
  # TODO research interaction random seeds and multiprocessing
  seed_value = config_wandb.get("seed", None)
  random.seed(seed_value)
  np.random.seed(seed_value)

  if not do_abm:
    if not os.path.exists(data_dir):
      raise FileNotFoundError((
        f"{data_dir} not found. Current wd: {os.getcwd()}"))

    observations_all, contacts_all, states_all = data_load.load_jsons(data_dir)
  else:
    observations_all = contacts_all = []
    states_all = np.zeros((1, 1, 1))

  experiment_fn = None
  if args.experiment_setup == "single":
    experiment_fn = compare_inference_algorithms
  elif args.experiment_setup == "prequential":
    experiment_fn = compare_prequential_quarantine
  else:
    raise ValueError(
      (f"Not recognized experiment {args.experiment_setup}. Should be one of"
       f"['single','prequential']"))

  try:
    experiment_fn(
      inf_method,
      num_users=config_wandb["data"]["num_users"],
      num_time_steps=config_wandb["data"]["num_time_steps"],
      observations=observations_all,
      contacts=contacts_all,
      states=states_all,
      cfg=config_wandb,
      runner=runner_global,
      results_dir=results_dir_global,
      trace_dir=trace_dir_global,
      quick=args.quick,
      do_diagnosis=args.do_diagnosis,
      use_abm_simulator=do_abm,
      )
  except Exception as e:
    # This exception sends an WandB alert with the traceback and sweepid
    logger.info(f'Error repr: {repr(e)}')
    traceback_report = traceback.format_exc()
    wandb.alert(
      title=f"Error {os.getenv('SWEEPID')}-{os.getenv('SLURM_JOB_ID')}",
      text=(
        f"'{configname_data}', '{configname_model}', '{inf_method}'\n"
        + traceback_report)
    )
    raise e

  runner_global.finish()
