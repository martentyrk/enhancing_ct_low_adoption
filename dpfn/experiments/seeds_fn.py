"""Runs inference methods with multiple random seeds."""
import argparse
from mpi4py import MPI  # pytype: disable=import-error
import numpy as np
from dpfn.config import config
from dpfn.data import data_load
from dpfn.experiments import prequential, util_experiments
from dpfn import constants
from dpfn import LOGGER_FILENAME, logger
from dpfn import util
from dpfn import util_wandb
import glob
import numba
import os
import random
from sklearn import metrics
import sys
import time
import traceback
from typing import Any, Dict, List, Optional
import wandb

TRACEMALLOC = False

comm_world = MPI.COMM_WORLD
mpi_rank = comm_world.Get_rank()
num_proc = comm_world.Get_size()
print(f"num_proc {num_proc} mpi_rank {mpi_rank}")


def strip_seed(filename: str) -> str:
  return os.path.splitext(os.path.basename(filename))[0][-8:]


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
      dp_noise=-1.,
      clip_margin=-1.,
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


def compare_seeds(
    seed: int,
    num_seeds: int,
    inference_method: str,
    num_users: int,
    num_time_steps: int,
    observations: List[constants.Observation],
    contacts: List[constants.Contact],
    states: np.ndarray,
    cfg: Dict[str, Any],
    runner,
    trace_dir: str):
  """Compares different inference algorithms on the supplied contact graph."""
  # Contacts on last day are not of influence
  def filter_fn(datum):
    return datum[2] < (num_time_steps - 1)
  contacts = list(filter(filter_fn, contacts))

  contacts = util.make_default_array(
    contacts, dtype=np.int32, rowlength=4)
  observations = util.make_default_array(
    observations, dtype=np.int32, rowlength=3)

  alpha = cfg["model"]["alpha"]
  beta = cfg["model"]["beta"]

  num_rounds = cfg["model"]["num_rounds"]
  num_users = int(num_users)

  # Data and simulator params
  inference_func, _ = make_inference_func(
    inference_method, num_users, cfg, trace_dir)

  if mpi_rank == 0:
    logger.info(f"Start inference method {inference_method}")

  for num_seed in range(num_seeds):
    seed_value = seed + 1000 * num_seed
    random.seed(seed_value)
    np.random.seed(seed_value)

    time_start = time.time()
    z_states_inferred = inference_func(
      np.array(observations),
      np.array(contacts),
      num_rounds,
      num_time_steps,
      users_stale=None,
      diagnostic=None)

    if mpi_rank == 0:
      np.testing.assert_array_almost_equal(
        z_states_inferred.shape, [num_users, num_time_steps, 4])
      time_spent = time.time() - time_start

      z_states_reshaped = z_states_inferred.reshape(
        (num_users*num_time_steps, 4))
      like = z_states_reshaped[
        range(num_users*num_time_steps), states.flatten()].reshape(states.shape)

      fname = f"{trace_dir}/states_{seed_value:08d}.npy"
      np.save(fname, z_states_inferred.astype(np.float32))

      # Calculate AUPR
      score_pos = np.array(z_states_inferred[:, :, 2][states == 2]).flatten()
      score_neg = np.array(z_states_inferred[:, :, 2][states != 2]).flatten()
      scores = np.concatenate((score_pos, score_neg))
      labels = np.concatenate(
        (np.ones_like(score_pos), np.zeros_like(score_neg)))
      auroc = metrics.roc_auc_score(labels, scores)
      av_precision = metrics.average_precision_score(labels, scores)

      log_like = np.mean(np.log(like+1E-9))

      log_like_obs = prequential.get_evidence_obs(
        observations, z_states_inferred, alpha, beta)

      logger.info((
        f"{num_rounds:5} rounds for {num_users:10} users in {time_spent:10.2f} "
        f"seconds with log-like {log_like:10.2f}/{log_like_obs:10.2f} nats "
        f"and AUROC {auroc:5.3f} and AP {av_precision:5.3f}"))

      runner.log({
        "seed_value": seed_value,
        "time_spent": time_spent,
        "log_likelihood": log_like,
        "log_like_obs": log_like_obs,
        "AUROC": auroc,
        "AP": av_precision})
    sys.stdout.flush()


def analyse_seeds(
    trace_dir: str):
  """Analyse the seeds."""
  logger.info(f"Analyse seeds in {trace_dir}")

  # Find fnames
  fnames = list(sorted(glob.glob(f"{trace_dir}/states_*.npy")))
  seed0 = strip_seed(fnames[0])

  z_ref = 0.0
  for num_file, fname in enumerate(fnames):

    if num_file == 0:
      z_ref = np.load(fname)
    else:
      z = np.load(fname)
      dist = np.mean(np.abs(z - z_ref))
      logger.info(
        f"Distance between {strip_seed(fname)} and {seed0} is {dist:8.3f}")


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    description='Compare statistics acrosss inference methods')
  parser.add_argument('--inference_method', type=str, default='fn',
                      choices=['fn', 'dummy', 'random', 'bp', 'dct', 'sib'],
                      help='Name of the inference method')
  parser.add_argument('--config_data', type=str, default='large_graph_02',
                      help='Name of the config file for the data')
  parser.add_argument('--config_model', type=str, default='model_02',
                      help='Name of the config file for the model')
  parser.add_argument('--name', type=str, default=None,
                      help=('Name of the experiments. WandB will set a random'
                            ' when left undefined'))
  parser.add_argument('--num_seeds', type=int, default=8,
                      help='Number of seeds to run')
  parser.add_argument('--do_inference', action='store_true',
                      help='Do inference')

  # TODO make a better heuristic for this:
  numba.set_num_threads(2)

  args = parser.parse_args()

  configname_data = args.config_data
  configname_model = args.config_model
  fname_config_data = f"dpfn/config/{configname_data}.ini"
  fname_config_model = f"dpfn/config/{configname_model}.ini"
  data_dir = f"dpfn/data/{configname_data}/"
  inf_method = args.inference_method
  # Set up locations to store results

  # Dump seeds here
  trace_dir_global = (
    f'results/seeds_{inf_method}/{configname_data}__{configname_model}/')
  logger.info(f"Dump traces to results_dir_global {trace_dir_global}")
  if mpi_rank == 0:
    util.maybe_make_dir(trace_dir_global)

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
  tags = ["seeds", inf_method, f"cpu{util.get_cpu_count()}"]
  tags.append("local" if (os.getenv('SLURM_JOB_ID') is None) else "slurm")

  if mpi_rank == 0:
    runner_global = wandb.init(
      settings=wandb.Settings(start_method="fork"),
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
  logger.info(f"sweep_id: {os.getenv('SWEEPID')}")
  logger.info(f"slurm_id: {os.getenv('SLURM_JOB_ID')}")
  logger.info(f"slurm_name: {os.getenv('SLURM_JOB_NAME')}")
  logger.info(f"slurm_ntasks: {os.getenv('SLURM_NTASKS')}")

  if mpi_rank == 0:
    util_experiments.make_git_log()

  # Set random seed
  # TODO research interaction random seeds and multiprocessing

  if not os.path.exists(data_dir):
    raise FileNotFoundError((
      f"{data_dir} not found. Current wd: {os.getcwd()}"))

  observations_all, contacts_all, states_all = data_load.load_jsons(data_dir)

  try:
    if args.do_inference:
      compare_seeds(
        config_wandb.get("seed", 123),
        args.num_seeds,
        inf_method,
        num_users=config_wandb["data"]["num_users"],
        num_time_steps=config_wandb["data"]["num_time_steps"],
        observations=observations_all,
        contacts=contacts_all,
        states=states_all,
        cfg=config_wandb,
        runner=runner_global,
        trace_dir=trace_dir_global,
        )

    analyse_seeds(trace_dir_global)
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
