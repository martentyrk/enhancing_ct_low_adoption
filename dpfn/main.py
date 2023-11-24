import argparse
import numpy as np
import numba
import os
import random
import socket
import traceback
import wandb

from collections import defaultdict
from config import config
from experiments import (util_experiments)
from dpfn import LOGGER_FILENAME, logger
from dpfn import util
from dpfn import util_wandb
from experiments.compare_covasim import compare_policy_covasim
from experiments.compare_abm import compare_abm


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    description='Compare statistics acrosss inference methods')
  parser.add_argument('--inference_method', type=str, default='fn',
                      choices=[
                        'fn', 'dummy', 'random', 'dpct', 'fncpp'],
                      help='Name of the inference method')
  parser.add_argument('--simulator', type=str, default='abm',
                      choices=['abm', 'covasim'],
                      help='Name of the simulator')
  parser.add_argument('--config_data', type=str, default='intermediate_graph_abm_02',
                      help='Name of the config file for the data')
  parser.add_argument('--config_model', type=str, default='model_ABM01',
                      help='Name of the config file for the model')
  parser.add_argument('--name', type=str, default=None,
                      help=('Name of the experiments. WandB will set a random'
                            ' when left undefined'))
  parser.add_argument('--do_diagnosis', action='store_true')
  parser.add_argument('--dump_traces', action='store_true')

  # TODO make a better heuristic for this:
  # num_threads = max((util.get_cpu_count()-1, 1))
  num_threads = 16
  numba.set_num_threads(num_threads)
  logger.info(f"SLURM env N_TASKS: {os.getenv('SLURM_NTASKS')}")
  logger.info(f"Start with {num_threads} threads")

  args = parser.parse_args()

  configname_data = args.config_data
  configname_model = args.config_model
  fname_config_data = f"dpfn/config/{configname_data}.ini"
  fname_config_model = f"dpfn/config/{configname_model}.ini"
  data_dir = f"dpfn/data/{configname_data}/"

  inf_method = args.inference_method
  # Set up locations to store results
  experiment_name = 'run_prequential'
  results_dir_global = (
    f'results/{experiment_name}/{configname_data}__{configname_model}/')

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
  config_wandb = defaultdict(None)
  config_wandb['config_data_name'] = configname_data
  config_wandb['config_model_name'] = configname_model
  config_wandb['cpu_count'] = util.get_cpu_count()
  config_wandb['data'] = config_data.to_dict()
  config_wandb['model'] = config_model.to_dict()

  # WandB tags
  tags = [
    str(args.simulator), inf_method, f"cpu{util.get_cpu_count()}",
    configname_data, configname_model]
  tags.append("dump_traces" if args.dump_traces else "nodump")

  tags.append("local" if (os.getenv('SLURM_JOB_ID') is None) else "slurm")

  do_wandb = ('carbon' not in socket.gethostname())
  if do_wandb:
    runner_global = wandb.init(
      project="dpfn",
      notes=" ",
      name=args.name,
      tags=tags,
      config=config_wandb,
    )
    config_wandb = config.clean_hierarchy(dict(runner_global.config))
  else:
    runner_global = util_wandb.WandbDummy()
  config_wandb = util_experiments.set_noisy_test_params(config_wandb)
  config_wandb = util_experiments.convert_log_params(config_wandb)
  logger.info(config_wandb)

  logger.info(f"Logger filename {LOGGER_FILENAME}")
  logger.info(f"Saving to results_dir_global {results_dir_global}")
  logger.info(f"sweep_id: {os.getenv('SWEEPID')}")
  logger.info(f"slurm_id: {os.getenv('SLURM_JOB_ID')}")
  logger.info(f"slurm_name: {os.getenv('SLURM_JOB_NAME')}")
  logger.info(f"slurm_ntasks: {os.getenv('SLURM_NTASKS')}")

  # This line prints all git differences etc. No need for now.
  # util_experiments.make_git_log()

  if 'carbon' not in socket.gethostname():
    wandb.mark_preempting()

  # Set random seed
  seed_value = config_wandb.get("seed", -1)
  if seed_value > 0:
    random.seed(seed_value)
    np.random.seed(seed_value)
  else:
    seed_value = random.randint(0, 999)
  # Random number generator to pass as argument to some imported functions
  arg_rng = np.random.default_rng(seed=seed_value)

  try:
    if args.simulator == "abm":
      comparison_fn = compare_abm
    elif args.simulator == "covasim":
      comparison_fn = compare_policy_covasim
    else:
      raise ValueError(f"Unknown simulator {args.simulator}")

    # Run full comparison
    comparison_fn(
      inf_method,
      cfg=config_wandb,
      runner=runner_global,
      results_dir=results_dir_global,
      arg_rng=arg_rng,
      trace_dir=trace_dir_global,
      do_diagnosis=args.do_diagnosis
      )

  except Exception as e:
    # This exception sends an WandB alert with the traceback and sweepid
    logger.info(f'Error repr: {repr(e)}')
    traceback_report = traceback.format_exc()
    if slurmid := os.getenv('SLURM_JOB_ID'):
      wandb.alert(
        title=f"Error {os.getenv('SWEEPID')}-{slurmid}",
        text=(
          f"'{configname_data}', '{configname_model}', '{inf_method}'\n"
          + traceback_report)
      )
    raise e

  runner_global.finish()