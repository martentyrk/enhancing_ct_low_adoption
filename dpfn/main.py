import argparse
import numpy as np
import numba
import os
import random
import socket
import traceback
import wandb
import json
from collections import defaultdict
from config import config
from experiments import (util_experiments)
from dpfn import LOGGER_FILENAME, logger
from dpfn import util
from dpfn import util_wandb
from experiments.compare_covasim import compare_policy_covasim
from experiments.compare_abm import compare_abm
from experiments.model_utils import get_model
import torch


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
  parser.add_argument('--app_users_fraction', type=float, default=None)
  parser.add_argument('--modify_contacts', action='store_true')
  parser.add_argument('--age_baseline', action='store_true')
  parser.add_argument('--mean_baseline', action='store_true')
  parser.add_argument('--static_baseline', action='store_true')
  parser.add_argument('--static_baseline_path', type=str, default='dpfn/data/static_age_baseline_results')
  parser.add_argument('--seed_value', type=int, default=None)
  parser.add_argument('--model', 
                      type=str, 
                      help="Type of deep learning model to apply to FN",
                      default=None,
                      choices=['gcn'])

  # TODO make a better heuristic for this:
  # num_threads = max((util.get_cpu_count()-1, 1))
  num_threads = 16
  numba.set_num_threads(num_threads)
  logger.info(f"SLURM env N_TASKS: {os.getenv('SLURM_NTASKS')}")
  logger.info(f"Start with {num_threads} threads")

  args = parser.parse_args()
  
  #Both baselines should not be used at the same time.
  assert sum([args.age_baseline, args.mean_baseline]) <= 1
  
  configname_data = args.config_data
  configname_model = args.config_model
  fname_config_data = f"dpfn/config/{configname_data}.ini"
  fname_config_model = f"dpfn/config/{configname_model}.ini"
  data_dir = f"dpfn/data/{configname_data}/"

  inf_method = args.inference_method

  config_data = config.ConfigBase(fname_config_data)
  config_model = config.ConfigBase(fname_config_model)

  # Start WandB
  config_wandb = defaultdict(None)
  config_wandb['config_data_name'] = configname_data
  config_wandb['config_model_name'] = configname_model
  config_wandb['cpu_count'] = util.get_cpu_count()
  config_wandb['data'] = config_data.to_dict()
  config_wandb['model'] = config_model.to_dict()
  
  if args.app_users_fraction:
    config_wandb["data"]["app_users_fraction"] = float(args.app_users_fraction)

  # WandB tags
  tags = [
    str(args.simulator), inf_method, f"cpu{util.get_cpu_count()}",
    configname_data, configname_model]
  tags.append("dump_traces" if args.dump_traces else "nodump")

  tags.append("local" if (os.getenv('SLURM_JOB_ID') is None) else "slurm")
  
  do_wandb = 'int' not in socket.gethostname()
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
  # Prepare model if model given
  
  if args.model:
    dl_model = get_model(args.model)
    dl_model.load_state_dict(torch.load(f"dpfn/config/{args.model}.pt"))
  else:
    dl_model = None
  
     # Set random seed
  seed_value = config_wandb.get("seed", -1)
  if seed_value > 0:
    random.seed(seed_value)
    np.random.seed(seed_value)
  else:
    if args.seed_value:
      seed_value = args.seed_value
    else:
      seed_value = random.randint(0, 999)
  # Random number generator to pass as argument to some imported functions
  arg_rng = np.random.default_rng(seed=seed_value)
  
  # Set up locations to store results
  if args.static_baseline:
    experiment_name = 'run_abm_static_age_seed_'+ str(seed_value)
  elif args.mean_baseline:
    experiment_name = 'run_abm_mean_collect_seed_'+ str(seed_value)
  elif args.age_baseline:
    experiment_name = 'run_abm_age_seed_'+ str(seed_value)
  elif args.model:
    experiment_name = 'run_abm_age_seed_model' + str(args.model) + '_seed_' + str(seed_value)
  else:
    experiment_name = 'run_abm_seed'+ str(seed_value)
  
  if args.app_users_fraction:
    experiment_name = experiment_name + "_adaption_" + str(args.app_users_fraction)
  elif 'app_users_fraction_wandb' in config_wandb:
    experiment_name = experiment_name + "_adaption_" + str(config_wandb.get('app_users_fraction_wandb', -1))
  
  results_dir_global = (
    f'results/{experiment_name}/{configname_data}_{configname_model}/')

  # This line prints all git differences etc. No need for now.
  # util_experiments.make_git_log()

  if do_wandb:
    wandb.mark_preempting()

  util.maybe_make_dir(results_dir_global)
  if args.dump_traces:
    trace_dir_global = (
      f'../../../../scratch-shared/mturk/datadump_long/trace_high_mem_{experiment_name}')
    util.maybe_make_dir(trace_dir_global)
    logger.info(f"Dump traces to results_dir_global {trace_dir_global}")
  else:
    trace_dir_global = None

  
  logger.info(f"Logger filename {LOGGER_FILENAME}")
  logger.info(f"Saving to results_dir_global {results_dir_global}")
  logger.info(f"sweep_id: {os.getenv('SWEEPID')}")
  logger.info(f"slurm_id: {os.getenv('SLURM_JOB_ID')}")
  logger.info(f"slurm_name: {os.getenv('SLURM_JOB_NAME')}")
  logger.info(f"slurm_ntasks: {os.getenv('SLURM_NTASKS')}")

  static_baseline_value = -1.0

  if args.static_baseline:
    if sum([args.age_baseline, args.mean_baseline]) != 1:
      raise ValueError('At least age_baseline or mean_baseline need to be selected.')
    overall_baseline = []
    
    for subdir, dirs, files in os.walk(args.static_baseline_path):
      for file in files:
        if file.split('.')[-1] == 'jl':
          filename = os.path.join(subdir, file)
          f = open(filename)
          static_values_dict = json.load(f)
    
          mean_baseline_value = static_values_dict['run_mean_baseline']
          if mean_baseline_value > 0:
            overall_baseline.append(mean_baseline_value)
          else:
            age_baseline_values = static_values_dict['running_mean_age_groups']
            overall_baseline.append(age_baseline_values)
            
    static_baseline_value = np.mean(np.array(overall_baseline), axis=0, dtype=np.float32)
  
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
      do_diagnosis=args.do_diagnosis,
      modify_contacts=args.modify_contacts,
      run_mean_baseline=args.mean_baseline,
      run_age_baseline=args.age_baseline,
      static_baseline_value=static_baseline_value,
      dl_model = dl_model,
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