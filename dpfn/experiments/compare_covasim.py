"""Compare inference methods on likelihood and AUROC, and run prequentially."""
import argparse
import covasim as cv
from mpi4py import MPI  # pytype: disable=import-error
import numpy as np
from dpfn.config import config
from dpfn.experiments import compare_stats, prequential, util_experiments
from dpfn import LOGGER_FILENAME, logger
from dpfn import util
from dpfn import util_wandb
import numba
import os
import psutil
import random
import threading
import time
import traceback
from typing import Any, Dict, Optional
import wandb


comm_world = MPI.COMM_WORLD
mpi_rank = comm_world.Get_rank()
num_proc = comm_world.Get_size()
print(f"num_proc {num_proc} mpi_rank {mpi_rank}")


class StoreSEIR(cv.Analyzer):
  """Store the SEIR rates for each day."""

  def __init__(self, num_days, *fargs, **kwargs):
    super().__init__(*fargs, **kwargs)
    self.t = np.zeros((num_days), dtype=np.float32)
    self.s_rate = np.zeros((num_days), dtype=np.float32)
    self.e_rate = np.zeros((num_days), dtype=np.float32)
    self.i_rate = np.zeros((num_days), dtype=np.float32)
    self.r_rate = np.zeros((num_days), dtype=np.float32)

    self.isolation_rate = np.zeros((num_days), dtype=np.float32)

    self.precisions = np.zeros((num_days), dtype=np.float32)
    self.recalls = np.zeros((num_days), dtype=np.float32)

    self.timestamps = np.zeros((num_days+1), dtype=np.float64)
    self._time_prev = time.time()

  def apply(self, sim):
    """Applies the analyser on the simulation object."""
    ppl = sim.people  # Shorthand
    num_people = len(ppl)
    day = sim.t

    self.t[day] = day
    self.s_rate[day] = ppl.susceptible.sum() / num_people
    self.e_rate[day] = (ppl.exposed.sum() - ppl.infectious.sum()) / num_people
    self.i_rate[day] = ppl.infectious.sum() / num_people
    self.r_rate[day] = ppl.recovered.sum() + ppl.dead.sum() / num_people

    isolated = np.logical_or(ppl.isolated, ppl.quarantined)
    true_positives = np.sum(np.logical_and(isolated, ppl.infectious))

    self.isolation_rate[day] = np.sum(isolated) / num_people

    # precision should be 1 when there are no false positives
    self.precisions[day] = (true_positives+1E-9) / (np.sum(isolated) + 1E-9)

    # Number of infected people in isolation over total number of infected
    self.recalls[day] = (true_positives+1E-9) / (np.sum(ppl.infectious) + 1E-9)
    self.timestamps[day] = time.time() - self._time_prev
    self._time_prev = time.time()

    logger.info((
      f"On day {day:3} recall is {self.recalls[day]:.2f} "
      f"at IR {self.i_rate[day] + self.e_rate[day]:.4f} "
      f"timediff {self.timestamps[day]:8.1f}"))


def compare_policy_covasim(
    inference_method: str,
    cfg: Dict[str, Any],
    runner,
    results_dir: str,
    trace_dir: Optional[str] = None,
    quick: bool = False,
    do_diagnosis: bool = False):
  """Compares different inference algorithms on the supplied contact graph."""
  del do_diagnosis

  num_time_steps = cfg["data"]["num_time_steps"]
  num_users = cfg["data"]["num_users"]
  # Daily fraction of the population that gets a test
  fraction_test = cfg["data"]["fraction_test"]
  # Probability of the person being lost-to-follow-up after a test
  loss_prob = cfg["data"]["loss_prob"]

  num_days_window = cfg["model"]["num_days_window"]
  quantization = cfg["model"]["quantization"]
  num_rounds = cfg["model"]["num_rounds"]

  seed = cfg.get("seed", 123)

  logger.info((
    f"Settings at experiment: {quantization:.0f} quant, at {fraction_test}% "
    f"seed {seed}"))

  inference_func, do_baseline = compare_stats.make_inference_func(
    inference_method, num_users, cfg, trace_dir=trace_dir)

  sensitivity = 1. - cfg["data"]["alpha"]

  if cfg["model"]["beta"] == 0:
    logger.warning("COVASIM does not model false positives yet, setting to 0")

  cfg["model"]["beta"] = 0
  cfg["data"]["beta"] = 0
  assert num_time_steps == 91, "hardcoded 91 days for now, TODO: fix this"

  if quick:
    num_rounds = 2

  # Arrays to accumulate statistics

  t0 = time.time()

  # Make simulator
  pop_infected = 25
  if num_users >= 100000:
    pop_infected = 50

  pars = {
    "pop_type": 'hybrid',
    "pop_size": num_users,
    "pop_infected": pop_infected,
    "start_day": '2020-02-01',
    "end_day": '2020-05-01',
  }

  def subtarget_func_baseline(sim, history):
    """Subtarget function for testing.

    This function is run every day after the contacts are sampled and before
    the tests are run. It returns a dictionary with the keys 'inds' and 'vals'.
    """
    # cv.true() returns indices of people matching this condition
    exposed = cv.true(sim.people.exposed)
    # Everyone in the population -- equivalent to np.arange(len(sim.people))
    inds = sim.people.uid
    vals = np.ones(len(sim.people))  # Create the array
    vals[exposed] = 100  # Probability for testing
    output = dict(inds=inds, vals=vals)
    return output, history

  def subtarget_func_inference(sim, history):
    """Subtarget function for testing.

    This function is run every day after the contacts are sampled and before
    the tests are run. It returns a dictionary with the keys 'inds' and 'vals'.
    """
    assert isinstance(history, dict)
    # Slice window

    if sim.t > 3:
      contacts = sim.people.contacts

      contacts_add = []
      for layerkey in contacts.keys():
        ones_vec = np.ones_like(contacts[layerkey]['p1'])
        contacts_add.append(np.stack((
          contacts[layerkey]['p1'],
          contacts[layerkey]['p2'],
          sim.t*ones_vec,
          ones_vec,
          ), axis=1))
        contacts_add.append(np.stack((
          contacts[layerkey]['p2'],
          contacts[layerkey]['p1'],
          sim.t*ones_vec,
          ones_vec,
          ), axis=1))

      contacts_today = np.concatenate(contacts_add, axis=0)

      history['contacts'] = np.concatenate(
        (history['contacts'], contacts_today), axis=0)

      # TODO make this more efficient
      num_days = min((num_days_window, sim.t))
      history['contacts'] = history['contacts'][
        history['contacts'][:, 2] > sim.t - num_days]
      history['observations'] = history['observations'][
        history['observations'][:, 1] > sim.t - num_days]

      contacts_rel = np.copy(history['contacts']).astype(np.int32)
      obs_rel = np.copy(history['observations']).astype(np.int32)

      contacts_rel[:, 2] = contacts_rel[:, 2] - sim.t + num_days - 1
      obs_rel[:, 1] = obs_rel[:, 1] - sim.t + num_days - 1

      # TODO remove this line. Follow up with github issue
      # https://github.com/InstituteforDiseaseModeling/covasim/issues/400
      is_not_double = np.logical_not(contacts_rel[:, 0] == contacts_rel[:, 1])
      contacts_rel = contacts_rel[is_not_double]

      # assert 0 <= contacts_rel[:, 2].min() <= sim.t, (
      #   f"Earliest contact {contacts_rel[:, 2].min()} is before {sim.t}")
      # assert 0 <= obs_rel[:, 1].min() <= sim.t, (
      #   f"Earliest obs {obs_rel[:, 1].min()} is before {sim.t}")

      # Add +1 so the model predicts one day into the future
      _, pred = inference_func(
        observations_list=obs_rel,
        contacts_list=contacts_rel,
        num_updates=num_rounds,
        num_time_steps=num_days + 1)
      rank_score = pred[:, -1, 1] + pred[:, -1, 2]
    else:
      # For the first few days of a simulation, just test randomly
      rank_score = np.ones(num_users) + np.random.rand(num_users)

    inds = sim.people.uid  # Everyone in the population
    output = dict(inds=inds, vals=rank_score)
    return output, history

  # TODO: fix this to new keyword
  subtarget_func = (
    subtarget_func_baseline if do_baseline else subtarget_func_inference)

  test_intervention = cv.test_num(
    daily_tests=int(fraction_test*num_users),
    do_plot=False,
    sensitivity=sensitivity,  # 1 - false_negative_rate
    loss_prob=loss_prob,  # probability of the person being lost-to-follow-up
    subtarget=subtarget_func)

  # Create, run, and plot the simulations
  sim = cv.Sim(
    pars,
    interventions=test_intervention,
    analyzers=StoreSEIR(num_days=num_time_steps, label='analysis'))

  # COVASIM run() runs the entire simulation, including the initialization
  sim.set_seed(seed=seed)
  sim.run(reset_seed=True)

  analysis = sim.get_analyzer('analysis')
  infection_rates = analysis.e_rate + analysis.i_rate

  prequential.dump_results_json(
    datadir=results_dir,
    cfg=cfg,
    precisions=analysis.precisions.tolist(),
    recalls=analysis.recalls.tolist(),
    exposed_rates=analysis.e_rate.tolist(),
    infection_rates=infection_rates.tolist(),
    num_quarantined=analysis.isolation_rate.tolist(),
    inference_method=inference_method,
    name=runner.name,
    pir=float(np.max(infection_rates)),
    quantization=quantization,
    seed=cfg.get("seed", -1))

  if mpi_rank == 0:
    time_pir, pir = np.argmax(infection_rates), np.max(infection_rates)
    logger.info(f"At day {time_pir} peak infection rate is {pir:.5f}")

    _, loadavg5, loadavg15 = os.getloadavg()
    swap_use = psutil.swap_memory().used / (1024.0 ** 3)

    time_spent = time.time() - t0
    logger.info(f"With {num_rounds} rounds, PIR {pir:5.2f}")
    runner.log({
      "time_spent": time_spent,
      "pir_mean": pir,
      "loadavg5": loadavg5,
      "loadavg15": loadavg15,
      "swap_use": swap_use,
      "recall": np.nanmean(analysis.recalls),
      "precision": np.nanmean(analysis.precisions)})


def log_to_wandb(wandb_runner):
  """Logs system statistics to wandb every minute."""
  while True:
    loadavg1, loadavg5, _ = os.getloadavg()
    swap_use = psutil.swap_memory().used / (1024.0 ** 3)

    wandb_runner.log({
      "loadavg1": loadavg1,
      "loadavg5": loadavg5,
      "swap_use": swap_use})
    time.sleep(60)


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    description='Compare statistics acrosss inference methods')
  parser.add_argument('--inference_method', type=str, default='fn',
                      choices=[
                        'fn', 'dummy', 'random', 'bp', 'dct', 'dpct', 'sib',
                        'gibbs'],
                      help='Name of the inference method')
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
  logger.info(f"SLURM env N_TASKS: {os.getenv('SLURM_NTASKS')}")
  numba.set_num_threads(max((util.get_cpu_count()-1, 1)))

  args = parser.parse_args()

  configname_data = args.config_data
  configname_model = args.config_model
  fname_config_data = f"dpfn/config/{configname_data}.ini"
  fname_config_model = f"dpfn/config/{configname_model}.ini"
  data_dir = f"dpfn/data/{configname_data}/"

  inf_method = args.inference_method
  # Set up locations to store results

  experiment_name = "prequential"
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
    experiment_name, inf_method, f"cpu{util.get_cpu_count()}",
    configname_data, configname_model]
  tags.append("quick" if args.quick else "noquick")
  tags.append("dump_traces" if args.dump_traces else "noquick")
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
    # config_wandb = {
    #   "data": config_data.to_dict(), "model": config_model.to_dict()}

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
  seed_value = config_wandb.get("seed", None)
  random.seed(seed_value)
  np.random.seed(seed_value)
  # Random number generator to pass as argument to some imported functions
  arg_rng = np.random.default_rng(seed=seed_value)

  try:
    daemon_wandb = threading.Thread(
      target=log_to_wandb, args=(runner_global,), daemon=True)
    daemon_wandb.start()

    compare_policy_covasim(
      inf_method,
      cfg=config_wandb,
      runner=runner_global,
      results_dir=results_dir_global,
      trace_dir=trace_dir_global,
      quick=args.quick,
      do_diagnosis=args.do_diagnosis,
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
