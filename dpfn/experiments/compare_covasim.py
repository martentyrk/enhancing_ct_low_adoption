import covasim as cv
import numpy as np
import os
import psutil
import threading
import time
from typing import Any, Dict, Optional
import warnings

from dpfn.experiments import (
  prequential, util_covasim)
from dpfn import logger

def compare_policy_covasim(
    inference_method: str,
    cfg: Dict[str, Any],
    runner,
    results_dir: str,
    trace_dir: Optional[str] = None,
    do_diagnosis: bool = False):
  """Compares different inference algorithms on the supplied contact graph."""
  del do_diagnosis

  # Start a daemon to log to wandb
  daemon_wandb = threading.Thread(
    target=util_covasim.log_to_wandb, args=(runner,), daemon=True)
  daemon_wandb.start()

  num_time_steps = cfg["data"]["num_time_steps"]
  num_users = cfg["data"]["num_users"]
  # Daily fraction of the population that gets a test
  fraction_test = cfg["data"]["fraction_test"]
  # Probability of the person being lost-to-follow-up after a test
  loss_prob = cfg["data"]["loss_prob"]
  policy_weight_01 = cfg["model"]["policy_weight_01"]
  policy_weight_02 = cfg["model"]["policy_weight_02"]
  policy_weight_03 = cfg["model"]["policy_weight_03"]
  t_start_quarantine = cfg["data"]["t_start_quarantine"]

  num_days_window = cfg["model"]["num_days_window"]
  quantization = cfg["model"]["quantization"]
  num_rounds = cfg["model"]["num_rounds"]

  seed = cfg.get("seed", 123)

  logger.info((
    f"Settings at experiment: {quantization:.0f} quant, at {fraction_test}% "
    f"seed {seed}"))

  inference_func, do_random = make_inference_func(
    inference_method, num_users, cfg, trace_dir=trace_dir)

  sensitivity = 1. - cfg["data"]["alpha"]

  if cfg["model"]["beta"] == 0:
    logger.warning("COVASIM does not model false positives yet, setting to 0")

  cfg["model"]["beta"] = 0
  cfg["data"]["beta"] = 0
  assert num_time_steps == 91, "hardcoded 91 days for now, TODO: fix this"

  t0 = time.time()

  # Make simulator
  pop_infected = 25
  if num_users >= 100000:
    pop_infected = 50
  if num_users > 500000:
    pop_infected = 100

  pars = {
    "pop_type": 'hybrid',
    "pop_size": num_users,
    "pop_infected": pop_infected,
    "start_day": '2020-02-01',
    "end_day": '2020-05-01',
  }

  def subtarget_func_random(sim, history):
    """Subtarget function for random testing.

    This function is run every day after the contacts are sampled and before
    the tests are run. It returns a dictionary with the keys 'inds' and 'vals'.
    """
    inds = sim.people.uid
    vals = np.random.rand(len(sim.people))  # Create the array

    # Uncomment these lines to deterministically test people that are exposed
    # vals = np.ones(len(sim.people))  # Create the array
    # exposed = cv.true(sim.people.exposed)
    # vals[exposed] = 100  # Probability for testing
    return {'inds': inds, 'vals': vals}, history

  def subtarget_func_inference(sim, history):
    """Subtarget function for testing.

    This function is run every day after the contacts are sampled and before
    the tests are run. It returns a dictionary with the keys 'inds' and 'vals'.
    """
    assert isinstance(history, dict)
    # Slice window

    # TODO(rob): no need to compute this every timestep
    users_age = np.digitize(
      sim.people.age, np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90,])) - 1
    users_age = users_age.astype(np.int32)

    if sim.t > t_start_quarantine:
      contacts = sim.people.contacts

      contacts_add = []
      for layerkey in contacts.keys():
        ones_vec = np.ones_like(contacts[layerkey]['p1'])
        contacts_add.append(np.stack((
          contacts[layerkey]['p1'], #p1 stands for "sources"
          contacts[layerkey]['p2'], #p2 stands for "targets"
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

      if trace_dir and sim.t == 15:
        fname = os.path.join(trace_dir, f"contacts_{sim.t}.npz")
        np.savez(
          fname, contacts=contacts_rel, observations=obs_rel)
        logger.info(f"Dumped traces to {fname}")

      # Add +1 so the model predicts one day into the future
      t_start = time.time()
      pred, contacts_age = inference_func(
        observations_list=obs_rel,
        contacts_list=contacts_rel,
        num_updates=num_rounds,
        num_time_steps=num_days + 1,
        users_age=users_age)
      rank_score = pred[:, -1, 1] + pred[:, -1, 2]
      time_spent = time.time() - t_start

      if np.any(np.abs(
          [policy_weight_01, policy_weight_02, policy_weight_03]) > 1E-9):
        assert contacts_age is not None, f"Contacts age is {contacts_age}"
        rank_score += (
          policy_weight_01 * contacts_age[:, 1] / 10
          + policy_weight_02 * contacts_age[:, 0] / 10
          + policy_weight_03 * users_age / 10)

      # Track some metrics here:
      states_today = 3*np.ones(num_users, dtype=np.int32)
      states_today[sim.people.exposed] = 2
      states_today[sim.people.infectious] = 1
      states_today[sim.people.susceptible] = 0

      p_at_state = pred[range(num_users), -1, states_today]
      history['likelihoods_state'][sim.t] = np.mean(np.log(p_at_state+1E-9))
      history['ave_prob_inf_at_inf'][sim.t] = np.mean(
        p_at_state[states_today == 2])
      history['time_inf_func'][sim.t] = time_spent
    else:
      # For the first few days of a simulation, just test randomly
      rank_score = np.ones(num_users) + np.random.rand(num_users)

    output = {'inds': sim.people.uid, 'vals': rank_score}
    return output, history

  # TODO: fix this to new keyword
  subtarget_func = (
    subtarget_func_random if do_random else subtarget_func_inference)

  test_intervention = cv.test_num(
    daily_tests=int(fraction_test*num_users),
    do_plot=False,
    sensitivity=sensitivity,  # 1 - false_negative_rate
    loss_prob=loss_prob,  # probability of the person being lost-to-follow-up
    subtarget=subtarget_func,
    label='intervention_history')

  # Create, run, and plot the simulations
  sim = cv.Sim(
    pars,
    interventions=test_intervention,
    analyzers=util_covasim.StoreSEIR(num_days=num_time_steps, label='analysis'))

  # COVASIM run() runs the entire simulation, including the initialization
  sim.set_seed(seed=seed)
  sim.run(reset_seed=True)

  analysis = sim.get_analyzer('analysis')
  history_intv = sim.get_intervention('intervention_history').history
  infection_rates = analysis.e_rate + analysis.i_rate
  peak_crit_rate = np.max(analysis.crit_rate)

  # Calculate PIR and Drate
  time_pir, pir = np.argmax(infection_rates), np.max(infection_rates)
  total_drate = sim.people.dead.sum() / len(sim.people)

  prequential.dump_results_json(
    datadir=results_dir,
    cfg=cfg,
    precisions=analysis.precisions.tolist(),
    recalls=analysis.recalls.tolist(),
    exposed_rates=analysis.e_rate.tolist(),
    infection_rates=infection_rates.tolist(),
    num_quarantined=analysis.isolation_rate.tolist(),
    critical_rates=analysis.crit_rate.tolist(),
    likelihoods_state=history_intv['likelihoods_state'].tolist(),
    ave_prob_inf_at_inf=history_intv['ave_prob_inf_at_inf'].tolist(),
    inference_method=inference_method,
    name=runner.name,
    pir=float(np.max(infection_rates)),
    pcr=float(np.max(analysis.crit_rate)),
    total_drate=float(total_drate),
    quantization=quantization,
    seed=cfg.get("seed", -1))

  logger.info((
    f"At day {time_pir} peak infection rate is {pir:.5f} "
    f"and total death rate is {total_drate:.5f}"))

  _, loadavg5, loadavg15 = os.getloadavg()
  swap_use = psutil.swap_memory().used / (1024.0 ** 3)

  time_spent = time.time() - t0
  logger.info(f"With {num_rounds} rounds, PIR {pir:5.2f}")
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    results = {
      "time_spent": time_spent,
      "pir_mean": pir,
      "pcr": peak_crit_rate,
      "total_drate": total_drate,
      "loadavg5": loadavg5,
      "loadavg15": loadavg15,
      "swap_use": swap_use,
      "recall": np.nanmean(analysis.recalls[10:]),
      "precision": np.nanmean(analysis.precisions[10:])}
  runner.log(results)

  return results