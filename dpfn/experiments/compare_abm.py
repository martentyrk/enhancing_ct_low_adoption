"""Compare inference methods on likelihood and AUROC, and run prequentially."""
import copy
import numpy as np
import os
import psutil
from sklearn import metrics
import time
import tqdm
from typing import Any, Dict, Optional
import warnings
from dpfn.experiments import (
  prequential, util_experiments, util_dataset)
from dpfn import logger
from dpfn import simulator

def compare_abm(
    inference_method: str,
    cfg: Dict[str, Any],
    runner,
    results_dir: str,
    arg_rng: int,
    trace_dir: Optional[str] = None,
    do_diagnosis: bool = False):
  """Compares different inference algorithms on the supplied contact graph."""
  num_users = cfg["data"]["num_users"]
  num_time_steps = cfg["data"]["num_time_steps"]

  num_days_window = cfg["model"]["num_days_window"]
  quantization = cfg["model"]["quantization"]

  num_rounds = cfg["model"]["num_rounds"]
  policy_weight_01 = cfg["model"]["policy_weight_01"]
  policy_weight_02 = cfg["model"]["policy_weight_02"]
  policy_weight_03 = cfg["model"]["policy_weight_03"]
  rng_seed = cfg.get("seed", 123)

  fraction_test = cfg["data"]["fraction_test"]
  
  app_users_fraction = cfg["data"]["app_users_fraction"]
  
  # When doing a sweep, then use parameters from there.
  if cfg.get('app_users_fraction_wandb'):
    app_users_fraction = cfg['app_users_fraction_wandb']
  
  assert app_users_fraction > 0 and app_users_fraction <= 1.0
  assert isinstance(app_users_fraction, float)

  # Data and simulator params
  num_days_quarantine = cfg["data"]["num_days_quarantine"]
  t_start_quarantine = cfg["data"]["t_start_quarantine"]

  logger.info((
    f"Settings at experiment: {quantization:.0f} quant, at {fraction_test}%"))

  diagnostic = runner if do_diagnosis else None

  sim = simulator.ABMSimulator(
    num_time_steps, num_users, app_users_fraction, rng_seed)
  users_age = -1*np.ones((num_users), dtype=np.int32)
  
  app_users = sim.get_app_users()
  app_user_ids = np.nonzero(app_users)[0]
  # How many users there actually are, take that from app_user_ids.
  app_user_frac_num = app_user_ids.shape[0]
  logger.info(f"Number of app users: {app_user_frac_num}")
  
  inference_func, do_random_quarantine = util_experiments.make_inference_func(
    inference_method, app_user_frac_num, cfg, user_ids=app_user_ids, trace_dir=trace_dir)

  # Set conditional distributions for observations
  p_obs_infected = np.array(
    [cfg["data"]["alpha"], 1-float(cfg["data"]["alpha"])], dtype=np.float32)
  p_obs_not_infected = np.array(
    [1-float(cfg["data"]["beta"]), cfg["data"]["beta"]], dtype=np.float32)

  # Arrays to accumulate statistics
  pir_running = 0.
  precisions = np.zeros((num_time_steps))
  recalls = np.zeros((num_time_steps))
  infection_rates = np.zeros((num_time_steps))
  exposed_rates = np.zeros((num_time_steps))
  critical_rates = np.zeros((num_time_steps))
  likelihoods_state = np.zeros((num_time_steps))
  ave_prob_inf = np.zeros((num_time_steps))
  ave_prob_inf_at_inf = np.zeros((num_time_steps))
  ave_precision = np.zeros((num_time_steps))
  num_quarantined = np.zeros((num_time_steps), dtype=np.int32)
  num_tested = np.zeros((num_time_steps), dtype=np.int32)

  # Placeholder for tests on first day
  z_states_inferred = np.zeros((num_users, 1, 4))
  user_quarantine_ends = -1*np.ones((num_users), dtype=np.int32)
  contacts_age = np.zeros((num_users, 2), dtype=np.int32)

  logger.info(f"Do random quarantine? {do_random_quarantine}")
  t0 = time.time()

  logger.info((
    f"Start simulation with {num_rounds} updates"))

  for t_now in tqdm.trange(1, num_time_steps):
    t_start_loop = time.time()

    sim.step()
    if t_now == 1:
      users_age = sim.get_age_users()

    # Number of days to use for inference
    num_days = min((t_now + 1, num_days_window))
    # When num_days exceeds t_now, then offset should start counting at 0
    days_offset = t_now + 1 - num_days
    assert 0 <= days_offset <= num_time_steps

    # For each day, t_now, only receive obs up to and including 't_now-1'
    assert sim.get_current_day() == t_now
    rank_score = (z_states_inferred[:, -1, 1] + z_states_inferred[:, -1, 2])
    
    if np.any(np.abs(
        [policy_weight_01, policy_weight_02, policy_weight_03]) > 1E-9):
      assert contacts_age is not None, f"Contacts age is {contacts_age}"
      rank_score += (
        policy_weight_01 * contacts_age[:, 1] / 10
        + policy_weight_02 * contacts_age[:, 0] / 10
        + policy_weight_03 * users_age / 10)

    # Do not test when user in quarantine
    rank_score *= (user_quarantine_ends < t_now)
    

    # Grab tests on the main process
    test_frac = int(fraction_test * num_users)
    num_tests = test_frac if test_frac <= app_user_frac_num else app_user_frac_num
    
    users_to_test = prequential.decide_tests(
      scores_infect=rank_score,
      num_tests=num_tests,
      user_ids=app_user_ids)

    # TODO: Marten, should users_to_test be app_user_ids instead? 
    # No, since it will be the same regardless, since we only look at users 
    # of app to test in the first place
    obs_today = sim.get_observations_today(
      users_to_test.astype(np.int32),
      p_obs_infected,
      p_obs_not_infected,
      arg_rng)

    sim.set_window(days_offset)

    if not do_random_quarantine:
      t_start = time.time()

      # Contacts do not need to be filtered, since self._contacts has alreay
      # been done in the sim.step() function.
      contacts_now = sim.get_contacts()
      observations_now = sim.get_observations_all()
      # observations_condition = np.isin(observations_now[:, 0], app_user_ids)
      # observations_now = observations_now[observations_condition]
      
      logger.info((
        f"Day {t_now}: {contacts_now.shape[0]} contacts, "
        f"{observations_now.shape[0]} obs"))

      t_start = time.time()
      z_states_inferred_temp, contacts_age = inference_func(
        observations_now,
        contacts_now,
        num_rounds,
        num_days,
        users_age=users_age,
        diagnostic=diagnostic)
      
      np.testing.assert_array_almost_equal(
        z_states_inferred_temp.shape, [app_user_frac_num, num_days, 4])
      
      z_states_inferred = np.concatenate((z_states_inferred, np.zeros((num_users, 1, 4))), axis=1)
      # Insert values from predictions to the places where contacts exist.
      z_states_inferred[app_user_ids] = z_states_inferred_temp

      # TODO: Marten, the regular non-c++ FN function returns contacts_age as None. So no need to update. 
      if inference_method == "fncpp":
        contacts_age[app_user_ids] = contacts_age
      else:
        contacts_age = None
      
      
      logger.info(f"Time spent on inference_func {time.time() - t_start:.0f}")

      if trace_dir is not None:
        # if t_now == 15:
        #   fname = os.path.join(trace_dir, "contacts_10k.npy")
        #   np.save(fname, contacts_now)
        #   fname = os.path.join(trace_dir, "observations_10k.npy")
        #   np.save(fname, observations_now)

        # Dump graphs to train with a GNN
        # if t_now > 10:
        #   util_dataset.dump_features_flat(
        #     contacts_now, observations_now, z_states_inferred,
        #     sim.get_states_today(),
        #     contacts_age, users_age, trace_dir, num_users, t_now)
        if t_now > 10:
          user_free = (user_quarantine_ends < t_now)
          util_dataset.dump_features_graph(
            contacts_now, observations_now, z_states_inferred, user_free,
            sim.get_states_today(), users_age, trace_dir, num_users,
            num_time_steps, t_now)

    else:
      z_states_inferred = np.zeros((num_users, num_days, 4))

    # Users that test positive go into quarantine
    users_to_quarantine = obs_today[np.where(obs_today[:, 2] > 0)[0], 0]

    # Only run quarantines after a warmup period
    if t_now < t_start_quarantine:
      users_to_quarantine = np.array([], dtype=np.int32)

    user_quarantine_ends[users_to_quarantine] = t_now + num_days_quarantine

    # This function will remove the contacts that happen TODAY (and which may
    # spread the virus and cause people to shift to E-state tomorrow).
    sim.quarantine_users(users_to_quarantine, num_days_quarantine)
    assert sim.get_current_day() == t_now

    # NOTE: fpr is only defined as long as num_users_quarantine is fixed.
    # else switch to precision and recall
    states_today = sim.get_states_today()
    # states_today_only_appusers = states_today[app_user_ids]
    precision, recall = prequential.calc_prec_recall(
      states_today, user_quarantine_ends > t_now)
    infection_rate = np.mean(states_today == 2)
    exposed_rate = np.mean(
      np.logical_or(states_today == 1, states_today == 2))
    pir_running = max((pir_running, infection_rate))
    logger.info((f"precision: {precision:5.2f}, recall: {recall: 5.2f}, "
                 f"infection rate: {infection_rate:5.3f}({pir_running:5.3f}),"
                 f"{exposed_rate:5.3f}, tests: {len(users_to_test):5.0f} "
                 f"Qs: {len(users_to_quarantine):5.0f}"))

    precisions[t_now] = precision
    recalls[t_now] = recall
    infection_rates[t_now] = infection_rate
    critical_rates[t_now] = sim.get_critical_rate()
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
      positive = np.logical_or(states_today == 1, states_today == 2)
      rank_score = z_states_inferred[:, -1, 1:3].sum(axis=1)
      ave_precision[t_now] = metrics.average_precision_score(
        y_true=positive, y_score=rank_score)
    else:
      ave_precision[t_now] = 0.

    time_full_loop = time.time() - t_start_loop
    logger.info(f"Time spent on full_loop {time_full_loop:.0f}")

    loadavg1, loadavg5, _ = os.getloadavg()
    swap_use = psutil.swap_memory().used / (1024.0 ** 3)
    runner.log({
      "time_step": time_full_loop,
      "infection_rate": infection_rate,
      "load1": loadavg1,
      "load5": loadavg5,
      "swap_use": swap_use,
      "recall": recall,
      })

  time_pir, pir = np.argmax(infection_rates), np.max(infection_rates)
  total_drate = sim.get_death_rate()

  logger.info((
    f"At day {time_pir} peak infection rate is {pir:.5f} "
    f"and total death rate is {total_drate:.5f}"))

  prequential.dump_results_json(
    datadir=results_dir,
    cfg=cfg,
    ave_prob_inf=ave_prob_inf.tolist(),
    ave_prob_inf_at_inf=ave_prob_inf_at_inf.tolist(),
    ave_precision=ave_precision.tolist(),
    exposed_rates=exposed_rates.tolist(),
    critical_rates=critical_rates.tolist(),
    inference_method=inference_method,
    infection_rates=infection_rates.tolist(),
    likelihoods_state=likelihoods_state.tolist(),
    name=runner.name,
    num_quarantined=num_quarantined.tolist(),
    num_tested=num_tested.tolist(),
    pir=float(pir),
    pcr=float(np.max(critical_rates)),
    total_drate=float(total_drate),
    precisions=precisions.tolist(),
    quantization=quantization,
    recalls=recalls.tolist(),
    seed=cfg.get("seed", -1),
  )

  time_spent = time.time() - t0
  logger.info(f"With {num_rounds} rounds, PIR {pir:5.2f}")
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    results = {
      "time_spent": time_spent,
      "pir_mean": pir,
      "pcr": np.max(critical_rates),
      "total_drate": total_drate,
      "recall": np.nanmean(recalls[10:]),
      "precision": np.nanmean(precisions[10:])}
  runner.log(results)

  # Overwrite every experiment, such that code could be pre-empted
  prequential.dump_results(
    results_dir, precisions=precisions, recalls=recalls,
    infection_rates=infection_rates)
  return results



