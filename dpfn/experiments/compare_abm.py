"""Compare inference methods on likelihood and AUROC, and run prequentially."""
import argparse
import copy
import covasim as cv
import numpy as np
from dpfn.config import config
from dpfn.experiments import (
  prequential, util_experiments, util_covasim, util_dataset)
from dpfn import LOGGER_FILENAME, logger
from dpfn import simulator
from dpfn import util
from dpfn import util_wandb
import numba
import os
import psutil
import random
from sklearn import metrics
import socket
import threading
import time
import tqdm
import traceback
from typing import Any, Dict, Optional
import wandb


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
  epsilon_dp = cfg["model"]["epsilon_dp"]
  delta_dp = cfg["model"]["delta_dp"]
  a_rdp = cfg["model"]["a_rdp"]
  clip_lower = cfg["model"]["clip_lower"]
  clip_upper = cfg["model"]["clip_upper"]

  dedup_contacts = cfg["model"]["dedup_contacts"]

  # DP method to use, explanation in constants.py, value of -1 means no DP
  dp_method = cfg["model"]["dp_method"]

  # TODO: put a flag here to use the analytic solution for a_rdp
  if dp_method == 5:
    logger.info("DP method 5, convert (eps,delta) to (a, rho)")
    # In this clause, we optimize for a_rdp analytically
    # For any other value of a_rdp, we use the value provided
    eps_orig = copy.copy(epsilon_dp)

    d_term = np.log(1/delta_dp)
    a_rdp = 1 + (d_term + np.sqrt(d_term**2 + d_term*epsilon_dp)) / epsilon_dp
    epsilon_dp = eps_orig - d_term / (a_rdp - 1)

    assert epsilon_dp > 0.
    assert a_rdp > 1.
    assert a_rdp / epsilon_dp > 0.

    logger.info((
      f"Optimize a_rdp manually at (e, d) ({eps_orig:.2e}, {delta_dp:.2e})\n"
      f"Optimisation returns (a, e) ({a_rdp:.2e}, {epsilon_dp:.2e})\n"
      f"At multiplier {a_rdp/epsilon_dp:.2f}"))
    delta_dp = -1.0  # Set to negative, as we optimised for a_rdp

  # Construct dynamics
  # Construct Geometric distro's for E and I states

  do_random_quarantine = False
  if inference_method == "bp":
    inference_func = util_experiments.wrap_belief_propagation(
      num_users=num_users,
      alpha=alpha,
      beta=beta,
      probab0=p0,
      probab1=p1,
      param_g=g,
      param_h=h,
      epsilon_dp=epsilon_dp,
      a_rdp=a_rdp,
      clip_lower=clip_lower,
      clip_upper=clip_upper,
      quantization=quantization,
      trace_dir=trace_dir)
  elif inference_method == "fn":
    inference_func = util_experiments.wrap_fact_neigh_inference(
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
      trace_dir=trace_dir)
  elif inference_method == "fncpp":
    inference_func = util_experiments.wrap_fact_neigh_cpp(
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
      dedup_contacts=dedup_contacts,
      model_fpath=cfg["model"]["model_fpath"])
  elif inference_method == "bpcpp":
    inference_func = util_experiments.wrap_bp_cpp(
      num_users=num_users,
      alpha=alpha,
      beta=beta,
      probab0=p0,
      probab1=p1,
      g_param=g,
      h_param=h,
      dp_method=dp_method,
      rho_rdp=epsilon_dp,
      delta_dp=delta_dp,
      a_rdp=a_rdp,
      clip_lower=clip_lower,
      clip_upper=clip_upper,
      quantization=quantization,
      trace_dir=trace_dir)
  elif inference_method == "gibbs":
    if epsilon_dp > 0:
      assert a_rdp < 0
      assert delta_dp < 0
      assert clip_lower > 0

    inference_func = util_experiments.wrap_gibbs_inference(
      num_users=num_users,
      g_param=g,
      h_param=h,
      clip_lower=clip_lower,
      epsilon_dp=epsilon_dp,
      alpha=alpha,
      beta=beta,
      probab_0=p0,
      probab_1=p1)
  elif inference_method == "random":
    inference_func = None
    do_random_quarantine = True
  elif inference_method == "dummy":
    inference_func = util_experiments.wrap_dummy_inference(
      num_users=num_users, trace_dir=trace_dir)
  elif inference_method == "dpct":
    assert a_rdp < 0, f"a rdp should be negative ({a_rdp})"
    assert delta_dp > 0
    inference_func = util_experiments.wrap_dpct_inference(
      num_users=num_users, epsilon_dp=epsilon_dp, delta_dp=delta_dp)
  else:
    raise ValueError((
      f"Not recognised inference method {inference_method}. Should be one of"
      f"['random', 'fn', 'dummy', 'dpct', 'bp', 'gibbs', 'fncpp', 'bpcpp']"
    ))
  return inference_func, do_random_quarantine


def compare_abm(
    inference_method: str,
    cfg: Dict[str, Any],
    runner,
    results_dir: str,
    trace_dir: Optional[str] = None,
    do_diagnosis: bool = False):
  """Compares different inference algorithms on the supplied contact graph."""
  num_users = config_wandb["data"]["num_users"]
  num_time_steps = config_wandb["data"]["num_time_steps"]

  num_days_window = cfg["model"]["num_days_window"]
  quantization = cfg["model"]["quantization"]

  num_rounds = cfg["model"]["num_rounds"]
  policy_weight_01 = cfg["model"]["policy_weight_01"]
  policy_weight_02 = cfg["model"]["policy_weight_02"]
  policy_weight_03 = cfg["model"]["policy_weight_03"]
  rng_seed = cfg.get("seed", 123)

  fraction_test = cfg["data"]["fraction_test"]

  # Data and simulator params
  num_days_quarantine = cfg["data"]["num_days_quarantine"]
  t_start_quarantine = cfg["data"]["t_start_quarantine"]

  logger.info((
    f"Settings at experiment: {quantization:.0f} quant, at {fraction_test}%"))

  diagnostic = runner if do_diagnosis else None

  inference_func, do_random_quarantine = make_inference_func(
    inference_method, num_users, cfg, trace_dir=trace_dir)

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

  sim = simulator.ABMSimulator(
    num_time_steps, num_users, rng_seed)
  users_age = -1*np.ones((num_users), dtype=np.int32)

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
    users_to_test = prequential.decide_tests(
      scores_infect=rank_score,
      num_tests=int(fraction_test * num_users))

    obs_today = sim.get_observations_today(
      users_to_test.astype(np.int32),
      p_obs_infected,
      p_obs_not_infected,
      arg_rng)

    sim.set_window(days_offset)

    if not do_random_quarantine:
      t_start = time.time()

      contacts_now = sim.get_contacts()
      observations_now = sim.get_observations_all()

      logger.info((
        f"Day {t_now}: {contacts_now.shape[0]} contacts, "
        f"{observations_now.shape[0]} obs"))

      t_start = time.time()
      z_states_inferred, contacts_age = inference_func(
        observations_now,
        contacts_now,
        num_rounds,
        num_days,
        users_age=users_age,
        diagnostic=diagnostic)

      np.testing.assert_array_almost_equal(
        z_states_inferred.shape, [num_users, num_days, 4])
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
            num_time_steps, t_now, int(rng_seed))

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
    target=util_covasim.log_to_wandb, args=(runner_global,), daemon=True)
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
  std_rank_noise = cfg["data"]["std_rank_noise"]
  t_start_quarantine = cfg["data"]["t_start_quarantine"]

  num_days_window = cfg["model"]["num_days_window"]
  quantization = cfg["model"]["quantization"]
  num_rounds = cfg["model"]["num_rounds"]

  rng_seed = cfg.get("seed", 123)

  logger.info((
    f"Settings at experiment: {quantization:.0f} quant, at {fraction_test}% "
    f"seed {rng_seed}"))

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

      if std_rank_noise > 0:
        rank_score += std_rank_noise * np.random.randn(num_users)

      if trace_dir is not None and sim.t > 10:
        user_free = np.logical_not(sim.people.isolated)

        # Get states today
        # Exposed is superset of infectious, but this is overwritten below
        states_today = np.zeros((num_users))
        states_today[sim.people.exposed] = 1
        states_today[sim.people.infectious] = 2
        states_today[sim.people.dead] = 3
        states_today[sim.people.recovered] = 3

        rate_user_free = np.mean(user_free)
        rate_infection = np.mean(np.logical_or(
          states_today == 1, states_today == 2))
        rate_diagnosed = np.mean(sim.people.diagnosed)
        logger.info((
          f"Day {sim.t}: {rate_user_free:.4f} free, "
          f"{rate_infection:.4f} infection, {rate_diagnosed:.4f} diagnosed"))

        util_dataset.dump_features_graph(
          contacts_rel, obs_rel, pred, user_free,
          states_today, users_age, trace_dir, num_users,
          num_time_steps, sim.t, int(rng_seed))

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
  sim.set_seed(seed=rng_seed)
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


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    description='Compare statistics acrosss inference methods')
  parser.add_argument('--inference_method', type=str, default='fn',
                      choices=[
                        'fn', 'dummy', 'random', 'bp', 'dpct',
                        'gibbs', 'fncpp', 'bpcpp'],
                      help='Name of the inference method')
  parser.add_argument('--simulator', type=str, default='abm',
                      choices=['abm', 'covasim'],
                      help='Name of the simulator')
  parser.add_argument('--config_data', type=str, default='small_graph',
                      help='Name of the config file for the data')
  parser.add_argument('--config_model', type=str, default='model_02',
                      help='Name of the config file for the model')
  parser.add_argument('--name', type=str, default=None,
                      help=('Name of the experiments. WandB will set a random'
                            ' when left undefined'))
  parser.add_argument('--do_diagnosis', action='store_true')
  parser.add_argument('--dump_traces', action='store_true')

  # TODO make a better heuristic for this:
  num_threads = max((util.get_cpu_count()-1, 1))
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
  config_wandb = {
    "config_data_name": configname_data,
    "config_model_name": configname_model,
    "cpu_count": util.get_cpu_count(),
    "data": config_data.to_dict(),
    "model": config_model.to_dict(),
  }

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

  util_experiments.make_git_log()

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
      trace_dir=trace_dir_global,
      do_diagnosis=args.do_diagnosis)

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
