"""Test functions for inference.py."""

from dpfn import inference, util
import numpy as np


def test_factorised_neighbor_step():
  """Tests Factorised Neighbors step."""

  contacts_all = [
    (0, 1, 2, 1),
    (1, 0, 2, 1),
    (3, 2, 2, 1),
    (2, 3, 2, 1),
    (4, 5, 2, 1),
    (5, 4, 2, 1),
    ]
  observations_all = np.array([
    (0, 2, 1)
  ], dtype=np.int32)
  num_users = 6
  num_time_steps = 5

  user_interval = (0, num_users)

  p0, p1 = 0.01, 0.3
  g_param, h_param = 0.2, 0.2
  alpha, beta = 0.0001, 0.001

  seq_array = np.stack(list(
    util.iter_sequences(time_total=num_time_steps, start_se=False)))
  seq_array_hot = np.transpose(util.state_seq_to_hot_time_seq(
    seq_array, time_total=num_time_steps), [1, 2, 0]).astype(np.int32)

  prior = [1-p0, p0, 0., 0.]

  log_A_start = util.enumerate_log_prior_values(
    prior, [1-p0, 1-g_param, 1-h_param],
    seq_array, num_time_steps)

  # Precompute log(C) terms, relating to observations
  obs_array = util.make_inf_obs_array(int(num_time_steps), alpha, beta)
  log_c_z_u = util.calc_c_z_u(
    user_interval,
    obs_array,
    observations_all)

  q_marginal_infected = np.zeros((num_users, num_time_steps), dtype=np.float32)

  infect_counter = util.InfectiousContactCount(
    contacts=contacts_all,
    samples=None,
    num_users=num_users,
    num_time_steps=num_time_steps,
  )
  past_contacts = infect_counter.get_past_contacts_slice(
    list(range(user_interval[0], user_interval[1])))

  obs_diff = np.max(log_c_z_u) - np.min(log_c_z_u)
  assert obs_diff > 1.0, f"Observation difference is too small {obs_diff}"

  # 1 update
  post_exp, tstart, tend = inference.fn_step_wrapped(
    user_interval,
    seq_array_hot,
    log_c_z_u,
    log_A_start,
    q_marginal_infected,
    num_time_steps,
    p0,
    p1,
    past_contacts_array=past_contacts,
    start_belief=np.ones((num_users, 4), dtype=np.float32),
    quantization=-1)

  time_spent = tend - tstart

  assert time_spent < 1.0, f"FN takes way too long: {time_spent}"
  np.testing.assert_array_almost_equal(
    post_exp.shape, [num_users, num_time_steps, 4])


def test_fact_neigh_with_start_belief():

  contacts_all = np.array([
    (0, 1, 2, 1),
    ], dtype=np.int32)
  observations_all = np.array([
    (0, 2, 1)
  ], dtype=np.int32)
  num_users = 2
  num_time_steps = 5

  p0, p1 = 0.01, 0.5

  start_belief = np.array(
    [[.1, .4, .5, .0],
     [.9, .1, .0, .0]], dtype=np.float32)

  _, post_exp = inference.fact_neigh(
    num_users=num_users,
    num_time_steps=num_time_steps,
    observations_all=observations_all,
    contacts_all=contacts_all,
    probab_0=p0,
    probab_1=p1,
    g_param=.5,
    h_param=.5,
    start_belief=start_belief,
    alpha=0.001,
    beta=0.01,
    num_updates=5)

  text = ("Note this is a stochastic test. And may fail one in a thousand times"
          "Please rerun a few times")
  # Start belief for u0 is high in E and I states, so after the contact between
  # u0 and u1 on day 2, then u1 should be in E state and I state after

  with np.printoptions(precision=3, suppress=True):
    assert post_exp[1][3][1] > .2, text + "\n" + repr(post_exp)
    assert post_exp[1][4][2] > .1, text + "\n" + repr(post_exp)


def test_factorised_neighbor_step_clipping():
  """Tests Factorised Neighbors step when clipping is applied."""

  contacts_all = [
    (0, 1, 2, 1),
    (1, 0, 2, 1),
    (3, 2, 2, 1),
    (2, 3, 2, 1),
    (4, 5, 2, 1),
    (5, 4, 2, 1),
    ]
  observations_all = np.array([
    (0, 2, 1)
  ], dtype=np.int32)
  num_users = 6
  num_time_steps = 5

  user_interval = (0, num_users)

  p0, p1 = 0.01, 0.3
  g_param, h_param = 0.2, 0.2
  alpha, beta = 0.0001, 0.001

  seq_array = np.stack(list(
    util.iter_sequences(time_total=num_time_steps, start_se=False)))
  seq_array_hot = np.transpose(util.state_seq_to_hot_time_seq(
    seq_array, time_total=num_time_steps), [1, 2, 0]).astype(np.int8)

  prior = [1-p0, p0, 0., 0.]

  log_A_start = util.enumerate_log_prior_values(
    prior, [1-p0, 1-g_param, 1-h_param],
    seq_array, num_time_steps)

  # Precompute log(C) terms, relating to observations
  obs_array = util.make_inf_obs_array(int(num_time_steps), alpha, beta)
  log_c_z_u = util.calc_c_z_u(
    user_interval,
    obs_array,
    observations_all)

  q_marginal_infected = np.random.rand(
    num_users, num_time_steps).astype(np.float32)

  infect_counter = util.InfectiousContactCount(
    contacts=contacts_all,
    samples=None,
    num_users=num_users,
    num_time_steps=num_time_steps,
  )
  past_contacts = infect_counter.get_past_contacts_slice(
    list(range(user_interval[0], user_interval[1])))

  obs_diff = np.max(log_c_z_u) - np.min(log_c_z_u)
  assert obs_diff > 1.0, f"Observation difference is too small {obs_diff}"

  # 1 update
  post_exp_no_clip, _, _ = inference.fn_step_wrapped(
    user_interval,
    seq_array_hot,
    log_c_z_u,
    log_A_start,
    q_marginal_infected,
    num_time_steps,
    p0,
    p1,
    past_contacts_array=past_contacts,
    start_belief=np.ones((num_users, 4), dtype=np.float32),
    quantization=-1)

  post_exp, tstart, tend = inference.fn_step_wrapped(
    user_interval,
    seq_array_hot,
    log_c_z_u,
    log_A_start,
    q_marginal_infected,
    num_time_steps,
    p0,
    p1,
    clip_lower=.2,
    clip_upper=.8,
    past_contacts_array=past_contacts,
    start_belief=np.ones((num_users, 4), dtype=np.float32),
    quantization=-1)

  time_spent = tend - tstart

  assert time_spent < 1.0, f"FN takes way too long: {time_spent}"

  # This is a stochastic test, but should be very unlikely to fail
  assert np.sum(np.abs(post_exp - post_exp_no_clip)) > 1E-9, (
    "Clipping did not change the posterior expectation")

  post_exp_noised, tstart, tend = inference.fn_step_wrapped(
    user_interval,
    seq_array_hot,
    log_c_z_u,
    log_A_start,
    q_marginal_infected,
    num_time_steps,
    p0,
    p1,
    clip_lower=.01,
    clip_upper=.8,
    past_contacts_array=past_contacts,
    dp_method=4,
    epsilon_dp=1.,
    delta_dp=1./100,
    start_belief=np.ones((num_users, 4), dtype=np.float32),
    quantization=-1)

  assert time_spent < 1.0, f"FN takes way too long: {time_spent}"

  # This is a stochastic test, but should be very unlikely to fail
  assert np.sum(np.abs(post_exp - post_exp_noised)) > 1E-9, (
    "DP noise did not change the posterior expectation")


def test_factorised_neighbor_diff_private():
  """Tests Factorised Neighbors step when differential privacy."""

  contacts_all = [
    (0, 1, 2, 1),
    (1, 0, 2, 1),
    (3, 2, 2, 1),
    (2, 3, 2, 1),
    (4, 5, 2, 1),
    (5, 4, 2, 1),
    ]
  observations_all = np.array([
    (0, 2, 1)
  ], dtype=np.int32)
  num_users = 6
  num_time_steps = 5

  user_interval = (0, num_users)

  p0, p1 = 0.01, 0.3
  g_param, h_param = 0.2, 0.2
  alpha, beta = 0.0001, 0.001

  seq_array = np.stack(list(
    util.iter_sequences(time_total=num_time_steps, start_se=False)))
  seq_array_hot = np.transpose(util.state_seq_to_hot_time_seq(
    seq_array, time_total=num_time_steps), [1, 2, 0]).astype(np.int8)

  prior = [1-p0, p0, 0., 0.]

  log_A_start = util.enumerate_log_prior_values(
    prior, [1-p0, 1-g_param, 1-h_param],
    seq_array, num_time_steps)

  # Precompute log(C) terms, relating to observations
  obs_array = util.make_inf_obs_array(int(num_time_steps), alpha, beta)
  log_c_z_u = util.calc_c_z_u(
    user_interval,
    obs_array,
    observations_all)

  q_marginal_infected = np.random.rand(
    num_users, num_time_steps).astype(np.float32)

  infect_counter = util.InfectiousContactCount(
    contacts=contacts_all,
    samples=None,
    num_users=num_users,
    num_time_steps=num_time_steps,
  )
  past_contacts = infect_counter.get_past_contacts_slice(
    list(range(user_interval[0], user_interval[1])))

  obs_diff = np.max(log_c_z_u) - np.min(log_c_z_u)
  assert obs_diff > 1.0, f"Observation difference is too small {obs_diff}"

  # 1 update
  post_exp_no_clip, _, _ = inference.fn_step_wrapped(
    user_interval,
    seq_array_hot,
    log_c_z_u,
    log_A_start,
    q_marginal_infected,
    num_time_steps,
    p0,
    p1,
    past_contacts_array=past_contacts,
    start_belief=np.ones((num_users, 4), dtype=np.float32),
    quantization=-1)

  post_exp_dp, tstart, tend = inference.fn_step_wrapped(
    user_interval,
    seq_array_hot,
    log_c_z_u,
    log_A_start,
    q_marginal_infected,
    num_time_steps,
    p0,
    p1,
    clip_lower=.01,
    clip_upper=.99,
    past_contacts_array=past_contacts,
    start_belief=np.ones((num_users, 4), dtype=np.float32),
    quantization=-1,
    dp_method=6,
    epsilon_dp=0.1,
    delta_dp=1./100,)

  post_exp_small_dp, tstart, tend = inference.fn_step_wrapped(
    user_interval,
    seq_array_hot,
    log_c_z_u,
    log_A_start,
    q_marginal_infected,
    num_time_steps,
    p0,
    p1,
    clip_lower=.0001,
    clip_upper=.9999,
    past_contacts_array=past_contacts,
    start_belief=np.ones((num_users, 4), dtype=np.float32),
    quantization=-1,
    dp_method=6,
    epsilon_dp=1000000,
    delta_dp=1./10,)

  time_spent = tend - tstart

  assert time_spent < 1.0, f"FN takes way too long: {time_spent}"

  # This is a stochastic test, but should be very unlikely to fail
  np.testing.assert_array_almost_equal(
    post_exp_small_dp, post_exp_no_clip, decimal=4,
    err_msg="Enormous epsilon in DP6 should not change posterior expectation")
  assert np.mean(np.abs(post_exp_dp - post_exp_no_clip)) > 1E-4, (
    "Appling DP6 did not change the posterior expectation")
