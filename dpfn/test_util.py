"""Test functions for util.py."""

from dpfn import util
import numpy as np
import random


def test_state_time():
  result = util.state_at_time([6, 0, 0], 4)
  assert float(result.flatten()[0]) == 0, f"result {result}"

  result = util.state_at_time([[1, 4, 5], [1, 3, 5], [1, 2, 5]], 4)
  assert result.tolist() == [1, 2, 2], f"result {result}"

  result = util.state_at_time([[1, 4, 5], [1, 3, 5], [1, 2, 5]], 1)
  assert result.tolist() == [1, 1, 1], f"result {result}"

  result = util.state_at_time([1, 1, 1], 4)
  assert float(result.flatten()[0]) == 3, f"result {result}"


def test_state_time_cache():
  result = util.state_at_time_cache(6, 0, 0, 4)
  assert result == 0

  result = util.state_at_time_cache(1, 4, 5, 4)
  assert result == 1

  result = util.state_at_time_cache(1, 3, 5, 4)
  assert result == 2

  result = util.state_at_time_cache(1, 2, 5, 4)
  assert result == 2

  result = util.state_at_time_cache(1, 2, 5, 1)
  assert result == 1

  result = util.state_at_time_cache(1, 2, 5, 8)
  assert result == 3


def test_calculate_log_c_z():
  num_time_steps = 4

  observations_all = np.array([
    (1, 2, 1),
    (2, 3, 0),
    ], dtype=np.int32)

  a, b = .1, .2

  obs_array = util.make_inf_obs_array(int(num_time_steps), a, b)
  result = util.calc_c_z_u(
    user_interval=(0, 3),
    obs_array=obs_array,
    observations=observations_all)
  expected = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [np.log(b), np.log(b), np.log(b), np.log(1-a), np.log(1-a), np.log(b),
     np.log(1-a), np.log(1-a), np.log(1-a), np.log(1-a), np.log(b), np.log(b),
     np.log(1-a), np.log(1-a), np.log(b), np.log(b), np.log(b), np.log(b),
     np.log(b), np.log(b)],
    [np.log(1-b), np.log(1-b), np.log(1-b), np.log(1-b), np.log(a), np.log(1-b),
     np.log(1-b), np.log(a), np.log(1-b), np.log(a), np.log(a), np.log(1-b),
     np.log(1-b), np.log(a), np.log(a), np.log(1-b), np.log(a), np.log(1-b),
     np.log(1-b), np.log(1-b)]])

  assert result.shape == expected.shape, (
    f"Shapes dont match: {result.shape} {expected.shape}")

  np.testing.assert_array_almost_equal(result, expected)


def test_generate_sequences():
  result = util.generate_sequence_days(time_total=4)
  expected = [(4, 0, 0), (3, 1, 0), (2, 1, 1), (2, 2, 0), (1, 1, 1), (1, 1, 2),
              (1, 2, 1), (1, 3, 0)]

  for x, y in zip(result, expected):
    assert x == y, f"{x} and {y} do not match"


def test_calc_log_a():
  potential_sequences = list(util.generate_sequence_days(time_total=4))
  seq_array = np.stack(potential_sequences, axis=0)

  g = 1 / 8
  h = 1 / 8
  p0 = 0.01

  log_A = util.calc_log_a_start(seq_array, p0, g, h)

  expected = [
    3*np.log(1-p0),
    2*np.log(1-p0),
    np.log(1-p0) + np.log(g),
    np.log(1-p0) + np.log(1-g),
    np.log(g) + np.log(h),
    np.log(g) + np.log(1-h),
    np.log(1-g) + np.log(g),
    2*np.log(1-g)
  ]
  np.testing.assert_array_almost_equal(log_A, expected)


def test_state_seq_to_time_seq():
  seq_days = list(util.generate_sequence_days(4))
  result = util.state_seq_to_time_seq(np.array(seq_days), 5)

  expected = [
    [0, 0, 0, 0, 3],
    [0, 0, 0, 1, 3],
    [0, 0, 1, 2, 3],
    [0, 0, 1, 1, 3],
    [0, 1, 2, 3, 3],
    [0, 1, 2, 2, 3],
    [0, 1, 1, 2, 3],
    [0, 1, 1, 1, 3]]

  np.testing.assert_array_almost_equal(result, np.array(expected))

  seq_days = list(util.generate_sequence_days(4))
  result = util.state_seq_to_hot_time_seq(np.array(seq_days), 5)
  np.testing.assert_array_almost_equal(result.shape, [8, 5, 4])


def test_iter_sequences():
  num_time_steps = 7
  num_seqs = len(list(util.iter_sequences(num_time_steps, start_se=True)))
  np.testing.assert_almost_equal(num_seqs, 64)

  # num_time_steps+1 more
  num_seqs = len(list(util.iter_sequences(num_time_steps, start_se=False)))
  np.testing.assert_almost_equal(num_seqs, 72)


def test_enumerate_log_prior_values_full_sums_1():
  num_time_steps = 7
  p0, g, h = 0.01, 0.2, 0.16

  seqs = np.stack(list(
    util.iter_sequences(time_total=num_time_steps, start_se=False)))

  log_p = util.enumerate_log_prior_values(
    [1-p0, p0, 0., 0.], [1-p0, 1-g, 1-h], seqs, num_time_steps)

  np.testing.assert_almost_equal(np.sum(np.exp(log_p)), 1.0, decimal=3)


def test_d_penalty_term():
  contacts_all = np.array([
    (0, 1, 2, 1),
    (1, 0, 2, 1),
    (3, 2, 2, 1),
    (2, 3, 2, 1),
    (4, 5, 2, 1),
    (5, 4, 2, 1),
    ], dtype=np.int32)
  num_users = 6
  num_time_steps = 5

  past_contacts, _ = util.get_past_contacts_static(
    (0, num_users), contacts_all, num_msg=num_time_steps*4)

  q_marginal_infected = np.array([
    [.8, .8, .8, .8, .8],
    [.1, .1, .8, .8, .8],
    [.1, .1, .1, .1, .1],
    [.1, .1, .1, .1, .1],
    [.1, .1, .1, .1, .1],
    [.1, .1, .1, .1, .1],
  ], dtype=np.float32)

  user = 1
  d_term, d_no_term = util.precompute_d_penalty_terms_fn(
    q_marginal_infected,
    p0=0.01,
    p1=.3,
    past_contacts=past_contacts[user],
    num_time_steps=num_time_steps)

  # Contact with user 0, which is infected with .8, so penalty for term should
  # be less (higher number) than no_term.
  assert np.all(d_term >= d_no_term)
  assert d_term[0] == 0
  assert d_no_term[0] == 0
  assert d_term[num_time_steps] == 0

  # Second test case
  q_marginal_infected = np.array([
    [.1, .1, .1, .1, .1],
    [.1, .1, .8, .8, .8],
    [.1, .1, .1, .1, .1],
    [.1, .1, .1, .1, .1],
    [.8, .8, .8, .8, .8],
    [.1, .1, .1, .1, .1],
  ], dtype=np.float32)

  user = 1
  d_term, d_no_term = util.precompute_d_penalty_terms_fn(
    q_marginal_infected,
    p0=0.01,
    p1=1E-5,
    past_contacts=past_contacts[user],
    num_time_steps=num_time_steps)

  # With small p1, penalty for termination should be small (low number)
  assert np.all(d_term < 0.001)

  past_contacts, _ = util.get_past_contacts_static(
    (0, num_users), contacts_all, num_msg=int(num_time_steps*100))

  d_term_new, d_no_term_new = util.precompute_d_penalty_terms_fn(
    q_marginal_infected,
    p0=0.01,
    p1=1E-5,
    past_contacts=past_contacts[user],
    num_time_steps=num_time_steps)
  np.testing.assert_array_almost_equal(d_term, d_term_new)
  np.testing.assert_array_almost_equal(d_no_term, d_no_term_new)


def test_d_penalty_term_numerical():
  contacts_all = np.array([
    (4, 1, 1, 1),
    (3, 2, 4, 1),
    ], dtype=np.int32)
  num_users = 6
  num_time_steps = 8

  # Second test case
  user = 1
  q_marginal_infected = np.array([
    [.1, .1, .1, .1, .1, .1, .1, .1],
    [.1, .1, .1, .8, .8, .8, .8, .8],
    [.1, .1, .1, .1, .1, .1, .1, .1],
    [.8, .8, .8, .8, .8, .8, .8, .8],
    [.8, .8, .8, .8, .8, .8, .8, .8],
    [.1, .1, .1, .1, .1, .1, .1, .1],
  ], dtype=np.float32)

  past_contacts, _ = util.get_past_contacts_static(
    (0, num_users), contacts_all, num_msg=int(num_time_steps*100))

  d_term, d_no_term = util.precompute_d_penalty_terms_fn(
    q_marginal_infected,
    p0=0.001,
    p1=0.3,
    past_contacts=past_contacts[user],
    num_time_steps=num_time_steps)

  d_term_new, d_no_term_new = util.precompute_d_penalty_terms_fn2(
    q_marginal_infected,
    p0=0.001,
    p1=0.3,
    past_contacts=past_contacts[user],
    num_time_steps=num_time_steps)

  np.testing.assert_array_almost_equal(d_term, d_term_new)
  np.testing.assert_array_almost_equal(d_no_term, d_no_term_new)
  assert d_no_term_new.dtype == np.float32
  assert d_term_new.dtype == np.float32


def test_d_penalty_term_regression():
  contacts_all = np.array([
    (4, 1, 1, 1),
    (3, 1, 4, 1),
    ], dtype=np.int32)
  num_users = 6
  num_time_steps = 8

  # Second test case
  user = 1
  q_marginal_infected = np.array([
    [.1, .1, .1, .1, .1, .1, .1, .1],
    [.1, .1, .1, .8, .8, .8, .8, .8],
    [.1, .1, .1, .1, .1, .1, .1, .1],
    [.9, .9, .9, .9, .9, .9, .9, .9],
    [.8, .8, .8, .8, .8, .8, .8, .8],
    [.1, .1, .1, .1, .1, .1, .1, .1],
  ], dtype=np.float32)

  past_contacts, _ = util.get_past_contacts_static(
    (0, num_users), contacts_all, num_msg=int(num_time_steps*100))

  d_term, d_no_term = util.precompute_d_penalty_terms_fn2(
    q_marginal_infected,
    p0=0.001,
    p1=0.3,
    past_contacts=past_contacts[user],
    num_time_steps=num_time_steps)

  assert d_no_term.dtype == np.float32
  assert d_term.dtype == np.float32

  # Note: these are the results from the old implementation
  # Dump made on January 27, 2023

  d_term_expected = np.array(
    [0., 0., 5.4838, 0., 0., 5.601122, 0., 0., 0.], dtype=np.float32)
  d_no_term_expected = np.array(
    [0., 0., -0.274437, 0., 0., -0.314711, 0., 0., 0.], dtype=np.float32)

  np.testing.assert_array_almost_equal(d_term, d_term_expected)
  np.testing.assert_array_almost_equal(d_no_term, d_no_term_expected)


def test_d_penalty_rdp_regression():
  contacts_all = np.array([
    (4, 1, 1, 1),
    (3, 1, 4, 1),
    ], dtype=np.int32)
  num_users = 6
  num_time_steps = 8

  # Second test case
  user = 1
  q_marginal_infected = np.array([
    [.1, .1, .1, .1, .1, .1, .1, .1],
    [.1, .1, .1, .8, .8, .8, .8, .8],
    [.1, .1, .1, .1, .1, .1, .1, .1],
    [.9, .9, .9, .9, .9, .9, .9, .9],
    [.8, .8, .8, .8, .8, .8, .8, .8],
    [.1, .1, .1, .1, .1, .1, .1, .1],
  ], dtype=np.float32)

  past_contacts, _ = util.get_past_contacts_static(
    (0, num_users), contacts_all, num_msg=int(num_time_steps*100))

  d_term, d_no_term = util.precompute_d_penalty_terms_rdp(
    q_marginal_infected,
    p0=0.001,
    p1=0.3,
    clip_lower=-1,
    clip_upper=1000,
    a_rdp=1.,
    epsilon_rdp=1000000000,
    past_contacts=past_contacts[user],
    num_time_steps=num_time_steps)

  assert d_no_term.dtype == np.float32
  assert d_term.dtype == np.float32

  # Note: these are the results from the old implementation
  # Dump made on January 27, 2023

  d_term_expected = np.array(
    [0., 0., 5.4838, 0., 0., 5.601122, 0., 0., 0.], dtype=np.float32)
  d_no_term_expected = np.array(
    [0., 0., -0.274437, 0., 0., -0.314711, 0., 0., 0.], dtype=np.float32)

  # Only check up to 4 decimal places, because tiny amount of noise is added,
  # even with epsilon_rdp=large
  np.testing.assert_array_almost_equal(
    d_term, d_term_expected, decimal=4)
  np.testing.assert_array_almost_equal(
    d_no_term, d_no_term_expected, decimal=4)


def test_d_penalty_rdp_noisy():
  contacts_all = np.array([
    (4, 1, 1, 1),
    (3, 1, 4, 1),
    ], dtype=np.int32)
  num_users = 6
  num_time_steps = 8

  # Second test case
  user = 1
  q_marginal_infected = np.array([
    [.1, .1, .1, .1, .1, .1, .1, .1],
    [.1, .1, .1, .8, .8, .8, .8, .8],
    [.1, .1, .1, .1, .1, .1, .1, .1],
    [.9, .9, .9, .9, .9, .9, .9, .9],
    [.8, .8, .8, .8, .8, .8, .8, .8],
    [.1, .1, .1, .1, .1, .1, .1, .1],
  ], dtype=np.float32)

  past_contacts, _ = util.get_past_contacts_static(
    (0, num_users), contacts_all, num_msg=int(num_time_steps*100))

  d_term1, d_no_term1 = util.precompute_d_penalty_terms_rdp(
    q_marginal_infected,
    p0=0.001,
    p1=0.3,
    clip_lower=-1,
    clip_upper=1000,
    a_rdp=1.,
    epsilon_rdp=1,
    past_contacts=past_contacts[user],
    num_time_steps=num_time_steps)

  d_term2, d_no_term2 = util.precompute_d_penalty_terms_rdp(
    q_marginal_infected,
    p0=0.001,
    p1=0.3,
    clip_lower=-1,
    clip_upper=1000,
    a_rdp=200.,
    epsilon_rdp=1,
    past_contacts=past_contacts[user],
    num_time_steps=num_time_steps)

  assert np.max(np.abs(d_term1 - d_term2)) > 1E-2
  assert np.max(np.abs(d_no_term1 - d_no_term2)) > 1E-2


def test_d_penalty_term_against_dp_gaussian():
  contacts_all = np.array([
    (4, 1, 1, 1),
    (3, 1, 4, 1),
    ], dtype=np.int32)
  num_users = 6
  num_time_steps = 8

  # Second test case
  user = 1
  q_marginal_infected = np.array([
    [.1, .1, .1, .1, .1, .1, .1, .1],
    [.1, .1, .1, .8, .8, .8, .8, .8],
    [.1, .1, .1, .1, .1, .1, .1, .1],
    [.9, .9, .9, .9, .9, .9, .9, .9],
    [.8, .8, .8, .8, .8, .8, .8, .8],
    [.1, .1, .1, .1, .1, .1, .1, .1],
  ], dtype=np.float32)

  past_contacts, _ = util.get_past_contacts_static(
    (0, num_users), contacts_all, num_msg=int(num_time_steps*100))

  d_term, d_no_term = util.precompute_d_penalty_terms_fn2(
    q_marginal_infected,
    p0=0.001,
    p1=0.1,
    past_contacts=past_contacts[user],
    num_time_steps=num_time_steps)

  assert d_no_term.dtype == np.float32
  assert d_term.dtype == np.float32

  d_term_dp, d_no_term_dp = util.precompute_d_penalty_terms_dp_gaussian(
    q_marginal_infected,
    p0=0.001,
    p1=0.1,
    epsilon_dp=10000,  # For high epsilon, results should be close to non-dp
    delta_dp=0.1,  # For high delta, results should be close to non-dp
    past_contacts=past_contacts[user],
    num_time_steps=num_time_steps)

  np.testing.assert_array_almost_equal(d_term, d_term_dp, decimal=2)
  np.testing.assert_array_almost_equal(d_no_term, d_no_term_dp, decimal=2)


def test_add_lognormal_noise_rdp():
  means = np.array([.1, .3, .6, .9, .99], dtype=np.float32)

  results = np.exp(util.add_lognormal_noise_rdp(
    np.log(means),
    num_contacts=5,
    epsilon_dp=10000,
    a_rdp=5,
    sensitivity=np.log(0.9),
  ))

  np.testing.assert_array_almost_equal(results, means, decimal=2)


def test_add_lognormal_noise_rdp_repeat():
  means = np.array([.1, .3, .6, .9, .99], dtype=np.float32)

  num_repeats = 30

  results = np.mean([np.exp(util.add_lognormal_noise_rdp(
    np.log(means),
    num_contacts=5,
    epsilon_dp=5,
    a_rdp=5,
    sensitivity=np.log(0.9))) for _ in range(num_repeats)], axis=0)

  np.testing.assert_array_almost_equal(results, means, decimal=2, err_msg=(
    "Stochastic test, but should be very unlikely to fail."))


def test_softmax():
  # Check that trick for numerical stability yields identical results

  def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

  logits = np.random.randn(13)

  np.testing.assert_array_almost_equal(
    softmax(logits),
    util.softmax(logits)
  )


def test_it_num_infected_probs():
  # For max entropy, all probs_total should be 2**-len(probs)
  probs = [.5, .5, .5, .5]
  for _, prob_total in util.it_num_infected_probs(probs):
    np.testing.assert_almost_equal(prob_total, 1/16)

  # For any (random) probs, the probs_total should sum to 1
  probs = list((random.random() for _ in range(5)))
  sums, probs_total = zip(*util.it_num_infected_probs(probs))
  assert len(list(sums)) == 2**5
  np.testing.assert_almost_equal(sum(probs_total), 1.0)

  # For one prob, should reduce to [(0, 1-p), (1, p)]
  prob = random.random()
  sums, probs_total = zip(*util.it_num_infected_probs([prob]))
  assert list(sums) == [0, 1]
  np.testing.assert_almost_equal(list(probs_total), [1-prob, prob])

  # Manual result
  probs = [.8, .7]
  expected = [(0, .2*.3), (1, .2*.7), (1, .8*.3), (2, .8*.7)]
  expected_sums, expected_probs_total = zip(*expected)
  result_sums, result_probs_total = zip(*util.it_num_infected_probs(probs))
  np.testing.assert_array_almost_equal(list(result_sums), list(expected_sums))
  np.testing.assert_array_almost_equal(
    list(result_probs_total), list(expected_probs_total))


def test_past_contact_array_fast():
  contacts_all = np.array([
    (1, 2, 4, 1),
    (2, 1, 4, 1),
    (0, 1, 4, 1)
    ])

  past_contacts, max_num_c = util.get_past_contacts_fast((0, 3), contacts_all)

  np.testing.assert_array_almost_equal(past_contacts.shape, [3, 2+1, 2])
  np.testing.assert_array_almost_equal(past_contacts[0], -1)
  np.testing.assert_array_almost_equal(past_contacts[2][1], -1)
  np.testing.assert_array_almost_equal(past_contacts[2][0], [4, 1])

  np.testing.assert_equal(past_contacts.dtype, np.int32)
  np.testing.assert_almost_equal(max_num_c, 2)


def test_past_contact_array_static():
  num_msg = 13
  contacts_all = np.array([
    (1, 2, 4, 1),
    (2, 1, 4, 1),
    (0, 1, 4, 1)
    ], dtype=np.int32)

  past_contacts, max_num_c = util.get_past_contacts_static(
    (0, 3), contacts_all, num_msg=num_msg)

  np.testing.assert_array_almost_equal(past_contacts.shape, [3, num_msg, 2])
  np.testing.assert_array_almost_equal(past_contacts[0], -1)
  np.testing.assert_array_almost_equal(past_contacts[2][1], -1)
  np.testing.assert_array_almost_equal(past_contacts[2][0], [4, 1])

  np.testing.assert_equal(past_contacts.dtype, np.int32)
  np.testing.assert_almost_equal(max_num_c, 2)


def test_past_contact_array_fast_copy_paste_static():
  num_msg = 13
  contacts_all = np.array([
    (1, 2, 4, 1),
    (1, 2, 3, 1),
    (1, 2, 2, 1),
    (1, 2, 1, 1),
    (2, 1, 4, 1),
    ], dtype=np.int32)

  past_contacts_static, max_num_static = util.get_past_contacts_static(
    (0, 3), contacts_all, num_msg=num_msg)

  past_contacts_fast, max_num_c = util.get_past_contacts_fast(
    (0, 3), contacts_all)
  np.testing.assert_almost_equal(max_num_c, 4)

  # Check that the max_contacts match
  np.testing.assert_almost_equal(
    max_num_static, max_num_c)

  # Silly to test set, but contacts could occur in any order ofcourse
  # Check that the values match
  np.testing.assert_equal(
    set(past_contacts_static[0].flatten().tolist()),
    set(past_contacts_fast[0].flatten().tolist())
  )
  np.testing.assert_equal(
    set(past_contacts_static[1].flatten().tolist()),
    set(past_contacts_fast[1].flatten().tolist())
  )
  np.testing.assert_equal(
    set(past_contacts_static[2].flatten().tolist()),
    set(past_contacts_fast[2].flatten().tolist())
  )


def test_spread_buckets():
  num_sample_array = util.spread_buckets(100, 10)
  expected = 10 * np.ones((10))
  np.testing.assert_array_almost_equal(num_sample_array, expected)

  num_sample_array = util.spread_buckets(97, 97)
  np.testing.assert_array_almost_equal(num_sample_array, np.ones((97)))

  num_samples = np.sum(util.spread_buckets(100, 13))
  np.testing.assert_almost_equal(num_samples, 100)

  with np.testing.assert_raises(AssertionError):
    util.spread_buckets(13, 17)


def test_spread_buckets_interval():
  user_id = util.spread_buckets_interval(100, 10)
  np.testing.assert_array_almost_equal(user_id, 10*np.arange(11))


def test_quantize():
  x = np.random.randn(13, 13).astype(np.float32)

  x_quantized = util.quantize(x, 8)
  assert x_quantized.dtype == np.float32

  x_quantized = util.quantize(x, -1)
  assert x_quantized.dtype == np.float32


def test_root_find_a_rdp_eps():

  delta = 1/200

  # Test at intermediate values
  a_value, eps_value = util.root_find_a_rdp(1., delta)
  mult1 = a_value / eps_value
  a_value, eps_value = util.root_find_a_rdp(2., delta)
  mult2 = a_value / eps_value

  assert mult1 > mult2

  # Test at extreme values
  a_value, eps_value = util.root_find_a_rdp(0.01, delta)
  mult1 = a_value / eps_value
  a_value, eps_value = util.root_find_a_rdp(0.02, delta)
  mult2 = a_value / eps_value

  assert mult1 > mult2


def test_root_find_a_rdp_delta():

  eps = 1.1

  # Test at intermediate values
  a_value, eps_value = util.root_find_a_rdp(eps, 1/200)
  mult1 = a_value / eps_value
  a_value, eps_value = util.root_find_a_rdp(eps, 1/100)
  mult2 = a_value / eps_value

  assert mult1 > mult2

  # Test at extreme values
  a_value, eps_value = util.root_find_a_rdp(eps, 1/2000)
  mult1 = a_value / eps_value
  a_value, eps_value = util.root_find_a_rdp(eps, 1/1000)
  mult2 = a_value / eps_value

  assert mult1 > mult2


def test_root_find_a_rdp_error():
  with np.testing.assert_raises(AssertionError):
    util.root_find_a_rdp(0., 1/200)
  with np.testing.assert_raises(AssertionError):
    util.root_find_a_rdp(1., 200)


def test_root_find_a_rdp_values():
  num_interp = 5

  for delta_inv in 10**np.linspace(1, 4, num=num_interp):
    for eps in 10**np.linspace(-4, 4, num=num_interp):
      _ = util.root_find_a_rdp(eps, 1 / delta_inv)
