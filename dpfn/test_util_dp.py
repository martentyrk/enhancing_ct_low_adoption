"""Test functions for util_dp.py."""

from dpfn import util, util_dp
import numpy as np


def test_get_num_contacts_min_max():
  """Tests get_num_contacts_min_max function."""
  pc_array = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
  num_time_steps = 6
  num_contacts_min, num_contacts_max = util_dp.get_num_contacts_min_max(
    pc_array, num_time_steps)
  assert num_contacts_min == 0
  assert num_contacts_max == 1

  pc_array = np.array([[0, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5]])
  num_time_steps = 6
  num_contacts_min, num_contacts_max = util_dp.get_num_contacts_min_max(
    pc_array, num_time_steps)
  assert num_contacts_min == 0
  assert num_contacts_max == 3


def test_get_sensitivity_log():
  num_contacts = 5
  probab0, probab1 = 1/1000, 1/100
  margin = 0.05

  sensitivity1 = util_dp.get_sensitivity_log(
    num_contacts, probab0, probab1, margin, 1.-margin)
  sensitivity2 = util_dp.get_sensitivity_log(
    num_contacts*10, probab0, probab1, margin, 1.-margin)

  assert sensitivity1 > sensitivity2


def test_noise_i_column_beta_edges():
  nrows = 13

  # Test with means close to 1
  data = np.random.rand(nrows, 4)
  data[:, 2] = 100.
  data /= np.sum(data, axis=-1, keepdims=True)

  # Simply test that the function runs
  data_noised = util_dp.noise_i_column_beta(data, sigma=0.1)
  assert data_noised.shape == data.shape

  # Test with means close to 0
  data = 100*np.random.rand(nrows, 4)
  data[:, 2] = 0.
  data /= np.sum(data, axis=-1, keepdims=True)

  # Simply test that the function runs
  data_noised = util_dp.noise_i_column_beta(data, sigma=0.1)
  assert data_noised.shape == data.shape


def test_noise_i_column_beta():
  nrows = 100
  data = 0.04 + np.random.rand(nrows, 4)*0.92
  data /= np.sum(data, axis=-1, keepdims=True)

  data_noised = util_dp.noise_i_column_beta(data, sigma=0.0001)
  assert data_noised.shape == data.shape
  np.testing.assert_array_almost_equal(data, data_noised, decimal=3)

  # With enormous amount of noise, most infection scores should be high
  data = np.random.rand(nrows, 4)
  data[:, 2] = 0.0
  data /= np.sum(data, axis=-1, keepdims=True)

  data_noised = util_dp.noise_i_column_beta(data, sigma=2.)
  assert np.median(data_noised[:, 2]) > .1


def test_logit_noise():

  num_users = 100
  num_time_steps = 10
  messages = np.random.rand(num_users, num_time_steps).astype(np.float32)

  messages = util_dp.add_noise_per_message_logit(
    messages, epsilon_dp=10., delta_dp=0.01,
    clip_lower=0.01, clip_upper=0.99)

  assert messages.dtype == np.float32
  np.testing.assert_array_equal(messages.shape, (num_users, num_time_steps))


def test_logit_no_noise():

  num_users = 100
  num_time_steps = 10
  messages = np.random.rand(num_users, num_time_steps).astype(np.float32)

  messages_out = util_dp.add_noise_per_message_logit(
    messages, epsilon_dp=1000000., delta_dp=0.01,
    clip_lower=0.01, clip_upper=0.99)

  np.testing.assert_array_almost_equal(messages, messages_out, decimal=3)


def test_logit_lots_noise():

  num_users = 100
  num_time_steps = 10
  messages = np.random.rand(num_users, num_time_steps).astype(np.float32)

  messages_out = util_dp.add_noise_per_message_logit(
    messages, epsilon_dp=0.1, delta_dp=0.001,
    clip_lower=0.01, clip_upper=0.99)

  diff = np.mean(np.abs(messages - messages_out))
  assert diff > 0.1, "For small epsilon, the noise should be large."


def test_fn_geometric_mean_noise():
  num_users = 100
  num_time_steps = 10
  probab1 = 0.1
  p_infected = np.random.rand(num_users, num_time_steps).astype(np.float32)

  contacts_all = np.array([
    (0, 1, 2, 1),
    (1, 0, 2, 1),
    (3, 2, 2, 1),
    (2, 3, 2, 1),
    (4, 5, 2, 1),
    (5, 4, 2, 1),
    ], dtype=np.int32)

  past_contacts, _ = util.get_past_contacts_static(
    (0, num_users), contacts_all, num_msg=10*num_time_steps)

  print(util_dp.fn_rdp_mean_noise.signatures)

  outcome = util_dp.fn_rdp_mean_noise(
    p_infected=p_infected,
    user_interval=(0, num_users),
    past_contacts=past_contacts,
    p1=probab1,
    epsilon_dp=1.,
    delta_dp=1/2000)

  assert np.all(outcome >= 0.)
  assert np.all(outcome <= 1.)


def test_fn_geometric_mean_noise_high_eps():
  num_users = 100
  num_time_steps = 10
  probab1 = 0.1
  p_infected = np.random.rand(num_users, num_time_steps).astype(np.float32)

  contacts_all = np.array([
    (0, 1, 2, 1),
    (1, 0, 2, 1),
    (3, 2, 2, 1),
    (2, 3, 2, 1),
    (4, 5, 2, 1),
    (5, 4, 2, 1),
    ], dtype=np.int32)

  past_contacts, _ = util.get_past_contacts_static(
    (0, num_users), contacts_all, num_msg=10*num_time_steps)

  outcome = util_dp.fn_rdp_mean_noise(
    p_infected=p_infected,
    user_interval=(0, num_users),
    past_contacts=past_contacts,
    p1=probab1,
    epsilon_dp=100.,
    delta_dp=1/200)

  # For a high value of epsilon, agents without a contact should have
  # a low covidscore
  assert np.all(outcome[7:] < 0.1)
