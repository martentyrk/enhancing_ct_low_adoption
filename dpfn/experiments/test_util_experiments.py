"""Tests for util_experiments.py"""
import numpy as np
from dpfn.experiments import util_experiments


def test_dummy_inference():
  num_users = 1000
  num_time_steps = 13

  inference_func = util_experiments.wrap_dummy_inference(num_users)

  z_states = inference_func(None, None, None, num_time_steps)

  np.testing.assert_array_almost_equal(
    z_states.shape, [num_users, num_time_steps, 4])


def test_dct_inference():
  num_users = 6
  num_time_steps = 5

  contacts_all = np.array([
    [0, 1, 3, 1],
    [0, 2, 3, 1],
  ], dtype=np.int32)

  observations_all = np.array([
    [0, 3, 1],
  ], dtype=np.int32)

  dct_func = util_experiments.wrap_dct_inference(num_users)

  scores = dct_func(
    observations_all, contacts_all, None, num_time_steps, None, None, None)

  # User 1 had an infected contact (user0)
  assert np.all(scores[1, :, :2] < 0.5)
  assert np.all(scores[1, :, 2] > 0.5)

  # User 2 had an infected contact (user0)
  assert np.all(scores[2, :, :2] < 0.5)
  assert np.all(scores[2, :, 2] > 0.5)

  # User 0 had a positive test
  assert np.all(scores[0, :, 2] > 0.5)

  # Users 3, 4, 5 had no infected contacts
  assert np.all(scores[3:, :, 2] < 0.5)
