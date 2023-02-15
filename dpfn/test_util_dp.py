"""Test functions for util_dp.py."""

from dpfn import util_dp
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
    num_contacts, probab0, probab1, margin)
  sensitivity2 = util_dp.get_sensitivity_log(
    num_contacts*10, probab0, probab1, margin)

  assert sensitivity1 > sensitivity2
