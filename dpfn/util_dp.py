"""Utility functions for Differential Privacy."""

import numba
import numpy as np
from typing import Tuple


@numba.njit
def get_num_contacts_min_max(
    pc_array: np.ndarray, num_time_steps: int) -> Tuple[int, int]:
  """Get the minimum and maximum number of contacts per any time step."""
  num_contacts = np.zeros((num_time_steps), dtype=np.int32)
  for row in pc_array:
    num_contacts[row[0]] += 1

  return int(num_contacts.min()), int(num_contacts.max())


@numba.njit
def get_sensitivity_log(
    num_contacts: int,
    probab0: float,
    probab1: float,
    margin: float) -> float:
  """Calculates the sensitivity in the logits."""
  assert 0 <= margin <= 1
  assert 0 <= probab1 <= 1
  assert 0 <= num_contacts <= 1000

  alpha = margin
  beta = 1 - margin

  M = (1 - probab0)*np.power(1 - alpha*probab1, num_contacts)
  numer = (1 - (beta*(1 - probab1) + (1 - beta))*M)
  denom = (1 - M)

  R2_value = np.abs(np.log(1 / (1-beta*probab1)))
  R3_value = np.abs(np.log(numer/denom))
  return float(max((R2_value, R3_value)))
