"""Utility functions for Differential Privacy."""

import numba
import numpy as np
from scipy import stats
from typing import Tuple


@numba.njit
def get_num_contacts_min_max(
    pc_array: np.ndarray, num_time_steps: int) -> Tuple[int, int]:
  """Get the minimum and maximum number of contacts per any time step."""
  num_contacts = np.zeros((num_time_steps), dtype=np.int32)

  for row in pc_array:
    timestep = int(row[0])

    if timestep > 0:
      num_contacts[timestep] += 1

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
  assert 0 <= num_contacts <= 2000

  alpha = margin
  beta = 1 - margin

  M = (1 - probab0)*np.power(1 - alpha*probab1, num_contacts)
  numer = (1 - (beta*(1 - probab1) + (1 - beta))*M)
  denom = 1 - M

  R2_value = np.abs(np.log(1 / (1-beta*probab1)))
  R3_value = np.abs(np.log(numer/denom))
  return float(max((R2_value, R3_value)))


# @numba.njit
def noise_i_column(data: np.ndarray, sigma: float) -> np.ndarray:
  assert len(data.shape) == 2
  num_rows = len(data)

  def make_rv(mean: np.ndarray, sigma: float):
    return stats.truncnorm(-mean/sigma, (1-mean)/sigma, loc=mean, scale=sigma)

  noise_additive = make_rv(mean=data[:, 2], sigma=sigma)
  data[:, 2] = noise_additive.rvs(size=(num_rows, ))
  return data / np.sum(data, axis=-1, keepdims=True)
