"""Utility functions for Differential Privacy."""

import numba
import numpy as np
from dpfn import logger
from scipy import stats
from typing import Tuple, Union


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

  mean = data[:, 2]
  rv = stats.truncnorm(-mean/sigma, (1-mean)/sigma, loc=mean, scale=sigma)

  data[:, 2] = rv.rvs(size=(len(data), ))
  return data / np.sum(data, axis=-1, keepdims=True)


def calc_ab_beta(
    mean: Union[float, np.ndarray], sigma: Union[float, np.ndarray]):
  """Calculates a heuristic for the beta distribution parameters."""
  mean = np.copy(np.clip(mean, 0.001, 0.999))
  sigma = np.copy(sigma)*np.ones_like(mean)

  a_param = mean*(mean*(1-mean)/sigma**2 - 1)

  if np.any(a_param < 1):
    # a<1 implies that this combination of mean and sigma is not possible.
    # We can try to shrink the mean to 0.5 to get a valid a.
    while np.any(a_param < 1):
      # While-loop is guaranteed to exit, because
      #   infinite small sigma is always possible
      mean += 0.01 * (a_param < 1) * (2 * (mean < 0.5) - 1)
      sigma -= 0.005 * (a_param < 1)
      a_param = mean*(mean*(1-mean)/sigma**2 - 1)

      if np.any(mean > 1.):
        with np.printoptions(threshold=np.inf, precision=3):
          logger.debug(f"mean{mean}")
          logger.debug(f"sigma{sigma}")
          logger.debug(f"a_param{a_param}")
        raise ValueError(f"a{a_param}, mean{mean}")
  return a_param, a_param*(1-mean)/mean


def noise_i_column_beta(data: np.ndarray, sigma: float) -> np.ndarray:
  assert len(data.shape) == 2

  mean = data[:, 2]
  a, b = calc_ab_beta(mean, sigma)

  data[:, 2] = stats.beta(a=a, b=b).rvs(size=(len(data), ))
  return data / np.sum(data, axis=-1, keepdims=True)
