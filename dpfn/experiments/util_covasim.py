"""Compare inference methods on likelihood and AUROC, and run prequentially."""
import covasim as cv
import numpy as np
from dpfn import logger
import os
import psutil
import time


class StoreSEIR(cv.Analyzer):
  """Store the SEIR rates for each day."""

  def __init__(self, num_days, *fargs, **kwargs):
    super().__init__(*fargs, **kwargs)
    self.t = np.zeros((num_days), dtype=np.float32)
    self.s_rate = np.zeros((num_days), dtype=np.float32)
    self.e_rate = np.zeros((num_days), dtype=np.float32)
    self.i_rate = np.zeros((num_days), dtype=np.float32)
    self.r_rate = np.zeros((num_days), dtype=np.float32)

    self.isolation_rate = np.zeros((num_days), dtype=np.float32)

    self.precisions = np.zeros((num_days), dtype=np.float32)
    self.recalls = np.zeros((num_days), dtype=np.float32)

    self.timestamps = np.zeros((num_days+1), dtype=np.float64)
    self._time_prev = time.time()

  def apply(self, sim):
    """Applies the analyser on the simulation object."""
    ppl = sim.people  # Shorthand
    num_people = len(ppl)
    day = sim.t

    self.t[day] = day
    self.s_rate[day] = ppl.susceptible.sum() / num_people
    self.e_rate[day] = (ppl.exposed.sum() - ppl.infectious.sum()) / num_people
    self.i_rate[day] = ppl.infectious.sum() / num_people
    self.r_rate[day] = ppl.recovered.sum() + ppl.dead.sum() / num_people

    isolated = np.logical_or(ppl.isolated, ppl.quarantined)
    true_positives = np.sum(np.logical_and(isolated, ppl.infectious))

    self.isolation_rate[day] = np.sum(isolated) / num_people

    # precision should be 1 when there are no false positives
    self.precisions[day] = (true_positives+1E-9) / (np.sum(isolated) + 1E-9)

    # Number of infected people in isolation over total number of infected
    self.recalls[day] = (true_positives+1E-9) / (np.sum(ppl.infectious) + 1E-9)
    self.timestamps[day] = time.time() - self._time_prev
    self._time_prev = time.time()

    time_inf_func = sim.get_intervention(
      'intervention_history').history['time_inf_func'][sim.t]

    logger.info((
      f"On day {day:3} recall is {self.recalls[day]:.2f} "
      f"at IR {self.i_rate[day] + self.e_rate[day]:.4f} "
      f"timediff {self.timestamps[day]:8.1f}"
      f"({time_inf_func:5.1f})"))


def log_to_wandb(wandb_runner):
  """Logs system statistics to wandb every minute."""
  while True:
    loadavg1, loadavg5, _ = os.getloadavg()
    swap_use = psutil.swap_memory().used / (1024.0 ** 3)

    wandb_runner.log({
      "loadavg1": loadavg1,
      "loadavg5": loadavg5,
      "swap_use": swap_use})
    time.sleep(60)
