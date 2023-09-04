"""Compare inference methods on likelihood and AUROC, and run prequentially."""
import covasim as cv
from dpfn.experiments import compare_covasim
import numpy as np


def test_covasim_tests():
  """Compares different inference algorithms on the supplied contact graph."""

  num_time_steps = 30
  num_users = 2000

  pars = {
    "pop_type": 'hybrid',
    "pop_size": num_users,
    "pop_infected": 100,
    "start_day": '2020-02-01',
    "end_day": '2020-03-01',
  }

  def subtarget_func(sim, history):
    """Subtarget function for testing.

    This function is run every day after the contacts are sampled and before
    the tests are run. It returns a dictionary with the keys 'inds' and 'vals'.
    """
    assert isinstance(history, dict)

    infected = np.logical_or(sim.people.exposed, sim.people.infectious)

    if len(history['observations'] > 1):
      obs_rel = np.copy(history['observations']).astype(np.int32)
      assert np.max(obs_rel) < num_users
      assert np.min(obs_rel) >= 0
      assert np.max(obs_rel[:, 1]) <= sim.t
      assert np.max(obs_rel[:, 2]) <= 1

      obs_rel = obs_rel[obs_rel[:, 1] == sim.t-1]

      inds_pos = obs_rel[obs_rel[:, 2] > 0][:, 0]
      if len(inds_pos) > 0:
        true_positives = np.sum(infected[inds_pos])
        precision = true_positives / len(inds_pos)

        assert precision > 0.7, (
          "Stochastic test. True positives are calculated one day after tests.")

        print((
          f"Precision in test {precision:8.3f} from {true_positives:4} TP in "
          f"{len(inds_pos):4} positive tests, at IR {np.mean(infected):8.3f}"))
      inds_neg = obs_rel[obs_rel[:, 2] < 1][:, 0]
      if len(inds_neg) > 0:
        true_negatives = np.sum(np.logical_not(infected[inds_neg]))
        recall = true_negatives / len(inds_neg)

        assert recall > 0.7, (
          "Stochastic test. True negatives are calculated one day after tests.")

        print((
          f"Recall in test {recall:8.3f} from {true_negatives:4} TN in "
          f"{len(inds_neg):4} negative tests, at IR {np.mean(infected):8.3f}"))

    inds = sim.people.uid  # Everyone in the population
    output = {
      'inds': inds, 'vals': np.ones_like(inds)+np.random.rand(len(inds))}
    return output, history

  test_intervention = cv.test_num(
    daily_tests=int(0.1*num_users),
    do_plot=False,
    sensitivity=1.0,
    subtarget=subtarget_func,
    label='intervention_history')

  # Create, run, and plot the simulations
  analyzer = compare_covasim.StoreSEIR(
    num_days=num_time_steps, label='analysis')
  sim = cv.Sim(
    pars,
    interventions=test_intervention,
    analyzers=analyzer)

  # COVASIM run() runs the entire simulation, including the initialization
  sim.run()
