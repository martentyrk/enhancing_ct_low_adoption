"""Simulating individuals in a pandemic with SEIR states."""
from COVID19 import model as abm_model
from COVID19 import simulation
import covid19

from dpfn import constants, logger, util
from dpfn.experiments import prequential
import numpy as np
import os
from typing import List, Union


class ABMSimulator():
  """Simulator based on OpenABM.

  Based on the simulator:
  Hinch, et al. 'OpenABM-Covid19—An agent-based model for non-pharmaceutical
  interventions against COVID-19 including contact tracing.'
  PLoS computational biology 2021.

  See README.md for installation instructions.
  """

  def __init__(
      self,
      num_time_steps: int,
      num_users: int,
      rng_seed: int = 123,
      ) -> None:
    self.num_time_steps = num_time_steps
    self.num_users = num_users
    self.rng_seed = rng_seed

    self._day_current = 0

    # Note that contacts are offset with self._day_start_window and contacts
    # prior to self._day_start_window have been discarded.
    self._day_start_window = 0
    # Array with rows (user, timestep, outcome), all integers
    self._observations_all = np.zeros((0, 3), dtype=np.int32)
    # Array with rows (user_from, user_to, timestep, feature)
    self._contacts = np.zeros((0, 4), dtype=np.int32)

    filename = "baseline_parameters.csv"
    filename_hh = "baseline_household_demographics.csv"

    input_param_file = os.path.join(constants.ABM_HOME, filename)
    input_households = os.path.join(constants.ABM_HOME, filename_hh)

    util.check_exists(input_param_file)
    util.check_exists(input_households)
    util.maybe_make_dir("results/tmp/")

    logger.info("Construct ABM simulator")
    params = abm_model.Parameters(  # TODO, import full module!
      input_param_file=input_param_file,
      param_line_number=1,
      output_file_dir="results/tmp/",
      input_households=input_households
    )

    if num_users < 10000:
      # TODO figure out why this fails
      logger.debug('ABM simulator might fail with <10k users')

    # Start with sufficient amount of initial infections. Start in E-state
    n_seed = 5
    if 20000 < num_users < 200000:
      n_seed = 25
    if num_users >= 200000:
      n_seed = 50

    params.set_param("n_total", num_users)
    params.set_param("n_seed_infection", n_seed)
    params.set_param("days_of_interactions", 7)
    params.set_param("rng_seed", rng_seed)

    model_init = abm_model.Model(params)
    self.model = simulation.COVID19IBM(model=model_init)
    self.sim = simulation.Simulation(env=self.model, end_time=num_time_steps)
    logger.info("Finished constructing ABM simulator")

  def get_states_today(self) -> np.ndarray:
    """Returns the states an np.ndarray in size [num_users].

    Each element in [0, 1, 2, 3].
    """
    return np.take(
      constants.state_to_seir,
      np.array(covid19.get_state(self.model.model.c_model)))

  def get_age_users(self) -> np.ndarray:
    """Returns the age categories of all users."""
    return np.array(covid19.get_age(self.model.model.c_model), dtype=np.int32)

  def _get_states_abm(self) -> np.ndarray:
    """Returns the states of the underlying abm simulator."""
    return np.array(covid19.get_state(self.model.model.c_model))

  def get_death_rate(self) -> float:
    """Returns the death rate of the underlying OpenABM simulator."""
    states = np.array(
      covid19.get_state(self.model.model.c_model), dtype=np.int32)
    # State 9 is death in OpenABM simulator
    return np.mean(states == 9)

  def get_critical_rate(self) -> float:
    """Returns the critical rate of the underlying OpenABM simulator."""
    states = np.array(
      covid19.get_state(self.model.model.c_model), dtype=np.int32)
    # State 9 is death in OpenABM simulator
    return np.mean(np.logical_or(states == 6, states == 7))

  def get_contacts(self) -> np.ndarray:
    """Returns contacts.

    Note that contacts are offset with self._day_start_window and contacts prior
    to self._day_start_window have been discarded.
    """
    return self._contacts

  def get_observations_all(self) -> np.ndarray:
    """Returns all observations."""
    return self._observations_all

  def get_observations_today(
      self,
      users_to_observe: np.ndarray,
      p_obs_infected: np.ndarray,
      p_obs_not_infected: np.ndarray,
      obs_rng: np.random._generator.Generator,
      ) -> np.ndarray:
    """Returns the observations for current day."""
    assert users_to_observe.dtype == np.int32

    day_relative = self.get_current_day() - self._day_start_window
    observations_new = prequential.get_observations_one_day(
      self.get_states_today(),
      users_to_observe,
      len(users_to_observe),
      day_relative,
      p_obs_infected,
      p_obs_not_infected,
      obs_rng)
    self._observations_all = np.concatenate(
      (self._observations_all, observations_new), axis=0)
    return observations_new

  def set_window(self, days_offset: int):
    """Sets the window with days_offset at day0.

    All days will start counting 0 at days_offset.
    The internal counter self._day_start_window keeps track of the previous
    counting for day0.
    """
    # Days_offset and day_start are absolute days, cut_off is relative
    to_cut_off = max((0, days_offset - self._day_start_window))
    assert to_cut_off <= self._day_current

    self._observations_all = self._observations_all[
      self._observations_all[:, 1] >= to_cut_off]
    self._observations_all[:, 1] -= to_cut_off

    self._contacts = self._contacts[
      self._contacts[:, 2] >= to_cut_off]
    self._contacts[:, 2] -= to_cut_off

    self._day_start_window = days_offset

  def get_current_day(self) -> int:
    """Returns the current day in absolute counting.

    Note, this day number is INDEPENDENT of the windowing.
    """
    # 0-based indexes!
    return self._day_current

  
  
  def step(self, num_steps: int = 1):
    """Advances the simulator by num_steps days.

    Contacts from OpenABM will be appended to the list of contacts.
    """
    self.sim.steps(num_steps)
    # an array of 1 and 0, where 1 denotes that user uses the app.
    app_users = np.array(covid19.get_app_users(self.model.model.c_model))
    print(app_users[:50])
    print(app_users.shape)
    app_users_ids = np.non_zero(app_users)
    print(app_users_ids[:50])
    
    contacts_incoming = np.array(covid19.get_contacts_daily(
      self.model.model.c_model, self._day_current), dtype=np.int32)
    
    print(contacts_incoming[:50])
    print(contacts_incoming.shape)

    #contacts incoming is 4dim [userID, interaction_individual_index, t_day, interaction type]
    # TODO: use these features
    contacts_incoming[:, 3] = 1
    contacts_incoming[:, 2] = self.get_current_day() - self._day_start_window

    self._contacts = np.concatenate((
      self._contacts, contacts_incoming), axis=0)
    self._day_current += 1

  def quarantine_users(
      self,
      users_to_quarantine: Union[np.ndarray, List[int]],
      num_days: int):
    """Quarantines the defined users.

    This function will remove the contacts that happen TODAY (and which may
    spread the virus and cause people to shift to E-state tomorrow).
    """
    # Timestep of the actual ABM simulator could be found at
    #   * self.model.model.c_model.time
    status = covid19.intervention_quarantine_list(
      self.model.model.c_model,
      list(users_to_quarantine),
      self.get_current_day()+1 + num_days)
    assert int(status) == 0
