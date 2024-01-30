import time
from abc import abstractmethod

import numpy as np

from Experiment.ExperimentData import ExperimentStorage
from utils.utils import calculate_mse



class BaseApproxMethod:

  def __init__(self, normalize=False):
    self.normalize = normalize
    self.grand_co_value = None
    self.n = None
    self.initial_budget = None
    self.budget = None
    self.game = None
    self.experiment_storage = None
    self.shapley_values = None
    self.player_steps = None
    self.start_time = None
    self.end_time = None
    self.saving_steps = None
    self.evaluation_penalty = None
    self.shapley_mean_values = None
    self.shapley_mean_square_values = None
    self.shapley_var_values = None
    self.estimate_counter = None


  def reset(self, game, budget, evaluation_penalty, step_size):
    # game information
    self.game = game
    self.n = self.game.get_player_number()

    # general information
    self.initial_budget = budget
    self.budget = budget
    self.step_size = step_size
    self.saving_steps = [i for i in range(self.budget, -1, -1) if i % self.step_size == 0]
    self.experiment_storage = ExperimentStorage(self.game)
    self.player_steps = [0] * self.n

    # shapley tracking
    self.shapley_values = np.zeros(self.n)
    self.grand_co_value = None
    if self.normalize:
      self.grand_co_value = self.game.get_value(self.get_all_players())
    self.shapley_mean_values = np.zeros(self.n)
    self.shapley_mean_square_values = np.zeros(self.n)
    self.shapley_var_values = np.zeros(self.n)

    # time tracking
    self.estimate_counter = 0
    self.start_time = None
    self.end_time = None
    self.evaluation_penalty = evaluation_penalty

  def get_only_game_value(self, S):
      return self.game.get_value(S)

  def get_game_value(self, S):
    assert self.budget > 0

    self.__track_mse_over_time()
    if len(S) == 0:
      return True, 0
    if len(S) == self.n:
      if self.normalize and self.grand_co_value is not None:
        return True, self.grand_co_value
      else:
        self.budget -= 1
        self.grand_co_value = self.game.get_value(S)
        return self.budget > 0, self.grand_co_value
    else:
      self.budget -= 1
      return self.budget > 0, self.game.get_value(S)


  def __track_mse_over_time(self):
    if self.budget == self.saving_steps[0]:

      if self.start_time is None:
        self.start_time = time.time()
        # track mse over time
        shapley_estimates = np.zeros(np.array(self.get_estimates().shape))
        self.experiment_storage.add_mse_value(calculate_mse(shapley_estimates, self.game.get_shapley_values()))
        self.experiment_storage.add_time_evaluation(0 + self.evaluation_penalty * self.step_size)
        # track mean and var shapley values over time
        self.__calculate_running_measurements(shapley_estimates)
        self.experiment_storage.add_mean_shapley_value(self.shapley_mean_values)
        self.experiment_storage.add_var_shapley_value(self.shapley_var_values)

      else:
        self.end_time = time.time()
        self.experiment_storage.add_time_evaluation(self.end_time - self.start_time + self.evaluation_penalty * self.step_size)
        # track mse over time
        shapley_estimates = self.get_estimates()
        self.experiment_storage.add_mse_value(calculate_mse(shapley_estimates, self.game.get_shapley_values()))
        # track mean and var shapley values over time
        self.__calculate_running_measurements(shapley_estimates)
        self.experiment_storage.add_mean_shapley_value(self.shapley_mean_values)
        self.experiment_storage.add_var_shapley_value(self.shapley_var_values)
        self.start_time = time.time()

      del self.saving_steps[0]


  def __calculate_running_measurements(self, estimate):
    self.shapley_mean_values = (self.shapley_mean_values * self.estimate_counter + estimate) / (self.estimate_counter + 1)
    self.shapley_mean_square_values = (self.shapley_mean_square_values * self.estimate_counter + np.square(estimate)) / \
                                      (self.estimate_counter + 1)
    self.shapley_var_values = self.shapley_mean_square_values - np.square(self.shapley_mean_values)
    self.estimate_counter += 1



  def update_shapley_value(self, player, estimate):
    step = self.player_steps[player]
    self.shapley_values[player] = (self.shapley_values[player] * step + estimate) / (step+1)
    self.player_steps[player] += 1


  def get_all_players(self):
    return list(range(self.n))


  @abstractmethod
  def approximate_shapley_values(self) -> dict:
    raise NotImplementedError

  @abstractmethod
  def get_estimates(self):
    raise NotImplementedError


  @abstractmethod
  def get_name(self) -> str:
    raise NotImplementedError
