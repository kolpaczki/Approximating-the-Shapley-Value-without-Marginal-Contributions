import math
from typing import List

import numpy as np
import time
from ApproxMethods.AbstractApproxMethod import BaseApproxMethod
from Experiment.GameEvaluationStorage import GameEvaluationStorage
from utils.utils import calculate_mse
from Games.AbstractGame import BaseGame



class GameEvaluator:

  def __init__(self,
               approx_methods: [BaseApproxMethod],
               budget, number_of_runs,
               evaluation_penalty, step_size):
    # evaluation parameters
    self.approx_methods = approx_methods
    self.budget = budget
    self.number_of_runs = number_of_runs
    self.evaluation_penalty = evaluation_penalty
    self.step_size = step_size

    # data storage
    self.storage = GameEvaluationStorage(number_of_runs, self.approx_methods, budget)


  def evaluate_exact_same_game(self, game):
    for i, method in enumerate(self.approx_methods):
      mse_list = list()
      print(f'### Approximation Method: {method.get_name()} ###')
      for run in range(self.number_of_runs):
        print(f'# --------- No. of run: {run} --------- #')
        # conduct run
        start_time = time.time()
        method.reset(game=game,
                     budget=self.budget,
                     evaluation_penalty=self.evaluation_penalty,
                     step_size=self.step_size)
        experiment_storage = method.approximate_shapley_values()
        end_time = time.time()
        experiment_storage['overall_time'] = end_time - start_time
        try:
          print(f'Mse: {experiment_storage["mse_history"][-1]}')
          mse_list.append(experiment_storage["mse_history"][-1])
        except:
          print('This method does not have a mse_history')
        print(f'Overall time: {experiment_storage["overall_time"]}')
        print('# --------------------------------- #')

        # add results
        self.storage.add_experiment_information(experiment_storage, i, run, game)
      try:
        print('*** Mean_MSE over all runs:', np.mean(mse_list))
        print('*** Std_Error_MSE over all runs:', math.sqrt((1.0 / (self.number_of_runs - 1.0)) * np.var(mse_list)))
      except:
        print('*** No MSE information available.')
    return self.storage.to_json()


  def evaluate_game(self, games:List):
    for i, method in enumerate(self.approx_methods):
      mse_list = list()
      print(f'### ----------- Approximation Method: {method.get_name()} ----------- ###')
      for run in range(self.number_of_runs):
        print(f'# --------- No. of run: {run} --------- #')
        # conduct run
        start_time = time.time()
        method.reset(game=games[run],
                     budget=self.budget,
                     evaluation_penalty=self.evaluation_penalty,
                     step_size=self.step_size)
        experiment_storage = method.approximate_shapley_values()
        end_time = time.time()
        experiment_storage['overall_time'] = end_time - start_time
        try:
          print(f'Mse: {experiment_storage["mse_history"][-1]}')
          mse_list.append(experiment_storage["mse_history"][-1])
        except:
          print('This method does not have a mse_history')
        print(f'Overall time: {experiment_storage["overall_time"]}')
        print('# --------------------------------- #')

        # add results
        self.storage.add_experiment_information(experiment_storage, i, run, games[run])
      try:
        print('*** Mean_MSE over all runs:', np.mean(mse_list))
        print('*** Std_MSE over all runs:', math.sqrt((1.0 / (self.number_of_runs - 1.0)) * np.var(mse_list)))
      except:
        print('*** No MSE information available.')
      print('### --------------------------------------------------------------- ###')
    return self.storage.to_json()

