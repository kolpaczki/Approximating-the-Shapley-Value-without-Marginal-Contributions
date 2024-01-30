import random

import numpy as np

from Games.AbstractGame import BaseGame

class SOUG(BaseGame):

  def __init__(self, number_of_players, set_number):
    # check that game initialisation makes sense
    assert set_number >= 1
    assert set_number <= 2 ** number_of_players

    self.n = number_of_players
    self.grand_worth = 100
    self.min_size = 1
    self.max_size = int(self.n)
    self.sets, self.coefficients = self.__randomly_generate_sets(set_number)
    self.shapley_values = self.calculate_shapley_values()

  def get_game_information(self):
    information = dict()
    information['name'] = 'Sum of Unanimity Games'
    information['number_of_players'] = self.n
    information['sets'] = self.sets
    information['coefficients'] = self.coefficients
    information['ground_truth_shapley_value'] = self.get_shapley_values()
    return information


  def get_value(self, S):
    value = 0
    for i, set in enumerate(self.sets):
      if self.__issubset(set, S):
        value += self.coefficients[i]
    return value


  def get_player_number(self):
    return self.n


  def get_shapley_values(self):
    return self.shapley_values


  def calculate_shapley_values(self):
    shapley_values = np.zeros(self.n)
    for i in range(self.n):
      shapley_values[i] = np.sum([np.where(i in set, 1, 0) * self.coefficients[j] / len(set) for j, set
                                  in enumerate(self.sets)])
    return shapley_values


  def __randomly_generate_sets(self, set_number):
    sets = list()
    coefficients = list()
    for i in range(set_number):
      size = random.randint(self.min_size, self.max_size)
      sets.append(np.sort(np.random.choice(self.n, size, replace=False)))
      coefficients.append(np.random.random_sample())
    coefficients = [self.grand_worth * i / sum(coefficients) for i in coefficients]
    return sets, coefficients


  def get_name(self):
    return 'SOUG'


  @staticmethod
  def __issubset(subset, main_set):
    return True if all((i in main_set) for i in subset) else False
