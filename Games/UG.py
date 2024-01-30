import numpy as np

from Games.AbstractGame import BaseGame



class UG(BaseGame):

  def __init__(self, number_of_players):
    self.N = number_of_players
    self.set = np.random.choice(list(range(self.N)), np.random.randint(low=1, high=self.N), replace=False)
    self.shapley_values = self.calculate_shapley_values()


  def get_game_information(self):
    information = dict()
    information['name'] = 'Unanimity Games'
    information['number_of_players'] = self.N
    information['set'] = self.set
    information['ground_truth_shapley_value'] = self.get_shapley_values()
    return information


  def get_value(self, S):
    if self.__issubset(self.set, S):
      return 1
    return 0


  def get_player_number(self):
    return self.N


  def get_shapley_values(self):
    return self.shapley_values

  def calculate_shapley_values(self):
    shapley_values = np.zeros(self.N)
    for i in range(self.N):
      if i in self.set:
        shapley_values[i] = 1 / len(self.set)
    return shapley_values


  def get_name(self):
    return 'Unanimity'


  @staticmethod
  def __issubset(subset, main_set):
    return True if all((i in main_set) for i in subset) else False
