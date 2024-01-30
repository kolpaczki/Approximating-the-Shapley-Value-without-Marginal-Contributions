from Games.AbstractGame import BaseGame

class ShoeGame(BaseGame):

  def __init__(self, n):
    assert n%2 == 0
    self.n = n

  def get_game_information(self):
    information = dict()
    information['name'] = 'Sum of Unanimity Games'
    information['number_of_players'] = self.n
    information['ground_truth_shapley_value'] = self.get_shapley_values()
    return information


  def get_value(self, S):
    S = list(S)
    S_left = len([i for i in S if i >= self.n/2])
    S_right = len([i for i in S if i < self.n/2])
    return min(S_left, S_right)

  def get_player_number(self):
    return self.n


  def get_shapley_values(self):
    return [0.5] * self.n


  def get_name(self):
    return 'Shoe'