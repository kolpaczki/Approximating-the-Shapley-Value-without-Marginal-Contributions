import numpy as np

from ApproxMethods.AbstractApproxMethod import BaseApproxMethod
from utils.experiment_util import normalize_shapley_value


# SVARM
class SVARM(BaseApproxMethod):
  def __init__(self, normalize=False, warm_up=False):
    super().__init__(normalize)
    self.warm_up = warm_up

  def approximate_shapley_values(self) -> dict:
    self.phi_i_plus = np.zeros(self.n)
    self.phi_i_minus = np.zeros(self.n)
    self.c_i_plus = np.zeros(self.n)
    self.c_i_minus = np.zeros(self.n)
    self.H_n = sum([1/s for s in range(1, self.n+1)])

    if self.warm_up:
      self.__conduct_warmup()

    more_budget = True
    while more_budget:
      A_plus = self.__sample_A_plus()
      more_budget = self.__positive_update(A_plus)
      if not more_budget:
        break

      A_minus = self.__sample_A_minus()
      more_budget = self.__negative_update(A_minus)

    self.experiment_storage.add_shapley_values(self.get_estimates())
    return self.experiment_storage.to_json()


  def __sample_A_plus(self):
    s_plus = np.random.choice(range(1, self.n+1), 1, p=[1/(s*self.H_n) for s in range(1, self.n+1)])
    return np.random.choice(self.get_all_players(), s_plus, replace=False)


  def __sample_A_minus(self):
    s_minus = np.random.choice(range(0, self.n), 1, p=[1/((self.n-s)*self.H_n) for s in range(0, self.n)])
    return np.random.choice(self.get_all_players(), s_minus, replace=False)


  def __positive_update(self, A):
    more_budget, value = self.get_game_value(A)
    for i in A:
      self.phi_i_plus[i] = (self.phi_i_plus[i]*self.c_i_plus[i] + value) / (self.c_i_plus[i] + 1)
      self.c_i_plus[i] += 1
    return more_budget


  def __negative_update(self, A):
    more_budget, value = self.get_game_value(A)
    players = [i for i in self.get_all_players() if i not in A]
    for i in players:
      self.phi_i_minus[i] = (self.phi_i_minus[i]*self.c_i_minus[i] + value) /(self.c_i_minus[i] + 1)
      self.c_i_minus[i] += 1
    return more_budget


  def __conduct_warmup(self):
    for i in self.get_all_players():
      players_without_i = [j for j in self.get_all_players() if j != i]

      # sample A_plus
      size_of_A_plus = np.random.choice(self.n, 1)
      A_plus = np.random.choice(players_without_i, size_of_A_plus, replace=False)

      # sample A_minus
      size_of_A_minus = np.random.choice(self.n, 1)
      A_minus = np.random.choice(players_without_i, size_of_A_minus, replace=False)

      # set values
      _, value = self.get_game_value(np.append(A_plus, i))
      self.phi_i_plus[i] = value
      self.c_i_plus[i] = 1

      _, value = self.get_game_value(A_minus)
      self.phi_i_minus[i] = value
      self.c_i_minus[i] = 1


  def get_estimates(self):
    self.shapley_values = self.phi_i_plus - self.phi_i_minus
    if self.normalize:
      return normalize_shapley_value(self.shapley_values, self.grand_co_value)
    return self.shapley_values


  def get_name(self) -> str:
    if self.normalize:
      if self.warm_up:
        return 'SVARM_warmup_nor'
      return 'SVARM_nor'
    if self.warm_up:
      return 'SVARM_warmup'
    return 'SVARM'
