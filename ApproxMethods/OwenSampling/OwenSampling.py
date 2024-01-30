import numpy as np
import numpy.random
from typing import List

from ApproxMethods.AbstractApproxMethod import BaseApproxMethod
from utils.experiment_util import normalize_shapley_value


# Owen Sampling (Okhrati and Lipani, 2020)
class OwenSampling(BaseApproxMethod):

  def __init__(self):
    super().__init__()

  def approximate_shapley_values(self) -> dict:
    # store shapley values
    self.Q = max(int(self.budget / (self.n * 4)),1)   # each stratum of a player gets 2 updates (M=2, see paper), some one more
    self.phi_i_q = np.zeros((self.n, self.Q+1))
    self.c_i_q = np.zeros((self.n, self.Q+1))

    more_budget = True
    while more_budget:
      for k in range(self.Q+1):
        q = k / self.Q
        if not more_budget:
          break
        for i in range(self.n):
          S_i_q = self.__sample_S(i, q)
          more_budget, first_value = self.get_game_value(S_i_q + [i])
          if not more_budget:
            break
          more_budget, second_value = self.get_game_value(S_i_q)
          delta_i = first_value - second_value

          # update shapley values
          self.phi_i_q[i][k] = (self.phi_i_q[i][k] * self.c_i_q[i][k] + delta_i) / (self.c_i_q[i][k] + 1)
          self.c_i_q[i][k] += 1
          if not more_budget:
            break

    self.experiment_storage.add_shapley_values(self.get_estimates())
    return self.experiment_storage.to_json()


  def __sample_S(self, player, q) -> List[int]:
    available_player = [i for i in range(self.n) if i != player]
    return_S = list()
    for p in available_player:
      if numpy.random.random_sample() < q:
        return_S.append(p)
    return return_S


  def get_estimates(self):
    self.shapley_values = 1 / np.sum(np.where(self.c_i_q > 0, 1, 0), axis=1) * np.sum(self.phi_i_q, axis=1)
    na_indices = np.argwhere(np.isnan(self.shapley_values))
    self.shapley_values[na_indices] = 0
    if self.normalize:
      return normalize_shapley_value(self.shapley_values, self.grand_co_value)
    return self.shapley_values


  def get_name(self) -> str:
    if self.normalize:
      return 'OwenSampling_nor'
    else:
      return 'OwenSampling'
