import numpy as np

from ApproxMethods.AbstractApproxMethod import BaseApproxMethod
from utils.experiment_util import normalize_shapley_value


# Structured Sampling (van Campen et al., 2018)
class StructuredSampling(BaseApproxMethod):

  def approximate_shapley_values(self):
    self.phi_i_l = np.zeros((self.n, self.n))
    self.c_i_l = np.zeros((self.n, self.n))

    more_budget = True
    while more_budget:
      for l in range(self.n):
        if not more_budget:
          break

        sample = np.random.choice(self.get_all_players(), self.n, replace=False).tolist()

        for i in range(self.n):
          swapped_sample = self.__swap_player_i_to_position_j(i, l, sample)
          S = swapped_sample[:l]
          if not more_budget:
            break
          more_budget, first_value = self.get_game_value(S + [i])
          if not more_budget:
            break
          more_budget, second_value = self.get_game_value(S)

          estimate = first_value - second_value
          self.phi_i_l[i][l] = (self.phi_i_l[i][l] * self.c_i_l[i][l] + estimate) / (self.c_i_l[i][l] + 1)
          self.c_i_l[i][l] += 1

    self.experiment_storage.add_shapley_values(self.get_estimates())
    return self.experiment_storage.to_json()


  @staticmethod
  def __swap_player_i_to_position_j(player, j, permutation):
    perm = permutation.copy()
    player_at_j = perm[j]
    position_of_player = perm.index(player)
    perm[j] = player
    perm[position_of_player] = player_at_j
    return perm


  def get_estimates(self):
    self.shapley_values = 1 / np.sum(np.where(self.c_i_l > 0, 1, 0), axis=1) * np.sum(self.phi_i_l, axis=1)
    na_indices = np.argwhere(np.isnan(self.shapley_values))
    self.shapley_values[na_indices] = 0
    if self.normalize:
      return normalize_shapley_value(self.shapley_values, self.grand_co_value)
    return self.shapley_values


  def get_name(self) -> str:
    if self.normalize:
      return 'StructuredSampling_nor'
    else:
      return 'StructuredSampling'
