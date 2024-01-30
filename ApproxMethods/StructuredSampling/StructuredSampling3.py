import numpy as np

from ApproxMethods.AbstractApproxMethod import BaseApproxMethod
from utils.experiment_util import normalize_shapley_value



class StructuredSampling3(BaseApproxMethod):

  def approximate_shapley_values(self) -> dict:
    # initialization
    self.phi_i_l = np.zeros((self.n, self.n-1))
    self.c_i_l = np.zeros((self.n, self.n-3))
    M = int(self.budget / (np.square(self.n) - self.n - 2))

    # pre-evaluation
    for i in range(self.n):
      _, self.phi_i_l[i][0] = self.get_game_value([i])
      tmp_players = self.get_all_players()
      tmp_players.remove(i)
      _, game_value = self.get_game_value(tmp_players)
      self.phi_i_l[i][-1] = self.grand_co_value - game_value

    # main loop
    more_budget = True
    while more_budget:
      for l in range(1, self.n-3):
        if not more_budget:
          break
        sample_permutation = np.random.choice(self.get_all_players(), self.n, replace=False).tolist()

        # store values
        more_budget, v_minus = self.get_game_value(sample_permutation[:l+1])
        if not more_budget:
          break
        more_budget, v_plus = self.get_game_value(sample_permutation[:l+2])
        if not more_budget:
          break

        for player in range(self.n):
          swapped_permutation = self.__swap_player_i_to_position_j(player, l+1, sample_permutation)
          if sample_permutation.index(player) < (l+1):
            more_budget, game_value = self.get_game_value(swapped_permutation[:l+1])
            delta_i = v_plus - game_value
          elif sample_permutation.index(player) > (l+1):
            more_budget, game_value = self.get_game_value(swapped_permutation[:l+2])
            delta_i = game_value - v_minus
          else:
            delta_i = v_plus - v_minus

          # update shapley value
          self.phi_i_l[player][l] = (self.c_i_l[player][l] * self.phi_i_l[player][l] + delta_i) / (self.c_i_l[player][l] + 1)
          self.c_i_l[player][l] += 1

          if not more_budget:
            break

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
    self.shapley_values = 1 / (np.sum(np.where(self.c_i_l > 0, 1, 0), axis=1) + 2) * np.sum(self.phi_i_l, axis=1)
    if self.normalize:
      return normalize_shapley_value(self.shapley_values, self.grand_co_value)
    return self.shapley_values


  def get_name(self) -> str:
    if self.normalize:
      return 'StructuredSampling3_nor'
    else:
      return 'StructuredSampling3'
