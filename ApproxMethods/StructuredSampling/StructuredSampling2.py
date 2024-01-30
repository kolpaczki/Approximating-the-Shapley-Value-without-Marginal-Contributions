import numpy as np

from ApproxMethods.AbstractApproxMethod import BaseApproxMethod
from utils.experiment_util import normalize_shapley_value



class StructuredSampling2(BaseApproxMethod):

  def approximate_shapley_values(self):
    M = int(self.budget / (np.square(self.n) - self.n - 2))
    permutations = np.empty(shape=(self.game.get_player_number(), M), dtype=list)

    # create permutations
    for m in range(M):
      for n in range(self.n):
        sample = np.random.choice(list(range(self.n)), self.n, replace=False).tolist()
        permutations[n][m] = sample

    # calculate shapley value from permutations
    available_M = [M] * self.n

    more_budget = True
    while more_budget:
      if sum(available_M) == 0:
        break

      sampled_position = np.random.choice([i for i in range(self.n) if available_M[i] != 0])
      available_M[sampled_position] -= 1
      permutation = permutations[sampled_position][available_M[sampled_position] - 1]
      more_budget, stored_value_one = self.get_game_value(permutation[:sampled_position])
      if not more_budget:
        break
      more_budget, stored_value_two = self.get_game_value(permutation[:sampled_position + 1])
      if not more_budget:
        break

      for player in range(self.n):
        swapped_perm = self.__swap_player_i_to_position_j(player, sampled_position, permutation)
        if player not in permutation[:sampled_position]:
          second_value = stored_value_one
        else:
          more_budget, second_value = self.get_game_value(swapped_perm[:sampled_position])
          if not more_budget:
            break
        if player in permutation[:sampled_position + 1]:
          first_value = stored_value_two
        else:
          more_budget, first_value = self.get_game_value(swapped_perm[:sampled_position + 1])
          if not more_budget:
            break

        estimate = first_value - second_value
        self.update_shapley_value(player, estimate)

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
    if self.normalize:
      return normalize_shapley_value(self.shapley_values, self.grand_co_value)
    return self.shapley_values

  def get_name(self) -> str:
    if self.normalize:
      return "StructuredSampling2_nor"
    else:
      return 'StructuredSampling2'
