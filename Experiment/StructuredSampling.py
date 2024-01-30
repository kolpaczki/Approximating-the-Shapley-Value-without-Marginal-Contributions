import numpy as np

from ApproxMethods.AbstractApproxMethod import BaseApproxMethod


class StructuredSampling(BaseApproxMethod):

  def approximate_shapley_values(self):
    M = int(self.budget / (2 * np.power(self.n, 2)))
    permutations = np.empty(shape=(self.n, M), dtype=list)

    for m in range(M):
      for n in range(self.n):
        sample = np.random.choice(self.get_all_players(), self.n, replace=False).tolist()
        permutations[n][m] = sample

    for j in range(self.n):
      for player in range(self.n):
        permutations[j][:] = self.__swap_player_i_to_position_j(player, j, permutations[j][:])

        for permutation in permutations[j]:
          _, first_value = self.get_game_value(permutation[:j + 1])
          _, second_value = self.get_game_value(permutation[:j])
          estimate = first_value - second_value
          self.update_shapley_value(player, estimate)

    self.experiment_storage.add_shapley_values(self.shapley_values, self.normalize, self.grand_co_value)
    return self.experiment_storage.to_json()

  @staticmethod
  def __swap_player_i_to_position_j(player, j, permutations):
    for permutation in permutations:
      player_at_j = permutation[j]
      position_of_player = permutation.index(player)
      permutation[j] = player
      permutation[position_of_player] = player_at_j
    return permutations

  def get_name(self) -> str:
    if self.normalize:
      return 'StructuredSampling_nor'
    else:
      return 'StructuredSampling'
