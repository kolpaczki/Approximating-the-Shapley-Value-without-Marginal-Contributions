import numpy as np

from ApproxMethods.AbstractApproxMethod import BaseApproxMethod
from utils.experiment_util import normalize_shapley_value


# ApproSHapley (Castro et al., 2009) with efficiency trick: one evaluation for each marginal contribution
class PermutationSampling(BaseApproxMethod):

  def approximate_shapley_values(self):
    player_set = np.arange(self.n)
    more_budget = True

    while more_budget:
      permutation = np.random.choice(player_set, self.n, replace=False)
      value_list = np.zeros(permutation.shape)

      # iterating over permutation
      for i in range(self.n):
        j = permutation[:i + 1][-1]
        more_budget, value_list[i] = self.get_game_value(permutation[:i + 1])
        if i == 0:
          delta = value_list[i]
        else:
          delta = value_list[i] - value_list[i-1]

        self.update_shapley_value(j, delta)
        if not more_budget:
          break

    self.experiment_storage.add_shapley_values(self.get_estimates())
    return self.experiment_storage.to_json()


  def get_estimates(self):
    if self.normalize:
      return normalize_shapley_value(self.shapley_values, self.grand_co_value)
    return self.shapley_values


  def get_name(self):
    if self.normalize:
      return 'PermutationSampling_nor'
    return "PermutationSampling"
