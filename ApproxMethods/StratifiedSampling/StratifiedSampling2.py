import numpy as np

from ApproxMethods.AbstractApproxMethod import BaseApproxMethod
from utils.experiment_util import normalize_shapley_value



class StratifiedSampling2(BaseApproxMethod):

  def approximate_shapley_values(self) -> dict:
    # initialize
    self.phi_i_l = np.zeros((self.n, self.n-1))
    m_i = [int(self.budget / self.n)] * self.n
    m_i_l = np.zeros((self.n, self.n-1))
    denominator = np.sum([np.power(l + 1, 2 / 3) for l in range(self.n)])
    for i in range(self.n):
      for l in range(self.n - 1):
        m_i_l[i][l] = int((m_i[i] * np.power(l + 1, 2 / 3)) / denominator)

    more_budget = True
    # main loop
    for m in range(1, int(m_i_l[1][-1])+1):
      for l in range(self.n-1):
        if m < m_i_l[1][l]:
          for i in range(self.n):
            if not more_budget:
              break
            S_i_l_m = self.__draw_S_uniformly_at_random(i, l)
            more_budget, first_value = self.get_game_value(S_i_l_m + [i])
            if not more_budget:
              break
            more_budget, second_value = self.get_game_value(S_i_l_m)
            if not more_budget:
              break
            delta_i = first_value - second_value

            # update shapley value
            self.phi_i_l[i][l] = ((m-1)*self.phi_i_l[i][l] + delta_i) / m

    self.experiment_storage.add_shapley_values(self.get_estimates())
    return self.experiment_storage.to_json()


  def __draw_S_uniformly_at_random(self, i, l):
    players = self.get_all_players()
    players.remove(i)
    sampled_players = np.random.choice(players, l, replace=False)
    return sampled_players


  def get_estimates(self):
    self.shapley_values = 1/self.n * np.sum(self.phi_i_l, axis=1)
    if self.normalize:
      return normalize_shapley_value(self.shapley_values, self.grand_co_value)
    return self.shapley_values


  def get_name(self) -> str:
    if self.normalize:
      return 'StratifiedSampling2_nor'
    return 'StratifiedSampling2'
