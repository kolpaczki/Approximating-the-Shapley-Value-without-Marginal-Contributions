from scipy.special import binom

from ApproxMethods.AbstractApproxMethod import BaseApproxMethod


from utils.utils import powerset


class ExhaustiveSearch(BaseApproxMethod):

    def __init__(self):
        super().__init__()

    def compute_exact_shapley_values(self):
        all_player_set = set(self.get_all_players())
        for subset_size in range(1, self.n + 1):
            weight_s_plus = 1 / (self.n * binom(self.n - 1, subset_size - 1))
            weight_s_minus = 1 / (self.n * binom(self.n - 1, subset_size))
            for subset in powerset(all_player_set, subset_size, subset_size):
                subset_set = set(subset)
                game_value = self.get_only_game_value(subset)
                for player in subset_set:
                    self.shapley_values[player] = self.shapley_values[player] + weight_s_plus * game_value
                for player in all_player_set - subset_set:
                    self.shapley_values[player] = self.shapley_values[player] - weight_s_minus * game_value
        return self.shapley_values

    def approximate_shapley_values(self) -> dict:
        raise NotImplementedError

    def get_estimates(self):
        return self.shapley_values

    def get_name(self) -> str:
        return "ExhaustiveSearch"
