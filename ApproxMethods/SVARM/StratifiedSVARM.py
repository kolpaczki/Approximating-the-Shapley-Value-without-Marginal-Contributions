import math

import numpy as np

from ApproxMethods.AbstractApproxMethod import BaseApproxMethod
from utils import utils
from utils.experiment_util import normalize_shapley_value

# Stratified SVARM
class StratifiedSVARM(BaseApproxMethod):
    def __init__(self, normalize=False, warm_up=False, rich_warm_up=False, paired_sampling=False, dist_type="paper"):
        super().__init__(normalize)
        self.warm_up = warm_up
        self.rich_warm_up = rich_warm_up
        self.paired_sampling = paired_sampling
        self.dist_type = dist_type

    def approximate_shapley_values(self) -> dict:
        self.phi_i_l_plus = np.zeros((self.n, self.n))
        self.phi_i_l_minus = np.zeros((self.n, self.n))
        self.c_i_l_plus = np.zeros((self.n, self.n))
        self.c_i_l_minus = np.zeros((self.n, self.n))

        # probability distribution over sizes (2,...,n-2) for sampling
        distribution = utils.generateDistribution(self.dist_type, self.n)
        probs = [distribution[s] for s in range(self.n + 1)]

        self.__exact_calculation()
        if self.warm_up:
            if self.rich_warm_up:
                self.__rich_warmup()
            else:
                self.__positive_warmup()
                self.__negative_warmup()

        while self.budget > 0:
            s = np.random.choice(range(0, self.n + 1), 1, p=probs)
            A = np.random.choice(self.get_all_players(), s, replace=False)
            self.__update_procedure(A)

            if not self.budget > 0:
                break

            if self.paired_sampling:
                players_without_A = [i for i in self.get_all_players() if i not in A]
                self.__update_procedure(players_without_A)

        self.experiment_storage.add_shapley_values(self.get_estimates())
        return self.experiment_storage.to_json()

    def __exact_calculation(self):
        # negative strata
        for i in range(self.n):
            self.phi_i_l_minus[i][0] = 0
            self.c_i_l_minus[i][0] = 1

        if not self.normalize:
            _, self.grand_co_value = self.get_game_value(self.get_all_players())

        # positive n-1 strata
        for i in range(self.n):
            self.phi_i_l_plus[i][self.n - 1] = self.grand_co_value
            self.c_i_l_plus[i][self.n - 1] = 1

        for i in range(self.n):
            players = self.get_all_players()
            players.remove(i)

            _, v_plus = self.get_game_value([i])

            # positive 0 strata
            self.phi_i_l_plus[i][0] = v_plus
            self.c_i_l_plus[i][0] = 1

            for j in players:
                # negative 1 strata
                self.phi_i_l_minus[j][1] = (self.phi_i_l_minus[j][1] * self.c_i_l_minus[j][1] + v_plus) / (
                            self.c_i_l_minus[j][1] + 1)
                self.c_i_l_minus[j][1] += 1

            _, v_minus = self.get_game_value(players)

            # negative n-1 strata
            self.phi_i_l_minus[i][self.n - 1] = v_minus
            self.c_i_l_minus[i][self.n - 1] = 1

            # positive n-2 strata
            for j in players:
                self.phi_i_l_plus[j][self.n - 2] = (self.phi_i_l_plus[j][self.n - 2] * self.c_i_l_plus[j][
                    self.n - 2] + v_minus) / (self.c_i_l_plus[j][self.n - 2] + 1)
                self.c_i_l_plus[j][self.n - 2] += 1

    def __positive_warmup(self):
        for s in range(2, self.n - 1):
            pi = np.random.choice(self.get_all_players(), self.n, replace=False)

            for k in range(0, int(self.n / s)):
                A = [pi[r + k * s - 1] for r in range(1, s + 1)]
                _, v = self.get_game_value(A)
                for i in A:
                    self.phi_i_l_plus[i][s - 1] = v
                    self.c_i_l_plus[i][s - 1] = 1

            if self.n % s != 0:
                A = list([pi[r - 1] for r in range(self.n - (self.n % s) + 1, self.n + 1)])
                players = [player for player in self.get_all_players() if player not in A]
                B = list(np.random.choice(players, s - (self.n % s), replace=False))
                _, v = self.get_game_value(list(set(A + B)))
                for i in A:
                    self.phi_i_l_plus[i][s - 1] = v
                    self.c_i_l_plus[i][s - 1] = 1

    def __negative_warmup(self):
        for s in range(2, self.n - 1):
            pi = np.random.choice(self.get_all_players(), self.n, replace=False)

            for k in range(0, int(self.n / s)):
                A = [pi[r + k * s - 1] for r in range(1, s + 1)]
                players = [player for player in self.get_all_players() if player not in A]
                _, v = self.get_game_value(players)
                for i in A:
                    self.phi_i_l_minus[i][self.n - s] = v
                    self.c_i_l_minus[i][self.n - s] = 1

            if self.n % s != 0:
                A = [pi[r - 1] for r in range(self.n - (self.n % s) + 1, self.n + 1)]
                players = [player for player in self.get_all_players() if player not in A]
                B = list(np.random.choice(players, s - (self.n % s), replace=False))
                players = [player for player in self.get_all_players() if player not in list(set(A + B))]
                _, v = self.get_game_value(players)
                for i in A:
                    self.phi_i_l_minus[i][self.n - s] = v
                    self.c_i_l_minus[i][self.n - s] = 1

    def __rich_warmup(self):
        for s in range(2, self.n - 1):
            pi = np.random.choice(self.get_all_players(), self.n, replace=False)
            for k in range(0, int(self.n / s)):
                A = [pi[r + k * s - 1] for r in range(1, s + 1)]
                self.__update_procedure(A)
                complement = [i for i in self.get_all_players() if i not in A]
                self.__update_procedure(complement)
            if self.n % s != 0:
                A = [pi[r - 1] for r in range(self.n - (self.n % s) + 1, self.n + 1)]
                players = [player for player in self.get_all_players() if player not in A]
                B = list(np.random.choice(players, s - (self.n % s), replace=False))
                A_cup_B = list(set(A + B))
                self.__update_procedure(A_cup_B)
                complement = [i for i in self.get_all_players() if i not in A_cup_B]
                self.__update_procedure(complement)

    def get_estimates(self):
        phi_i_plus = 1 / (np.sum(np.where(self.c_i_l_plus > 0, 1, 0), axis=1)) * np.sum(self.phi_i_l_plus, axis=1)
        phi_i_minus = 1 / (np.sum(np.where(self.c_i_l_minus > 0, 1, 0), axis=1)) * np.sum(self.phi_i_l_minus, axis=1)
        self.shapley_values = phi_i_plus - phi_i_minus
        if self.normalize:
            return normalize_shapley_value(self.shapley_values, self.grand_co_value)
        return self.shapley_values

    def __update_procedure(self, A):
        _, v = self.get_game_value(A)
        s = len(A)
        for i in A:
            self.phi_i_l_plus[i][s - 1] = (self.phi_i_l_plus[i][s - 1] * self.c_i_l_plus[i][s - 1] + v) / (
                        self.c_i_l_plus[i][s - 1] + 1)
            self.c_i_l_plus[i][s - 1] += 1

        not_A = [i for i in self.get_all_players() if i not in A]
        for i in not_A:
            self.phi_i_l_minus[i][s] = (self.phi_i_l_minus[i][s] * self.c_i_l_minus[i][s] + v) / (
                        self.c_i_l_minus[i][s] + 1)
            self.c_i_l_minus[i][s] += 1

    def get_name(self) -> str:
        name = 'StratSVARM_' + self.dist_type

        if self.warm_up:
            name += '_warmup'
            if self.rich_warm_up:
                name += '_rich'

        if self.paired_sampling:
            name += '_paired'

        if self.normalize:
            name += '_nor'

        return name
