import itertools
import math
import random

import numpy as np

from ApproxMethods.AbstractApproxMethod import BaseApproxMethod
from utils import utils
from utils.experiment_util import normalize_shapley_value

# Stratified SVARM+
class StratifiedSVARMnoReplacement(BaseApproxMethod):
    def __init__(self, normalize=False, dynamic=False, dist_type="paper", smart_factor=1.0):
        super().__init__(normalize)
        self.dynamic = dynamic
        self.dist_type = dist_type
        self.smart_factor = smart_factor


    def approximate_shapley_values(self) -> dict:
        # initializes positive and negative sub-Shapley value estimates with zero
        # initializes sample counters of each estimate with zero
        self.phi_i_l_plus = np.zeros((self.n, self.n))
        self.phi_i_l_minus = np.zeros((self.n, self.n))
        self.c_i_l_plus = np.zeros((self.n, self.n))
        self.c_i_l_minus = np.zeros((self.n, self.n))

        # list containing a list for each size s between 2 and n-2
        # each lists stores the so far sampled coalitions of size as sets
        self.sampled_lists = [[] for s in range(0, self.n + 1)]
        self.switch = [False for s in range(0, self.n+1)]
        self.left_to_sample_lists = [[] for s in range(0, self.n+1)]

        # probability distribution over sizes (2,...,n-2) for sampling
        distribution = utils.generateDistribution(self.dist_type, self.n)
        self.probs = [distribution[s] for s in range(self.n + 1)]
        self.original_probs = [distribution[s] for s in range(self.n + 1)]

        # executes exact calculation: evaluate all sets of size 0,1,n-1,n
        self.__exact_calculation()

        # main loop
        while self.budget > 0:
            probs_sum = sum(self.probs)
            if probs_sum == 0:
                # all coalitions have been sampled, evaluate a random set to run out of budget
                size = random.randint(2, self.n-2)
                coalition = np.random.choice(self.get_all_players(), size, replace=False)
                _, v = self.get_game_value(coalition)
            else:
                # draw size s according to probability distribution
                size = np.random.choice(range(0, self.n + 1), 1, p=self.probs)[0]

                # update switch state and list if necessary
                if not self.switch[size] and len(self.sampled_lists[size]) >= self.smart_factor * math.comb(self.n, size):
                    for subset in list(itertools.combinations([i for i in range(self.n)], size)):
                        if set(subset) not in self.sampled_lists[size]:
                            self.left_to_sample_lists[size].append(set(subset))
                    self.switch[size] = True

                # draw coalition from list of remaining ones ore naively
                if self.switch[size]:
                    coalition = self.__sample_from_remaining(size)
                else:
                    coalition = self.__sample_naively(size)

                # add the new coalition to the list, update estimates with the coalition, update sampling distribution
                self.sampled_lists[size].append(set(coalition))
                self.__update_procedure(coalition)
                self.__update_probs(size, probs_sum)

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


    def get_estimates(self):
        phi_i_plus = 1 / (np.sum(np.where(self.c_i_l_plus > 0, 1, 0), axis=1)) * np.sum(self.phi_i_l_plus, axis=1)
        phi_i_minus = 1 / (np.sum(np.where(self.c_i_l_minus > 0, 1, 0), axis=1)) * np.sum(self.phi_i_l_minus, axis=1)
        self.shapley_values = phi_i_plus - phi_i_minus
        if self.normalize:
            return normalize_shapley_value(self.shapley_values, self.grand_co_value)
        return self.shapley_values


    def __update_procedure(self, coalition):
        _, v = self.get_game_value(coalition)
        s = len(coalition)

        # update all players contained in the coalition
        for i in coalition:
            self.phi_i_l_plus[i][s - 1] = (self.phi_i_l_plus[i][s - 1] * self.c_i_l_plus[i][s - 1] + v) / (self.c_i_l_plus[i][s - 1] + 1)
            self.c_i_l_plus[i][s - 1] += 1

        # update all players not contained in the coalition
        not_coalition = [i for i in self.get_all_players() if i not in coalition]
        for i in not_coalition:
            self.phi_i_l_minus[i][s] = (self.phi_i_l_minus[i][s] * self.c_i_l_minus[i][s] + v) / (self.c_i_l_minus[i][s] + 1)
            self.c_i_l_minus[i][s] += 1


    def __alreadySampled(self, coalition):
        return set(coalition) in self.sampled_lists[len(coalition)]


    def __update_probs(self, size, probs_sum):
        s = size
        stratum_size = math.comb(self.n, s)
        old_prob_s = self.probs[s]
        if self.dynamic:
            sampled = len(self.sampled_lists[s])
            if sampled == stratum_size:
                self.probs[s] = 0
                if probs_sum != old_prob_s:
                    new_sum = sum(self.probs)
                    for i in range(0, self.n + 1):
                        self.probs[i] /= new_sum
            else:
                self.probs[s] = self.original_probs[s] - ((self.original_probs[s] * sampled) / stratum_size)
                sum_others = sum(self.probs) - self.probs[s]
                if probs_sum != old_prob_s:
                    for i in range(0, s):
                        self.probs[i] *= (1 - self.probs[s]) / sum_others
                    for i in range(s + 1, self.n - 1):
                        self.probs[i] *= (1 - self.probs[s]) / sum_others
                else:
                    self.probs[s] = 1
        else:
            # adjust sample distribution, all coalitions of size s could have been sampled, set to zero, rescale the others
            if len(self.sampled_lists[s]) == stratum_size:
                # check whether this was the last size, if so then set all probabilities to zero, sampling is finished
                self.probs[s] = 0
                if probs_sum != old_prob_s:
                    new_sum = sum(self.probs)
                    for i in range(0, self.n + 1):
                        self.probs[i] /= new_sum


    def __sample_from_remaining(self, size):
        index = random.randint(0, len(self.left_to_sample_lists[size]) - 1)
        return self.left_to_sample_lists[size].pop(index)


    def __sample_naively(self, size):
        while True:
            coalition = np.random.choice(self.get_all_players(), size, replace=False)
            if not self.__alreadySampled(coalition):
                return coalition

    def get_name(self) -> str:
        name = 'StratSVARMnoReplacement_' + self.dist_type

        if self.dynamic:
            name += '_dyn'

        if self.normalize:
            name += '_nor'

        return name
