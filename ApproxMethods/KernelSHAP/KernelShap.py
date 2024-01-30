import copy
import itertools
import random
import typing

import numpy as np
from scipy.special import binom

from ApproxMethods.AbstractApproxMethod import BaseApproxMethod
from utils.experiment_util import normalize_shapley_value

# KernelSHAP (Lundberg and Lee, 2017)
class KernelShap(BaseApproxMethod):
    def __init__(self, normalize, pairing=False):
        super().__init__(normalize)
        self.pairing = pairing
        self.big_M = 10_000_000_000

    def approximate_shapley_values(self) -> dict:
        weights = self.__get_weights()
        sampling_weight = (np.asarray([0] + [*weights] + [0])) / sum(weights)

        regression_weights = (np.asarray([self.big_M] + [*weights] + [self.big_M]))
        game_function = self.game.get_value

        sampling_budget = self.budget - 2  # because of full and emtpy set (always need this)
        S_list, game_values, kernel_weights = self.__get_S_and_game_values(
            budget=sampling_budget,
            num_players=self.n,
            weight_vector=sampling_weight,
            N=self.get_all_players(),
            pairing=self.pairing,
            game_fun=game_function
        )

        # get empty and full model evaluations
        empty_value = game_function({})
        full_value = game_function(self.get_all_players())
        S_list.append(set())
        S_list.append(self.get_all_players())
        game_values.append(empty_value)
        game_values.append(full_value)
        kernel_weights[()] = self.big_M
        kernel_weights[tuple(self.get_all_players())] = self.big_M

        # transform s and v into np.ndarrays
        self.all_S = np.zeros(shape=(len(S_list), self.n), dtype=bool)
        for i, subset in enumerate(S_list):
            if len(subset) == 0:
                continue
            subset = np.asarray(list(subset))
            self.all_S[i, subset] = 1
        self.game_values = np.asarray(game_values) - empty_value

        # calculate weight
        self.W = np.zeros(shape=np.array(game_values).shape, dtype=float)
        for i, S in enumerate(self.all_S):
            # subset_size = sum(S)
            weight = kernel_weights[tuple(sorted(np.where(S)[0]))]
            # weight = min(self.big_M, weight)
            self.W[i] = weight
            # self.W[i] = regression_weights[sum(S)]

        self.experiment_storage.add_shapley_values(self.get_estimates())
        return self.experiment_storage.to_json()

    def __get_weights(self):
        """Get Shapley Weights."""
        weights = np.arange(1, self.n)
        weights = (self.n - 1) / (weights * (self.n - weights)) # correct is (n - 1) / (weights * (self.n - weights)) but this is not necessary with the constant term
        weights = weights / np.sum(weights)
        return weights

    def __get_S_and_game_values(self,
                                budget: int,
                                num_players: int,
                                weight_vector: np.ndarray,
                                N: set,
                                pairing: bool,
                                game_fun: typing.Callable
                                ):
        """Run the Sampling process and get game values and Subset"""
        complete_subsets, incomplete_subsets, budget = self.__determine_complete_subsets(
            budget=budget, n=num_players, s=1, q=weight_vector)

        all_subsets_to_sample = []
        kernel_weights = {}

        for complete_subset in complete_subsets:
            combinations = itertools.combinations(N, complete_subset)
            for subset in combinations:
                subset = set(subset)
                all_subsets_to_sample.append(subset)
                kernel_weights[tuple(sorted(subset))] = weight_vector[len(subset)] / binom(num_players, len(subset))

        remaining_weight = weight_vector[incomplete_subsets] / sum(
            weight_vector[incomplete_subsets])
        kernel_weights_sampling = {}

        if len(incomplete_subsets) > 0:
            sampled_subsets = set()
            n_sampled_subsets = 0
            while len(sampled_subsets) < budget:
                subset_size = random.choices(incomplete_subsets, remaining_weight, k=1)
                ids = np.random.choice(num_players, size=subset_size, replace=False)
                sampled_subset = tuple(sorted(ids))
                if sampled_subset not in sampled_subsets:
                    sampled_subsets.add(sampled_subset)
                    kernel_weights_sampling[sampled_subset] = 1.
                else:
                    kernel_weights_sampling[sampled_subset] += 1.
                n_sampled_subsets += 1
                if pairing:
                    if len(sampled_subsets) < budget:
                        sampled_subset_paired = tuple(sorted(set(N) - set(ids)))
                        if sampled_subset_paired not in sampled_subsets:
                            sampled_subsets.add(sampled_subset_paired)
                            kernel_weights_sampling[sampled_subset_paired] = 1.
                        else:
                            kernel_weights_sampling[sampled_subset_paired] += 1.
                        n_sampled_subsets += 1
            for subset in sampled_subsets:
                all_subsets_to_sample.append(set(subset))

            # re-normalize kernel weights
            weight_left = np.sum(weight_vector[incomplete_subsets])
            kernel_weights_sampling = {subset: weight * (weight_left / n_sampled_subsets) for
                                       subset, weight in kernel_weights_sampling.items()}
            kernel_weights.update(kernel_weights_sampling)

        game_values = [game_fun(subset) for subset in all_subsets_to_sample]
        return all_subsets_to_sample, game_values, kernel_weights

    def __determine_complete_subsets(self, s: int, n: int, budget: int, q: np.ndarray):
        """Get subsets that should be computable with given budget."""
        complete_subsets = []
        paired_subsets, unpaired_subset = self.__get_paired_subsets(s, n)

        incomplete_subsets = list(range(s, n - s + 1))
        weight_vector = copy.copy(q)
        sum_weight_vector = np.sum(weight_vector)
        weight_vector = np.divide(weight_vector, sum_weight_vector,
                                  out=weight_vector, where=sum_weight_vector != 0)
        allowed_budget = weight_vector * budget
        for subset_size_1, subset_size_2 in paired_subsets:
            subset_budget = int(binom(n, subset_size_1))
            if allowed_budget[subset_size_1] >= subset_budget and allowed_budget[subset_size_1] > 0:
                complete_subsets.extend((subset_size_1, subset_size_2))
                incomplete_subsets.remove(subset_size_1)
                incomplete_subsets.remove(subset_size_2)
                weight_vector[subset_size_1] = 0
                weight_vector[subset_size_2] = 0
                if not np.sum(weight_vector) == 0:
                    weight_vector /= np.sum(weight_vector)
                budget -= subset_budget * 2
            else:
                return complete_subsets, incomplete_subsets, budget
            allowed_budget = weight_vector * budget
        if unpaired_subset is not None:
            subset_budget = int(binom(n, unpaired_subset))
            if budget - subset_budget >= 0:
                complete_subsets.append(unpaired_subset)
                incomplete_subsets.remove(unpaired_subset)
                budget -= subset_budget
        return complete_subsets, incomplete_subsets, budget

    def __get_paired_subsets(self, s: int, n: int):
        """Get pairs of subsets. (for paired sampling)"""
        subset_sizes = list(range(s, n - s + 1))
        n_paired_subsets = int(len(subset_sizes) / 2)
        paired_subsets = [(subset_sizes[subset_size - 1], subset_sizes[-subset_size])
                          for subset_size in range(1, n_paired_subsets + 1)]
        unpaired_subset = None
        if n_paired_subsets < len(subset_sizes) / 2:
            unpaired_subset = int(np.median(subset_sizes))
        return paired_subsets, unpaired_subset

    def get_estimates(self):
        if self.W is None:
            return 0
        # do the regression
        A = self.all_S
        B = self.game_values
        W = np.sqrt(np.diag(self.W))
        Aw = np.dot(W, A)
        Bw = np.dot(B, W)
        phi, residuals, rank, singular_values = np.linalg.lstsq(Aw, Bw, rcond=None)
        if self.normalize:
            return normalize_shapley_value(phi, self.grand_co_value)
        return phi

    def get_name(self) -> str:
        if self.normalize:
            return 'kernelSHAP_nor'
        return 'kernelSHAP'
