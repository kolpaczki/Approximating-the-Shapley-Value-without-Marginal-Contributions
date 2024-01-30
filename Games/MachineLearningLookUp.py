import os
import random

import numpy as np
import pandas as pd
from scipy.special import binom

from Games.AbstractGame import BaseGame
from Games.SparseLinearModels import ShapleyInteractionsEstimator


class TabularLookUpGame(BaseGame):

    def __init__(self, n: int, data_folder: str = "adult_42", data_id: int = None, used_ids: set = None, set_zero: bool = False):
        if used_ids is None:
            used_ids = set()
        self.used_ids = used_ids
        if data_id is None:
            files = os.listdir(os.path.join("data_ml", data_folder, str(n)))
            files = list(set(files) - used_ids)
            if len(files) == 0:
                files = os.listdir(os.path.join("data_ml", data_folder, str(n)))
                self.used_ids = set()
            data_id = random.choice(files)
            data_id = int(data_id.split(".")[0])
        self.used_ids.add(str(data_id) + ".csv")
        data_path = os.path.join("data_ml", data_folder, str(n), str(data_id) + ".csv")
        self.df = pd.read_csv(data_path)
        self.game_name = "tabular_game"
        self.n = n

        self.storage = {}
        for _, sample in self.df.iterrows():
            S_id = sample["set"]
            value = float(sample["value"])
            self.storage[S_id] = value

        self.empty_value = 0
        if set_zero:
            self.empty_value = self.set_call(set())

    def set_call(self, S):
        S_id = 's'
        for player in sorted(S):
            S_id += str(player)
        return self.storage[S_id] - self.empty_value

    def get_game_information(self):
        information = dict()
        information['name'] = self.game_name
        information['number_of_players'] = self.n
        information['ground_truth_shapley_value'] = self.get_shapley_values()
        return information

    def get_value(self, S):
        return self.set_call(S)

    def get_player_number(self):
        return self.n

    def get_name(self):
        return 'MachineLearningGame'

    def get_shapley_values(self):
        N = set(range(self.n))
        estimator = ShapleyInteractionsEstimator(N, 1, 1, interaction_type="SII")
        phi = estimator.compute_interactions_complete(game=self.set_call)[1]
        return phi


if __name__ == "__main__":
    pass