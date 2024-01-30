import numpy as np


def normalize_shapley_value(shapley_value: np.array, value_grand_coalition):
    if value_grand_coalition is None:
        return shapley_value
    if all(shapley_value == np.zeros(shapley_value.shape)):
        return shapley_value
    shapley_value *= (value_grand_coalition / np.sum(shapley_value))
    return shapley_value
