import itertools
import json
import math
import os
from datetime import datetime
import numpy
import numpy as np


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_experiment_to_file(data):
    timestamp = datetime.now().strftime("%m-%d-%Y_%H%M%S")
    if not os.path.exists('Data'):
        os.mkdir('Data')
    with open(os.path.join('Data', f'evaluation__{timestamp}.json'), mode='w') as file:
        json.dump(data, file, cls=NumpyArrayEncoder, indent=2)


def to_array(x):
    return np.array(x)


def calculate_mse(estimate, actual):
    return np.square(np.subtract(estimate, actual)).mean()


def powerset(iterable, min_size=-1, max_size=None):
    if max_size is None and min_size > -1:
        max_size = min_size
    s = list(iterable)
    if max_size is None:
        max_size = len(s)
    else:
        max_size = min(max_size, len(s))
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(max(min_size, 0), max_size + 1))


# uniform distribution over sizes between 2 and n-2
def generateUniformDistribution(n):
    dist = [0 for i in range(n+1)]
    for s in range(2, n-1):
        dist[s] = 1 / (n-3)
    return dist


# probability distribution over sizes for sampling according to paper
def generatePaperDistribution(n):
    dist = [0 for i in range(n+1)]

    if n % 2 == 0:
        nlogn = n * math.log(n)
        H = sum([1 / s for s in range(1, int(n / 2))])
        nominator = nlogn - 1
        denominator = 2 * nlogn * (H - 1)
        frac = nominator / denominator
        for s in range(2, int(n / 2)):
            dist[s] = frac / s
            dist[n - s] = frac / s
        dist[int(n / 2)] = 1 / nlogn
    else:
        H = sum([1 / s for s in range(1, int((n - 1) / 2 + 1))])
        frac = 1 / (2 * (H - 1))
        for s in range(2, int((n - 1) / 2 + 1)):
            dist[s] = frac / s
            dist[n - s] = frac / s

    return dist


def generateLinearlyDescendingDistribution(n, factor):
    dist = [0 for i in range(n + 1)]
    m = 1.0 / ((n-3) * (factor*(n-2)-2)/(factor-1) - (n-2)*(n-1)/2 + 1)
    b = m * (factor * (n-2) - 2) / (factor-1)
    for s in range(2, n-1):
        dist[s] = b - m * s
    return dist

def generateDistribution(name, n):
    if name == "paper":
        return generatePaperDistribution(n)
    elif name == "uniform":
        return generateUniformDistribution(n)
    elif name == "descending":
        return generateLinearlyDescendingDistribution(n, 3.0)
    else:
        return []