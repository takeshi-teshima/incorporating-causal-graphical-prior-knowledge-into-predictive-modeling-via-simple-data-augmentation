import numpy as np
from .base import WeightComputer


class UnconditionalWeightComputer(WeightComputer):
    def __init__(self, n_candidates: int):
        self.n_candidates = n_candidates

    def __call__(self, _=None):
        return np.ones((1, self.n_candidates)) / self.n_candidates
