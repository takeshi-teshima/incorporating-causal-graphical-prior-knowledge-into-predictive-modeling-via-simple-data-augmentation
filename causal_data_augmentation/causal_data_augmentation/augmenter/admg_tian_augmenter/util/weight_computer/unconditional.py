import numpy as np
from .base import WeightComputer


class UnconditionalWeightComputer(WeightComputer):
    """Mockup class for returning the uniform weight."""
    def __init__(self, n_candidates: int):
        """Constructor."""
        self.n_candidates = n_candidates

    def __call__(self, _=None) -> np.ndarray:
        """Return the uniform weight.

        Returns:
            The uniform weight array (each being equal to 1/n) for the number of the candidates.
        """
        return np.ones((1, self.n_candidates)) / self.n_candidates
