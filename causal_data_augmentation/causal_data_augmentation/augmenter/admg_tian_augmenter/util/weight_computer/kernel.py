import numpy as np
from .base import WeightComputer

# Type hinting
from typing import Callable


class KernelWeightComputer(WeightComputer):
    """Kernel-based weight computer class."""

    def __init__(self, kernel: Callable[[np.ndarray], np.ndarray]):
        """
        Parameters:
            kernel : a callable object that returns the kernel matrix given the data.
        """
        self.kernel = kernel

    def __call__(self, data) -> np.ndarray:
        """Compute the weighting matrix.

        Parameters:
            data : ndarray of shape ``(n_data, n_dim)``.

        Returns:
            ndarray of shape ``(n_data, n_data_ref)``.
        """
        _gram_matrix = self.kernel(data)

        divisor = _gram_matrix.sum(axis=1, keepdims=True)
        # Without "out" argument, where the "where" condition is not met, the value is not initialized.
        weights = np.divide(_gram_matrix,
                            divisor,
                            where=divisor != 0,
                            out=np.zeros(_gram_matrix.shape))
        # with np.errstate(divide='ignore'):
        assert np.sum(np.isnan(weights)) == 0

        # All rows should have values that sum to either 1 or 0
        rowsums = weights.sum(axis=1)
        zero_or_one = np.logical_or(np.isclose(rowsums, 1.),
                                    np.isclose(rowsums, 0.))
        assert np.sum(np.logical_not(zero_or_one)) == 0
        return weights
