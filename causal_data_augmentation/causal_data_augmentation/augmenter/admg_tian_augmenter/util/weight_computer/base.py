import numpy as np


class WeightComputer:
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Compute the weighting matrix.

        Parameters:
            data : ndarray of shape ``(n_data, n_dim)``.

        Returns: ndarray of shape ``(n_data, n_data_ref)``
        """
        raise NotImplementedError()
