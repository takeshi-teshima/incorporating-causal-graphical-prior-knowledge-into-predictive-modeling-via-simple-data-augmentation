"""Augmenter for ADMG. Only uses the Tian factorization."""
import numpy as np
import pandas as pd
from .base import ADMGTianAugmenterBase
from .util.pandas import product_df, summarize_duplicates_df

# Type hinting
from typing import Tuple, Iterable, Optional, List
from ananke.graphs import ADMG
from pandas import DataFrame as DF
from causal_data_augmentation.contrib.aug_predictors.util import Timer
from .util.augmenter_kernel import AugmenterKernel


def summarize_log_weight(df, current_logweight):
    # Summarize log weights
    LOG_WEIGHT_TEMPORARY_KEY = '___log_weight'
    df[LOG_WEIGHT_TEMPORARY_KEY] = current_logweight
    df = summarize_duplicates_df(df, LOG_WEIGHT_TEMPORARY_KEY, 'sum')
    current_weight = df[LOG_WEIGHT_TEMPORARY_KEY].to_numpy()
    df = df.drop(LOG_WEIGHT_TEMPORARY_KEY, axis=1)
    return df, current_weight


@Timer.set(lambda t: print('[Timer] full_augmentation finished: ', t.time))
def _full_augmented_data(data: pd.DataFrame,
                         kernels: Iterable[AugmenterKernel],
                         log_weight_threshold: np.float64,
                         weight_threshold_type: str
                         ) -> Tuple[pd.DataFrame, np.ndarray]:
    """Compute the full augmentation.

    Params:
      data : the base data to use for the augmentation (in the current implementation, this has to be the same data set to which the kernels have been fit).
      kernels : vector-valued functions $f(c)$ that returns $(\tilde{p}(x_i | c))_i$ where $\tilde{p}(x_i | c)$ is the weighting of selecting $x_i$ (column $x$ of the $i$-th row of ``data``).
      log_weight_threshold : the lower log-weight threshold for performing the augmentation.

    Returns:
      df : the data frame containing the augmented data. This can have duplicate values with the original data.
           The augmentation candidates whose weight is lower than the threshold are pruned.
      weight : the relative weight of each augmented data computed from the kernels.
               The values may not sum to one because of the pruning (the values are not normalized after the pruning).
    """
    @Timer.set(lambda t: print('[Timer] _one_step took ', t.time))
    def _one_step(current_df, current_logweight, data_ref, kernel):
        c = current_df[kernel.c_names]
        w = kernel.c_weighter(np.array(c))  # (len_current_df, n)

        # Allow log(0) to be -inf.
        with np.errstate(divide='ignore'):
            logweights = np.log(w)

        # Augment the DataFrame
        assert len(data_ref[kernel.v_names]) == len(data_ref)
        _current_df = product_df(current_df, data_ref[kernel.v_names])

        # Update weights
        _current_logweight_base = np.repeat(current_logweight,
                                            len(data_ref),
                                            axis=1)
        assert np.sum(np.isnan(logweights)) == 0
        assert np.sum(logweights > 0) == 0
        _current_logweight_base += logweights
        _current_logweight = np.ravel(_current_logweight_base, order='C')

        if weight_threshold_type == 'total':
            not_pruned = _current_logweight >= log_weight_threshold
        elif weight_threshold_type == 'factor_wise':
            not_pruned = logweights.flatten() >= log_weight_threshold
        else:
            raise NotImplementedError()
        _current_df = _current_df[not_pruned]
        _current_logweight = _current_logweight[not_pruned]

        _current_df = _current_df.reset_index(drop=True)
        return _current_df, _current_logweight[:, None]

    # Initialize buffers.
    df = pd.DataFrame(index=pd.RangeIndex(1))  # Empty data frame
    current_logweight = np.log(np.ones((1, 1)))
    for kernel in kernels:
        df, current_logweight = _one_step(df, current_logweight, data, kernel)
        print('current_size', len(df))
        if len(df) == 0:
            return pd.DataFrame(columns=data.columns), np.empty(0)
    df, current_logweight = summarize_log_weight(df, current_logweight)

    current_weight = np.exp(current_logweight)
    if np.sum(np.isnan(current_weight)) != 0:
        current_weight = np.nan_to_num(current_weight)
    return df, current_weight


def _log_threshold(weight_threshold: Optional[float]) -> np.float64:
    if (weight_threshold is None) or (weight_threshold == 0):
        log_threshold = -np.inf
    else:
        log_threshold = np.log(weight_threshold)
    return log_threshold


class ADMGTianFullAugmenter(ADMGTianAugmenterBase):
    def augment(self, weight_threshold: Optional[float],
                weight_threshold_type: str,
                normalize_threshold_by_data_size: bool
                ) -> Tuple[DF, np.ndarray]:
        """Perform Tian factorization and augment the data accordingly.

        Params:
            weight_threshold : the lower weight threshold for performing the augmentation.
            normalize_threshold_by_data_size : whether to normalize the threshold by the data size.
                                               (``True``: divide the threshold by the data size)

        Returns:
            augmented_data : the fully augmented data.
            weights : the instance weights based on the kernel values.
        """
        log_weight_threshold = _log_threshold(weight_threshold)
        if weight_threshold_type == 'total':
            if normalize_threshold_by_data_size:
                log_weight_threshold -= np.log(len(self.data))
        elif weight_threshold_type == 'factor_wise':
            pass
        elif weight_threshold_type == 'factor_wise_sample_normalized':
            log_weight_threshold -= np.log(len(self.data))

        augmented_data, weights = _full_augmented_data(
            self.data, self.factorization_kernels, log_weight_threshold,
            weight_threshold_type)
        # Reorder DataFrame columns
        augmented_data = augmented_data[self.data.columns]
        augmented_data = augmented_data.astype(self.data.dtypes)
        return augmented_data, weights
