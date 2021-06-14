"""
Scikit-learn's cross validation does not take into account the sample weights for computing the validation scores.

References:
    https://stackoverflow.com/questions/49581104/sklearn-gridsearchcv-not-using-sample-weight-in-score-function
    https://github.com/scikit-learn/scikit-learn/issues/4632
    https://github.com/scikit-learn/scikit-learn/issues/2879#issuecomment-487789482
"""
import numpy as np
import sklearn.base
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Type hinting
import pandas as pd
# Type hinting
from typing import Iterable, Dict, Optional, Union, Any, List, Callable, Tuple


class InstanceWeightedGridSearchCVBase:
    """Base class for the instance weighted grid-search cross-validation."""
    SAMPLE_WEIGHT_COLUMN = '__sample_weight'
    PREDICTOR_PIPELINE_NAME = 'predictor'

    def __init__(self,
                 estimator,
                 param_grid,
                 *args,
                 cv=3,
                 greater_is_better=False,
                 metrics=[mean_squared_error],
                 **kwargs):
        """Constructor.

        Parameters
            estimator : Trainable predictor compatible with Sklearn's ``BaseEstimator``.
            param_grid : The parameter grid to select the hyper-parameter from.
            cv : The number of splits to use for the cross validation.
            *args : The other unnamed arguments.
            greater_is_better : Whether to the greater metric is better.
            metrics : The metric to use for evaluating the performance.
            kwargs : The other keyword arguments.
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.greater_is_better = greater_is_better
        self.metrics = metrics
        self.args = args
        self.kwargs = kwargs

    def _get_param_estimator(self, estimator, params: dict) -> Any:
        """Create a trainable predictor configured to the specific set of parameters.

        Parameters:
            estimator : Trainable predictor compatible with Sklearn's ``BaseEstimator``.
            params : Dictionary of parameters to configure the trainable predictor.

        Returns:
            The configured trainable predictor.
        """
        cloned_params = {
            k: sklearn.base.clone(v, safe=False)
            for k, v in params.items()
        }
        estimator = sklearn.base.clone(estimator).set_params(**cloned_params)
        return estimator

    def _select_best_params(self, _res: Iterable[float]) -> Tuple[dict, float]:
        """Find the best-performing parameters and its score.

        Parameters:
            _res : Scores obtained from the loop.

        Returns:
            ``best_params`` : The best-performing parameters.
            ``best_score`` : The score corresponding to the best-performing parameters.
        """
        best_params = None
        if self.greater_is_better:
            best_score = -np.inf
        else:
            best_score = np.inf

        for params, score in zip(ParameterGrid(self.param_grid), _res):
            if self.greater_is_better:
                if score > best_score:
                    best_params = params
                    best_score = score
            else:
                if best_score > score:
                    best_params = params
                    best_score = score
        return best_params, best_score

    def fit(self, X: Any, Y: Any, sample_weight: np.ndarray, *args, **kwargs) -> None:
        """Perform grid-search based on the cross-validation scores.

        Parameters:
            X : Training input data.
            Y : Training target data.
            sample_weight : Array of instance weights.
            *args : Placeholder.
            **kwargs : Placeholder.
        """
        # Accumulate CV scores
        _res = []
        for params in tqdm(list(ParameterGrid(self.param_grid))):
            estimator = self._get_param_estimator(self.estimator, params)
            _scores = self._cross_val_scores_weighted(estimator, X, Y,
                                                      sample_weight, self.cv,
                                                      self.metrics)

            _res.append(np.array(_scores).mean(axis=1)[0])

        # Select best params
        self.best_params_, self.best_score_ = self._select_best_params(_res)

        # Refit best param predictor
        _best_estimator = self._get_param_estimator(self.estimator,
                                                    self.best_params_)
        _best_estimator.fit(X, Y, sample_weight)
        self.best_estimator_ = _best_estimator

    def _cross_val_scores_weighted(self, model, X: Any, y: Any, weights: np.ndarray, cv: int, metrics: Iterable[Callable[[np.ndarray, np.ndarray], float]]) -> List[List[float]]:
        """Compute and accumulate the cross-validation scores for the given model configured to the specific candidate parameter set.

        Parameters:
            model : Trainable predictor configured to the specific parameter candidate set.
            X : Training input data.
            y : Training target data.
            weights : Instance weights.
            cv : Number of cross-validation splits.
            metrics : The metrics to be used for computing the scores.

        Returns:
            Nested list of cross-validation scores.
        """
        kf = KFold(n_splits=cv)
        kf.get_n_splits(X)
        scores = [[] for metric in metrics]
        for train_index, test_index in kf.split(X):
            model_clone = sklearn.base.clone(model)
            X_train = self._get_sliced_data(X, train_index)
            X_test = self._get_sliced_data(X, test_index)
            y_train = self._get_sliced_data(y, train_index)
            y_test = self._get_sliced_data(y, test_index)
            weights_train, weights_test = weights[train_index], weights[
                test_index]
            model_clone.fit(X_train, y_train, sample_weight=weights_train)
            y_pred = model_clone.predict(X_test)
            for i, metric in enumerate(metrics):
                score = metric(y_test, y_pred, sample_weight=weights_test)
                scores[i].append(score)
        return scores

    def get_best_estimator(self) -> Any:
        """Get the best-performing estimator.

        Returns:
            Best performing estimator.
        """
        return self.best_estimator_

    def _get_sliced_data(self, data: Any, index) -> Any:
        """
        Parameters:
            data : The training data to be split.
            index : The index to be selected.

        Returns:
            The sliced data.
        """
        raise NotImplementedError()


class PandasInstanceWeightedGridSearchCV(InstanceWeightedGridSearchCVBase):
    """Perform cross-validation given the data in the ``pd.DataFrame`` format."""
    def _get_sliced_data(self, data: pd.DataFrame, index) -> pd.DataFrame:
        """Get the sliced data for the index.

        Parameters:
            data : The DataFrame representing the data set.
            index : The index to be selected.

        Returns:
            The sliced DataFrame.
        """
        return data.iloc[index]
