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


class InstanceWeightedGridSearchCVBase:
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
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.greater_is_better = greater_is_better
        self.metrics = metrics
        self.args = args
        self.kwargs = kwargs

    def _get_param_estimator(self, estimator, params):
        cloned_params = {
            k: sklearn.base.clone(v, safe=False)
            for k, v in params.items()
        }
        estimator = sklearn.base.clone(estimator).set_params(**cloned_params)
        return estimator

    def _select_best_params(self, _res):
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

    def fit(self, X, Y, sample_weight, *args, **kwargs):
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

    def _cross_val_scores_weighted(self, model, X, y, weights, cv, metrics):
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

    def get_best_estimator(self):
        return self.best_estimator_

    def _get_sliced_data(self, data, index):
        raise NotImplementedError()


class PandasInstanceWeightedGridSearchCV(InstanceWeightedGridSearchCVBase):
    def _get_sliced_data(self, data: pd.DataFrame, index) -> pd.DataFrame:
        return data.iloc[index]


class NumpyInstanceWeightedGridSearchCV(InstanceWeightedGridSearchCVBase):
    def _get_sliced_data(self, data: np.ndarray, index) -> np.ndarray:
        return data[index]
