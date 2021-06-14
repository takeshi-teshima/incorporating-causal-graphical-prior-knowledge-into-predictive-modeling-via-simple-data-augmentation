import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import euclidean_distances
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from .util import Timer
from .util.grid_search_cv import PandasInstanceWeightedGridSearchCV

# Type hinting
from typing import Optional, Iterable, Union, Dict, List


class AugXGBRegressor:
    """Boosted-tree regressor."""
    def __init__(self,
                 predicted_var_name:str,
                 grid_params:Optional[Dict[str, List]]=None,
                 val_data:Optional[np.ndarray]=None):
        """Constructor.
        Parameters:
            grid_params : Hyper-parameters to cross-validate over.
        """
        self.predicted_var_name = predicted_var_name
        self.grid_params = grid_params
        if val_data is not None:
            X, y = val_data
            self.val_data = [(X, y)]
        else:
            self.val_data = None

    def fit(self,
            train_data: pd.DataFrame,
            aux_data: Optional[pd.DataFrame] = None,
            sample_weight: Optional[np.ndarray] = None,
            aux_sample_weight: Optional[np.ndarray] = None):
        """Fit the predictor.

        Parameters:
            train_data : the original data.
            aux_data : the augmented data.
            sample_weight: sample weight array.

        Notes:
            API reference: https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
        """
        if aux_data is not None:
            train_data = train_data.append(aux_data)
        if (sample_weight is not None) and (aux_sample_weight is not None):
            sample_weight = np.hstack((sample_weight, aux_sample_weight))

        if sample_weight is not None:
            train_data = train_data[sample_weight > 0]
            sample_weight = sample_weight[sample_weight > 0]
        try:
            X = train_data.drop(self.predicted_var_name, axis=1)
            Y = train_data[[self.predicted_var_name]]
            if self.val_data is None:
                if self.grid_params is not None:
                    gs = PandasInstanceWeightedGridSearchCV(XGBRegressor(),
                                                            self.grid_params,
                                                            cv=3,
                                                            n_jobs=1,
                                                            verbose=2)
                    gs.fit(X, Y, sample_weight=sample_weight, verbose=True)
                    self.model = gs.get_best_estimator()
                else:
                    self.model = XGBRegressor()
                    self.model.fit(X,
                                   Y,
                                   sample_weight=sample_weight,
                                   verbose=True)
            else:
                raise NotImplementedError(
                    'Implement the code to use self.val_data to select the parameters.'
                )
                self.model = XGBRegressor()
        except Exception as e:
            raise
            # print(e)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make a prediction.

        Parameters:
            X: input data (shape ``(n_sample, n_dim)``).

        Returns:
            The output of the model.
        """
        return self.model.predict(X)
