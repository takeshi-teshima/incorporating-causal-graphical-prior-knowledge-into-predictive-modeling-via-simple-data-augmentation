from causal_data_augmentation.api_support.eager_augmentation_evaluators import Evaluator

# Type hinting
from typing import Callable, Any, Tuple  # , Iterable, Dict, Optional, Union
import numpy as np


class PredictionEvaluator(Evaluator):
    """Evaluator wrapper class to evaluate the prediction."""
    def __init__(self, scorer: Callable[[np.ndarray, np.ndarray], Any],
                 test_data: Tuple[np.ndarray, np.ndarray], run_logger,
                 name: str):
        """Constructor.

        Parameters:
            scorer : The scorer that returns the score of the evaluation.
            test_data: Tuple of ``(X, Y)`` of the test data.
            run_logger : The experiment logger.
            name : Name of the evaluator.
        """
        self.scorer = scorer
        self.test_data = test_data
        super().__init__(run_logger, name)

    def evaluate(self, predictor_model):
        """Return an evaluated score.

        Parameters:
            predictor_model : should implement ``predict()``.
        """
        X, Y = self.test_data
        return self.scorer(Y, predictor_model.predict(X))


class PropertyEvaluator(Evaluator):
    """Evaluator wrapper class to evaluate the property of the trained predictor model."""
    def __init__(self, prop_name: str, run_logger, name: str):
        """Constructor.

        Parameters:
            prop_name
            run_logger
            name
        """
        self.prop_name = prop_name
        super().__init__(run_logger, name)

    def evaluate(self, predictor_model):
        """Return an evaluated score.

        Parameters:
            predictor_model.
        """
        try:
            val = getattr(predictor_model, self.prop_name)
        except AttributeError:
            print(f"predictor_model.{self.prop_name} not found")
            val = None
        return val
