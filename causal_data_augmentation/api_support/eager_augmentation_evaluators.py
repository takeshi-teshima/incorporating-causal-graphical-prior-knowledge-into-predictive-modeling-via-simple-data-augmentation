from abc import abstractmethod

# Type hinting
from typing import Optional, Dict, Any


class Evaluator:
    def __init__(self, run_logger, name: str):
        """Constructor."""
        self.run_logger = run_logger
        self.name = name

    @abstractmethod
    def evaluate(self, predictor_model):
        raise NotImplementedError()

    def save_result(self, value: Any):
        """Save the results using ``self.run_logger``.

        Parameters:
            value: the result to be stored.
        """
        self.run_logger.set_tags({self.name: value})

    def __call__(self, predictor_model):
        """Evaluate the score and record it.

        Params:
            predictor_model : should implement ``predict()``.
        """
        self.save_result(self.evaluate(predictor_model))
