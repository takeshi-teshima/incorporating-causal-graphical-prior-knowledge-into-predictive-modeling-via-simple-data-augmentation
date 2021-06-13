import numpy as np
from causal_data_augmentation.api_support.experiments.logging.run_wrapper import RunWrapper

# Type hinting
from typing import List, Tuple, Iterable
import pandas as pd
from causal_data_augmentation.api_support.eager_augmentation_evaluators import Evaluator


class XGBExperimentAPI:
    """XGBoost method without any device."""
    def __init__(self, debug: bool = False):
        self.debug = debug

    def run_method_and_eval(self, data: pd.DataFrame, predicted_var_name: str,
                            predictor_model, evaluators: Iterable[Evaluator],
                            run_logger):
        """
        Params:
            data: The source domain data to be used for fitting the novelty detector.
            predicted_var_name: The name of the predicted variable.
            predictor_model: Trainable predictor model to be trained on the augmented data.
                             Should implement ``fit()`` and ``predict()``.
            evaluators: a series of evaluators applied to the trained predictor.
            run_logger: The logger to record the experiment.
        """
        args = data, predicted_var_name, predictor_model, evaluators

        def _run(data, predicted_var_name, predictor_model, evaluators):
            # Perform training
            sample_weight = np.ones((len(data)))

            predictor_model.fit(data, None, sample_weight=sample_weight)

            for evaluator in evaluators:
                evaluator(predictor_model)

        single_run_wrapper = RunWrapper(_run, args, dict())
        try:
            run_logger.perform_run(lambda idx, _: single_run_wrapper(
                idx, _, run_logger=run_logger))
        except Exception as err:
            if self.debug:
                raise
            else:
                print(err)
