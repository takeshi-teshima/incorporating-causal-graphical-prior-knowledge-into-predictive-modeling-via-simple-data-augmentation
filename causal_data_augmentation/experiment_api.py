import numpy as np
from ananke.graphs import ADMG
from .causal_data_augmentation.api import EagerCausalDataAugmentation
from .causal_data_augmentation.api import AugmenterConfig, FullAugmentKind

from .api_support.experiments.logging.run_wrapper import RunWrapper
from .api_support.pandas import df_difference

# Type hinting
from typing import List, Tuple, Iterable, Optional, Union
import pandas as pd
from .api_support.eager_augmentation_evaluators import Evaluator

# Type hinting
from causal_data_augmentation.api_support.typing import GraphType


class CausalDataAugmentationEagerTrainingExperimentAPI:
    """The interface class where the method's codes are integrated for the experiment."""
    def __init__(self,
                 augmenter_config: AugmenterConfig,
                 fit_to_aug_only: bool,
                 aug_coeff: Optional[Union[float, Iterable[float]]] = None,
                 debug: bool = False):
        self.augmenter_config = augmenter_config
        self.fit_to_aug_only = fit_to_aug_only
        self.aug_coeff = 1. if aug_coeff is None else aug_coeff
        self.debug = debug

    def apply_augmenter(self, augmenter_config, method, data, admg):
        if isinstance(augmenter_config, FullAugmentKind):
            augmented_data, aug_weights = method.augment(data, admg)
            aug_weights = aug_weights.flatten()
        else:
            raise NotImplementedError()

        return augmented_data, aug_weights

    def _augment(self, data, graph, augmenter_config):
        vertices, di_edges, bi_edges = graph
        admg = ADMG(vertices, di_edges=di_edges, bi_edges=bi_edges)
        method = EagerCausalDataAugmentation(augmenter_config)

        # Augment
        augmented_data, aug_weights = self.apply_augmenter(
            augmenter_config, method, data, admg)
        return augmented_data, aug_weights

    def _multi_coeff_eval(self, augmented_data, aug_weights, aug_coeff, data,
                          graph, augmenter_config, predicted_var_name,
                          predictor_model, evaluators, run_logger):
        run_logger.set_tags({'aug_coeff': aug_coeff})
        if self.fit_to_aug_only:
            augmented_data = None
            orig_weights = np.zeros(len(data))
        else:
            X = np.array(data.drop(predicted_var_name, axis=1))
            Y = np.array(data[[predicted_var_name]])
            aug_X = np.array(augmented_data.drop(predicted_var_name, axis=1))
            aug_Y = np.array(augmented_data[[predicted_var_name]])
            orig_weights = np.ones(len(data)) / len(data)
            if aug_weights.size > 0:
                orig_weights *= 1 - aug_coeff
                aug_weights *= aug_coeff

        orig_weights *= len(data)
        aug_weights *= len(data)

        predictor_model.fit(data, augmented_data, orig_weights, aug_weights)

        for evaluator in evaluators:
            evaluator(predictor_model)

    def _measure_augmentation(self, augmented_data, aug_weights, data):
        props = {}
        _left, _right, _both = df_difference(data, augmented_data)
        props.update({
            'n_orig_only': len(_left),
            'n_aug_only': len(_right),
            'n_both': len(_both)
        })
        props.update({'augmented_size': len(augmented_data)})
        return props

    def _run(self, data, graph, augmenter_config, predicted_var_name,
             predictor_model, evaluators, run_logger):
        augmented_data, aug_weights = self._augment(data, graph,
                                                    augmenter_config)
        run_logger.set_tags(
            self._measure_augmentation(augmented_data, aug_weights, data))

        # Perform training
        run_logger.set_tags({
            'fit_to_aug_only': self.fit_to_aug_only,
            'aug_coeff': self.aug_coeff
        })

        if self.fit_to_aug_only:
            X = np.array(augmented_data.drop(predicted_var_name, axis=1))
            Y = np.array(augmented_data[[predicted_var_name]])
            aug_X, aug_Y = None, None
            weights = aug_weights
        else:
            X = np.array(data.drop(predicted_var_name, axis=1))
            Y = np.array(data[[predicted_var_name]])
            aug_X = np.array(augmented_data.drop(predicted_var_name, axis=1))
            aug_Y = np.array(augmented_data[[predicted_var_name]])
            orig_weights = np.ones(len(data)) / len(data)
            orig_weights = orig_weights * (1 - self.aug_coeff)
            aug_weights = self.aug_coeff * aug_weights
            weights = np.hstack((orig_weights, aug_weights))

        weights = weights * len(data)
        # predictor_model.fit(X, Y, aug_X, aug_Y, weights)
        predictor_model.fit(data, augmented_data, weights)

        for evaluator in evaluators:
            evaluator(predictor_model)

    def run_method_and_eval(self, data: pd.DataFrame, graph: GraphType,
                            predicted_var_name: str, predictor_model,
                            evaluators: Iterable[Evaluator], run_logger):
        """Run the method and record the results.

        Params:
            data: The source domain data to be used for fitting the novelty detector.
            graph: The ADMG object used for performing the augmentation.
            predicted_var_name: The name of the predicted variable.
            predictor_model: Trainable predictor model to be trained on the augmented data.
                             Should implement ``fit()`` and ``predict()``.
            evaluators: a series of evaluators applied to the trained predictor.
            run_logger: The logger to record the experiment.
        """

        try:
            if isinstance(self.aug_coeff, float):
                args = data, graph, self.augmenter_config, predicted_var_name, predictor_model, evaluators, run_logger
                single_run_wrapper = RunWrapper(self._run, args, dict())
                run_logger.perform_run(lambda idx, _: single_run_wrapper(
                    idx, _, run_logger=run_logger))
            else:
                augmented_data, aug_weights = self._augment(
                    data, graph, self.augmenter_config)
                run_logger.set_tags_exp_wide(
                    self._measure_augmentation(augmented_data, aug_weights,
                                               data))
                run_logger.set_tags_exp_wide(
                    {'fit_to_aug_only': self.fit_to_aug_only})

                from copy import deepcopy
                predictor = deepcopy(predictor_model)
                for aug_coeff in self.aug_coeff:
                    args = augmented_data, aug_weights, aug_coeff, data, graph, self.augmenter_config, predicted_var_name, predictor, evaluators, run_logger
                    single_run_multi_coeff_wrapper = RunWrapper(
                        self._multi_coeff_eval, args, dict())
                    # Perform training
                    run_logger.perform_run(
                        lambda idx, _: single_run_multi_coeff_wrapper(
                            idx, _, run_logger=run_logger))
        except Exception as err:
            if self.debug:
                raise
            else:
                print(err)
