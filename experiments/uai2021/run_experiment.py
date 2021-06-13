import os
from copy import copy
from pathlib import Path
from importlib import import_module
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from causal_data_augmentation.api_support.experiments.logging import MongoAndSacredRunLogger, PandasParamHistoryManager
import causal_data_augmentation.causal_data_augmentation.api_support.method_config as method_config_module
from support.database.records_aggregator import MongoAggregator
from support.database.mongo import get_mongo_observer, get_table
import support.estimate_causal_graph as estimate_causal_graph
from causal_data_augmentation.api_support.experiments.logging.pickler import Pickler

# Importing experiment suite from the parent directory.
import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))
import suite

# Importing this here enables on-the-fly import of the baseline module.
import baseline_methods


def _evaluate_proposed(cfg, dataset, run_logger):
    from causal_data_augmentation.experiment_api import CausalDataAugmentationEagerTrainingExperimentAPI

    ###################
    ## Get dataset
    ###################
    train_data, test_data, graph, predicted_var_name = dataset
    test_X = test_data.drop(predicted_var_name, axis=1)
    test_Y = test_data[predicted_var_name]
    test_data = (test_X, test_Y)

    ###################
    ## Prepare evaluation
    ###################
    from causal_data_augmentation.contrib.aug_predictors.xgb import AugXGBRegressor
    param_grid = OmegaConf.to_container(
        cfg.method.predictor) if cfg.method.predictor else None
    predictor_model = AugXGBRegressor(predicted_var_name, param_grid)
    from sklearn.metrics import mean_squared_error
    from causal_data_augmentation.contrib.eager_augmentation_evaluators import PredictionEvaluator, PropertyEvaluator
    AugmenterConfigClass = getattr(method_config_module,
                                   cfg.method.augmenter_config_name)
    augmenter_config = AugmenterConfigClass(**cfg.method.augmenter_config)
    run_logger.set_tags_exp_wide(
        dict(augmenter_config_weight_threshold=cfg.method.augmenter_config.
             weight_threshold,
             augmenter_conti_kernel=cfg.method.augmenter_config.
             weight_kernel_cfg.conti_kertype))

    ###################
    ## Run
    ###################
    run_logger.set_tags_exp_wide(dict(validation_run=False))
    evaluators = [
        PredictionEvaluator(mean_squared_error, test_data, run_logger,
                            'XGB_MSE'),
    ]
    method_experiment_api = CausalDataAugmentationEagerTrainingExperimentAPI(
        augmenter_config, cfg.method.fit_to_aug_only, cfg.method.aug_coeff,
        cfg.debug)
    method_experiment_api.run_method_and_eval(train_data, graph,
                                              predicted_var_name,
                                              predictor_model, evaluators,
                                              run_logger)


def _evaluate_baseline_xgb(cfg, dataset, run_logger):
    train_data, test_data, graph, predicted_var_name = dataset
    test_X = test_data.drop(predicted_var_name, axis=1)
    test_Y = test_data[predicted_var_name]
    # test_data = (np.array(test_X), np.array(test_Y))
    test_data = (test_X, test_Y)

    # Prepare evaluation
    from causal_data_augmentation.contrib.aug_predictors.xgb import AugXGBRegressor
    param_grid = OmegaConf.to_container(
        cfg.method.predictor) if cfg.method.predictor else None
    predictor_model = AugXGBRegressor(predicted_var_name, param_grid)
    from sklearn.metrics import mean_squared_error
    from causal_data_augmentation.contrib.eager_augmentation_evaluators import PredictionEvaluator, PropertyEvaluator
    evaluators = [
        PredictionEvaluator(mean_squared_error, test_data, run_logger,
                            'XGB_MSE'),
    ]

    from baseline_methods.xgb.api import XGBExperimentAPI
    method_api = XGBExperimentAPI(cfg.debug)
    method_api.run_method_and_eval(train_data, predicted_var_name,
                                   predictor_model, evaluators, run_logger)


def get_run_logger(cfg):
    mongo_params = cfg.database.mongo_host, cfg.database.mongo_port, cfg.database.mongo_user, cfg.database.mongo_pass, cfg.database.mongo_dbname
    # max_threads = cfg.misc.max_threads
    run_logger = MongoAndSacredRunLogger(
        cfg.recording.experiment_name, get_mongo_observer(*mongo_params),
        get_table(cfg.recording.table_name, *mongo_params),
        f'{cfg.recording.sacred_artifact_dir}/{cfg.recording.experiment_name}_{cfg.recording.recording_set}',
        f'{cfg.recording.shared_pickle_dir}')
    run_logger.set_tags_exp_wide(
        dict(
            # max_threads=max_threads,
            recording_set=cfg.recording.recording_set,
            method=cfg.method.name))
    return run_logger


def get_data(cfg, run_logger):
    data_name = cfg.data.name
    data_module = cfg.data.module
    data_path = hydra.utils.to_absolute_path('../suite/' + cfg.data.data_path)
    graph_path = hydra.utils.to_absolute_path(cfg.data.graph.path)
    data_run_id = cfg.parallelization.data_run_id
    data_cache_base_path = hydra.utils.to_absolute_path(
        cfg.recording.data_cache_base_path)

    run_logger.set_tags_exp_wide(
        dict(data=data_name,
             data_module=data_module,
             data_path=data_path,
             data_run_id=data_run_id,
             data_config=OmegaConf.to_container(cfg.data)))
    data_module = import_module(f'suite.data.{data_module}.load_data')

    ################
    ## Prepare data
    ################
    def normalize(df):
        conti_columns = df.select_dtypes(float).columns
        _df = df[conti_columns]
        _df = (_df - _df.mean()) / _df.std(ddof=0)
        df[conti_columns] = _df
        return df

    def log_transform(df, columns):
        df.loc[:, columns] = np.log(df[columns])
        return df

    if ('preprocess' in cfg.data) and (cfg.data.preprocess is not None):
        preprocessing_cfg = cfg.data.preprocess
        run_logger.set_tags_exp_wide(
            {'preprocessing': OmegaConf.to_container(preprocessing_cfg)})
    else:
        preprocessing_cfg = []
    preprocess_suffix = ''
    for preprocess in preprocessing_cfg:
        key, val = list(preprocess.items())[0]
        preprocess_suffix += f'_{key}_{val}'
    fulldata_name = f'fulldata_{data_name}{preprocess_suffix}'
    suffix = copy(preprocess_suffix)
    suffix += f'_{data_run_id}'
    suffix += f'_ts{cfg.data.train_size}'
    if 'validation_size' in cfg.data:
        suffix += f'_vs{cfg.data.validation_size}'
    suffix += f'_run{cfg.recording.recording_set}'
    data_cache_name = f'{data_name}{suffix}'

    def _get_fulldata():
        _data = data_module.load_data(data_path)

        for preprocess in preprocessing_cfg:
            key, val = list(preprocess.items())[0]
            # if NORMALIZE:
            if key == 'normalize' and val == True:
                _data = normalize(_data)
            if (key == 'log_transform') and val:
                _data = log_transform(_data, val)
        return _data

    _fulldata_pickler = Pickler(fulldata_name,
                                Path(data_cache_base_path) / data_name)

    def _get_split_indices():
        _data = _fulldata_pickler.find_or_create(_get_fulldata)

        train_ind, test_ind = train_test_split(_data.index,
                                               train_size=cfg.data.train_size)
        if 'validation_size' in cfg.data:
            valid_ind, train_ind = train_test_split(
                train_ind, train_size=cfg.data.validation_size)
            return (train_ind, valid_ind), test_ind
        else:
            return train_ind, test_ind

    _split_indices_pickler = Pickler(
        f'{data_cache_name}_indices',
        Path(data_cache_base_path) / data_name / 'indices')

    def _get_split_data():
        _data = _fulldata_pickler.find_or_create(_get_fulldata)
        train_ind, test_ind = _split_indices_pickler.find_or_create(
            _get_split_indices)

        if 'validation_size' in cfg.data:
            train_ind, valid_ind = train_ind
            train, valid, test = _data.loc[train_ind], _data.loc[
                valid_ind], _data.loc[test_ind]
            return (train, valid), test
        else:
            train, test = _data.loc[train_ind], _data.loc[test_ind]
            return train, test

    data_pickler = Pickler(f'{data_cache_name}_data',
                           Path(data_cache_base_path) / data_name)
    train_data, test_data = data_pickler.find_or_create(_get_split_data)
    run_logger.set_tags_exp_wide(
        dict(data_cache_name=data_cache_name,
             data_cache_path=str(data_pickler.cache_path),
             graph_path=graph_path,
             graph_loader=cfg.data.graph.loader))

    #################
    ## Prepare graph
    #################
    if hasattr(data_module, cfg.data.graph.loader):
        # Load graph
        graph_cache_name = f'{data_name}_graph_{Path(graph_path).stem}_{cfg.data.graph.loader}'
        _graph_getter = lambda: getattr(data_module, cfg.data.graph.loader)(
            graph_path)
    else:
        # Estimate graph
        _data = _fulldata_pickler.find_or_create(_get_fulldata)
        graph_cache_name = f'graph_{fulldata_name}_{cfg.data.graph.loader}'
        run_logger.set_tags_exp_wide({'graph_cache': graph_cache_name})
        _graph_getter = lambda: getattr(estimate_causal_graph, cfg.data.graph.
                                        loader)(_data)
    _graph = Pickler(graph_cache_name,
                     Path(data_cache_base_path) /
                     data_name).find_or_create(_graph_getter)
    _predicted_var_name = data_module.get_predicted_variable_name()

    if 'validation_size' not in cfg.data:
        run_logger.log_params_exp_wide({
            'train_size': len(train_data),
            'test_size': len(test_data)
        })
    else:
        run_logger.log_params_exp_wide({
            'train_size': len(train_data[0]),
            'validation_size': len(train_data[1]),
            'test_size': len(test_data)
        })
    return train_data, test_data, _graph, _predicted_var_name


@hydra.main(config_path='config/config.yaml')
def main(cfg: DictConfig):
    # Setup run logger
    run_logger = get_run_logger(cfg)

    dataset = get_data(cfg, run_logger)

    # Save method info
    run_logger.set_tags_exp_wide({
        f'method_{key}': val
        for key, val in OmegaConf.to_container(cfg.method).items()
    })

    if cfg.method.name == 'proposed':
        _evaluate_proposed(cfg, dataset, run_logger)
    elif cfg.method.name == 'base_xgb':
        _evaluate_baseline_xgb(cfg, dataset, run_logger)


if __name__ == '__main__':
    main()
