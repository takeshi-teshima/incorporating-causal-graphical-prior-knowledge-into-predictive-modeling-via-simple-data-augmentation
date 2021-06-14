"""Augmenter for ADMG. Only uses the Tian factorization.
"""
import numpy as np
import pandas as pd
from .util.augmenter_kernel import AugmenterKernel
from .util.weight_computer.kernel import KernelWeightComputer
from .util.weight_computer.unconditional import UnconditionalWeightComputer
from .util.weight_computer.kernel_fn.vanilla import VanillaProductKernel, VanillaProductKernelConfig

# Type hinting
from typing import Tuple, Iterable, Optional, List
from ananke.graphs import ADMG
from pandas import DataFrame as DF
from tqdm import tqdm

CONTI_OR_DISC = {
    'int64': 'u',
    'int32': 'u',
    'float64': 'c',
    'float32': 'c',
}


class ADMGTianAugmenterBase:
    """Base class for the proposed augmentation method based on the Tian factorization (topological ADMG factorization)."""
    def __init__(self, graph: ADMG, top_order: Optional[List[str]] = None):
        """Constructor.

        Parameters:
            graph : the ADMG model to be used for the data augmentation.
            top_order : a valid topological order on the graph (a list of the vertex names).
                        If ``None`` is provided, it is automatically computed from the graph (default: ``None``).
        """
        self.graph = graph
        if top_order is None:
            top_order = self.graph.topological_sort()
        self.graph_top_order = top_order

    def prepare(self, data: DF, weight_kernel_cfg: dict):
        """Perform Tian factorization and fit weight functions for the conditioning variables.

        Parameters:
            data : Data to be passed to the augmenter to prepare for the augmentation (typically the training data).
            weight_kernel_cfg : Configuration of the kernels.

        Notes:
            See https://gitlab.com/causal/ananke/-/blob/master/ananke/graphs/sg.py for ``districts`` of ADMG.
            See https://gitlab.com/causal/ananke/-/blob/master/ananke/graphs/admg.py for ``markov_pillow()`` of ADMG.
        """
        # Prepare dict (column name -> data type)
        dtypes = data.dtypes.apply(lambda x: x.name).apply(
            lambda x: CONTI_OR_DISC[x]).to_dict()

        # Prepare factorization kernels
        factorization_kernels = []
        for v in tqdm(self.graph_top_order):
            mp = list(self.graph.markov_pillow([v], self.graph_top_order))
            mp_data = np.array(data[mp])
            var_types = ''.join([dtypes[var] for var in mp])
            if len(mp) > 0:
                if weight_kernel_cfg['type'] == 'vanilla_kernel':
                    config = VanillaProductKernelConfig()
                    if weight_kernel_cfg.get('const_bandwidth', False):
                        config.conti_bw_method = lambda _: weight_kernel_cfg[
                            'bandwidth_temperature']
                    config.conti_bw_temperature = weight_kernel_cfg.get(
                        'bandwidth_temperature', 1.)
                    config.conti_kertype = weight_kernel_cfg.get(
                        'conti_kertype', 'gaussian')
                    _kernel = VanillaProductKernel(data_ref=mp_data,
                                                   vartypes=var_types,
                                                   config=config)
                    weighter = KernelWeightComputer(_kernel)
            else:
                n = len(data)
                weighter = UnconditionalWeightComputer(n)
            factorization_kernels.append(AugmenterKernel([v], mp, weighter))
        self.factorization_kernels = factorization_kernels
        self.data = data
