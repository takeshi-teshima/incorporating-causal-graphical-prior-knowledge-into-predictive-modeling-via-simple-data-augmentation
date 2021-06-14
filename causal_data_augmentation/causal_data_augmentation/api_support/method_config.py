from dataclasses import dataclass, field

# Type hinting
from typing import Iterable, Optional, Union, Any, Dict

BatchSizeType = Optional[Union[int, float]]


@dataclass
class AugmenterConfig:
    """Base class for defining the augmenter configurations."""
    pass


@dataclass
class FullAugmentKind(AugmenterConfig):
    """Intermediate class for defining the augmenter configurations for those augmenters that
    augment the data to the fullest possible extent.
    """
    pass


@dataclass
class FullAugment(FullAugmentKind):
    """Augmenter configuration class for those augmenters that augment the data to the fullest possible extent.

    Parameters:
        weight_threshold : The weight threshold.
        weight_threshold_type : How the weight threshold should be applied. Defaults to ``'total'`` indicating that the threshold is imposed on the weights obtained after fully filling in the probability tree.
        normalize_threshold_by_data_size : Whether to divide the given threshold by the training data set size to make the threshold compensate for the initial uniform weights (the depth-1 edge weights in the probability tree are $1/n$ where $n$ is the data set size).
        weight_kernel_cfg : Additional configurations to be passed to the weight kernels.
    """
    weight_threshold: Optional[float] = None
    weight_threshold_type: str = ['total'][0]
    normalize_threshold_by_data_size: bool = True
    weight_kernel_cfg: dict = field(default_factory=dict)
