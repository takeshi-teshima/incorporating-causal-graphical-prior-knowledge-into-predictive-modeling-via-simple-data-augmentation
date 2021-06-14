from dataclasses import dataclass, field

# Type hinting
from typing import Iterable, Optional, Union, Any, Dict

BatchSizeType = Optional[Union[int, float]]


@dataclass
class AugmenterConfig:
    pass


@dataclass
class FullAugmentKind(AugmenterConfig):
    pass


@dataclass
class FullAugment(FullAugmentKind):
    weight_threshold: Optional[float] = None
    weight_threshold_type: str = ['factor_wise', 'total'][0]
    normalize_threshold_by_data_size: bool = True
    weight_kernel_cfg: dict = field(default_factory=dict)
