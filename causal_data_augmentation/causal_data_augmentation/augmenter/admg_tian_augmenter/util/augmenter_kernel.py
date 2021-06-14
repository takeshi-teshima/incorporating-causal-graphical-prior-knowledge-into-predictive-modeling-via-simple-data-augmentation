import numpy as np
from dataclasses import dataclass, field  # https://docs.python.org/ja/3/library/dataclasses.html
from .weight_computer.base import WeightComputer

# Type hinting
from typing import Tuple, List, Optional, Callable


@dataclass
class AugmenterKernel:
    """Class for holding the weight computer.
    Corresponds to $p(v | c)$.

    Parameters:
        v_names     : names of the variables to be augmented ($v$).
        c_names     : names of the variables to be conditioned ($c$).
        c_weighter : kernel object used for the conditional resampling.
                   This object needs to have been already fit to the parameters.
    """
    v_names: List[str]
    c_names: List[str]
    c_weighter: WeightComputer
