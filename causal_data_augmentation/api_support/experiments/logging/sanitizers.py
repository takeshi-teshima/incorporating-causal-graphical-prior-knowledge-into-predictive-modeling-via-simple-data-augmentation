import numpy as np
from pathlib import Path

# Type hinting
from typing import Dict, Any, Iterable, Callable


def sanitize_path(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    else:
        return v


def sanitize_number(v):
    """Sanitize the data values to cast to the types that can be handled by MongoDB (via PyMongo)."""
    if isinstance(v, np.int64):
        v = int(v)
    elif isinstance(v, np.float64):
        v = float(v)
    elif isinstance(v, np.float32):
        v = float(v)
    elif isinstance(v, np.ndarray):
        v = v.tolist()
    return v


DEFAULT_SANITIZERS = [sanitize_path, sanitize_number]


class Sanitizer:
    def __init__(
            self,
            sanitizers: Iterable[Callable[[Any], Any]] = DEFAULT_SANITIZERS):
        self.sanitizers = sanitizers

    def __call__(self, r: Dict[str, Any]):
        rr = {}
        for k, v in r.items():
            for sanitizer in self.sanitizers:
                v = sanitizer(v)
            rr[k] = v
        return rr
