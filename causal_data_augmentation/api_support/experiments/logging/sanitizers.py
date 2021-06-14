import numpy as np
from pathlib import Path

# Type hinting
from typing import Dict, Any, Iterable, Callable


def sanitize_path(v: Any) -> Any:
    """Sanitize path.

    Parameters:
        v : Maybe a Path. If ``v`` is a ``Path`` object, it is converted to a string.

    Returns:
        The sanitized object.
    """
    if isinstance(v, Path):
        return str(v)
    else:
        return v


def sanitize_number(v: Any) -> Any:
    """Sanitize the data values to cast to the types that can be handled by MongoDB (via PyMongo).

    Parameters:
        v: The value to be recorded.
    """
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
    """Sanitizer class to wrap the sanitization procedure."""
    def __init__(self, sanitizers: Iterable[Callable[[Any], Any]] = DEFAULT_SANITIZERS):
        """Constructor.

        Parameters:
            sanitizers : The iterable of sanitizers.
        """
        self.sanitizers = sanitizers

    def __call__(self, r: Dict[str, Any]) -> Dict[str, Any]:
        """Perform sanitization to the values of the dictionary.

        Parameters:
            r : Dictionary whose values should be sanitized.

        Returns:
            Sanitized dictionary.
        """
        rr = {}
        for k, v in r.items():
            for sanitizer in self.sanitizers:
                v = sanitizer(v)
            rr[k] = v
        return rr
