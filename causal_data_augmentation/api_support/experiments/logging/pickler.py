from pathlib import Path
import dill as pickle

# Type hinting
from typing import Any, Callable


class Pickler:
    """Utility class to access the ``dill`` library."""
    def __init__(self, path: str, base_path:Path=Path('pickle')):
        """Constructor.

        Parameters:
            path : Relative path to save the file to.
            base_path : Base path to save the file to (the file is saved to ``base_path/path``).
        """
        self.base_path = base_path
        self.cache_path = base_path / f'{path}.dill'
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Any:
        """Load the frozen object."""
        with self.cache_path.open('rb') as _f:
            res = pickle.load(_f)
        return res

    def save(self, content: Any) -> None:
        """Save the object.

        Parameters:
            content : Object to save to the file.
        """
        with self.cache_path.open('wb') as _f:
            pickle.dump(content, _f)

    def find_or_create(self, func: Callable[[],Any]) -> Any:
        """Find the frozen object or create and freeze the object.

        Parameters:
            func : Callable used to newly create the object.

        Returns:
            The created or retrieved object.
        """
        if not self.cache_path.exists():
            self.save(func())

        res = self._load()
        return res

    def load_or_none(self) -> Any:
        """Load the frozen object or return ``None`` if it does not exist.

        Returns:
            The object or ``None`` if it does not exist.
        """
        if self.cache_path.exists():
            return self._load()
        else:
            return None
