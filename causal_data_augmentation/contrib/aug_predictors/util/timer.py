import timeit

# Type hinting
from typing import Callable, Optional


class _TimerDecorator:
    """Provides the decorator functionality to the Timer class."""
    def __init__(self,
                 callback: Optional[Callable] = None,
                 with_args: bool = False):
        """
        Parameters:
            callback: the function to be wrapped.
            with_args: whether to call the ``callback`` with the arguments.
        """
        self.callback = callback
        self.with_args = with_args

    def default_callback(self, t, func: Callable):
        print(f'Function {func.__name__}() took {t.time} seconds.')

    def __call__(self, func: Callable):
        """
        Parameters:
            func: the function to be benchmarked.
        """
        if self.with_args:

            def _decorated_func(*args, **kwargs):
                """A local function to wrap the evaluation of the function."""
                with Timer() as t:
                    _res = func(*args, **kwargs)
                if self.callback is not None:
                    self.callback(t, *args, **kwargs)
                else:
                    self.default_callback(t, func)
                return _res
        else:

            def _decorated_func(*args, **kwargs):
                """A local function to wrap the evaluation of the function."""
                with Timer() as t:
                    _res = func(*args, **kwargs)
                if self.callback is not None:
                    self.callback(t)
                else:
                    self.default_callback(t, func)
                return _res

        return _decorated_func


class Timer:
    """The timer class. It can be used as a decorator by ``set()``.

    Example:
        >>> @Timer.set(lambda t: print('[Timer] The method took:', t.time))
        ... def evaluate(a: int):
        ...     return a ** 2
    """
    def __enter__(self):
        """Start the timer."""
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        """End the timer."""
        self.stop = timeit.default_timer()
        self.time = self.stop - self.start

    def __str__(self):
        """Stringify the timer values."""
        return f'Duration: {self.time}, Start: {self.start}, Stop: {self.stop}'

    @classmethod
    def set(cls, *args, **kwargs):
        """Utility to decorate functions."""
        return _TimerDecorator(*args, **kwargs)
