class RunWrapper:
    """A wrapper class to hold the unchanged arguments throughout all runs of Sacred."""
    def __init__(self,
                 func,
                 static_args,
                 static_kwargs,
                 pass_extra_params=False):
        """
        Parameters:
            func: the function to be wrapped.
            static_args: the static arguments.
            static_kwargs: the static keyword arguments.
        """
        self.func = func
        self.static_args = static_args
        self.static_kwargs = static_kwargs
        self.pass_extra_params = pass_extra_params

    def __call__(self, idx, params_injector, run_logger):
        """Run the function with the saved parameters."""
        if self.pass_extra_params:
            return self.func(idx, params_injector, run_logger,
                             *self.static_args, **self.static_kwargs)
        else:
            return self.func(*self.static_args, **self.static_kwargs)
