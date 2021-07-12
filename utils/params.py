import functools
from typing import Callable

import yaml


def unpack_experiment_file(func: Callable) -> Callable:
    """Load parameters from an input file and inject it to function keyword arguments"""
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        experiment_file = kwargs.get('experiment_file')
        if experiment_file:
            with open(experiment_file, 'r') as f:
                run_args = yaml.full_load(f)
                kwargs.update(run_args)
        ret = func(*args, **kwargs)
        return ret

    return _wrapper


def check_param(param_name, required=False, param_type=None, choices=None, default=None):
    """function parameter validator"""
    def decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            param = kwargs.get(param_name, default)
            if required and param is None:
                raise RuntimeError
            if choices and param not in choices:
                raise RuntimeError
            if param_type:
                param = param_type(param)
                kwargs[param_name] = param
            ret = func(*args, **kwargs)
            return ret

        return _wrapper

    return decorator
