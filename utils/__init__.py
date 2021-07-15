import datetime
import functools
import logging
import os
import re
from importlib import import_module
from typing import Any, Callable, Dict, Pattern

import h5py as h5
import torch
import torch.nn
import torch.optim
import yaml
from torch.optim.optimizer import Optimizer


def load_project_config(config_file: str = 'project_config.yaml',
                        local_config_file: str = 'project_config.local.yaml'):
    """
    Args:
        config_file: config file available in git repository
        local_config_file: local yaml file - not available in git

    Returns:
        project config which is the same for all experiments
    """
    dirname = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(dirname, '..', config_file)
    local_config_file = os.path.join(dirname, '..', local_config_file)
    with open(config_file) as f:
        logging.info(f'load base project config from {config_file}')
        params = yaml.full_load(f)
    if os.path.isfile(local_config_file):
        logging.info(f'load local project config from {local_config_file}')
        with open(local_config_file) as f:
            local_params = yaml.full_load(f)
    else:
        local_params = {}
    params.update(local_params)
    logging.info(f'project config parameters {params}')
    return params


def load_secrets(secrets_file: str = 'secrets.yaml'):
    """
    Args:
        secrets_file: local yaml file - not available in git

    Returns:
        project config which is the same for all experiments
    """
    dirname = os.path.dirname(os.path.realpath(__file__))
    secrets_file = os.path.join(dirname, '..', secrets_file)
    if os.path.isfile(secrets_file):
        logging.info(f'load local secrets from {secrets_file}')
        with open(secrets_file) as f:
            secrets = yaml.full_load(f)
    else:
        secrets = {}
    return secrets


def setattr_nested(base: object, path: str, value: object):
    """Accept a dotted path to a nested attribute to set"""
    path, _, target = path.rpartition('.')
    for attrname in path.split('.'):
        base = getattr(base, attrname)
    setattr(base, target, value)


def import_function(class_path: str) -> Callable:
    """Function take module with to class or function and imports it dynamically"""
    modules = class_path.split('.')
    module_str = '.'.join(modules[:-1])
    cls = modules[-1]
    module = import_module(module_str)
    return getattr(module, cls)


def load_weights(model: torch.nn.Module, path: str) -> None:
    """
    Function loads h5 weights from file and load it to nn.Module
    H5 is data efficient and it can be easily loaded without PyTorch
    """
    state_dict = {}
    with h5.File(path, 'r') as file:
        for key, val in file.items():
            state_dict[key] = torch.from_numpy(val[...])
    model.load_state_dict(state_dict)


def store_weights(model: torch.nn.Module, path: str) -> None:
    """
    Function loads h5 weights from file and load it to nn.Module
    H5 is data efficient and it can be easily loaded without PyTorch
    """
    state_dict = model.state_dict()
    with h5.File(path, 'w') as f:
        for key, val in state_dict.items():
            f.create_dataset(key, data=val.detach().cpu().numpy())


def timer(process_name: str) -> Callable:
    """Decorator compute time of the execution of the function"""
    def decorator_timer(func: Callable) -> Callable:
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            tic = datetime.datetime.now()
            date_time = tic.strftime('%Y-%m-%d %H:%M:%S')
            print(f'{process_name} started at time {date_time}')
            func(*args, **kwargs)
            toc = datetime.datetime.now()
            date_time = toc.strftime('%Y-%m-%d %H:%M:%S')
            print(f'{process_name} ended at time {date_time}')
            print(f'{process_name} executed in {tic - toc}')

        return _wrapper

    return decorator_timer


def get_optimizer(model: torch.nn.Module, optimizer: str, *args,
                  **kwargs) -> Optimizer:
    """Function find optimizer in torch.optim and create it using passed args"""
    return getattr(torch.optim, optimizer)(model.parameters(), *args, **kwargs)


def build_loggers(loggers, **kwargs):
    loggers_module = 'utils.loggers'
    return [
        import_function('.'.join([loggers_module, logger]))(**kwargs)
        for logger in loggers
    ]


def dictionary_flatten(d: Dict[str, Any],
                       parent_key: str = '',
                       sep: str = '_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(dictionary_flatten(v, new_key, sep=sep).items())
        elif isinstance(v, list) and isinstance(v[0], dict):
            for idx, el in enumerate(v):
                sub_dict = dict((f'{idx}_{sub_key}', sub_val)
                                for sub_key, sub_val in el.items())
                items.extend(
                    dictionary_flatten(sub_dict, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def join_path(*args, create: bool = False):
    """
    Args:
        *args: paths to join
        create: create if does not exist

    Returns:
        path joined with '/' as separator
    """
    path = os.path.join(*args).replace('\\', '/')
    if create and '.' in path:
        directory = '/'.join(path.split('/')[:-1])
        os.makedirs(directory, exist_ok=True)
    elif create:
        os.makedirs(path, exist_ok=True)
    return path


def create_artifacts_dir(
    runs_dir: str,
    run_template: Pattern = re.compile(r'(GEAR-)([0-9]+)')) -> str:
    os.makedirs(runs_dir, exist_ok=True)
    runs = [
        re.match(run_template, run) for run in os.listdir(runs_dir)
        if re.match(run_template, run)
    ]
    if len(runs) == 0:
        next_run_dir = 'GEAR-0'
    else:
        last_run_match = max(runs, key=lambda r: int(r.group(2)))
        next_run_id = int(last_run_match.group(2)) + 1
        next_run_dir = last_run_match.group(1) + str(next_run_id)
    next_run_dir = join_path(runs_dir, next_run_dir)
    os.makedirs(next_run_dir)
    return next_run_dir
