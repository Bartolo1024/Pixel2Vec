import json
import logging
import os
from typing import Any, Callable, Dict, Tuple

import torch
import torchvision.transforms

import dataflow.transforms
import dataflow.utils
from utils import import_function, join_path, load_project_config


def restore_params(run_id: str, run_pattern: str = 'GEAR-{}') -> Dict:
    """Restore params from run id"""
    project_config = load_project_config()
    run_dir = join_path(project_config['runs_directory'],
                        run_pattern.format(run_id))
    params_path = os.path.join(run_dir, 'params.json')
    with open(params_path, 'r') as f:
        d = json.load(f)
    d['run_dir'] = run_dir
    return d


def restore_model(model_spec: Dict, run_dir: str, weights: str):
    """Restore model using given class and params, and load weights from storage or neptune
    Args:
        model_spec: directory that has model class (ex. models.simple_fcn.SimpleFCN) and dictionary parameters
        run_dir: run directory
        weights: weights filename

    Returns:
        nn.Module with loaded weights
    """
    model = import_function(model_spec['class'])(**model_spec['params'])
    weights_dir = os.path.join(run_dir, 'artifacts')
    weights_path = os.path.join(weights_dir, weights)
    logging.info(f'weights {weights_path} will be used')
    assert os.path.isfile(weights_path)
    state_dict = torch.load(weights_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    return model


def get_transforms(
    data_flow_spec: Dict,
    skip_resizing: bool = False
) -> Tuple[Callable[[Any], torch.Tensor], Callable[[Any], torch.Tensor]]:
    """Create transforms from the given data flow specification"""
    norm_stats = data_flow_spec['params']['normalization_stats']
    img_size = data_flow_spec['params'].get('img_size')
    to_grayscale = data_flow_spec['params'].get('to_grayscale')
    skip_resizing = not img_size or skip_resizing
    trn = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size)
        if not skip_resizing else lambda x: x,
        torchvision.transforms.Grayscale(
            num_output_channels=len(norm_stats['mean']))
        if to_grayscale else lambda x: x,
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(**norm_stats)
    ])
    inv_trn = dataflow.transforms.InvertNormalization(norm_stats)
    return trn, inv_trn


def choose_best_weights(artifacts_dir: str,
                        use_metrics: Dict[str, Callable]) -> Dict[str, str]:
    """List directory with weights and choose best for each metric
    Arguments:
        artifacts_dir: directory with weights
        use_metrics: metrics with suitable choose best functions

    Returns:
        best weights for each metric
    """
    ret = {}
    files = os.listdir(artifacts_dir)
    for metric_name, best_fn in use_metrics.items():
        matched_files = [
            s for s in files if metric_name in s and s.endswith('.pth')
        ]
        filenames_with_values = {
            s: float(s.split('=')[-1].replace('.pth', ''))
            for s in matched_files
        }
        if len(filenames_with_values) == 0:
            logging.warning(
                f'Can not find any weights for metric: {metric_name}')
            continue
        filename, _ = best_fn(filenames_with_values.items(),
                              key=lambda v: v[1])
        ret[metric_name] = filename
    return ret
