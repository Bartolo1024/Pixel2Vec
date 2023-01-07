import logging
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch.utils.data
from ignite.contrib.handlers import ProgressBar
from ignite.metrics.metric import Metric
from livelossplot.inputs.pytorch_ignite import PlotLossesCallback

import dataflow
import utils
import utils.epoch_progress_bar
import utils.metrics.loss
import utils.predictor
from engines import evaluator as evaluator_module
from utils import (average_output_metrics, feature_map_saver, reduced_dimensions_visualizer)
from utils.metrics import triplet_dot_product_accuracy


def create_data_flow(
    data_root: str,
    data_flow_params: Dict[str, Any],
    mode: str = 'patches',
) -> Dict[str, Iterable]:
    """Function that creates iterators - nested data-flows imports in order to skip non used dependencies
    Args:
        data_flow_params - dictionary with all data flow creator parameters:
        artifacts_dir - directory to store list of images:
    Returns:
        train iterator and test iterator
    """
    logging.info('load data')
    if mode == 'patches':
        loaders = dataflow.create_patches_dataflow(root=data_root, **data_flow_params)
    elif mode == 'images':
        loaders = dataflow.create_images_dataflow(root=data_root, **data_flow_params)
    elif mode == 'sliced_images':
        loaders = dataflow.create_sliced_images_dataflow(root=data_root, **data_flow_params)
    else:
        raise NotImplementedError(f'{mode} mode not implemented')
    return loaders


def build_plugins(
    predictor: utils.predictor.Predictor,
    artifacts_dir: str,
    visualization_images_dir: str,
    store_predictions_on_the_end: bool = True,
    t_sne_visualizer: bool = True,
    pca_visualizer: bool = True,
    choose_predictor_weights_by_metric: str = 'val_dot_product_accuracy',
    tsne_downsampling_kernel_size: Optional[int] = None
):
    """
    Args:
        predictor: predictor
        artifacts_dir: dir with stored weights
        visualization_images_dir: directory with images for predictions visualization
        store_predictions_on_the_end: compute predictions for all images on the end, store it
        t_sne_visualizer: compute predictions for all images on the end, store it
        pca_visualizer: compute predictions for all images on the end, store it
        choose_predictor_weights_by_metric: load the best model by provided metric
        tsne_downsampling_kernel_size: provide it if you want to decrease time of t-SNE computations

    Returns:
        list with plugins for trainer - plugins have to be attached after evaluators
    """
    epoch_progress_bar = utils.epoch_progress_bar.EpochProgressBar(desc='Epochs: ')
    p_bar = ProgressBar(persist=False, desc='Training epoch: ')
    output_metrics_average_plugin = average_output_metrics.AverageOutputMetrics()
    optional_plugins: List[Any] = []
    best_metric_fn = min if choose_predictor_weights_by_metric == 'loss' else max
    if store_predictions_on_the_end:
        logging.info('Attach final predictions plugin')
        store_feature_map = feature_map_saver.FeatureMapSaver(
            predictor=predictor,
            use_metrics={choose_predictor_weights_by_metric: best_metric_fn},
            artifacts_dir=artifacts_dir,
            visualization_images_dir=visualization_images_dir
        )
        optional_plugins.append(store_feature_map)
    if t_sne_visualizer:
        logging.info('Attach t-SNE visualization')
        t_sne_rgb_visualizer = reduced_dimensions_visualizer.ReducedDimensionsRGBVisualizer(
            reduction_mode='t-SNE',
            downsampling_kernel=tsne_downsampling_kernel_size,
            predictor=predictor,
            use_metrics={choose_predictor_weights_by_metric: best_metric_fn},
            artifacts_dir=artifacts_dir,
            visualization_images_dir=visualization_images_dir
        )
        optional_plugins.append(t_sne_rgb_visualizer)
    if pca_visualizer:
        logging.info('Attach PCA visualization')
        pca_rgb_visualizer = reduced_dimensions_visualizer.ReducedDimensionsRGBVisualizer(
            reduction_mode='PCA',
            predictor=predictor,
            use_metrics={choose_predictor_weights_by_metric: best_metric_fn},
            artifacts_dir=artifacts_dir,
            visualization_images_dir=visualization_images_dir
        )
        optional_plugins.append(pca_rgb_visualizer)
    return [epoch_progress_bar, p_bar, output_metrics_average_plugin, *optional_plugins]


def create_metrics(metrics: Iterable[str], loss_fn: Callable[[Any], torch.Tensor], prefix: str) -> Dict[str, Metric]:
    """
    Args:
        metrics: list of string metric names
        loss_fn: loss function

    Returns:
        list of ignite metrics
    """
    ret = {}
    for metric in metrics:
        if metric == 'loss':
            ret[f'{prefix}_{metric}'] = utils.metrics.loss.LossMetric(loss_fn)
        elif metric == 'dot_product_accuracy':
            ret[f'{prefix}_{metric}'] = triplet_dot_product_accuracy.TripletDotProductAccuracy()
        else:
            raise NotImplementedError(f'Metric {metric} is not implemented')
    return ret


def build_evaluators(
    predictor: utils.predictor.Predictor,
    loss_fn: Callable[[Any], torch.Tensor],
    loaders: Dict[str, Iterable],
    logger: PlotLossesCallback,
    device: torch.device,
    metrics: Iterable[str],
    attach_progress_bar: bool = False,
    mode: str = 'patches',
) -> Dict[str, evaluator_module.Evaluator]:
    """
    Args:
        predictor: predictor
        loss_fn: function or callable for loss value computation
        loaders: iterators with images
        logger: for metrics storation
        device: cuda, mps or cpu
        metrics: list with listed metrics
        attach_progress_bar: attach progress bar to each evaluator
        mode: patches or images

    Returns:
        directory with evaluators
    """
    evaluators = {}
    logging.info(f'create {loaders.keys()} validation engines')
    for loader_name in loaders.keys():
        logging.info(f'create {loader_name} validation engine')
        _metrics = create_metrics(metrics, loss_fn, prefix=loader_name)
        model = predictor.model
        evaluator = evaluator_module.Evaluator(model, _metrics, loaders[loader_name], device, mode=mode)
        if attach_progress_bar:
            evaluator.engine.attach(ProgressBar(persist=False, desc=f'Evaluation on the {loader_name} loader: '))
        evaluator.engine.attach(logger)
        evaluators[loader_name] = evaluator
    return evaluators
