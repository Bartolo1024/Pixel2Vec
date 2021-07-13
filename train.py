import logging
import os
from typing import Any, Callable, Dict, Iterable, List, Optional

import click
import torch.nn
import torch.optim
import torch.utils.data
from ignite.contrib.handlers import ProgressBar
from ignite.metrics.metric import Metric
from livelossplot.inputs.pytorch_ignite import PlotLossesCallback

import utils
import utils.epoch_progress_bar
import utils.metrics.loss
import utils.predictor
from engines import evaluator as evaluator_module
from engines import unsupervised
from utils import (
    average_output_metrics, create_artifacts_dir, feature_map_saver, params, reduced_dimensions_visualizer, saver,
    load_project_config
)
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
        import dataflow.pytorch
        loaders = dataflow.pytorch.create_patches_dataflow(root=data_root, **data_flow_params)
    elif mode == 'images':
        import dataflow.pytorch
        loaders = dataflow.pytorch.create_images_dataflow(root=data_root, **data_flow_params)
    elif mode == 'sliced_images':
        import dataflow.pytorch
        loaders = dataflow.pytorch.create_sliced_images_dataflow(root=data_root, **data_flow_params)
    else:
        raise NotImplementedError(f'{mode} mode not implemented')
    return loaders


def build_plugins(
    predictor: utils.predictor.Predictor,
    artifacts_dir: str,
    visualization_images_dir: str,
    data_flow_spec: Dict[str, Any],
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
        data_flow_spec: dictionary with dataflow create function and parameters
        device: cuda or cpu
        predictor_mode: images, sliced_images or patches - image will be processed in single shot or in chunks
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
        device: cuda or cpu
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


@click.command()
@click.option('-e', '--experiment-file', type=str, default='experiments/train_on_game_image.yaml')
@params.unpack_experiment_file
def main(
    data_dir: str,
    data_flow: Dict[str, Any],
    model_spec: Dict[str, Any],
    loss_fn_spec: Dict[str, Any],
    max_epochs: int,
    device: str = 'cuda',
    attach_eval_progress_bar: bool = True,
    plugins_builder_kwargs: Optional[Dict[str, Any]] = None,
    training_mode: str = 'patches',
    metrics: Iterable[str] = ('dot_product_accuracy', 'loss'),
    **kwargs
):
    """
    Args:
        data_dir: directory with datasets folders
        data_flow: dictionary with dataflow params and operations
        model_spec: dictionary with model specification contains create function location, arguments,
         optimizer and optimizer arguments:
        loss_fn_spec: dictionary with loss function specification contains create function location and arguments:
        max_epochs: number of train epochs:
        device: cuda or cpu
        attach_eval_progress_bar: flag that attach progress bar to evaluation
        plugins_builder_kwargs: build_plugins keyword arguments
        training_mode: patches, images or sliced_images
        metrics: listed metrics for evaluation stages

    Keyword Args:
        device: pytorch device - currently cuda or cpu
        attach_eval_progress_bar: flag attach progress bar to evaluators
    """
    project_config = load_project_config()
    if len(kwargs) > 0:
        logging.warning(f'In input yaml not used keyword arguments: {kwargs} were passed')
    device = torch.device('cpu') if not torch.cuda.is_available() or device == 'cpu' else torch.device('cuda')
    logging.info(f'training will be performed on {device} device')

    artifacts_dir = create_artifacts_dir(project_config['runs_directory'])
    logging.info(f'artifacts dir: {artifacts_dir}')
    os.makedirs(artifacts_dir, exist_ok=True)

    data_dir = os.path.join(project_config['data_root'], data_dir)
    loaders = create_data_flow(data_root=data_dir, data_flow_params=data_flow['params'], mode=training_mode)

    # load model from ./models/name_of_some_model.py
    logging.info(f"create model from {model_spec['class']}")
    model = utils.import_function(model_spec['class'])(**model_spec['params']).to(device)

    # load losses from ./losses/name_of_some_losses.py
    logging.info(f"create loss function: {loss_fn_spec['create_fn']}")
    loss_fn = utils.import_function(loss_fn_spec['create_fn'])(**loss_fn_spec['params']).to(device)

    # load optimizer from torch.optim
    logging.info(f"create optimizer: {model_spec['optimizer']}")
    optimizer = utils.get_optimizer(model, model_spec['optimizer'], **model_spec['optimizer_params'])

    logging.info(f"create trainer for mode: {training_mode}")
    create_trainer = unsupervised.create_triplet_trainer if training_mode == 'patches' \
        else unsupervised.create_single_img_trainer
    trainer = create_trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=device)

    logger = PlotLossesCallback(train_engine=trainer, outputs=('ExtremaPrinter', ))
    predictor = utils.predictor.Predictor(model, training_mode, data_flow, device)

    evaluators = build_evaluators(
        predictor,
        loss_fn,
        loaders,
        logger,
        device,
        attach_progress_bar=attach_eval_progress_bar,
        mode=training_mode,
        metrics=metrics
    )

    logging.info('attach plugins for weights checkpointing')
    for metric in metrics:
        mode = min if metric == 'loss' else max
        evaluators['val'].engine.attach(
            saver.create_best_metric_saver(model, trainer, artifacts_dir, f'val_{metric}', mode=mode)
        )
        evaluators['train'].engine.attach(
            saver.create_best_metric_saver(model, trainer, artifacts_dir, f'train_{metric}', mode=mode)
        )

    logging.info(f'attach evaluators: {evaluators.keys()} and logger')
    trainer.attach(logger, *evaluators.values())
    logging.info('create and attach plugins')
    plugins_builder_kwargs = plugins_builder_kwargs if plugins_builder_kwargs is not None else {}
    plugins = build_plugins(
        predictor=predictor,
        artifacts_dir=artifacts_dir,
        visualization_images_dir=data_dir,
        data_flow_spec=data_flow,
        **plugins_builder_kwargs
    )
    trainer.attach(*plugins)
    logging.info(f'start training with num epochs {max_epochs}')
    trainer.logger.setLevel(logging.ERROR)
    trainer.run(loaders['train'], max_epochs=max_epochs)


if __name__ == '__main__':
    _logger = logging.getLogger()
    _logger.setLevel(logging.INFO)
    main()
