import logging
import os
from typing import Any, Dict, Iterable, Optional

import click
import torch.nn
import torch.optim
import torch.utils.data
from livelossplot.inputs.pytorch_ignite import PlotLossesCallback

import pixel2vec.utils
import pixel2vec.utils.epoch_progress_bar
import pixel2vec.utils.metrics.loss
import pixel2vec.utils.predictor
from pixel2vec.engines import builders, unsupervised
from pixel2vec.utils import (create_artifacts_dir, load_project_config, params, saver)


@click.command()
@click.option('-e', '--experiment-file', type=str, default='experiments/minesweeper_sota.yaml')
@params.unpack_experiment_file
def main(
    data_dir: str,
    data_flow: Dict[str, Any],
    model_spec: Dict[str, Any],
    loss_fn_spec: Dict[str, Any],
    max_epochs: int,
    device: str = 'gpu',
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
        device: cuda, mps or cpu; gpu picks from cuda or mps
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

    if device in ['gpu', 'cuda'] and torch.cuda.is_available():
        device = torch.device('cuda')
    elif device in ['gpu', 'mps'] and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        if device != 'cpu':
            logging.warning(f'{device} not found, falling back to CPU.')
        device = torch.device('cpu')

    logging.info(f'training will be performed on {device} device')

    artifacts_dir = create_artifacts_dir(project_config['runs_directory'])
    logging.info(f'artifacts dir: {artifacts_dir}')
    os.makedirs(artifacts_dir, exist_ok=True)

    data_dir = os.path.join(project_config['data_root'], data_dir)
    loaders = builders.create_data_flow(data_root=data_dir, data_flow_params=data_flow['params'], mode=training_mode)

    # load model from ./models/name_of_some_model.py
    logging.info(f"create model from {model_spec['class']}")
    model = pixel2vec.utils.import_function(model_spec['class'])(**model_spec['params']).to(device)

    # load losses from ./losses/name_of_some_losses.py
    logging.info(f"create loss function: {loss_fn_spec['create_fn']}")
    loss_fn = pixel2vec.utils.import_function(loss_fn_spec['create_fn'])(**loss_fn_spec['params']).to(device)

    # load optimizer from torch.optim
    logging.info(f"create optimizer: {model_spec['optimizer']}")
    optimizer = pixel2vec.utils.get_optimizer(model, model_spec['optimizer'], **model_spec['optimizer_params'])

    logging.info(f"create trainer for mode: {training_mode}")
    create_trainer = unsupervised.create_triplet_trainer if training_mode == 'patches' \
        else unsupervised.create_single_img_trainer
    trainer = create_trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=device)

    logger = PlotLossesCallback(train_engine=trainer, outputs=('ExtremaPrinter', ))
    predictor = pixel2vec.utils.predictor.Predictor(model, training_mode, data_flow, device)

    evaluators = builders.build_evaluators(
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
    plugins = builders.build_plugins(
        predictor=predictor, artifacts_dir=artifacts_dir, visualization_images_dir=data_dir, **plugins_builder_kwargs
    )
    trainer.attach(*plugins)
    logging.info(f'start training with num epochs {max_epochs}')
    trainer.logger.setLevel(logging.ERROR)
    trainer.run(loaders['train'], max_epochs=max_epochs)


if __name__ == '__main__':
    _logger = logging.getLogger()
    _logger.setLevel(logging.INFO)
    main()
