import logging
import os
from typing import Iterable

import torch.utils.data
from livelossplot.inputs.pytorch_ignite import PlotLossesCallback
from pytest import fixture

import utils
import utils.epoch_progress_bar
import utils.metrics.loss
import utils.predictor
from engines import builders, unsupervised
from utils import create_artifacts_dir, load_project_config, saver


@fixture
def experiment_config():
    import yaml
    with open('experiments/test.yaml', 'r') as f:
        data = yaml.load(f)
    return data


@fixture
def device():
    return torch.device('cpu')


@fixture
def data_flow_spec():
    return {
        'params': {
            'batch_size': 16,
            'num_workers': 0,
            'normalization_stats': {
                'mean': (.5, .5, .5),
                'std': (.5, .5, .5)
            },
            'patch_size': (8, 8),
            'min_negative_distance': 3,
            'min_positive_distance': 0,
            'max_positive_distance': 2
        }
    }


@fixture()
def loaders(data_flow_spec):
    data_dir = os.path.join('data', 'test')
    loaders = builders.create_data_flow(
        data_root=data_dir,
        data_flow_params=data_flow_spec['params'],
        mode='patches')
    return loaders


@fixture()
def model(device):
    model_spec = {
        'class': 'models.simple_resnet.SimpleResNet',
        'params': {
            'features': 16,
            'num_blocks': 8
        },
        'optimizer': 'Adam',
        'optimizer_params': {
            'lr': 0.0001
        }
    }
    # load model from ./models/name_of_some_model.py
    logging.info(f"create model from {model_spec['class']}")
    model = utils.import_function(
        model_spec['class'])(**model_spec['params']).to(device)
    return model


@fixture
def predictor(model, data_flow_spec, device):
    predictor = utils.predictor.Predictor(model, 'patches', data_flow_spec,
                                          device)
    return predictor


@fixture()
def optimizer(model):
    optim_spec = {'optimizer': 'Adam', 'optimizer_params': {'lr': 0.0001}}
    # load model from ./models/name_of_some_model.py
    logging.info(f"create optimizer: {optim_spec['optimizer']}")
    optimizer = utils.get_optimizer(model, optim_spec['optimizer'],
                                    **optim_spec['optimizer_params'])
    return optimizer


@fixture()
def loss_fn(device):
    loss_fn_spec = {
        'create_fn': 'losses.contrastive_loss.ContrastiveLoss',
        'params': {}
    }
    # load model from ./models/name_of_some_model.py
    # load losses from ./losses/name_of_some_losses.py
    logging.info(f"create loss function: {loss_fn_spec['create_fn']}")
    loss_fn = utils.import_function(
        loss_fn_spec['create_fn'])(**loss_fn_spec['params']).to(device)
    return loss_fn


def test_train_engine(loaders, model, optimizer, loss_fn, predictor, device):
    data_dir: str = 'data/test/'
    max_epochs: int = 2
    training_mode: str = 'patches'
    metrics: Iterable[str] = ('loss', 'dot_product_accuracy')
    project_config = load_project_config()

    artifacts_dir = create_artifacts_dir(project_config['runs_directory'])
    logging.info(f'artifacts dir: {artifacts_dir}')
    os.makedirs(artifacts_dir, exist_ok=True)

    trainer = unsupervised.create_triplet_trainer(model=model,
                                                  optimizer=optimizer,
                                                  loss_fn=loss_fn,
                                                  device=device)
    logger = PlotLossesCallback(train_engine=trainer,
                                outputs=('ExtremaPrinter', ))
    evaluators = builders.build_evaluators(predictor,
                                           loss_fn,
                                           loaders,
                                           logger,
                                           device,
                                           attach_progress_bar=True,
                                           mode=training_mode,
                                           metrics=metrics)

    logging.info('attach plugins for weights checkpointing')
    for metric in metrics:
        mode = min if metric == 'loss' else max
        evaluators['val'].engine.attach(
            saver.create_best_metric_saver(model,
                                           trainer,
                                           artifacts_dir,
                                           f'val_{metric}',
                                           mode=mode))

    logging.info(f'attach evaluators: {evaluators.keys()} and logger')
    trainer.attach(logger, *evaluators.values())
    logging.info('create and attach plugins')
    plugins = builders.build_plugins(predictor=predictor,
                                     artifacts_dir=artifacts_dir,
                                     visualization_images_dir=data_dir,
                                     t_sne_visualizer=False)
    trainer.attach(*plugins)
    logging.info(f'start training with num epochs {max_epochs}')
    trainer.logger.setLevel(logging.ERROR)
    trainer.run(loaders['train'], max_epochs=max_epochs)

    out_files = os.listdir(artifacts_dir)
    assert len([f for f in out_files if 'loss' in f]) == 2
    assert len([f for f in out_files if 'val_dot_product_accuracy' in f]) == 3
    assert len([
        f for f in out_files if 'val_dot_product_accuracy_predictions' in f
    ]) == 1
    pca_file = os.path.join(artifacts_dir, 'PCA-rgb',
                            'best_val_dot_product_accuracy', 'screen.png')
    assert os.path.isfile(pca_file)
