import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import ignite.engine
import torch
from ignite.metrics import Metric
from torch import nn

from utils import dictionary_flatten
from utils.predictor import Predictor

from . import prepare_tensor_batch


def create_evaluator(
    model: nn.Module,
    step_func: Callable[[ignite.engine.Engine, List[torch.Tensor]], Any],
    metrics: Dict[str, Metric],
):
    """As create_supervised_evaluator from ignite but with passed step function and custom attach"""
    metrics = metrics or {}

    engine = ignite.engine.Engine(step_func)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    def _attach(*plugins):
        for plugin in plugins:
            plugin.attach(engine)

    engine.attach = _attach

    @engine.on(ignite.engine.Events.STARTED)
    def _on_started(_engine):
        _engine.state.model = model

    return engine


def create_supervised_evaluator(
    model: Union[nn.Module, Predictor],
    metrics: Dict[str, Metric],
    device: torch.device,
    prepare_batch: Optional[Callable[[List[torch.Tensor], torch.device, bool], torch.Tensor]],
    non_blocking: bool = False,
    output_transform: Callable[[torch.Tensor, torch.Tensor, Any], Any] = lambda batch, features, annotations:
    (features, annotations)
):
    def _supervised_inference(_, batch):
        if isinstance(model, torch.nn.Module):
            model.eval()
        with torch.no_grad():
            images, annotations = batch
            if prepare_batch is not None:
                batch = prepare_batch(images, device, non_blocking)
            elif isinstance(images, torch.Tensor):
                batch = prepare_tensor_batch(images, device=device, non_blocking=non_blocking)
            else:
                batch = images
            features = model(batch)
            return output_transform(batch, features, annotations)

    return create_evaluator(model, _supervised_inference, metrics)


def create_triplets_evaluator(
    model: nn.Module,
    metrics: Dict[str, Metric],
    device: torch.device,
    prepare_batch: Optional[Callable[[List[torch.Tensor], torch.device, bool], Tuple[torch.Tensor, torch.Tensor,
                                                                                     torch.Tensor]]],
    non_blocking=False,
    output_transform=lambda anchors, positives, negatives, anchors_emb, positives_emb, negatives_emb:
    (anchors_emb, positives_emb, negatives_emb)
):
    def _unsupervised_inference(_: ignite.engine.Engine, batch: List[torch.Tensor]):
        model.eval()
        with torch.no_grad():
            if prepare_batch is not None:
                anchors, positives, negatives = prepare_batch(batch, device, non_blocking)
            else:
                anchors, positives, negatives = prepare_tensor_batch(batch, device=device, non_blocking=non_blocking)
            anchors_emb = model(anchors).flatten(['C', 'H', 'W'], 'C')
            positives_emb = model(positives).flatten(['C', 'H', 'W'], 'C')
            negatives_emb = model(negatives).flatten(['C', 'H', 'W'], 'C')
            return output_transform(anchors, positives, negatives, anchors_emb, positives_emb, negatives_emb)

    return create_evaluator(model, _unsupervised_inference, metrics)


def create_single_img_evaluator(
    model: nn.Module,
    metrics: Dict[str, Metric],
    device: torch.device,
    prepare_batch: Optional[Callable[[List[torch.Tensor], torch.device, bool], torch.Tensor]],
    non_blocking=False,
    output_transform=lambda batch, features: (features, )
):
    def _unsupervised_inference(_: ignite.engine.Engine, batch: List[torch.Tensor]):
        model.eval()
        with torch.no_grad():
            if prepare_batch is not None:
                batch = prepare_batch(batch, device, non_blocking)
            else:
                batch = prepare_tensor_batch(batch, device=device, non_blocking=non_blocking)
            features = model(batch)
            return output_transform(batch, features)

    return create_evaluator(model, _unsupervised_inference, metrics)


class Evaluator:
    def __init__(
        self,
        model: Union[nn.Module, Predictor],
        metrics: Dict[str, Metric],
        test_loader: Iterable,
        device: torch.device,
        prepare_batch_fn: Optional[Callable] = None,
        mode='patches',
    ):
        """
        Args:
            model: model or predictor
            metrics: dictionary with ignite metrics
            test_loader: data loader for testing
            device: cuda or cpu
            prepare_batch_fn: function for moving tensors to device
            mode: patches, images, sliced_images or supervised
        """
        if mode == 'patches':
            create_evaluator_fn = create_triplets_evaluator
        elif mode == 'images' or mode == 'sliced_images':
            create_evaluator_fn = create_single_img_evaluator
        elif mode == 'supervised':
            create_evaluator_fn = create_supervised_evaluator
        else:
            raise RuntimeError(f'Evaluator mode {mode} is not supported')
        self.create_evaluator_fn = create_evaluator_fn
        self.engine = create_evaluator_fn(model, metrics, device, prepare_batch=prepare_batch_fn)
        self.engine.logger.setLevel(logging.WARNING)
        self.test_loader = test_loader
        self.metrics = metrics

    def attach(self, engine: ignite.engine.Engine):
        engine.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, self.on_epoch_end)
        engine.add_event_handler(ignite.engine.Events.COMPLETED, self.on_epoch_end)

    def on_epoch_end(self, _):
        self.eval()

    def eval(self):
        state = self.engine.run(self.test_loader)
        self.engine.state.metrics = dictionary_flatten(state.metrics, sep='/')
