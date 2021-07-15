from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
from ignite.engine import Engine, Events
from torch import device, nn
from torch.optim.optimizer import Optimizer

from . import prepare_tensor_batch

TripletBatch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def create_trainer(
    step_fn: Callable[..., Dict[str, float]], model: nn.Module, device: Optional[device] = None
) -> Engine:
    """Like from ignite.engine import create_supervised_trainer, but with custom attach func"""
    if device:
        model.to(device)

    trainer = Engine(step_fn)

    def _attach(*plugins: Any):
        for plugin in plugins:
            plugin.attach(trainer)

    trainer.attach = _attach

    @trainer.on(Events.STARTED)
    def _on_started(engine: Engine):
        engine.state.model = model

    @trainer.on(Events.EPOCH_STARTED)
    def _on_epoch_started(engine: Engine):
        engine.state.metrics = {}

    return trainer


def create_triplet_trainer(
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: Optional[device] = None,
    prepare_batch: Optional[Callable[[Sequence[torch.Tensor], torch.device], TripletBatch]] = None,
) -> Engine:
    def step_fn(_: Engine, batch: Sequence[torch.Tensor]):
        model.train()
        optimizer.zero_grad()
        if prepare_batch is not None:
            anchors, positives, negatives = prepare_batch(batch, device)
        else:
            anchors, positives, negatives = prepare_tensor_batch(batch, device=device)
        batch = torch.cat((anchors, positives, negatives))
        anchors, positives, negatives = model(batch).flatten(['C', 'H', 'W'], 'C').chunk(3)
        loss = loss_fn(anchors, positives, negatives)
        loss.backward()
        optimizer.step()
        return {'loss': loss.item()}

    return create_trainer(step_fn, model, device)


def create_single_img_trainer(
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: Optional[device] = None,
    prepare_batch: Optional[Callable[[Sequence[torch.Tensor], torch.device], torch.Tensor]] = None,
) -> Engine:
    def step_fn(_: Engine, batch: torch.Tensor):
        model.train()
        optimizer.zero_grad()
        if prepare_batch is not None:
            batch = prepare_batch(batch, device)
        else:
            batch = prepare_tensor_batch(batch, device=device)
        out = model(batch)
        loss = loss_fn(out)
        loss.backward()
        optimizer.step()
        return {'loss': loss.item()}

    return create_trainer(step_fn, model, device)
