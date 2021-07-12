import os
from typing import Callable

from ignite.engine import Engine, Events
from torch import nn

from . import store_weights


class PeriodicalH5NetSaver:
    """This saver saves the model on every n-th epoch"""
    def __init__(self, save_period: int, out__dir: str, model_variable_name: str = 'model') -> None:
        """
        Args:
            save_period: n-th epoch
            out_dir: dir to save
            model_variable_name: model variable in the engine
        """
        self.model_variable_name = model_variable_name
        self.save_period = save_period
        self.out_dir = out__dir
        os.makedirs(self.out_dir, exist_ok=True)

    def attach(self, engine: Engine) -> None:
        """attach to engine"""
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.on_epoch_end)
        engine.add_event_handler(Events.COMPLETED, self.store)

    def on_epoch_end(self, engine: Engine) -> None:
        """Store in every n-th epoch"""
        if engine.state.epoch != 0 and (engine.state.epoch + 1) % self.save_period == 0:
            self.store(engine, engine.state.epoch)

    def store(self, engine: Engine, suffix: str = 'final') -> None:
        """store model weights in h5 format"""
        model = getattr(engine.state, self.model_variable_name)
        store_weights(model, f'{self.out_dir}/weights_{suffix}.h5')


def create_best_metric_saver(
    model: nn.Module,
    trainer: Engine,
    artifacts_dir: str,
    metric_name: str,
    mode: Callable[[float, float], float] = min
) -> object:
    """
    Fabric of Plugins, that are store model with the higher value of the chosen metric
        model: nn.Module to save
        trainer - ignite train engine
        artifacts_dir - dir to store all weights
        metric_name: name of tracked metric
        mode: min or max - best value of metric for example max for accuracy, min for loss
        (min values will be saved with '-')
    Return:
        saver - plugin that saves best weights by value of chosen metric
    """
    assert mode in (min, max)

    from ignite.engine import Engine
    from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine

    os.makedirs(artifacts_dir, exist_ok=True)

    def score_function(engine: Engine):
        """Revert result for min function"""
        return mode(-1, 1) * engine.state.metrics[metric_name]

    to_save = {'model': model}
    handler = Checkpoint(
        to_save,
        DiskSaver(artifacts_dir, create_dir=True),
        n_saved=2,
        filename_prefix='best',
        score_function=score_function,
        score_name=metric_name,
        global_step_transform=global_step_from_engine(trainer)
    )
    # To be consistent between .pth and .pt format in next versions of ignite
    assert hasattr(handler, 'ext')
    handler.ext = 'pth'

    def _attach(engine) -> None:
        """Monkey patch for the checkpoint handler"""
        engine.add_event_handler(Events.COMPLETED, handler)

    handler.attach = _attach

    return handler
