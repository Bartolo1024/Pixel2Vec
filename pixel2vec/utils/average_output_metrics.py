from typing import Dict, List

from ignite.engine import Engine, Events


class AverageOutputMetrics:
    """Plugin collects output from train loop and compute average of all elements from epoch"""
    def __init__(self, iteration_metric_log: bool = False):
        self.metrics: Dict[str, List[float]] = {}
        self.iteration_metric_log = iteration_metric_log

    def reset(self):
        self.metrics: Dict[str, List[float]] = {}

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.on_iteration_end)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.on_epoch_end)

    def on_iteration_end(self, engine: Engine):
        """append iteration metrics (outputs with float value) to cache"""
        for key, val in engine.state.output.items():
            if not isinstance(val, float):
                continue
            if not self.metrics.get(key):
                self.metrics[key] = []
            self.metrics[key].append(val)
            if self.iteration_metric_log:
                engine.state.metrics[f'online_batch_{key}'] = val

    def on_epoch_end(self, engine: Engine):
        """Create metrics with an average of each cached value"""
        for key, val in self.metrics.items():
            num_batches = len(val)
            engine.state.metrics[f'online_epoch_{key}'] = sum(val) / num_batches
        self.reset()
