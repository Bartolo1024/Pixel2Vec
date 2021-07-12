from typing import Optional

import tqdm
from ignite.engine import Engine, Events


class EpochProgressBar:
    """This plugin pin progress bar to epochs"""
    def __init__(self, **p_bar_kwargs):
        self.p_bar_kwargs = p_bar_kwargs

    def attach(self, engine: Engine) -> None:
        """attach to engine"""
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.on_epoch_end)
        self.bar: Optional[tqdm.tqdm] = None

    def on_epoch_end(self, engine: Engine) -> None:
        """Update in every n-th epoch"""
        if self.bar is None:
            self.bar = tqdm.tqdm(total=engine.state.max_epochs, **self.p_bar_kwargs)
        self.bar.update()
