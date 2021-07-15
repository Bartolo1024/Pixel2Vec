from collections.abc import Iterable
from typing import List, Optional, Sequence, Union

import torch
from ignite.engine import convert_tensor


def prepare_tensor_batch(
    batch: Sequence[torch.Tensor],
    device: Optional[torch.device] = None,
    non_blocking: bool = False
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Prepare batch for training: pass to a device with options."""
    if isinstance(batch, torch.Tensor):
        return convert_tensor(batch, device=device, non_blocking=non_blocking)
    if isinstance(batch, Iterable):
        return [convert_tensor(el, device=device, non_blocking=non_blocking) for el in batch]
    raise NotImplementedError(f'Not implemented for batch of type {type(batch)}')
