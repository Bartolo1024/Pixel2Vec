from typing import Callable, Tuple

import torch
from ignite import metrics
from ignite.metrics.metric import reinit__is_reduced
from torch.nn import CosineSimilarity


class CosineDistanceAccuracy(metrics.Accuracy):
    def __init__(self,
                 threshold: float = .5,
                 output_transform: Callable = lambda x: x):
        super().__init__()
        self.similarity_metric = CosineSimilarity()
        self.threshold = threshold
        self._output_transform = output_transform

    @reinit__is_reduced
    def update(self, output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        emb_1, emb_2, target = output
        similarity = torch.abs(self.similarity_metric(emb_1, emb_2))
        target[target == -1] = 0
        target = target.long()
        correct = torch.eq(similarity.view(-1).to(target), target.view(-1))

        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]
