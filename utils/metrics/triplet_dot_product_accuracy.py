from typing import Callable, Tuple

import torch
from ignite import metrics
from ignite.metrics.metric import reinit__is_reduced


class TripletDotProductAccuracy(metrics.Accuracy):
    def __init__(self, output_transform: Callable = lambda x: x):
        super().__init__()
        self._output_transform = output_transform

    @reinit__is_reduced
    def update(self, output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """ Examine if positive has better similarity than negative and update accuracy
        Args:
            output - batch of anchors, positives and negatives - ([N, C], [N, C], [N, C]):
        """
        anchor_embedding, positive_embedding, negative_embedding = output
        anchor_embedding = anchor_embedding.align_to('N', 'X', 'C')
        positive_embedding = positive_embedding.align_to('N', 'C', 'Y')
        negative_embedding = negative_embedding.align_to('N', 'C', 'Y')
        positive_prediction = anchor_embedding.bmm(positive_embedding).flatten(
            ['N', 'X', 'Y'], 'N')
        negative_prediction = anchor_embedding.bmm(negative_embedding).flatten(
            ['N', 'X', 'Y'], 'N')
        correct = positive_prediction > negative_prediction

        self._num_correct += correct.sum().item()
        self._num_examples += correct.shape[0]
