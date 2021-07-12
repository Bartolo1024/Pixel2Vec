import torch
from pytorch_named_dims import name_module_class
from torch import nn


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        LogSigmoid = name_module_class(nn.LogSigmoid, [['N', '*']], ['N', '*'])
        self.log_sigmoid = LogSigmoid()

    def forward(
        self, anchor_embedding: torch.Tensor, positive_embedding: torch.Tensor, negative_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor_embedding: embeddings of chosen patches, shape [N, C]
            positive_embedding: embeddings of patches that are close to corresponding anchors, shape [N, C]
            negative_embedding: embeddings of patches that are far from corresponding anchors, shape [N, C]

        Returns:
            contrastive loss
        """
        anchor_embedding = anchor_embedding.align_to('N', 'X', 'C')
        positive_embedding = positive_embedding.align_to('N', 'C', 'Y')
        negative_embedding = negative_embedding.align_to('N', 'C', 'Y')
        positive_product = anchor_embedding.bmm(positive_embedding).flatten(['N', 'X', 'Y'], 'N')
        negative_product = anchor_embedding.bmm(negative_embedding).flatten(['N', 'X', 'Y'], 'N')
        return (-self.log_sigmoid(positive_product) - self.log_sigmoid(-negative_product)).mean()
