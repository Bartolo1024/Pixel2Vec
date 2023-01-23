import torch
from torch import nn


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

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
        # nn.LogSigmoid is not supported on M1 MPS, so we use .sigmoid().log()
        return (-positive_product.sigmoid().log() - (-negative_product).sigmoid().log()).mean()
