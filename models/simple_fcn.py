import torch
from pytorch_named_dims import nm
from torch import nn


class SimpleFCN(nn.Module):
    """Simple fully convolutional VGG-like model"""
    def __init__(self,
                 depth: int = 4,
                 features: int = 64,
                 in_channels: int = 3,
                 use_batch_norms=False,
                 vgg_like_initialization: bool = False):
        super().__init__()
        body = []
        for idx in range(depth):
            body.append(
                self.make_block(in_channels,
                                features,
                                use_batch_norms=use_batch_norms))
            in_channels = features
            features *= 2
        features = features // 2
        self.body = nn.Sequential(*body)
        self.head = nm.Conv2d(features, features, 1)
        if vgg_like_initialization:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        return self.head(x)

    @staticmethod
    def make_block(in_channels: int,
                   features: int,
                   use_batch_norms: bool = False):
        layers = [
            nm.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nm.BatchNorm2d(features) if use_batch_norms else nn.Sequential(),
            nm.ReLU(),
            nm.Conv2d(features, features, kernel_size=3, padding=1),
            nm.BatchNorm2d(features) if use_batch_norms else nn.Sequential(),
            nm.ReLU(),
            nm.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nm.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nm.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nm.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
