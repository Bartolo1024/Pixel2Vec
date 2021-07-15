import math

import torch
from pytorch_named_dims import nm
from torch import nn

__ALL__ = ['BaseResBlock', 'PreActivationResBlock', 'BottleneckResBlock', 'SimpleResNet']


class BaseResBlock(nn.Module):
    """
    Original implementation of the ResNet block with skip options for batch norm and last activation
    """
    def __init__(
        self,
        features: int,
        dilation: int = 1,
        kernel_size: int = 3,
        use_batch_norm: bool = False,
        block_output_activation: bool = True,
        block_gain: float = 1.
    ):
        super(BaseResBlock, self).__init__()
        padding = kernel_size // 2 + dilation // 2
        block = [
            nm.Conv2d(features, features, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nm.ReLU(),
            nm.BatchNorm2d(features) if use_batch_norm else nn.Sequential(),
            nm.Conv2d(features, features, kernel_size=1),
            nm.BatchNorm2d(features) if use_batch_norm else nn.Sequential(),
        ]
        self.block = nn.Sequential(*block)
        self.block_gain = block_gain
        self.activation = nn.ReLU() if block_output_activation else nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.block(x) * self.block_gain + identity
        return self.activation(x)


class PreActivationResBlock(nn.Module):
    """
    Implementation of block from https://arxiv.org/abs/1603.05027
    """
    def __init__(
        self,
        features: int,
        dilation: int = 1,
        kernel_size: int = 3,
        use_batch_norm: bool = False,
    ):
        super(PreActivationResBlock, self).__init__()
        padding = kernel_size // 2 + dilation // 2
        block = [
            nm.BatchNorm2d(features) if use_batch_norm else nn.Sequential(),
            nm.ReLU(),
            nm.Conv2d(features, features, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nm.ReLU(),
            nm.BatchNorm2d(features) if use_batch_norm else nn.Sequential(),
            nm.Conv2d(features, features, kernel_size=1),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        return self.block(x) + identity


class BottleneckResBlock(nn.Module):
    """
        Implementation of block from https://arxiv.org/abs/1603.05027
        """
    def __init__(
        self,
        features: int,
        bottleneck_features: int,
        dilation: int = 1,
        kernel_size: int = 3,
        use_batch_norm: bool = False,
        block_output_activation: bool = True,
    ):
        super(BottleneckResBlock, self).__init__()
        padding = kernel_size // 2 + dilation // 2
        block = [
            nm.Conv2d(features, features, kernel_size=1),
            nm.BatchNorm2d(features) if use_batch_norm else nn.Sequential(),
            nm.ReLU(),
            nm.Conv2d(features, bottleneck_features, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nm.BatchNorm2d(bottleneck_features) if use_batch_norm else nn.Sequential(),
            nm.ReLU(),
            nm.Conv2d(bottleneck_features, features, kernel_size=1),
            nm.BatchNorm2d(features) if use_batch_norm else nn.Sequential(),
        ]
        self.block = nn.Sequential(*block)
        self.activation = nn.ReLU() if block_output_activation else nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.block(x) + identity
        return self.activation(x)


class SimpleResNet(nn.Module):
    _AVAILABLE_BLOCKS = {
        'BaseResBlock': BaseResBlock,
        'PreActivationResBlock': PreActivationResBlock,
        'BottleneckResBlock': BottleneckResBlock
    }

    def __init__(
        self,
        features: int,
        num_blocks: int,
        block_type: str = 'BaseResBlock',
        in_channels: int = 3,
        pool: int = 1,
        **block_kwargs
    ):
        """
        Args:
            features: number of features in each convolutional filter
            num_blocks: number of ResNet blocks
            block_type: name of a block class implemented in this modules
            in_channels: number of input channels
            pool: perform adaptive pooling ont the end with specified size - type 0 if you want to skip pooling
            **block_kwargs: keyword arguments that will be passed to a block constructor
        """
        super(SimpleResNet, self).__init__()
        self.features = features
        self.stump = nm.Conv2d(in_channels, features, kernel_size=1)
        block_fn = self._AVAILABLE_BLOCKS[block_type]
        blocks = [block_fn(features, **block_kwargs) for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)
        self.pool = nm.AdaptiveMaxPool2d(pool) if pool > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: images batch of shape ['N', 'C', 'H', 'W']

        Returns:
            feature map of shape ['N', 'C', 'H', 'W']

        Notes:
            feature map is normalized by square root of number of features
             in order to decrease expected value of feature vector product
        """
        x = self.stump(x)
        x = self.blocks(x)
        if self.pool:
            x = self.pool(x)
        return x / math.sqrt(self.features)
