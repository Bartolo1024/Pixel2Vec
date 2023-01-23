from importlib import import_module
from math import sqrt
from typing import Any, Dict

import torch
from pytorch_named_dims import nm
from torch import nn
from torch.nn import functional


class BaseEncoderBlock(nn.Sequential):
    """
    Original implementation of the ResNet block with skip options for batch norm and last activation
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dilation: int = 1,
        kernel_size: int = 3,
        use_batch_norm: bool = False,
        skip_padding: bool = True
    ):
        """
        Args:
            in_features: number of input features
            out_features: number of output features
            dilation: use dilation in the first layer
            kernel_size: size of the first convolution kernel
            use_batch_norm: use batch norm after each convolution
            skip_padding: skip padding
        """
        self.skip_padding = skip_padding
        padding = kernel_size // 2 + dilation // 2 if not skip_padding else 0
        block = [
            nm.Conv2d(in_features, out_features, kernel_size=kernel_size, dilation=dilation, padding=padding),
            nm.BatchNorm2d(out_features) if use_batch_norm else nn.Sequential(),
            nm.ReLU(),
            nm.Conv2d(out_features, out_features, kernel_size=1),
            nm.BatchNorm2d(out_features) if use_batch_norm else nn.Sequential(),
            nm.ReLU(),
        ]
        super(BaseEncoderBlock, self).__init__(*block)


class BaseDecoderBlock(nn.Sequential):
    """
    Basic upscale block with skip options for batch norm and last activation
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 3,
        use_batch_norm: bool = False,
        skip_padding: bool = True
    ):
        """
        Args:
            in_features: number of input features
            out_features: number of output features
            kernel_size: size of the first convolution kernel
            use_batch_norm: use batch norm after convolution
            skip_padding: skip padding
        """
        super(BaseDecoderBlock, self).__init__()
        self.skip_padding = skip_padding
        padding = kernel_size // 2 if not skip_padding else 0
        super().__init__(
            nm.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding),
            nm.BatchNorm2d(out_features) if use_batch_norm else nn.Sequential(),
            nm.ReLU(),
            nm.Conv2d(out_features, out_features, kernel_size=1),
            nm.BatchNorm2d(out_features) if use_batch_norm else nn.Sequential(),
            nm.ReLU(),
        )


class UNetResBlock(nn.Module):
    """
    Original implementation of the ResNet block with skip options for batch norm and last activation
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dilation: int = 1,
        kernel_size: int = 3,
        use_batch_norm: bool = False,
        block_output_activation: bool = True,
        block_gain: float = 1.
    ):
        super(UNetResBlock, self).__init__()
        padding = kernel_size // 2 + dilation // 2
        block = [
            nm.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nm.ReLU(),
            nm.BatchNorm2d(out_features) if use_batch_norm else nn.Sequential(),
            nm.Conv2d(out_features, out_features, kernel_size=1),
            nm.BatchNorm2d(out_features) if use_batch_norm else nn.Sequential(),
        ]
        self.adapter = nm.Conv2d(in_features, out_features, kernel_size=1) if in_features != out_features else None
        self.block = nn.Sequential(*block)
        self.block_gain = block_gain
        self.activation = nn.ReLU() if block_output_activation else nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if not self.adapter else self.adapter(x)
        x = self.block(x) * self.block_gain + identity
        return self.activation(x)


class SimpleUNet(nn.Module):
    def __init__(
        self,
        features,
        num_blocks: int,
        encoder_block_type: str = 'BaseEncoderBlock',
        decoder_block_type: str = 'BaseDecoderBlock',
        scale: int = 2,
        pool_fn: str = 'MaxPool2d',
        in_channels: int = 3,
        encoder_block_kwargs: Dict[str, Any] = {},
        decoder_block_kwargs: Dict[str, Any] = {},
    ):
        """
        Args:
            features: base number of features
            num_blocks: depth of the U-Net
            encoder_block_type: A block class name available there
            decoder_block_type: A block class name available there
            scale: downscale and upscale ratio deeper U-Net blocks
            pool_fn: pooling layer constructor from pytorch_named_dims.nm
            in_channels: number of input channels
            encoder_block_kwargs: keyword arguments for each encoder block
            decoder_block_kwargs: keyword arguments for each decoder block
        """
        super().__init__()
        self.features = features
        self.stump = nm.Conv2d(in_channels, features, kernel_size=3)
        encoder_block_fn = getattr(import_module(__name__), encoder_block_type)
        decoder_block_fn = getattr(import_module(__name__), decoder_block_type)

        self.encoder_blocks = nn.ModuleList(
            [
                encoder_block_fn(
                    in_features=features * 2**idx, out_features=2 * features * 2**idx, **encoder_block_kwargs
                ) for idx in range(num_blocks)
            ]
        )

        self.pad = nm.ReflectionPad2d(
            sum([2 if getattr(b, 'skip_padding', False) else 0 for b in self.encoder_blocks]) + 2
        )
        self.pool = getattr(nm, pool_fn)(scale) if scale > 1 else nn.Sequential()

        self.decoder_blocks = nn.ModuleList(
            [
                decoder_block_fn(
                    in_features=features * 2**idx * (2 if idx == num_blocks - 1 else 4),
                    out_features=features * 2**idx,
                    **decoder_block_kwargs
                ) for idx in reversed(range(num_blocks))
            ]
        )

        self.head = nm.Conv2d(features * 2, features, kernel_size=1)

    def forward(self, x):
        x = self.pad(x)
        x = self.stump(x)
        outs = [x]

        for idx, block in enumerate(self.encoder_blocks):
            x = block(x)
            x = self.pool(x)
            if idx < len(self.encoder_blocks) - 1:
                outs.append(x)

        for block, out in zip(self.decoder_blocks, reversed(outs)):
            out_size = tuple(d + 2 if getattr(block, 'skip_padding', False) else d for d in out.shape[-2:])
            x = x.rename(None)  # interpolate needs not named tensor
            x = functional.interpolate(x, out_size, mode='bilinear', align_corners=True).rename('N', 'C', 'H', 'W')
            x = block(x)
            x = torch.cat((x, out), dim='C')

        return self.head(x) / sqrt(self.features)
