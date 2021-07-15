from __future__ import annotations

from typing import Any, Callable, List

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageListDataset(Dataset):
    def __init__(self,
                 image_paths: List[str],
                 transform: Callable[[Any], torch.Tensor],
                 convert_to: str = 'RGB'):
        """Dataset store all images paths and load it with PIL in runtime."""
        self.transform = transform
        self.image_paths = image_paths
        self.convert_to = convert_to

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Args:
            idx: index in paths

        Returns:
            transformed image
        """
        img = self.open_image(self.image_paths[idx])
        img = self.transform(img)
        return img

    def open_image(self, path):
        """Open image and convert to format specified in the constructor"""
        return Image.open(path).convert(self.convert_to)

    def __len__(self) -> int:
        return len(self.image_paths)


class ImageListInMemoryDataset(Dataset):
    def __init__(self, images: List[Image.Image],
                 transform: Callable[[Any], torch.Tensor]):
        """Dataset which store all images in memory"""
        self.images = images
        self.transform = transform

    @classmethod
    def from_paths(cls,
                   image_paths: List[str],
                   transform: Callable[[Any], torch.Tensor],
                   convert_to: str = 'RGB') -> ImageListInMemoryDataset:
        images = [Image.open(path).convert(convert_to) for path in image_paths]
        return ImageListInMemoryDataset(images=images, transform=transform)

    @classmethod
    def from_pil_images(
            cls, images: List[Image.Image],
            transform: Callable[[Any],
                                torch.Tensor]) -> ImageListInMemoryDataset:
        return ImageListInMemoryDataset(images=images, transform=transform)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        :param idx - index in paths:
        :return image tensor CHW:
        """
        img = self.images[idx]
        img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.images)
