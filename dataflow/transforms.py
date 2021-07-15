import logging
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torchvision.transforms

from dataflow import utils


class InvertNormalization:
    """Invert normalization - convert to 0., 1. float tensor image"""
    def __init__(self, normalization_stats: Dict[str, Tuple[float]], device: Optional[torch.device] = None):
        self.mean = torch.tensor(normalization_stats['mean']).to(device)
        self.std = torch.tensor(normalization_stats['std']).to(device)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Take CHW tensor and returns non normalized tensor with the same shape"""
        return ((img * self.std[:, None, None] + self.mean[:, None, None])).clamp(0, 1)


class GenerateRandomTriplets:
    """Convert one big image to a lot of random triplets witch anchor, a near patch and a patch which is far away"""
    def __init__(
        self,
        num_of_triplets: int,
        patch_size: Tuple[int, int],
        min_negative_distance: int = 1,
        min_positive_distance: int = 0,
        max_positive_distance: int = 1,
        skip_equal_negatives: bool = True,
        max_negative_retries: int = 100,
        skip_background: bool = False
    ):
        """
        :param num_of_triplets - number of generated triplets:
        :param patch_size - size of one img patch:
        :param min_negative_distance - min negative distance according to patch size:
        :param max_positive_distance - max positive distance according one patch patch size:
        """
        self.num_of_triplets = num_of_triplets
        self.patch_size = patch_size
        self.min_positive_distance = min_positive_distance
        self.min_negative_distance = min_negative_distance
        self.max_positive_distance = max_positive_distance
        self.skip_equal_negatives = skip_equal_negatives
        self.max_negative_retries = max_negative_retries
        self.skip_background = skip_background

    def __call__(self, img: torch.Tensor):
        """Split image into grid and sample random triplets from it"""
        _, img_h, img_w = img.shape
        grid = utils.extract_patches_from_tensor(img, self.patch_size)
        grid_idxes = list(range(len(grid)))
        max_column_patches, max_row_patches = utils.compute_max_patches((img_h, img_w), self.patch_size)
        padded_grid_idxes: List[int] = self._remove_margins(
            grid_idxes, max_column_patches=max_column_patches, max_row_patches=max_row_patches
        )
        anchor_idxes = random.sample(padded_grid_idxes, min(self.num_of_triplets, len(padded_grid_idxes)))
        samples: List[Dict[str, torch.Tensor]] = []
        for anchor_idx in anchor_idxes:
            positive_patch_idx = utils.random_index_from_area(
                anchor_idx, (max_column_patches, max_row_patches),
                min_radius=self.min_positive_distance,
                max_radius=self.max_positive_distance
            )
            negative_patch_idx = self.choose_negative_patch(anchor_idx, max_column_patches, max_row_patches, grid)
            samples.append(
                {
                    'anchor': grid[anchor_idx],
                    'positive': grid[positive_patch_idx],
                    'negative': grid[negative_patch_idx],
                }
            )
        return samples

    @staticmethod
    def _remove_margins(grid_idxes: List[Any], max_column_patches: int, max_row_patches: int) -> List[Any]:
        """Remove margin regions from flattened grid (grid indexes)"""
        grid_idxes = grid_idxes[max_row_patches:-max_row_patches]
        grid_idxes = grid_idxes[max_column_patches::]
        return grid_idxes

    def choose_negative_patch(
        self,
        anchor_idx: int,
        max_column_patches: int,
        max_row_patches: int,
        grid: torch.Tensor,
    ):
        negative_patch_idx = anchor_idx
        ctr = 0
        while torch.allclose(grid[negative_patch_idx], grid[anchor_idx]):
            negative_patch_idx = utils.random_index_from_area(
                anchor_idx, (max_column_patches, max_row_patches),
                min_radius=self.min_negative_distance,
                max_radius=max(max_column_patches, max_row_patches)
            )
            ctr += 1
            if ctr > self.max_negative_retries:
                logging.warning(
                    f'Patch was skipped {self.max_negative_retries} times - values were the same sa in anchor'
                )
                break
        return negative_patch_idx


class ExtractTripletsFromImage:
    def __init__(self, patch_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None, **kwargs):
        """
        :param transform - function for an each patch postprocessing:
        :param kwargs - keyword arguments for the triplet generator:
        """
        self.triplets_generator = GenerateRandomTriplets(**kwargs)
        self.patch_transform = patch_transform if patch_transform else torchvision.transforms.Compose(
            [
                RandomVerticalFlip(),
                RandomHorizontalFlip(),
                RandomBrightnessContrastAdjust(),
            ]
        )

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate triplets for all images, then concatenate it to three tensors"""
        triplets = self.triplets_generator(img)
        ret = {}
        for group_name in ['anchor', 'positive', 'negative']:
            group = [triplet[group_name] for triplet in triplets]
            group = [self.patch_transform(s) for s in group]
            group = torch.stack(group)
            ret[group_name] = group
        return ret['anchor'], ret['positive'], ret['negative']


class RandomHorizontalFlip:
    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def __call__(self, x: torch.Tensor):
        """
        :param x tensor CHW:
        :return maybe flipped tensor:
        """
        assert len(x.shape) == 3
        return x.flip(2) if random.random() < self.probability else x


class RandomVerticalFlip:
    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x tensor CHW:
        :return maybe flipped tensor CHW:
        """
        assert len(x.shape) == 3
        return x.flip(1) if random.random() < self.probability else x


class RandomBrightnessContrastAdjust:
    def __init__(
        self,
        alpha_range: Tuple[float, float] = (0.9, 1.1),
        beta_range: Tuple[float, float] = (0., 0.1),
        clamp_to_range: Optional[Tuple[float, float]] = None
    ):
        """
        Args:
            alpha_range: range of image random gain
            beta_range: range of image random bias
            clamp_to_range: on the end clamp image values to chosen range
        """
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.clamp_to_range = clamp_to_range

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image tensor of shape ['C', 'H', 'W']
        Returns:
            image with changed contrast
        """
        img = img.clone()
        min_alpha, max_alpha = self.alpha_range
        img *= random.random() * (max_alpha - min_alpha) + min_alpha
        min_beta, max_beta = self.beta_range
        img += random.random() * (max_beta - min_beta) + min_beta * torch.mean(img)
        if self.clamp_to_range:
            return img.clamp(*self.clamp_to_range)
        return img


class RandomRotateStraightAngle:
    """Random rotation 90 or -90 degrees"""
    def __init__(self, probability: float = .5, negative_angle_probability: float = .5):
        self.probability = probability
        self.negative_angle_probability = negative_angle_probability

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.probability:
            return x
        x = x.transpose(1, 2)
        return x.flip(2) if random.random() < self.negative_angle_probability else x.flip(1)


class BlotRightBottomWithAverageEdgeValue:
    def __init__(self, crop_percentage: float):
        """
        Args:
            crop_percentage: percentage of the area of cropped rectangle according to whole image
        """
        assert crop_percentage > 0 and crop_percentage < 1.
        self.crop_percentage = crop_percentage

    def __call__(self, x):
        edge_value = (x[:, 0, :-1].mean() + x[:, :-1, -1].mean() + x[:, -1, 1:].mean() + x[:, 1:, 0].mean()) / 4
        _, h, w, = x.shape
        start_vertical = int(h * math.sqrt(self.crop_percentage))
        start_horizontal = int(w * math.sqrt(self.crop_percentage))
        x[:, start_vertical:, start_horizontal:] = edge_value
        return x


class CropRightBottom:
    def __init__(self, crop_percentage: float):
        """
        Args:
            crop_percentage: percentage of the area of cropped rectangle according to whole image
        """
        assert crop_percentage > 0 and crop_percentage < 1.
        self.crop_percentage = crop_percentage

    def __call__(self, x):
        _, h, w, = x.shape
        start_vertical = int(h * math.sqrt(self.crop_percentage))
        start_horizontal = int(w * math.sqrt(self.crop_percentage))
        return x[:, start_vertical:, start_horizontal:]
