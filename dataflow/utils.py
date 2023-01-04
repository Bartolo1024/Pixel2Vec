import math
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import sklearn.feature_extraction.image as sk_ext_image
import torch
import torchvision.transforms
from numpy import ndarray
from torch.nn import functional as F

import dataflow.transforms

_IMG_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']


def extract_patches_from_numpy_array(img: ndarray, patch_size: Tuple[int, int]) -> List[ndarray]:
    """
    shrink image into patches with no intersection area
    unfortunately extract_patches_2d from sklearn works randomly
    """
    assert img.shape[-1] == 3
    patch_shape = [*patch_size, img.shape[-1]]
    grid = sk_ext_image.extract_patches(img, patch_shape, extraction_step=patch_shape).reshape(-1, *patch_shape)
    return grid


def extract_patches_from_tensor(
    img: torch.Tensor,
    patch_size: Tuple[int, int],
    step: Optional[Tuple[int, int]] = None,
    pad: Optional[List[int]] = None,
    flatten: bool = True
) -> torch.Tensor:
    """
    Args:
        img tensor: in CHW format
        patch_size: size of patch in HW format
        step: step od between next patches
        pad: padding for the image
        flatten: flatten ['grid_height', 'grid_width'] to ['N']

    Returns:
        tensor of shape ['grid_height', 'grid_width', 'C', 'H', 'W'] or ['N', 'C', 'H', 'W'],
        by default 'grid_height' and 'grid_width' are flattened to 'N'
        N is number of all patches, non full-size patches are skipped
    """
    assert len(img.shape) == 3
    patch_height, patch_width = patch_size
    h_step, w_step = step if step else patch_size
    img = F.pad(img, pad) if pad else img
    names = img.names
    img = img.rename(None)
    grid = img.unfold(1, patch_height, h_step).unfold(2, patch_width, w_step)
    grid = grid.permute(1, 2, 0, 3, 4)
    if all([n is None for n in names]) and flatten:
        return grid.reshape(-1, 3, *patch_size)
    elif all([n is None for n in names]):
        return grid
    elif flatten:
        return grid.reshape(-1, 3, *patch_size).rename('N', 'C', 'H', 'W')
    else:
        return grid.rename('grid_height', 'grid_width', 'C', 'H', 'W')


def merge_grid_patches(grid: List[torch.Tensor], grid_size: Tuple[int, int]) -> torch.Tensor:
    """
    Args:
        grid: List of tensors of shape ['C', 'H', 'W']
        img_shape: shape of the output image

    Returns:
        image with merged patches
    """
    grid_h, grid_w = grid_size
    num_channels, patch_h, patch_w = grid[0].shape
    max_h = grid_h // patch_h
    max_w = grid_w // patch_w
    out = torch.zeros((num_channels, grid_h, grid_w))
    for idx, patch in enumerate(grid):
        w_idx = idx % max_w
        h_idx = (idx // max_w) % max_h
        out[:, h_idx * patch_h:(h_idx + 1) * patch_h, w_idx * patch_w:(w_idx + 1) * patch_w] = patch.rename(None)
    return out


def set_adjacent(arr: Union[np.ndarray, torch.Tensor], patch_idx: int, value: int, radius: int = 1) -> np.ndarray:
    """Set cells that are adjacent to the given index to the given value"""
    num_of_patches_in_column, num_of_patches_in_row = arr.shape
    patch_column_idx = patch_idx // num_of_patches_in_row
    patch_row_idx = patch_idx % num_of_patches_in_row
    arr[max(patch_column_idx - radius, 0):min(patch_column_idx + 1 + radius, num_of_patches_in_column),
        max(patch_row_idx - radius, 0):min(patch_row_idx + 1 + radius, num_of_patches_in_row)] = value
    return arr


def random_index_from_area(
    patch_idx: int,
    max_patches: Tuple[int, int],
    min_radius: int = 0,
    max_radius: int = 1,
    mask: Optional[np.ndarray] = None,
) -> int:
    """
    Args:
        patch_idx: idx of chosen patch in 1d array:
        max_patches: num of patches in columns and rows
        min_radius: min distance of chosen patch
        max_radius: max distance of chosen patch
        mask: additional mask for area

    Returns:
        idxes in 1d array of random patch from area defined by distance range and mask
    """
    num_of_patches_in_column, num_of_patches_in_row = max_patches
    idxes = np.arange(num_of_patches_in_column * num_of_patches_in_row)
    flattened_weights = np.zeros(num_of_patches_in_column * num_of_patches_in_row, dtype=bool)
    weights = flattened_weights.reshape(num_of_patches_in_column, num_of_patches_in_row)
    weights = set_adjacent(weights, patch_idx, 1, max_radius)
    weights = set_adjacent(weights, patch_idx, 0, min_radius)
    if mask is not None and np.sum(weights * mask) > 0:
        weights[~mask] = False
    return np.random.choice(idxes[flattened_weights])


def compute_max_patches(img_size: Tuple[int, int], patch_size: Tuple[int, int]):
    """Compute patches count in column and row"""
    height, width = img_size
    patch_height, patch_width = patch_size
    max_column_patches = height // patch_height
    max_row_patches = width // patch_width
    return max_column_patches, max_row_patches


def get_img_list_from_folder(img_folder: str, img_extensions: Optional[List[str]] = None) -> List[str]:
    """Scan folder end return all images with specified extension"""
    paths = []
    if img_extensions is None:
        img_extensions = _IMG_EXTENSIONS
    for r, d, f in os.walk(img_folder):
        for file in f:
            if file.split('.')[-1] in img_extensions:
                paths.append(os.path.join(r, file))
    return paths


def read_images_from_folder(img_folder: str,
                            img_extensions: Optional[List[str]] = None,
                            convert_to: str = 'RGB') -> List[PIL.Image.Image]:
    img_paths = get_img_list_from_folder(img_folder, img_extensions)
    images = [PIL.Image.open(path).convert(convert_to) for path in img_paths]
    return images


class PrepareImagePatchTriplets:
    def __init__(
        self, device: torch.device, transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None, **kwargs
    ):
        """
        :param device - cuda, mps or cpu torch device:
        :param transform - function for an each patch postprocessing:
        :param kwargs - keyword arguments for the triplet generator:
        """
        self.triplets_generator = dataflow.transforms.GenerateRandomTriplets(**kwargs)
        self.patch_transform = transform if transform else torchvision.transforms.Compose(
            [
                dataflow.transforms.RandomVerticalFlip(),
                dataflow.transforms.RandomHorizontalFlip(),
                dataflow.transforms.RandomBrightnessContrastAdjust(),
            ]
        )
        self.device = device

    def __call__(self, batch: List[torch.Tensor], *_, **__) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate triplets for all images, then concatenate it to three tensors"""
        triplets_batch = [self.triplets_generator(img) for img in batch]
        ret = {}
        for group_name in ['anchor', 'positive', 'negative']:
            group = [triplet[group_name] for img_triplets in triplets_batch for triplet in img_triplets]
            group = [self.patch_transform(s) for s in group]
            group = torch.stack(group).to(self.device)
            ret[group_name] = group
        return ret['anchor'], ret['positive'], ret['negative']


def extract_patches_from_pil_image(img: PIL.Image.Image, patch_size: Tuple[int, int],
                                   patch_coverage: int) -> List[PIL.Image.Image]:
    """
    Args:
        img: PIL image
        patch_size: tuple with size of patches
        patch_coverage: intersection between patches

    Returns:
        list of PIL images (patches) that cover all input images
    """
    patches = []
    patch_height, patch_width = patch_size
    w_step = patch_width - patch_coverage
    h_step = patch_height - patch_coverage
    width, height = img.size
    extended_width = math.ceil(width / w_step) * w_step
    extended_height = math.ceil(height / h_step) * h_step
    for start_col, end_col in zip(range(0, extended_width, w_step), range(w_step, extended_width + w_step, w_step)):
        for start_row, end_row in zip(
            range(0, extended_height, h_step), range(patch_height, extended_height + h_step, h_step)
        ):
            patch = img.crop((start_col, start_row, min(end_col, width), min(end_row, height)))
            patches.append(patch)
    return patches


def merge_feature_maps(
    feature_maps: List[torch.Tensor], img_size: Tuple[int, int], patch_size: Tuple[int, int], patch_coverage: int
):
    """
    Args:
        feature_maps: list of feature maps of shape ['C', 'H', 'W'] each
        img_size: size of the output image
        patch_size: size of one patch
        patch_coverage: coverage between patches

    Returns:
        one tensor merged from provided feature maps

    Notes:
        TODO: implement patches blending
    """
    features = feature_maps[0].shape[1]
    width, height = img_size
    ret = torch.zeros(features, height, width, dtype=torch.float)
    patch_height, patch_width = patch_size
    w_step = patch_width - patch_coverage
    h_step = patch_height - patch_coverage
    extended_width = math.ceil(width / w_step) * w_step
    extended_height = math.ceil(height / h_step) * h_step
    ctr = 0
    for start_col, end_col in zip(range(0, extended_width, w_step), range(w_step, extended_width + w_step, w_step)):
        for start_row, end_row in zip(
            range(0, extended_height, h_step), range(patch_height, extended_height + h_step, h_step)
        ):
            feature_map = feature_maps[ctr]
            ret[:, start_row:min(end_row, height), start_col:min(end_col, width)] = feature_map.rename(None)
            ctr += 1
    return ret
