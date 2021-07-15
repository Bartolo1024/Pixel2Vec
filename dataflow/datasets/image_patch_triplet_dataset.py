from typing import Any, Callable, List, Tuple

import PIL.Image
import torch
import tqdm
from torch.utils.data import Dataset

import dataflow.utils


def clone(p: torch.Tensor) -> torch.Tensor:
    return p.clone()


class ImagePatchTripletDatset(Dataset):
    def __init__(self,
                 image_paths: List[str],
                 patch_size: Tuple[int, int],
                 transform: Callable[[Any], torch.Tensor],
                 patch_transform: Callable[[torch.Tensor],
                                           torch.Tensor] = clone,
                 convert_to: str = 'RGB',
                 min_negative_distance: int = 1,
                 min_positive_distance: int = 0,
                 max_positive_distance: int = 1,
                 max_negative_retries: int = 100,
                 skip_background: bool = False,
                 repeat_transforms: bool = False):
        """
        Args:
            image_paths: path to images
            patch_size: size of extracted anchors, positives and negatives
            transform: transform for image that is performed once per iteration
            patch_transform: transform performed on each patch during __getitem__ method call
            convert_to: format of an image ex. RGB
            min_negative_distance: minimum distance of a negative patch
            min_positive_distance: minimum distance of a positive patch
            max_positive_distance: maximum distance of a positive patch
            skip_background: skip detected background and sample only from race tracks
        """
        self.transform = transform
        self.convert_to = convert_to
        self.patch_size = patch_size
        self.min_positive_distance = min_positive_distance
        self.min_negative_distance = min_negative_distance
        self.max_positive_distance = max_positive_distance
        self.max_negative_retries = max_negative_retries
        self.skip_background = skip_background
        self.patch_transform = patch_transform
        self.patch_counter = 0
        self.images = [self.open_image(path) for path in image_paths]
        self.grids: List[torch.Tensor] = []
        self.repeat_transforms = repeat_transforms
        self.transformed_images = [self.transform(img) for img in self.images]
        self.pre_compute_grids()

    def reset(self):
        """Function that performs transform again on start of each epoch"""
        if self.repeat_transforms:
            self.transformed_images = [
                self.transform(img) for img in self.images
            ]
            self.pre_compute_grids()
        self.patch_counter = 0

    def pre_compute_grids(self):
        """Compute grids and create item idx, to grid patch mapping hashtable
        Notes :
            Named tensors do not support serialization, so grid has None names
        """
        self.grids: List[torch.Tensor] = []
        self.idx_mapping = {}
        patch_idx = 0
        for grid_idx, img in tqdm.tqdm(enumerate(self.transformed_images),
                                       desc='Prepare grids'):
            grid = dataflow.utils.extract_patches_from_tensor(img,
                                                              self.patch_size,
                                                              flatten=False)
            self.grids.append(grid)
            for h_idx, row in enumerate(grid):
                for w_idx, col in enumerate(row):
                    mapping = {
                        'grid_idx': grid_idx,
                        'h_idx': h_idx,
                        'w_idx': w_idx,
                    }
                    self.idx_mapping[patch_idx] = mapping
                    patch_idx += 1
        self.flattened_grids = [grid.flatten(0, 1) for grid in self.grids]
        self.max_patches = patch_idx

    def __getitem__(
            self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate triplets for provided index
        Returns:
            anchor defined by the index, positive patch that is close to the anchor, and negative patch that is far
        """
        if self.patch_counter > self.max_patches:
            self.reset()
        mapping = self.idx_mapping[idx]
        grid = self.grids[mapping['grid_idx']]
        flattened_grid = self.flattened_grids[mapping['grid_idx']]
        grid_size = grid.shape[:2]
        anchor = grid[mapping['h_idx'], mapping['w_idx']]
        anchor_idx = mapping['h_idx'] * grid.shape[1] + mapping['w_idx']
        positive_patch_idxs = dataflow.utils.random_index_from_area(
            anchor_idx,
            grid.shape[:2],
            min_radius=self.min_positive_distance,
            max_radius=self.max_positive_distance)
        positive = flattened_grid[positive_patch_idxs]
        negative_patch_idx = dataflow.utils.random_index_from_area(
            anchor_idx,
            grid_size,
            min_radius=self.min_negative_distance,
            max_radius=max(grid.shape[:2]))
        negative = flattened_grid[negative_patch_idx]
        return self.patch_transform(anchor), self.patch_transform(
            positive), self.patch_transform(negative)

    def open_image(self, path):
        """Open image and convert to format specified in the constructor"""
        return PIL.Image.open(path).convert(self.convert_to)

    def __len__(self) -> int:
        """
        Returns:
            num of patches counted in the pre_compute_grids function
        """
        return self.max_patches
