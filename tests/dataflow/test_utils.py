import numpy as np
import torch

from pixel2vec.dataflow import utils


def test_extract_patches_from_tensor():
    height, width = 4, 4
    patch_size = 2, 2
    img = torch.stack([torch.arange(height * width).reshape(height, width) for _ in range(3)])
    grid = utils.extract_patches_from_tensor(img, patch_size)
    target_array = torch.stack([torch.tensor([[0, 1], [4, 5]]) for _ in range(3)])
    assert np.array_equal(grid[0].numpy(), target_array)
    target_array = torch.stack([torch.tensor([[2, 3], [6, 7]]) for _ in range(3)])
    assert np.array_equal(grid[1].numpy(), target_array)


def test_set_adjacent_to_ones():
    target = [
        [0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]
    ]
    target = np.array(target)
    result = np.zeros_like(target)
    utils.set_adjacent(result, 24, value=1, radius=2)
    utils.set_adjacent(result, 24, value=0, radius=1)
    assert np.allclose(result, target)
