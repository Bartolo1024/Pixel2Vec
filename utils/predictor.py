from typing import Any, Dict

import torch
from torch import nn

from dataflow.utils import (
    extract_patches_from_pil_image, extract_patches_from_tensor, merge_feature_maps, merge_grid_patches
)
from utils.restoration import get_transforms


class Predictor:
    def __init__(
        self,
        model: nn.Module,
        mode: str,
        data_flow_spec: Dict[str, Any],
        device: torch.device = torch.device('cpu'),
    ):
        assert mode in ('images', 'patches', 'sliced_images')
        self.model = model
        self.mode = mode
        self.transform, _ = get_transforms(data_flow_spec)
        self.device = device
        self.chunk_size = data_flow_spec['params']['batch_size']
        self.patch_size = data_flow_spec['params'].get('patch_size')
        self.patch_coverage = data_flow_spec['params'].get('patch_coverage')
        if mode == 'patches' or mode == 'sliced_images':
            assert self.patch_size is not None

    def __call__(self, x):
        self.model = self.model.to(self.device)
        self.model.eval()
        if isinstance(x, list):
            ret = [self.__call__(img).rename(None) for img in x]
            return torch.stack(ret).rename('N', 'C', 'H', 'W')
        if self.mode == 'images':
            x = self.transform(x).rename('C', 'H', 'W')
            return self.single_inference(x)
        elif self.mode == 'patches':
            x = self.transform(x).rename('C', 'H', 'W')
            return self.predict_on_patches(x)
        elif self.mode == 'sliced_images':
            return self.predict_on_sliced_images(x)
        raise NotImplementedError

    def single_inference(self, img):
        with torch.no_grad():
            return self.model(img.align_to('N', 'C', 'H', 'W').to(self.device)).squeeze('N')

    def predict_on_patches(self, img):
        patches = extract_patches_from_tensor(img, self.patch_size, flatten=False)
        grid_size = patches.shape[:2]
        patches = patches.flatten(['grid_height', 'grid_width'], 'N')
        batch_size = patches.shape[0]
        num_chunks = batch_size // self.chunk_size
        ret = []
        with torch.no_grad():
            for chunk in patches.chunk(num_chunks):
                result = self.model(chunk.to(self.device))
                ret.extend(result.unbind())
        ret = merge_grid_patches(ret, grid_size)
        return ret.rename('C', 'H', 'W')

    def predict_on_sliced_images(self, img):
        img_size = img.size
        patches = extract_patches_from_pil_image(img, self.patch_size, self.patch_coverage)
        feature_maps = []
        with torch.no_grad():
            for patch in patches:
                patch = self.transform(patch).rename('C', 'H', 'W')
                patch = patch.align_to('N', 'C', 'H', 'W').to(self.device)
                feature_map = self.model(patch)
                feature_maps.append(feature_map.cpu())
        return merge_feature_maps(feature_maps, img_size, self.patch_size, self.patch_coverage).rename('C', 'H', 'W')
