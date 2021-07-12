import logging
import os
from typing import Callable, Dict, Optional

import h5py as h5
import numpy as np
import PIL.Image
import torch
import tqdm
from ignite.engine import Engine, Events
from sklearn.decomposition import PCA

from dataflow.utils import get_img_list_from_folder
from utils.predictor import Predictor
from utils.restoration import choose_best_weights


def pca_projection(feature_map: torch.Tensor, mask: torch.Tensor, num_components: int):
    """
    Args:
        feature_map: tensor of shape ['C', 'H', 'W']
        mask: tensor of shape ['H', 'W']
        num_components: number of components after PCA

    Returns:
        Grid projected into grid with a lower number of features
    """
    projector = PCA(n_components=num_components)
    _, height, width = feature_map.shape
    nonzero_h, nonzero_w = torch.nonzero(mask.rename(None), as_tuple=True)
    feature_vectors = feature_map.rename(None)[:, nonzero_h, nonzero_w].transpose(0, 1).cpu()
    x_projected = projector.fit_transform(feature_vectors)
    grid = torch.zeros(num_components, height, width)
    grid[:, nonzero_h, nonzero_w] = torch.tensor(x_projected, dtype=torch.float).transpose(1, 0)
    grid = grid.permute(1, 2, 0).numpy()
    return grid


def store_small_images_with_feature_maps(
    predictor: Predictor,
    images_dir: str,
    num_components: Optional[int] = None,
    out_file_path: str = 'images_with_featuremaps.h5'
):
    """Compute feature maps for images from given directory and store it in hdf5 file
    Args:
        predictor: a callable class which transform a PIL.Image into a feature map
        images_dir: directory with images
        num_components: number of channels after PCA projection
        out_file_path: output file path
    """
    image_paths = get_img_list_from_folder(images_dir)
    with h5.File(out_file_path, 'w') as hf:
        for img_path in tqdm.tqdm(image_paths, unit='img', desc=f'feature maps computations'):
            img_name = ''.join(img_path.split('/')[-1].split('.')[:-1])
            img_raw = PIL.Image.open(img_path).convert('RGB')
            feature_map = predictor(img_raw)
            channels, f_map_height, f_map_width = feature_map.shape
            mask = torch.ones(feature_map.shape[-2:], dtype=torch.bool)
            feature_map = feature_map.cpu()
            if num_components:
                feature_map = pca_projection(feature_map, mask, min(channels, num_components))
            else:
                logging.info('PCA projection will be skipped')
                feature_map = feature_map.align_to('H', 'W', 'C').numpy()
            img_raw = img_raw.resize((f_map_width, f_map_height))
            hf.create_dataset(f'{img_name}_rgb', data=np.array(img_raw))
            hf.create_dataset(f'{img_name}_features', data=feature_map)


class FeatureMapSaver:
    def __init__(
        self,
        predictor: Predictor,
        use_metrics: Dict[str, Callable[[float], bool]],
        artifacts_dir: str,
        visualization_images_dir: str,
        num_features: Optional[int] = 16
    ):
        """
        Args:
            predictor: Callable predictor
            use_metrics: list of metrics with functions that choose the best value
            artifacts_dir: directory with stored models
            num_features: number of features after PCA
        """
        self.predictor = predictor
        self.use_metrics = use_metrics
        self.artifacts_dir = artifacts_dir
        self.visualization_images_dir = visualization_images_dir
        self.num_features = num_features

    def attach(self, engine: Engine):
        """
        Args:
            engine: training engine

        Notes:
            attach to the engine complete and exception events has to be attached after checkpointer
        """
        engine.add_event_handler(Events.COMPLETED, self.on_training_end)
        engine.add_event_handler(Events.EXCEPTION_RAISED, self.on_training_end)

    def on_training_end(self, *_):
        metrics_with_the_best_weights = choose_best_weights(self.artifacts_dir, self.use_metrics)
        logging.info(f'found best weights: {metrics_with_the_best_weights}')
        old_state_dict = self.predictor.model.state_dict()
        for metric_name, weights in metrics_with_the_best_weights.items():
            logging.info(f'loading best weights {weights}')
            state_dict = torch.load(os.path.join(self.artifacts_dir, weights))
            self.predictor.model.load_state_dict(state_dict)
            out_path = os.path.join(self.artifacts_dir, f'model_best_{metric_name}_predictions.h5')
            store_small_images_with_feature_maps(
                self.predictor, self.visualization_images_dir, self.num_features, out_path
            )
        self.predictor.model.load_state_dict(old_state_dict)
