import logging
import os
from typing import Any, Dict, Optional, Union

import numpy as np
import PIL.Image
import torch
import tqdm
from ignite.engine import Engine, Events
from matplotlib import pyplot as plt
from pytorch_named_dims import nm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from dataflow.utils import get_img_list_from_folder
from utils.predictor import Predictor
from utils.restoration import choose_best_weights


def get_projection(feature_map: torch.Tensor, projector: Union[TSNE, PCA]):
    """
        Args:
            feature_map: feature map of shape ['C', 'H', 'W']
            projector: projector from sklearn with fit_transform function

        Returns:
            PIL image with t-sne visualization
        """
    _, height, width = feature_map.shape
    feature_vectors = feature_map.flatten(['H', 'W'], 'N').transpose('C', 'N').cpu().numpy()
    x_projected = projector.fit_transform(feature_vectors)
    grid = x_projected.reshape(height, width, 3)
    grid_min = grid.min()
    grid_max = grid.max()
    projection_img = ((grid - grid_min) / (grid_max - grid_min) * 255).astype(np.uint8)
    projection_img = PIL.Image.fromarray(projection_img).resize((width, height), resample=PIL.Image.NEAREST)
    return projection_img


def get_tsne_rgb(feature_map: torch.Tensor):
    """
    Args:
        feature_map: feature map of shape ['C', 'H', 'W']

    Returns:
        PIL image with t-sne visualization
    """
    tsne = TSNE(n_components=3, init='pca', random_state=0)
    return get_projection(feature_map, tsne)


def get_pca_rgb(feature_map: torch.Tensor):
    """
    Args:
        feature_map: feature map of shape ['C', 'H', 'W']

    Returns:
        PIL image with pca visualization
    """
    pca = PCA(n_components=3, random_state=0)
    return get_projection(feature_map, pca)


class ReducedDimensionsRGBVisualizer:
    def __init__(
        self,
        predictor: Predictor,
        use_metrics: Dict[str, Any],
        artifacts_dir: str,
        visualization_images_dir: str,
        out_image_scale: float = .12,
        reduction_mode: str = 't-SNE',
        downsampling_kernel: Optional[int] = None,
    ):
        """
        Args:
            use_metrics: list of metrics with functions that choose the best value
            artifacts_dir: directory with stored models
            visualization_images_dir: dictionary with images for computations
            out_image_scale: scale of output matplotlib figure
            reduction_mode: t-SNE or PCA
            downsampling_kernel: downsample feature map in order to reduce computation time
        """
        self.use_metrics = use_metrics
        self.artifacts_dir = artifacts_dir
        self.visualization_images_dir = visualization_images_dir
        self.out_image_scale = out_image_scale
        assert reduction_mode in ('t-SNE', 'PCA')
        self.reduction_mode = reduction_mode
        self.downsampling_kernel = nm.AvgPool2d(downsampling_kernel) if downsampling_kernel else None
        self.feature_map_reduction_fn = get_tsne_rgb if reduction_mode == 't-SNE' else get_pca_rgb
        self.predictor = predictor

    def attach(self, engine: Engine):
        """
        Args:
            engine: training engine

        Notes:
            attach to the engine complete and exception events has to be attached after checkpointer
        """
        engine.add_event_handler(Events.COMPLETED, self.choose_models_and_compute)
        engine.add_event_handler(Events.EXCEPTION_RAISED, self.choose_models_and_compute)

    def choose_models_and_compute(self, *_):
        metrics_with_the_best_weights = choose_best_weights(self.artifacts_dir, self.use_metrics)
        logging.info(f'found best weights: {metrics_with_the_best_weights}')
        old_state_dict = self.predictor.model.state_dict()
        for metric_name, weights in metrics_with_the_best_weights.items():
            logging.info(f'loading best weights {weights}')
            state_dict = torch.load(os.path.join(self.artifacts_dir, weights))
            self.predictor.model.load_state_dict(state_dict)
            self.compute_and_store_rgb_projection(f'best_{metric_name}')
        self.predictor.model.load_state_dict(old_state_dict)

    def compute_and_store_rgb_projection(self, model_name: str):
        """Computes rgb visualization for provided model
        Args:
            model_name: model identifier
        """
        image_paths = get_img_list_from_folder(self.visualization_images_dir)
        os.makedirs(os.path.join(self.artifacts_dir, f'{self.reduction_mode}-rgb', model_name), exist_ok=True)
        for img_path in tqdm.tqdm(
            image_paths, unit='img', desc=f'{self.reduction_mode} computations for: {model_name}'
        ):
            img_name = ''.join(img_path.split('/')[-1])
            raw_img = PIL.Image.open(img_path).convert('RGB')
            feature_map = self.predictor(raw_img)
            if self.downsampling_kernel:
                feature_map = self.downsampling_kernel(feature_map.align_to('N', 'C', 'H', 'W')).squeeze('N')
                logging.info(f'Reduction will be performed with downsampled feature map of shape {feature_map.shape}')
            projected_img = self.feature_map_reduction_fn(feature_map)
            self.save_matplotlib_results(raw_img, projected_img, model_name, img_name)

    def save_matplotlib_results(self, source_img: PIL.Image, t_sne_img: PIL.Image, model_name: str, img_name: str):
        """
        Args:
            source_img: input rgb image
            t_sne_img: image with 3 dimensional t-SNE visualization
            model_name: model identifier
            img_name: name of output image
        """
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(source_img)
        ax[0].set_title('input image')
        ax[0].set_axis_off()
        ax[1].imshow(t_sne_img)
        ax[1].set_title(f'{self.reduction_mode}')
        ax[1].set_axis_off()
        w, h = t_sne_img.size
        w_inches = int(w * self.out_image_scale)
        h_inches = int(h * self.out_image_scale)
        fig.set_size_inches(w_inches, h_inches, forward=True)
        out_path = os.path.join(self.artifacts_dir, f'{self.reduction_mode}-rgb', model_name, img_name)
        plt.savefig(out_path, bbox_inches='tight')
