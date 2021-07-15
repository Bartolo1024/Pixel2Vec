from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from ignite.metrics.metric import Metric
from torch.nn import functional as F

from utils.annotation_samplers import MaskRandomSampler, SkeletonSampler
from utils.centres_distance import (choose_best_idxes_with_cosine_similarity,
                                    choose_best_idxes_with_euclidean_distances)

EvaluatorOutputType = Tuple[torch.Tensor, List[List[Dict[str, Any]]]]


class SparseAnnotationAccuracy(Metric):
    def __init__(self,
                 similarity_mode: str = 'cosine',
                 annotation_mode: str = 'skeleton',
                 *args,
                 **kwargs):
        assert similarity_mode in ('cosine', 'euclidean')
        self._num_correct = 0
        self._num_examples = 0
        self._similarity_mode = similarity_mode
        self.annotations_sampler = SkeletonSampler(
        ) if annotation_mode == 'skeleton' else MaskRandomSampler(10)
        self._annotation_mode = annotation_mode
        super().__init__(*args, **kwargs)

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output: EvaluatorOutputType):
        features_batch, annotations_batch = output
        img_width, img_height = annotations_batch[0][0]['image_size']
        features_batch = F.interpolate(features_batch.rename(None),
                                       (img_height, img_width)).rename(
                                           'N', 'C', 'H', 'W')
        for features, annotations in zip(features_batch, annotations_batch):
            masks = self.create_masks(annotations)
            sparse_annotations = self.annotations_sampler(annotations)
            predictions = self.create_predictions_map(sparse_annotations,
                                                      features)
            labels = self.masks_to_labels(masks, device=features.device)
            correct = predictions.eq(labels).view(-1)
            self._num_correct += torch.sum(correct).item()
            self._num_examples += correct.shape[0]

    @staticmethod
    def create_masks(annotations: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Args:
            annotations: list of dictionaries with coco annotations

        Returns:
            list of masks per each object
        """
        masks: List[np.ndarray] = []
        img_width, img_height = annotations[0]['image_size']
        background_mask = np.zeros((img_height, img_width), dtype=np.int32)
        for ann in annotations:
            mask = np.zeros((img_height, img_width), dtype=np.int32)
            contours = np.array(ann['segmentation']).reshape(-1, 1, 2).astype(
                np.int32)
            mask = cv2.drawContours(mask, [contours], -1, 1., thickness=-1)
            background_mask = cv2.drawContours(background_mask, [contours],
                                               -1, (1.),
                                               thickness=-1)
            masks.append(mask)
        background_mask = background_mask.astype(np.bool) ^ 1
        masks.insert(0, background_mask)
        return masks

    def create_predictions_map(self, collected_data: List[Dict[str, int]],
                               features: torch.Tensor):
        """
        Args:
            collected_data: list of points represented by dictionaries with x, y and label
            features: tensor of shape ['C', 'H', 'W']

        Returns:
            creates feature map with predicted labels
        """
        centres = [features[..., it['y'], it['x']] for it in collected_data]
        centres = torch.stack([el.rename(None) for el in centres])
        if self._similarity_mode == 'cosine':
            best_idxes = choose_best_idxes_with_cosine_similarity(
                centres, features)
        else:
            best_idxes = choose_best_idxes_with_euclidean_distances(
                centres, features)
        predictions = torch.zeros_like(best_idxes).rename(None)
        labels = [it['label'] for it in collected_data]
        for idx, label in enumerate(labels):
            predictions[best_idxes.rename(None) == idx] = label
        return predictions

    @staticmethod
    def masks_to_labels(masks: List[np.ndarray],
                        device: torch.device) -> torch.Tensor:
        """
        Args:
            masks: boolean masks
            device: cuda or cpu

        Returns:
            long tensor with labels
        """
        masks = [mask * (idx + 1) for idx, mask in enumerate(masks)]
        labels = np.stack(masks).sum(axis=0) - 1
        return torch.tensor(labels, dtype=torch.long, device=device)

    def compute(self):
        """Compute accuracy"""
        return self._num_correct / self._num_examples
