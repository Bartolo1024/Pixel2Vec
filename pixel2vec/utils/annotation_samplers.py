from abc import abstractmethod
from typing import Any, Dict, List

import cv2
import numpy as np


class AnnotationSampler:
    @abstractmethod
    def __call__(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, int]]:
        """
        Args:
            annotations: coco annotations

        Returns:
            A list of dictionaries with x, y and label for each chosen point
        """
        pass

    @staticmethod
    def annotation_to_mask(annotation: Dict[str, Any]) -> np.ndarray:
        """
        Args:
            annotation: coco annotation

        Returns:
            numpy boolean mask
        """
        img_width, img_height = annotation['image_size']
        mask = np.zeros((img_height, img_width), dtype=np.int32)
        contours = np.array(annotation['segmentation']).reshape(-1, 1, 2).astype(np.int32)
        mask = cv2.drawContours(mask, [contours], -1, 1., thickness=-1)
        return mask


class MaskRandomSampler(AnnotationSampler):
    def __init__(self, samples_per_class: int):
        """
        Args:
            samples_per_class: number of points gathered per one mask
        """
        self._samples_per_class = samples_per_class

    def __call__(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, int]]:
        """
        Args:
            annotations: coco annotations

        Returns:
            A list of dictionaries with x, y and label for each chosen point
        """
        collected_data = []
        background_mask = np.zeros(annotations[0]['image_size'], dtype=np.int32)
        for ann_idx, ann in enumerate(annotations):
            mask = self.annotation_to_mask(ann)
            background_mask += mask
            collected_data.extend(self.sample_points_from_mask(mask, ann_idx + 1))
        background_mask = background_mask.astype(np.bool) ^ 1
        collected_data.extend(self.sample_points_from_mask(background_mask, 0))
        return collected_data

    def choose_idxes_from_boolean_mask(self, mask):
        choosen_idxes = []
        img_height, img_width = mask.shape
        mask = mask.reshape(-1)
        for n in range(self._samples_per_class):
            idx = np.random.choice(range(img_height * img_width), p=mask.reshape(-1) / mask.sum())
            mask[idx] = 0.
            choosen_idxes.append(idx)
        return choosen_idxes

    def sample_points_from_mask(self, mask, label):
        collected_data = []
        img_height, img_width = mask.shape
        idxes = self.choose_idxes_from_boolean_mask(mask=mask)
        for idx in idxes:
            ys = int(idx / img_width)
            xs = idx % img_width
            collected_data.append({'x': xs, 'y': ys, 'label': label})
        return collected_data


class SkeletonSampler(AnnotationSampler):
    def __call__(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, int]]:
        """
        Args:
            annotations: coco annotations

        Returns:
            A list of dictionaries with x, y and label for each chosen point
        """
        collected_data = []
        image_width, image_height = annotations[0]['image_size']
        background_mask = np.zeros((image_height, image_width), dtype=np.int32)
        for ann_idx, ann in enumerate(annotations):
            mask = self.annotation_to_mask(ann)
            background_mask += mask
            skeleton = self.get_skeleton(mask)
            collected_data.extend(self._skeleton_to_points(skeleton, ann_idx + 1))
        background_mask = background_mask.astype(np.bool) ^ 1
        skeleton = self.get_skeleton(background_mask)
        collected_data.extend(self._skeleton_to_points(skeleton, 0))
        return collected_data

    @staticmethod
    def get_skeleton(mask: np.ndarray) -> np.ndarray:
        """
        Args:
            mask: boolean mask

        Returns:
            boolean mask with skeleton
        """
        img = mask.astype(np.uint8)
        skel = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while cv2.countNonZero(img) != 0:
            opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(img, opened)
            eroded = cv2.erode(img, element)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
        return skel

    def _skeleton_to_points(self, skeleton: np.ndarray, label: int) -> List[Dict[str, int]]:
        """
        Args:
            skeleton: boolean mask with skeleton
            label: label of the given skeleton

        Returns:
            list of points defined by x, y, label
        """
        y_points, x_points = skeleton.astype(np.bool).nonzero()
        collected_data = [{'x': x, 'y': y, 'label': label} for y, x in zip(y_points, x_points)]
        return collected_data
