import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import sklearn.model_selection
import torch
import torchvision.transforms
from torch.utils.data.dataloader import DataLoader

import dataflow.transforms
from dataflow.datasets import ImagePatchTripletDatset
from dataflow.datasets import ImageListInMemoryDataset
from dataflow.utils import (extract_patches_from_pil_image,
                            get_img_list_from_folder, read_images_from_folder)


def warn_if_no_workers(func):
    def _wrapper(*args, **kwargs):
        num_workers = kwargs.get('num_workers')
        if num_workers == 0:
            logging.warning(
                f'Dataloader from function {func} will be working with 0 workers!!! (bottleneck)'
            )
        ret = func(*args, **kwargs)
        return ret

    return _wrapper


def do_not_split(paths: List[str], **_):
    return paths, paths


def build_transforms(
    normalization_stats: Dict[str, Tuple[float, float, float]],
    img_size: Optional[Tuple[int, int]] = None,
    crop_percentage: float = 0.25,
    color_jitter_kwargs: Optional[Dict[str, Any]] = None,
    to_grayscale: bool = False
) -> Tuple[Callable[[Any], torch.Tensor], Callable[[Any], torch.Tensor]]:
    """Build transforms
    Args:
        normalization_stats: dictionary with mean and standard deviation for an image normalization
        img_size: height and width of image
        crop_percentage: evaluation crop percentage of the image
        color_jitter_kwargs: keyword arguments for the color jitter - if None transform won't be applied
        to_grayscale: convert image to grayscale

    Returns:
        Two callable that transforms PIL.Image into a tensor of shape ('C', 'H', 'W')
    """
    train_transform = []
    test_transform = []

    if img_size:
        train_transform.append(torchvision.transforms.Resize(size=img_size))
        test_transform.append(torchvision.transforms.Resize(size=img_size))

    if color_jitter_kwargs:
        train_transform.append(
            torchvision.transforms.ColorJitter(**color_jitter_kwargs))

    if to_grayscale:
        num_channels = len(normalization_stats['mean'])
        train_transform.append(
            torchvision.transforms.Grayscale(num_output_channels=num_channels))
        test_transform.append(
            torchvision.transforms.Grayscale(num_output_channels=num_channels))

    train_transform.append(torchvision.transforms.ToTensor())
    test_transform.append(torchvision.transforms.ToTensor())

    if crop_percentage <= 0.99:
        train_crop = dataflow.transforms.BlotRightBottomWithAverageEdgeValue(
            crop_percentage)
        train_transform.append(train_crop)
        test_crop = dataflow.transforms.CropRightBottom(crop_percentage)
        test_transform.append(test_crop)
    else:
        logging.info('Images will not be cropped for evaluation')

    normalizer = torchvision.transforms.Normalize(**normalization_stats)
    train_transform.append(normalizer)
    test_transform.append(normalizer)
    return torchvision.transforms.Compose(
        train_transform), torchvision.transforms.Compose(test_transform)


def _patches_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor,
                                          torch.Tensor]]):
    anchors, positives, negatives = zip(*batch)
    anchors = torch.stack([t.rename(None) for t in anchors])
    positives = torch.stack([t.rename(None) for t in positives])
    negatives = torch.stack([t.rename(None) for t in negatives])
    return anchors, positives, negatives


@warn_if_no_workers
def create_patches_dataflow(
    root: str,
    normalization_stats: Dict[str, Tuple[float, float, float]],
    batch_size: int,
    img_size: Optional[Tuple[int, int]] = None,
    test_size: float = 0.2,
    random_state: int = 88,
    num_workers: int = 0,
    shuffle: bool = True,
    validation_crop_percentage: float = 0.25,
    **dataset_kwargs,
) -> Dict[str, Iterable[torch.Tensor]]:
    """
    Args:
        root: directory with images
        normalization_stats: dictionary with mean and standard deviation for an image normalization
        batch_size: size of batch of images
        img_size: height and width of image - can be skipped if you want to have different image sizes
        test_size: size of test set as all images fraction
        random_state: random seed for train test split
        num_workers: num of processes for data flows
        shuffle: flag to shuffle images in each train set iteration
        validation_crop_percentage: part of an image for a validation crop (do not crop if higher than 0.99)
        **dataset_kwargs: dataset keyword arguments

    Returns:
        two iterators with tensors

    Notes:
        Currently test set may be created from separated images or image subparts
    """
    img_paths = get_img_list_from_folder(root)
    train_test_split_fn = sklearn.model_selection.train_test_split if not validation_crop_percentage else do_not_split
    train_paths, test_paths = train_test_split_fn(img_paths,
                                                  test_size=test_size,
                                                  random_state=random_state)
    train_transform, test_transform = build_transforms(
        normalization_stats, img_size, validation_crop_percentage)

    patch_transform = torchvision.transforms.Compose([
        dataflow.transforms.RandomVerticalFlip(),
        dataflow.transforms.RandomHorizontalFlip(),
        dataflow.transforms.RandomBrightnessContrastAdjust(),
    ])
    train_dataset = ImagePatchTripletDatset(image_paths=train_paths,
                                            transform=train_transform,
                                            patch_transform=patch_transform,
                                            **dataset_kwargs)
    test_dataset = ImagePatchTripletDatset(image_paths=test_paths,
                                           transform=test_transform,
                                           **dataset_kwargs)
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   shuffle=shuffle,
                                   collate_fn=_patches_collate_fn)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=False,
                                  collate_fn=_patches_collate_fn)
    return {'train': train_data_loader, 'val': test_data_loader}


def _images_collate_fn(batch: List[torch.Tensor]):
    return torch.stack([t.rename(None) for t in batch])


@warn_if_no_workers
def create_images_dataflow(
    root: str,
    normalization_stats: Dict[str, Tuple[float, float, float]],
    batch_size: int,
    img_size: Optional[Tuple[int, int]] = None,
    test_size: float = 0.2,
    random_state: int = 88,
    num_workers: int = 0,
    shuffle: bool = True,
    validation_crop_percentage: float = 0.25,
    color_jitter_kwargs: Optional[Dict[str, Any]] = None,
    to_grayscale: bool = False,
    **dataset_kwargs,
) -> Dict[str, Iterable[torch.Tensor]]:
    """
    Args:
        root: directory with images
        normalization_stats: dictionary with mean and standard deviation for an image normalization
        batch_size: size of batch of images
        img_size: height and width of image - can be skipped if you want to have different image sizes
        test_size: size of test set as all images fraction
        random_state: random seed for train test split
        num_workers: num of processes for data flows
        shuffle: flag to shuffle images in each train set iteration
        validation_crop_percentage: part of an image for a validation crop (do not crop if higher than 0.99)
        color_jitter_kwargs: keyword arguments for the color jitter, transform won't be applied if None
        to_grayscale: convert image to grayscale
        **dataset_kwargs: dataset keyword arguments

    Returns:
        two iterators with tensors

    Notes:
        Currently test set may be created from separated images or image subparts
    """
    img_paths = get_img_list_from_folder(root)
    train_test_split_fn = sklearn.model_selection.train_test_split if not validation_crop_percentage else do_not_split
    train_paths, test_paths = train_test_split_fn(img_paths,
                                                  test_size=test_size,
                                                  random_state=random_state)
    train_transform, test_transform = build_transforms(
        normalization_stats,
        img_size,
        validation_crop_percentage,
        color_jitter_kwargs=color_jitter_kwargs,
        to_grayscale=to_grayscale)
    train_dataset = ImageListInMemoryDataset.from_paths(
        image_paths=train_paths, transform=train_transform, **dataset_kwargs)
    test_dataset = ImageListInMemoryDataset.from_paths(
        image_paths=test_paths, transform=test_transform, **dataset_kwargs)
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   shuffle=shuffle,
                                   collate_fn=_images_collate_fn)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=False,
                                  collate_fn=_images_collate_fn)
    return {'train': train_data_loader, 'val': test_data_loader}


@warn_if_no_workers
def create_sliced_images_dataflow(
    root: str,
    normalization_stats: Dict[str, Tuple[float, float, float]],
    batch_size: int,
    patch_size: Tuple[int, int],
    patch_coverage: int,
    random_state: int = 88,
    test_size: float = 0.2,
    num_workers: int = 0,
    shuffle: bool = True,
    color_jitter_kwargs: Optional[Dict[str, Any]] = None,
    to_grayscale: bool = False,
    convert_to: str = 'RGB',
) -> Dict[str, Iterable[torch.Tensor]]:
    """
    Args:
        root: directory with images
        normalization_stats: dictionary with mean and standard deviation for an image normalization
        batch_size: size of batch of images
        patch_size: size of image patch
        patch_coverage: intersection between patches
        random_state: random seed for train test split
        test_size: size of test set as all images fraction
        num_workers: num of processes for data flows
        shuffle: flag to shuffle images in each train set iteration
        color_jitter_kwargs: keyword arguments for the color jitter, transform won't be applied if None
        to_grayscale: convert image to grayscale
        convert_to: PIL image format

    Returns:
        two iterators with tensors

    """
    train_transform, test_transform = build_transforms(
        normalization_stats,
        img_size=patch_size,
        crop_percentage=1.,
        color_jitter_kwargs=color_jitter_kwargs,
        to_grayscale=to_grayscale)
    images = read_images_from_folder(root, convert_to=convert_to)
    images = [
        patch for img in images for patch in extract_patches_from_pil_image(
            img, patch_size, patch_coverage)
    ]
    train_images, test_images = sklearn.model_selection.train_test_split(
        images, test_size=test_size, random_state=random_state)
    train_dataset = ImageListInMemoryDataset.from_pil_images(
        images=train_images, transform=train_transform)
    test_dataset = ImageListInMemoryDataset.from_pil_images(
        images=test_images, transform=test_transform)
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   shuffle=shuffle,
                                   collate_fn=_images_collate_fn)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=False,
                                  collate_fn=_images_collate_fn)
    return {'train': train_data_loader, 'val': test_data_loader}
