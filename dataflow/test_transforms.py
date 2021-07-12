import torch
from PIL import Image
from torchvision.transforms import (RandomHorizontalFlip, RandomVerticalFlip,
                                    ToTensor)

import dataflow.transforms


def create_test_image():
    img1 = Image.new('RGB', size=(100, 100), color=(255, 0, 0))
    img2 = Image.new('RGB', size=(120, 130), color=(0, 255, 0))
    shift = (50, 60)
    img1.paste(img2, shift)
    return img1


def test_vertical_tensor_flip():
    pil_img = create_test_image()
    tensor_img = ToTensor()(pil_img)
    vertical_flipped = dataflow.transforms.RandomVerticalFlip(probability=1.)(tensor_img)
    vertical_flipped_pil = ToTensor()(RandomVerticalFlip(p=1.)(pil_img))
    assert torch.allclose(vertical_flipped, vertical_flipped_pil)


def test_horizontal_tensor_flip():
    pil_img = create_test_image()
    tensor_img = ToTensor()(pil_img)
    horizontal_flipped = dataflow.transforms.RandomHorizontalFlip(probability=1.)(tensor_img)
    horizontal_flipped_pil = ToTensor()(RandomHorizontalFlip(p=1.)(pil_img))
    assert torch.allclose(horizontal_flipped, horizontal_flipped_pil)


def test_tensor_positive_rotation():
    x = torch.zeros(3, 10, 10)
    x[..., 9] = torch.arange(10)
    y = torch.zeros(3, 10, 10)
    y[:, 0, :] = torch.arange(10)
    trn = dataflow.transforms.RandomRotateStraightAngle(probability=1., negative_angle_probability=0.)
    rot = trn(x)
    assert torch.allclose(rot, y)


def test_tensor_negative_rotation():
    x = torch.zeros(3, 10, 10)
    x[..., 9] = torch.arange(10)
    y = torch.zeros(3, 10, 10)
    y[:, 9, :] = torch.arange(10).flip(0)
    trn = dataflow.transforms.RandomRotateStraightAngle(probability=1., negative_angle_probability=1.)
    rot = trn(x)
    assert torch.allclose(rot, y)


def test_right_bottom_crop():
    trn = dataflow.transforms.CropRightBottom(crop_percentage=0.25)
    x = torch.arange(4 * 4).view(1, 4, 4)
    target = torch.tensor([[10, 11], [14, 15]]).unsqueeze(0)
    cropped = trn(x)
    assert torch.allclose(cropped, target)


def test_blot_right_bottom_with_edges_average_with_ones():
    crop_percentage = 0.25
    trn = dataflow.transforms.BlotRightBottomWithAverageEdgeValue(crop_percentage=crop_percentage)
    cropper = dataflow.transforms.CropRightBottom(crop_percentage=crop_percentage)
    x = torch.ones((1, 8, 8))
    target = torch.ones((1, 4, 4))
    cropped = cropper(trn(x))
    assert torch.allclose(cropped, target)
