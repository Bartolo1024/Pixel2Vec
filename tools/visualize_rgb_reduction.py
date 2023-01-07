import logging
import os
import random

import click
import torch
import torchvision.transforms
from matplotlib import pyplot as plt
from PIL import Image

from utils.predictor import Predictor
from utils.reduced_dimensions_visualizer import get_pca_rgb, get_tsne_rgb
from utils.restoration import restore_model, restore_params


@click.command()
@click.argument('run_id')
@click.argument('weights')
@click.argument('images_path')
@click.option('--image-name', default=None, type=str)
@click.option('--device', default=torch.device('cuda'), type=str)
@click.option('--dims-reduction-method', type=click.Choice(('t-SNE', 'PCA')), default='t-SNE')
@click.option('--scale', default=.12, type=float)
@click.option('--predictor-mode', type=click.Choice(('patches', 'single_inference')), default='patches')
@click.option('--input-noise', type=float, default=0.)
def main(
    run_id: str, weights: str, images_path: str, image_name: str, device: torch.device, dims_reduction_method: str,
    scale: float, predictor_mode: str, input_noise: float
):
    """Compute dot product of one image patch with others and draw heatmap
    Args:
        run_id: from neptune
        weights: name of weights in mlruns/<run_id>/artifacts/
        images_path: path to test images folder
        image_name: image name from folder (random if not provided)
        device: 'cuda', 'mps' (Apple M1 GPU) or 'cpu'
        dims_reduction_method: t-SNE or PCA
        scale: scale of the output image
        predictor_mode: predict on patches or whole image
    """
    if image_name is None:
        image_name = random.choice(os.listdir(images_path))
        logging.info(f'random image {image_name}')

    params = restore_params(run_id)
    out_dir = os.path.join(params['run_dir'], f'{dims_reduction_method}-rgb')
    os.makedirs(out_dir, exist_ok=True)
    out_image_path = os.path.join(out_dir, image_name)
    logging.info(f'results will we stored as {out_image_path}')

    model = restore_model(params['model_spec'], params['run_dir'], weights).to(device)
    predictor = Predictor(model, predictor_mode, params['data_flow'], device)

    predictor.transform = torchvision.transforms.Compose(
        [predictor.transform, lambda s: s + input_noise * torch.rand_like(s)]
    )

    reduce_fn = get_tsne_rgb if dims_reduction_method == 't-SNE' else get_pca_rgb

    raw_img = Image.open(os.path.join(images_path, image_name)).convert('RGB')
    feature_map = predictor(raw_img)

    reduced_img = reduce_fn(feature_map)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(raw_img)
    ax[0].set_title('input image')
    ax[0].set_axis_off()
    ax[1].imshow(reduced_img)
    ax[1].set_title(dims_reduction_method)
    ax[1].set_axis_off()
    _, h, w = feature_map.shape
    w_inches = int(w * scale)
    h_inches = int(h * scale)
    fig.set_size_inches(w_inches, h_inches, forward=True)
    plt.savefig(out_image_path, bbox_inches='tight')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
