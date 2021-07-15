import logging
import os

import click
from PIL import Image
from tqdm import tqdm

ALLOWED_EXTENSIONS = ('tif', 'jpg', 'png', 'jpeg')


@click.command()
@click.argument('in_path')
@click.argument('out_path')
@click.option('-h', '--out-height', default=None, type=float)
@click.option('-w', '--out-width', default=None, type=float)
@click.option('--mode', type=click.Choice(['values', 'ratio'], case_sensitive=False), default='values')
def main(in_path: str, out_path: str, out_height: float, out_width: float, mode: str):
    """
    Args:
        in_path: input directory
        out_path: output directory
        out_height: height pixels or height ratio
        out_width: width pixels or width ratio
        mode: resize with scale or resize with fixed width and height

    Saves:
        Resized images
    """
    assert not (out_height is None and out_width is None)
    os.makedirs(out_path, exist_ok=True)
    images = os.listdir(in_path)
    logging.info(f'resized images: {images} will be saved in {out_path}')
    for file_name in tqdm(images):
        img_extension = file_name.split('.')[-1]
        if img_extension not in ALLOWED_EXTENSIONS:
            print(f'File {file_name} will be skipped: extension {img_extension} not in {ALLOWED_EXTENSIONS}')
            continue
        file_path = os.path.join(in_path, file_name)
        img = Image.open(file_path)
        width, height = img.size
        if mode == 'values':
            out_height = out_height if out_height else height * out_width / width
            out_width = out_width if out_width else width * out_height / height
            img = img.resize((int(out_width), int(out_height)))
        elif mode == 'ratio':
            assert out_width and out_height
            result_height = int(height * out_height)
            result_width = int(width * out_width)
            img = img.resize((result_width, result_height))
        else:
            raise NotImplementedError
        out_file_path = os.path.join(out_path, file_name)
        img.save(out_file_path)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
