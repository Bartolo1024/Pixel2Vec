# TileNet

Unsupervised semantic segmentation trained on the single image

## Run

To train the SOTA model on the minesweeper image run command:\
  ```python train.py --experiment-file experiments/train_on_game_image.yaml ```

Note: on Windows 10 you want to set in `experiments/[name].yaml` file `num_workers: 0`, otherwise it will be super slow, vide https://github.com/pytorch/pytorch/issues/12831.

## Data

Example inputs are placed in the `data\` directory.