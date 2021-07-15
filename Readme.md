# Pixel2Vec

Self supervised feature extraction trained on the single image

## Run

To train the SOTA model on the minesweeper image run command:\
  ```python train.py --experiment-file experiments/train_on_game_image.yaml ```

Note: on Windows 10 you want to set in `experiments/[name].yaml` file `num_workers: 0`, otherwise it will be super slow, vide https://github.com/pytorch/pytorch/issues/12831.

## Data

Example inputs are placed in the `data\` directory.

## Example results

The model was trained on the given Simcity image. All feature vectors were projected into 3d space, and presented as the RGB image. The result is shown on the bottom.  

![Input image](./data/examples/sim.png)
![PCA](./data/examples/sim_pca.jpg)

## Funding

Project is supported by [Program Operacyjny Inteligentny Rozwój grant for ECC Games for GearShift project](https://mapadotacji.gov.pl/projekty/874596/?lang=en).
