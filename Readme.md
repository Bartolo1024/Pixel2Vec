# TileNet

## Run

We assume that you have data, and pre-processed it. If not, see for the **Data** section.

* Train on the minesweeper image (sota run)
    - Replace `data_root` with `data/minesweeper_2_256/` in the `project_config.yaml` or in the `project_config.local.yaml` file
    - Run command ```python train.py --experiment-file experiments/train_on_game_image.yaml ```

Note: on Windows 10 you want to set in `experiments/[name].yaml` file `num_workers: 0`, otherwise it will be super slow, vide https://github.com/pytorch/pytorch/issues/12831.

## Data

### One Image Datasets for quick tests

* Tycoon image
```shell script
wget https://user-images.githubusercontent.com/1001610/85272345-58624100-b47c-11ea-8326-7c4bb19c4d05.png -P data/test_tycoon
```
* Tycoon small board image
```shell script
wget https://user-images.githubusercontent.com/24765461/85410634-a0aa5d80-b567-11ea-8522-5653bfa0bda1.png -P data/test_tycoon_small
```
* Minesweeper
```shell script
wget https://user-images.githubusercontent.com/1001610/85146528-f2907200-b24d-11ea-96bb-582edf2bdc0e.png -P data/minesweeper
```
* Old Simcity
```shell script
wget https://user-images.githubusercontent.com/1001610/85148345-50be5480-b250-11ea-8c69-a735fb2074d8.png -P data/old_simcity
```