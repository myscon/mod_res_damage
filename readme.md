# mod_res_damage

## Introduction
This repo contains a simple framework for training a NN for moderate resolution damage mapping using SAR. Included is a BNN built on top of the TerraMind encoder as well as monte carlo inference logit. Also included is an example pipeline from ASF ('mod_res_damage/utils/modify_bright.py') to download SAR using their hyp3 API.

## Usage
There is an example config file at ('configs/terramind.yaml') that dictates how to structure config for deciding model, criterion, dataset, modules etc. This config is used in ('mod_res_damage/main.py') to instantiate all necessary objects for training from the model itself to evaluating the model.

To run

```
cd mod_res_damage # cd into repo. no pip install is available
pip install -e .  # install repo as editable so import paths work

python main.py --config-name=['name_of_config_file_without_suffix_in_repo_configs_directory']
```

If you are running on a slurm managed cluster. There is a quick run.slurm example included in the root directory of the repo.

```
sbatch run.slurm  # run this as you would any other batch script
```
