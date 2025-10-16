#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=ceoas-arm
#SBATCH --nodelist=youmu
#SBATCH --output="/ceoas/Vandenhoek_Lab/mod_res_damage/data/slurm_logs/unet_sml_ukraine-conflict.log"
#SBATCH --error="/ceoas/Vandenhoek_Lab/mod_res_damage/data/slurm_logs/unet_sml_ukraine-conflict.err"

conda activate sundial


export SLURM_ARRAY_TASK_ID=0
export HUGGINGFACE_HUB_CACHE=/ceoas/Vandenhoek_Lab/mod_res_damage/.hub
export HYDRA_FULL_ERROR=1 

srun \
  --job-name="unet_sml_ukraine-conflict" \
  python mod_res_damage \
      --config-name=unet \
        experiment_name="ukraine-conflict-unet-sml"
