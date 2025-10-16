#!/bin/bash -l

#SBATCH --job-name=prbms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=ewg
#SBATCH --output="/ceoas/Vandenhoek_Lab/mod_res_damage/data/slurm_logs/train_prbms.log"
#SBATCH --error="/ceoas/Vandenhoek_Lab/mod_res_damage/data/slurm_logs/train_prbms.err"

conda activate mod_res_damage

export HUGGINGFACE_HUB_CACHE=/ceoas/Vandenhoek_Lab/mod_res_damage/.hub
export HYDRA_FULL_ERROR=1 

srun --job-name="prbm" \
  python /ceoas/Vandenhoek_Lab/mod_res_damage/mod_res_damage/utils/train_prbms.py
