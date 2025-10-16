#!/bin/bash -l

#SBATCH --job-name=unet_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ceoas-arm
#SBATCH --nodelist=youmu
#SBATCH --array=0-14%3
#SBATCH --output="/ceoas/Vandenhoek_Lab/mod_res_damage/data/slurm_logs/unet_parent.log"
#SBATCH --error="/ceoas/Vandenhoek_Lab/mod_res_damage/data/slurm_logs/unet_parent.err"

conda activate sundial

HOLDOUTS=(
  '["ukraine-conflict"]'
  '["bata-explosion"]'
  '["congo-volcano"]'
  '["haiti-earthquake"]'
  '["hawaii-wildfire"]'
  '["la_palma-volcano"]'
  '["libya-flood"]'
  '["marshall-wildfire"]'
  '["mexico-hurricane"]'
  '["morocco-earthquake"]'
  '["myanmar-hurricane"]'
  '["noto-earthquake"]'
  '["mexico-hurricane_group-1"]'
  '["turkey-earthquake_group-1"]'
)

holdout_list=${HOLDOUTS[$SLURM_ARRAY_TASK_ID]}
holdout=${holdout_list//[\[\]\"]/}

export HUGGINGFACE_HUB_CACHE=/ceoas/Vandenhoek_Lab/mod_res_damage/.hub

conda activate sundial & srun \
  --job-name="unet_${holdout}" \
  --output="/ceoas/Vandenhoek_Lab/mod_res_damage/data/slurm_logs/unet_${holdout}.log" \
  --error="/ceoas/Vandenhoek_Lab/mod_res_damage/data/slurm_logs/unet_${holdout}.err" \
  python mod_res_damage \
      --config-name=unet dataset.holdout="$holdout_list" \
        experiment_name="$holdout"
