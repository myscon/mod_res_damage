#!/bin/bash -l

#SBATCH --job-name=bayes_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ewg
#SBATCH --array=0-15%8
#SBATCH --output="/ceoas/Vandenhoek_Lab/mod_res_damage/data/slurm_logs/bayesiammtl_parent.log"
#SBATCH --error="/ceoas/Vandenhoek_Lab/mod_res_damage/data/slurm_logs/bayesiammtl_parent.err"

conda activate mod_res_damage

HOLDOUTS=(
  '["ukraine-conflict"]'
  '["bata-explosion"]'
  '["beirut-explosion"]'
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
  '["ukraine-conflict_group-1"]'
  '["mexico-hurricane_group-1"]'
  '["turkey-earthquake_group-1"]'
)

holdout_list=${HOLDOUTS[$SLURM_ARRAY_TASK_ID]}
holdout=${holdout_list//[\[\]\"]/}

export HUGGINGFACE_HUB_CACHE=/ceoas/Vandenhoek_Lab/mod_res_damage/.hub

srun \
  --job-name="bayesiammtl_${holdout}" \
  --output="/ceoas/Vandenhoek_Lab/mod_res_damage/data/slurm_logs/bayesiammtl_${holdout}.log" \
  --error="/ceoas/Vandenhoek_Lab/mod_res_damage/data/slurm_logs/bayesiammtl_${holdout}.err" \
  python mod_res_damage \
      --config-name=terramind dataset.holdout="$holdout_list" \
        experiment_name="$holdout"
