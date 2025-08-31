#!/bin/bash -l

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

for holdout_list in "${HOLDOUTS[@]}"; do
  holdout=${holdout_list//[\[\]\"]/}
  echo "=== Running for $holdout ==="
  python  mod_res_damage --config-name=terramind dataset.holdout="$holdout_list" experiment_name="$holdout"
done
