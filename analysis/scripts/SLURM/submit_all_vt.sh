#!/bin/bash
# submit_all_vt.sh
# This script submits multiple jobs to SLURM for different v_t values.
# It uses a for loop to iterate over a list of v_t values and submits a job for each value.
# Each job is submitted with a unique job name based on the v_t value.
# It should get the participant id from the command line arguments.
# Usage: ./submit_all_vt.sh <participant_id>

v_t_list=(100 250 500 750 1000 1250 1500)

SCRIPT_DIR="$(dirname "$0")"
id=$1
for vt in "${v_t_list[@]}"
do
  sbatch --job-name=parameter_sweep${vt} "${SCRIPT_DIR}/submit_sweep.sh" $id $vt
done