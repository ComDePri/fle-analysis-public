#!/bin/bash

#SBATCH --job-name=parameter_sweep
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=24:00:00

# This script is submitted to SLURM for a single job.
# It should be called from the submit_all_vt.sh script.
# It should get the participant id and v_t from the command line arguments.
# Usage: ./submit_sweep.sh <participant_id> <v_t>


# Define root path
ROOT_PATH="/path/to/project/root"  # should have a subdirectory called "tracking_epochs" with the data and "et_llk" for results
# Activate virtual environment
source path/to/your/venv/bin/activate
cd path/to/your/code/fle-analysis || exit

if [ -z "$1" ]
  then
    echo "Error: Missing argument for id."
    exit 1
fi
if [ -z "$2" ]
  then
    echo "Error: Missing argument for v_t."
    exit 1
fi
id=$1
v_t=$2
python -m analysis.SLURM.run_sweep --id ${id} --v_t ${v_t} --root_path ${ROOT_PATH} --config analysis/SLURM/config.yaml

