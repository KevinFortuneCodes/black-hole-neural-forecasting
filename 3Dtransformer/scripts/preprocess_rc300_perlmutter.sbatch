#!/bin/bash
# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.
# Preprocess the rc300 reduced run on one Perlmutter CPU node.
#
# Submit from the repo root, after the login-node test (doc section 5):
#   mkdir -p 3Dtransformer/outputs/slurm
#   sbatch 3Dtransformer/scripts/preprocess_rc300_perlmutter.sbatch
# Arguments after the script name go to preprocess_rc300.py, e.g. --set r_max=null.
# Set --account to your NERSC allocation (sacctmgr show assoc user=$USER format=account).
# Reruns skip dumps that are already written, so a job that hits the time limit can just be resubmitted.
#SBATCH --job-name=prep_rc300
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --time=02:00:00
#SBATCH --output=3Dtransformer/outputs/slurm/%x_%j.log

set -euo pipefail
module load python
export OMP_NUM_THREADS=1
python -u 3Dtransformer/data/preprocess_rc300.py --workers "${WORKERS:-64}" "$@"
