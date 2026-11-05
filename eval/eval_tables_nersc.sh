#!/bin/bash
# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.

#SBATCH --account=YOUR_ALLOCATION
#SBATCH --job-name=eval_tables
#SBATCH --constraint=gpu&hbm80g
#SBATCH -G 1
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=eval_tables_%j.log

export PYTHONPATH=~/black-hole-neural-forecasting:$PYTHONPATH

PYTHON=~/miniforge3/envs/neural_operator/bin/python
SCRIPT=~/black-hole-neural-forecasting/eval/evaluate_forecast.py
SAVE=~/black-hole-neural-forecasting/eval_results
START=5000

mkdir -p "$SAVE"

# Table 3: Volume-Averaged |div B| Residual (horizons 1,10,50,100 hardcoded)
srun -n 1 -c 32 --gpu-bind=none \
    $PYTHON $SCRIPT --mode divb_table --start $START --save_dir $SAVE

# Table 4: Physics Consistency Metrics (horizons 1,10,50,100 hardcoded)
srun -n 1 -c 32 --gpu-bind=none \
    $PYTHON $SCRIPT --mode physics_table --start $START --save_dir $SAVE

# Commit and push results to GitHub
cd ~/black-hole-neural-forecasting
git pull --rebase origin SFNO
git add eval_results/
git commit -m "eval tables 3+4: $(date +%Y-%m-%d) job ${SLURM_JOB_ID}" || true
git push origin SFNO
