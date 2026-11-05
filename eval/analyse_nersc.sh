#!/bin/bash
# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.

#SBATCH --account=YOUR_ALLOCATION
#SBATCH --job-name=analyse_forecast
#SBATCH --constraint=gpu&hbm80g
#SBATCH -G 1
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=analyse_%j.log

export PYTHONPATH=~/black-hole-neural-forecasting:$PYTHONPATH

srun -n 1 -c 32 --gpu-bind=none \
    ~/miniforge3/envs/neural_operator/bin/python \
    ~/black-hole-neural-forecasting/SFNO/analyse_forecast.py \
    --var rho --plane both --start 5000 --steps 10 --rmax 50
