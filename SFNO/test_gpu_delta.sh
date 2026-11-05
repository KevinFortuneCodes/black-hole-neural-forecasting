#!/bin/bash
# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.

#SBATCH --account=YOUR_ALLOCATION   # Run `accounts` on Delta to get your GPU allocation name
#SBATCH --job-name=test_gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:01:00
#SBATCH --output=test_gpu_%j.log

echo "Job ${SLURM_JOB_ID} on $(hostname) | partition=${SLURM_JOB_PARTITION} | account=${SLURM_JOB_ACCOUNT}"
nvidia-smi
