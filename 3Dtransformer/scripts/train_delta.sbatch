#!/bin/bash
# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.
# Train one seed of one experiment on a 4-GPU A100 node. Outputs go to
# 3Dtransformer/outputs/<EXPERIMENT>/seed<SEED>/; resubmitting resumes from its latest.pth.
# Prefer submit_chain.sh, which also names the job after the experiment and seed.
# Direct use, from the repo root:
#     mkdir -p 3Dtransformer/outputs/slurm
#     sbatch --job-name=vit_baseline_v1_s0 --export=ALL,EXPERIMENT=vit_baseline_v1,SEED=0 3Dtransformer/scripts/train_delta.sbatch
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --job-name=vit_train
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=208G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=3Dtransformer/outputs/slurm/%x_%j.log

EXPERIMENT=${EXPERIMENT:-vit_baseline_v1}
SEED=${SEED:-0}
cd "$SLURM_SUBMIT_DIR"
source 3Dtransformer/scripts/env_delta.sh
echo "Experiment $EXPERIMENT, seed $SEED on $(hostname), job $SLURM_JOB_ID"

torchrun --standalone --nproc_per_node=4 3Dtransformer/vit/train.py --experiment "$EXPERIMENT" --seed "$SEED"
