#!/bin/bash
# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.
# Smoke test on one A100: synthetic checks, then full-size training steps on real dumps.
# Submit from the repo root:
#     mkdir -p 3Dtransformer/outputs/slurm
#     sbatch 3Dtransformer/scripts/smoke_delta.sbatch
# Time another experiment's data and model, or try a smaller batch if it runs out of memory:
#     sbatch --export=ALL,EXPERIMENT=vit_rc300_v1 3Dtransformer/scripts/smoke_delta.sbatch
#     sbatch --export=ALL,BATCH_SIZE=4 3Dtransformer/scripts/smoke_delta.sbatch
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --job-name=vit_smoke
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=3Dtransformer/outputs/slurm/%x_%j.log

cd "$SLURM_SUBMIT_DIR"
source 3Dtransformer/scripts/env_delta.sh
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "=== Synthetic checks"
python 3Dtransformer/smoke_test.py || exit 1

echo "=== Real data (${EXPERIMENT:-vit_baseline_v1}), batch ${BATCH_SIZE:-8} per GPU"
python 3Dtransformer/smoke_test.py --real --experiment "${EXPERIMENT:-vit_baseline_v1}" --batch-size "${BATCH_SIZE:-8}"
