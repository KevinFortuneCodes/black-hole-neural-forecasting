#!/bin/bash
# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.
# Score one run's checkpoint on the blind test set with TRON's evaluation suite, on one A100.
# Results go to 3Dtransformer/outputs/<EXPERIMENT>/seed<SEED>/eval/<CHECKPOINT>_blind/ (out_res.txt, figures, .npz).
# From the repo root:
#     sbatch --job-name=eval_vit_rc300_v1_s0 --export=ALL,EXPERIMENT=vit_rc300_v1,SEED=0 3Dtransformer/scripts/eval_delta.sbatch
# EVAL_ARGS go to the evaluator, e.g. a quick check: EVAL_ARGS="--mode single_step"
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --job-name=vit_eval
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=3Dtransformer/outputs/slurm/%x_%j.log

EXPERIMENT=${EXPERIMENT:?set EXPERIMENT}
SEED=${SEED:-0}
CHECKPOINT=${CHECKPOINT:-best}
EVAL_ARGS=${EVAL_ARGS:---mode all --max_steps 50}
cd "$SLURM_SUBMIT_DIR"
PYTORCH_MODULE=pytorch-conda/2.8   # its matplotlib saves PNGs; 2.12's fails (env_delta.sh)
source 3Dtransformer/scripts/env_delta.sh
echo "Evaluating $EXPERIMENT seed $SEED ($CHECKPOINT.pth) on $(hostname), job $SLURM_JOB_ID: $EVAL_ARGS"

python 3Dtransformer/eval/evaluate_vit.py --experiment "$EXPERIMENT" --train_seed "$SEED" \
    --checkpoint "$CHECKPOINT" $EVAL_ARGS
