#!/bin/bash
# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.

#SBATCH --account=YOUR_ALLOCATION   # GPU accounts use _g suffix on Perlmutter
#SBATCH --job-name=test_forecast
#SBATCH --constraint=gpu
#SBATCH --partition=debug                  # 30-min debug queue — fastest allocation
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64                 # All 64 cores on a Perlmutter GPU node
#SBATCH --time=00:30:00
#SBATCH --output=test_forecast_%j.log

# ── Resolve the head node ─────────────────────────────────────────────────────
head_node=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)

# ── Load Environment ──────────────────────────────────────────────────────────
source ~/miniforge3/etc/profile.d/conda.sh
conda activate neural_operator

# ── NCCL & Network Settings (Perlmutter Slingshot 11) ─────────────────────────
export NCCL_NET_GDR_LEVEL=PHB   # GPU-direct over HSN on Perlmutter
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export LOGLEVEL=INFO

# ── Write smoke-test wrapper ──────────────────────────────────────────────────
# Patches CONFIG for a fast 3-epoch run without touching train_forecast.py
WRAPPER=~/test_wrapper_${SLURM_JOB_ID}.py

cat > "$WRAPPER" << 'PYEOF'
import sys, os
sys.path.insert(0, os.path.expanduser("~/black-hole-neural-forecasting/SFNO"))
import train_forecast

# Smoke-test overrides — just enough to validate the full pipeline
train_forecast.CONFIG["epochs"] = 2
train_forecast.CONFIG["patience"] = 999   # disable early stopping
train_forecast.CONFIG["batch_size"] = 4   # debug nodes are 40GB; production uses 8
train_forecast.CONFIG["calib_batches"] = 5  # keep calibration fast in debug run

train_forecast.main()
PYEOF

# ── Launch test run ───────────────────────────────────────────────────────────
srun -n 2 --cpu-bind=cores \
    torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${head_node}:29500" \
    "$WRAPPER"
