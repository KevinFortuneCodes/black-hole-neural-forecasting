#!/bin/bash
# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.

#SBATCH --account=YOUR_ALLOCATION
#SBATCH --job-name=ddp_forecast
#SBATCH --constraint=gpu&hbm80g
#SBATCH -G 16
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=ddp_forecast_%j.log

# ── Resolve the head node ─────────────────────────────────────────────────────
head_node=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)

# ── Load Environment ──────────────────────────────────────────────────────────
source ~/miniforge3/etc/profile.d/conda.sh
conda activate neural_operator

# ── NCCL & Network Settings (Perlmutter Slingshot 11) ─────────────────────────
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export LOGLEVEL=INFO
export OMP_NUM_THREADS=1

# ── Launch full training run ──────────────────────────────────────────────────
srun -n 4 -c 128 --cpu-bind=cores --gpu-bind=none \
    torchrun \
    --nnodes=4 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${head_node}:29500" \
    ~/black-hole-neural-forecasting/SFNO/train_forecast.py
