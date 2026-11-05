#!/bin/bash
# Written by Sachin Venkatesh, with minor changes by Kevin Fortune; included with credit.
# Kevin Fortune claims no ownership of the original. See LICENSE.

#SBATCH --account=YOUR_ALLOCATION          # Verify this is your deltaAI allocation name
#SBATCH --job-name=ddp_curr
#SBATCH --partition=ghx4          # deltaAI Grace Hopper partition
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=47:00:00
#SBATCH --output=ddp_curr_%j.log

# ── Resolve the head node ─────────────────────────────────────────────────
nodes=( $( scontrol show hostnames ${SLURM_JOB_NODELIST} ) )
head_node="${nodes[0]}"
# Using hostname -I to get the specific HSN IP address for rendezvous
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "${head_node}" hostname -I | awk '{print $1}')

# ── Load Environment ──────────────────────────────────────────────────────
# Note: deltaAI uses specific modules for PyTorch optimization
module load python/miniforge3_pytorch

# ── NCCL & Network Settings (Slingshot 11) ────────────────────────────────
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn     # Critical for deltaAI interconnect
export LOGLEVEL=INFO

# ── Launch DDP training ───────────────────────────────────────────────────
# We use -n 4 (one task per node) and let torchrun manage the 4 GPUs/node
srun -n 4 --cpu-bind=cores \
    torchrun \
    --nnodes=4 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${head_node_ip}:29500" \
    SFNO/train_forecast.py
