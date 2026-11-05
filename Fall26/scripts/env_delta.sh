# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.
# Sourced by the job scripts: loads PyTorch on Delta.
# pytorch-conda/2.12 only loads on RH9.6 nodes; RH9.4 nodes need pytorch-conda/2.8.
# Uses `module reset`, never `module purge` (purge drops craype, which pytorch-conda needs).
# PYTORCH_MODULE picks another module: eval_delta.sbatch uses pytorch-conda/2.8 because
# matplotlib in pytorch-conda/2.12 cannot save PNGs (FreeType "raster overflow", 2026-10-22).
PYTORCH_MODULE=${PYTORCH_MODULE:-pytorch-conda/2.12}

module reset > /dev/null 2>&1
module load "$PYTORCH_MODULE" > /dev/null 2>&1
if ! python -c "import torch" > /dev/null 2>&1; then
    echo "$PYTORCH_MODULE unavailable on $(hostname) ($(cat /etc/redhat-release)); using pytorch-conda/2.8"
    module reset > /dev/null 2>&1
    module load pytorch-conda/2.8
fi

export OMP_NUM_THREADS=4
python -c "import torch; print(f'torch {torch.__version__} | CUDA {torch.version.cuda} | {torch.cuda.device_count()} GPU(s)')"
