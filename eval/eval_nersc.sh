#!/bin/bash
# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.

#SBATCH --account=YOUR_ALLOCATION
#SBATCH --job-name=eval_forecast
#SBATCH --constraint=gpu&hbm80g
#SBATCH -G 1
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=eval_%j.log

export PYTHONPATH=~/black-hole-neural-forecasting:$PYTHONPATH

srun -n 1 -c 32 --gpu-bind=none \
    ~/miniforge3/envs/neural_operator/bin/python \
    ~/black-hole-neural-forecasting/eval/evaluate_forecast.py \
    --mode all --start 5000 --max_steps 50 --save_dir eval_results

# Copy training loss history (includes loss_scales normalization weights)
LOSS_SRC=/path/to/sfno_data/processed_forecast/logs/loss_history.npz
if [ -f "$LOSS_SRC" ]; then
    cp "$LOSS_SRC" ~/black-hole-neural-forecasting/eval_results/
    echo "Copied loss_history.npz"
else
    echo "Warning: $LOSS_SRC not found — skipping"
fi

# Generate loss curves plot from loss history
~/miniforge3/envs/neural_operator/bin/python ~/black-hole-neural-forecasting/eval/plot_loss.py

# Commit and push eval results to GitHub
cd ~/black-hole-neural-forecasting
git pull --rebase origin SFNO
git add eval_results/
git commit -m "eval results: $(date +%Y-%m-%d) job ${SLURM_JOB_ID}"
git push origin SFNO
