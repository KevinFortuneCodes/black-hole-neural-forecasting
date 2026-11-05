#!/bin/bash
# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.
# Submit a chain of 48-hour training jobs for one experiment and seed; each starts when the
# previous one ends and resumes from its latest checkpoint. Jobs after the run finishes or
# stops early exit immediately. Jobs are named <experiment>_s<seed>, and so are their logs
# in Fall26/outputs/slurm/.
# Usage, from the repo root:  Fall26/scripts/submit_chain.sh <experiment> <seed> [jobs, default 2]
set -e
experiment=${1:?usage: submit_chain.sh <experiment> <seed> [jobs]}
seed=${2:?usage: submit_chain.sh <experiment> <seed> [jobs]}
jobs=${3:-2}
python3 - "$experiment" <<'EOF' || { echo "unknown experiment: $experiment (see EXPERIMENTS in Fall26/vit/config.py)"; exit 1; }
import sys
sys.path.insert(0, "Fall26")
from vit.config import EXPERIMENTS          # plain Python; no torch needed
sys.exit(0 if sys.argv[1] in EXPERIMENTS else 1)
EOF
mkdir -p Fall26/outputs/slurm
prev=""
for i in $(seq 1 "$jobs"); do
    dep=${prev:+--dependency=afterany:$prev}
    prev=$(sbatch --parsable $dep --job-name="${experiment}_s${seed}" \
           --export=ALL,EXPERIMENT="$experiment",SEED="$seed" Fall26/scripts/train_delta.sbatch)
    echo "$experiment seed $seed, job $i of $jobs: $prev"
done
