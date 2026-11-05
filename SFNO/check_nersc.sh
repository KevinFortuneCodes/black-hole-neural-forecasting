#!/bin/bash
# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.
# Run on NERSC login node before any sbatch to catch path/env errors early.
set -e

REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
PYTHON=~/miniforge3/envs/neural_operator/bin/python
echo "Repo root: $REPO"

# ── Script paths ──────────────────────────────────────────────────────────────
for script in ddp_script_nersc.sh test_train_nersc.sh; do
    echo -n "Checking paths in $script ... "
    while IFS= read -r path; do
        expanded="${path/\~/$HOME}"
        if [[ ! -e "$expanded" ]]; then
            echo "FAIL"
            echo "  Missing: $expanded  (referenced in $script)"
            exit 1
        fi
    done < <(grep -oP '~/\S+\.py' "$REPO/SFNO/$script" | grep -v '\$')
    echo "OK"
done

# ── Python entrypoints ────────────────────────────────────────────────────────
echo -n "Checking train_forecast.py importable ... "
$PYTHON -c "
import sys
sys.path.insert(0, '$REPO/SFNO')
import train_forecast
print('OK')
"

# ── Data directory ────────────────────────────────────────────────────────────
echo -n "Checking data_dir accessible ... "
DATA_DIR=$($PYTHON -c "import sys; sys.path.insert(0, '$REPO/SFNO'); import train_forecast; print(train_forecast.CONFIG['data_dir'])")
if [[ ! -d "$DATA_DIR" ]]; then
    echo "FAIL — $DATA_DIR not found"
    exit 1
fi
N=$(ls "$DATA_DIR"/dump_*.npy 2>/dev/null | wc -l)
if [[ $N -eq 0 ]]; then
    echo "FAIL — no dump_*.npy files in $DATA_DIR"
    exit 1
fi
echo "OK ($N files)"

# ── Checkpoint / log dirs writable ───────────────────────────────────────────
echo -n "Checking checkpoint/log dirs writable ... "
CKPT_DIR=$($PYTHON -c "import sys; sys.path.insert(0, '$REPO/SFNO'); import train_forecast; print(train_forecast.CONFIG['checkpoint_dir'])")
LOG_DIR=$($PYTHON -c "import sys; sys.path.insert(0, '$REPO/SFNO'); import train_forecast; print(train_forecast.CONFIG['log_dir'])")
mkdir -p "$CKPT_DIR" "$LOG_DIR"
touch "$CKPT_DIR/.write_test" && rm "$CKPT_DIR/.write_test"
touch "$LOG_DIR/.write_test"  && rm "$LOG_DIR/.write_test"
echo "OK"

echo ""
echo "All checks passed — safe to sbatch."
