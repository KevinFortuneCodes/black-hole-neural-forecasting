#!/bin/bash
# Copyright (c) 2026 Kevin Fortune. All rights reserved.
# Shared for review only; no license to use, copy, modify or distribute is granted. See LICENSE.
# Usage: bash SFNO/submit.sh [test|prod]
# Runs pre-flight checks before submitting any sbatch job.
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODE="${1:-prod}"

bash "$SCRIPT_DIR/check_nersc.sh"

case "$MODE" in
    test) sbatch "$SCRIPT_DIR/test_train_nersc.sh" ;;
    prod) sbatch "$SCRIPT_DIR/ddp_script_nersc.sh" ;;
    *)    echo "Unknown mode: $MODE (use 'test' or 'prod')"; exit 1 ;;
esac
