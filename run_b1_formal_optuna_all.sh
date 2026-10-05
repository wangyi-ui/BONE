#!/usr/bin/env bash

cd /root/autodl-tmp/BONE || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/bone

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

set +e

ROOT="outputs/bone_p3/p3_c1_b1_optuna/formal_tuning"

DATASETS=(
  "HandWrite"
  "BDGP"
  "MSRC"
  "COIL20"
  "Caltech-5V"
  "MNIST-USPS"
  "ALOI"
  "CCV"
)

TOTAL=8
COUNT=0

echo "======================================================================"
echo "P3-C1-B1-A1 FORMAL OPTUNA MAIN GRID"
echo "25 CANDIDATES / DATASET"
echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

for dataset in "${DATASETS[@]}"; do

    COUNT=$((COUNT + 1))

    echo
    echo "######################################################################"
    echo "[$COUNT/$TOTAL] FORMAL OPTUNA DATASET=$dataset"
    echo "START=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "######################################################################"

    START_TS=$(date +%s)

    PYTHONPATH=. python -u \
      experiments/utility_relation/p3_c1_b1_optuna/run_formal_optuna_tuning.py \
      --dataset "$dataset" \
      --device cuda:0 \
      --formal-tuning-root "$ROOT"

    RC=$?

    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))

    echo
    echo "[$COUNT/$TOTAL] DATASET=$dataset"
    echo "EXIT_CODE=$RC"
    echo "ELAPSED_SECONDS=$ELAPSED"
    echo "END=$(date '+%Y-%m-%d %H:%M:%S')"

    if [ "$RC" -ne 0 ]; then
        echo
        echo "######################################################################"
        echo "FORMAL OPTUNA FAIL-CLOSED"
        echo "FAILED_DATASET=$dataset"
        echo "EXIT_CODE=$RC"
        echo
        echo "DO NOT DELETE study.db."
        echo "DO NOT DELETE completed candidate artifacts."
        echo "Resume should reuse verified SQLite/candidate cache."
        echo "######################################################################"
        exit "$RC"
    fi

    # Free cached GPU memory between datasets.
    python - <<'PY'
import gc
import torch

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
PY

done

echo
echo "======================================================================"
echo "ALL 8 FORMAL OPTUNA MAIN-GRID RUNS RETURNED EXIT=0"
echo "END_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

exit 0
