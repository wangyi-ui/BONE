#!/usr/bin/env bash

cd /root/autodl-tmp/BONE || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/bone

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

set +e

ROOT="outputs/bone_p3/p3_c1_b1_optuna/formal_tuning"
FREEZE="experiment_freeze/bone_p3_c1_b1_a1_formal_optuna_runner_r2_20261003"

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

echo "======================================================================"
echo "P3-C1-B1-A1 FORMAL OPTUNA R2 MAIN GRID"
echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

echo
echo "========== R2 FREEZE CHECK =========="

grep -qx \
  'STATUS=FORMAL_OPTUNA_RUNNER_R2_FROZEN' \
  "$FREEZE/STATUS.txt"

RC_FREEZE=$?

if [ "$RC_FREEZE" -ne 0 ]; then
    echo "FAIL-CLOSED: R2 freeze invalid"
    exit 10
fi

echo "R2_FREEZE=PASS"

echo
echo "========== GPU CHECK =========="

python - <<'PY'
import sys
import torch

print("torch =", torch.__version__)
print("CUDA available =", torch.cuda.is_available())

if not torch.cuda.is_available():
    sys.exit(1)

print("GPU =", torch.cuda.get_device_name(0))
PY

RC_GPU=$?

if [ "$RC_GPU" -ne 0 ]; then
    echo "FAIL-CLOSED: CUDA unavailable"
    exit 11
fi

TOTAL=8
COUNT=0

for dataset in "${DATASETS[@]}"; do

    COUNT=$((COUNT + 1))

    echo
    echo "######################################################################"
    echo "[$COUNT/$TOTAL] DATASET=$dataset"
    echo "START=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "######################################################################"

    START_TS=$(date +%s)

    PYTHONPATH=. python -u -m \
      experiments.utility_relation.p3_c1_b1_optuna.run_formal_optuna_tuning_r2 \
      --dataset "$dataset" \
      --device cuda:0 \
      --formal-tuning-root "$ROOT"

    RC=$?

    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))

    echo
    echo "======================================================================"
    echo "DATASET=$dataset"
    echo "EXIT_CODE=$RC"
    echo "ELAPSED_SECONDS=$ELAPSED"
    echo "END=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================================================"

    if [ "$RC" -ne 0 ]; then
        echo
        echo "######################################################################"
        echo "FORMAL OPTUNA R2 FAIL-CLOSED"
        echo "FAILED_DATASET=$dataset"
        echo "EXIT_CODE=$RC"
        echo
        echo "DO NOT DELETE:"
        echo "  study_r2.db"
        echo "  candidate artifacts"
        echo "  base_reference"
        echo "  carriers"
        echo "######################################################################"
        exit "$RC"
    fi

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
echo "ALL 8 R2 MAIN-GRID RUNS RETURNED EXIT=0"
echo "END_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

exit 0
