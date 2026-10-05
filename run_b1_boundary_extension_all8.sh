#!/usr/bin/env bash

cd /root/autodl-tmp/BONE || exit 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/bone

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

set +e

ROOT="outputs/bone_p3/p3_c1_b1_optuna/formal_tuning"
REPO="/root/autodl-tmp/BONE"

FREEZE="experiment_freeze/bone_p3_c1_b1_a2_upper_lr_boundary_extension_preregistered_20261004"

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
echo "P3-C1-B1-A2 UPPER-LR BOUNDARY EXTENSION"
echo "LR = 1e-3"
echo "EXPOSURES = 1,3,5,10,20"
echo "8 DATASETS × 5 CANDIDATES = 40"
echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

echo
echo "========== 1. FREEZE GATE =========="

grep -qx \
  'STATUS=P3_C1_B1_A2_UPPER_LR_BOUNDARY_EXTENSION_FROZEN' \
  "$FREEZE/STATUS.txt"

RC_FREEZE=$?

grep -qx \
  'ALL_EIGHT_EXTENSION_READY=true' \
  "$FREEZE/STATUS.txt"

RC_READY=$?

grep -qx \
  'EXTENSION_LR=0.001' \
  "$FREEZE/STATUS.txt"

RC_LR=$?

grep -qx \
  'TOTAL_EXTENSION_CANDIDATES=40' \
  "$FREEZE/STATUS.txt"

RC_COUNT=$?

if [ "$RC_FREEZE" -ne 0 ] || \
   [ "$RC_READY" -ne 0 ] || \
   [ "$RC_LR" -ne 0 ] || \
   [ "$RC_COUNT" -ne 0 ]; then

    echo "FAIL-CLOSED: EXTENSION FREEZE GATE FAILED"
    exit 10
fi

echo "EXTENSION_FREEZE_GATE=PASS"

echo
echo "========== 2. RUNNER ENTRY GATE =========="

PYTHONPATH=. python -m \
  experiments.utility_relation.p3_c1_b1_optuna.run_boundary_extension \
  --help >/dev/null 2>&1

RC_RUNNER=$?

if [ "$RC_RUNNER" -ne 0 ]; then
    echo "FAIL-CLOSED: BOUNDARY RUNNER UNAVAILABLE"
    exit 11
fi

echo "BOUNDARY_RUNNER_GATE=PASS"

echo
echo "========== 3. GPU GATE =========="

python - <<'PY'
import sys
import torch

print("torch =", torch.__version__)
print("CUDA available =", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("CUDA_GATE=FAIL")
    sys.exit(1)

print("GPU =", torch.cuda.get_device_name(0))
print("CUDA_GATE=PASS")
PY

RC_GPU=$?

if [ "$RC_GPU" -ne 0 ]; then
    exit 12
fi

echo
echo "========== 4. BEGIN REAL EXTENSION =========="

COUNT=0
TOTAL=8

for dataset in "${DATASETS[@]}"
do
    COUNT=$((COUNT + 1))

    echo
    echo "######################################################################"
    echo "[$COUNT/$TOTAL] DATASET=$dataset"
    echo "EXTENSION_LR=1e-3"
    echo "EXTENSION_EXPOSURES=1,3,5,10,20"
    echo "START=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "######################################################################"

    START_TS=$(date +%s)

    PYTHONPATH=. python -u -m \
      experiments.utility_relation.p3_c1_b1_optuna.run_boundary_extension \
      --dataset "$dataset" \
      --device cuda:0 \
      --formal-tuning-root "$ROOT" \
      --repository-root "$REPO"

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
        echo "BOUNDARY EXTENSION FAIL-CLOSED"
        echo "FAILED_DATASET=$dataset"
        echo "EXIT_CODE=$RC"
        echo
        echo "DO NOT DELETE:"
        echo "  boundary extension study"
        echo "  extension candidate artifacts"
        echo "  original 25 candidates"
        echo "  BASE references"
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
echo "ALL 8 BOUNDARY EXTENSION RUNS RETURNED EXIT=0"
echo "TOTAL_EXPECTED_NEW_CANDIDATES=40"
echo "END_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

exit 0
