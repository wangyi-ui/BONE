#!/usr/bin/env bash

cd /root/autodl-tmp/BONE || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/bone

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

set +e

OUTROOT="outputs/bone_p3/p3_c1_b1_optuna/formal_tuning"

R1_FREEZE="experiment_freeze/bone_p3_c1_b1_a1_formal_carrier_materializer_r1_20261003"

LOG_ROOT="logs/bone_p3/p3_c1_b1_optuna"

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

SEEDS=(11 37)

echo "======================================================================"
echo "P3-C1-B1-A1 FORMAL DEV CARRIER MATERIALIZATION R1"
echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

# ---------------------------------------------------------------------
# 1. R1 freeze status
# ---------------------------------------------------------------------
echo
echo "========== 1. R1 FREEZE STATUS =========="

if [ ! -f "$R1_FREEZE/STATUS.txt" ]; then
    echo "FAIL-CLOSED: R1 STATUS.txt missing"
    exit 10
fi

cat "$R1_FREEZE/STATUS.txt"

grep -qx 'STATUS=FORMAL_CARRIER_MATERIALIZER_R1_FROZEN' \
    "$R1_FREEZE/STATUS.txt"

RC_STATUS=$?

echo "R1_STATUS_CHECK=$RC_STATUS"

if [ "$RC_STATUS" -ne 0 ]; then
    echo "FAIL-CLOSED: R1 freeze status invalid"
    exit 11
fi


# ---------------------------------------------------------------------
# 2. Verify R1 freeze manifests
# ---------------------------------------------------------------------
echo
echo "========== 2. R1 FREEZE HASH VERIFICATION =========="

for MANIFEST in \
    new_source_sha256.txt \
    frozen_dependency_sha256.txt \
    parent_source_sha256.txt
do
    FILE="$R1_FREEZE/$MANIFEST"

    if [ ! -f "$FILE" ]; then
        echo "FAIL-CLOSED: missing $FILE"
        exit 12
    fi

    echo
    echo "--- $MANIFEST ---"

    sha256sum -c "$FILE"
    RC=$?

    if [ "$RC" -ne 0 ]; then
        echo "FAIL-CLOSED: manifest verification failed: $MANIFEST"
        exit 13
    fi
done


# ---------------------------------------------------------------------
# 3. GPU gate
# ---------------------------------------------------------------------
echo
echo "========== 3. GPU CHECK =========="

python - <<'PY'
import sys
import torch

print("torch =", torch.__version__)
print("CUDA available =", torch.cuda.is_available())
print("CUDA device count =", torch.cuda.device_count())

if not torch.cuda.is_available():
    print("FAIL-CLOSED: CUDA unavailable")
    sys.exit(1)

print("GPU =", torch.cuda.get_device_name(0))
PY

RC_GPU=$?

if [ "$RC_GPU" -ne 0 ]; then
    echo "FAIL-CLOSED: GPU check failed"
    exit 14
fi


# ---------------------------------------------------------------------
# 4. Verify R1 CLI
# ---------------------------------------------------------------------
echo
echo "========== 4. R1 CLI CHECK =========="

R1_RUNNER="experiments/utility_relation/p3_c1_b1_optuna/materialize_formal_dev_carrier_r1.py"

if [ ! -f "$R1_RUNNER" ]; then
    echo "FAIL-CLOSED: R1 runner missing"
    exit 15
fi

PYTHONPATH=. python "$R1_RUNNER" --help
RC_HELP=$?

if [ "$RC_HELP" -ne 0 ]; then
    echo "FAIL-CLOSED: R1 CLI help failed"
    exit 16
fi


# ---------------------------------------------------------------------
# 5. Read-only inventory before resume
# ---------------------------------------------------------------------
echo
echo "========== 5. INVENTORY BEFORE RESUME =========="

mkdir -p "$LOG_ROOT"

PYTHONPATH=. python \
  experiments/utility_relation/p3_c1_b1_optuna/audit_formal_dev_carriers.py \
  --output-root "$OUTROOT" \
  --output \
  outputs/bone_p3/p3_c1_b1_optuna/formal_dev_carrier_inventory_before_r1_resume.json

RC_INITIAL_AUDIT=$?

echo "INITIAL_AUDIT_EXIT=$RC_INITIAL_AUDIT"
echo "NOTE: inventory is expected to be incomplete before resume."


# ---------------------------------------------------------------------
# 6. Resume all expected identities
# ---------------------------------------------------------------------
echo
echo "========== 6. FORMAL CARRIER RESUME =========="

TOTAL=16
COUNT=0

for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do

    COUNT=$((COUNT + 1))

    echo
    echo "======================================================================"
    echo "[$COUNT/$TOTAL] DATASET=$dataset DEV_SEED=$seed"
    echo "START=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================================================"

    START_TS=$(date +%s)

    PYTHONPATH=. python "$R1_RUNNER" \
      --dataset "$dataset" \
      --dev-seed "$seed" \
      --device cuda:0 \
      --output-root "$OUTROOT"

    RC=$?

    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))

    echo
    echo "[$COUNT/$TOTAL] EXIT_CODE=$RC"
    echo "[$COUNT/$TOTAL] ELAPSED_SECONDS=$ELAPSED"
    echo "[$COUNT/$TOTAL] END=$(date '+%Y-%m-%d %H:%M:%S')"

    if [ "$RC" -ne 0 ]; then
        echo
        echo "######################################################################"
        echo "FAIL-CLOSED"
        echo "FAILED_DATASET=$dataset"
        echo "FAILED_DEV_SEED=$seed"
        echo "EXIT_CODE=$RC"
        echo
        echo "Previously verified COMPLETE carriers remain untouched."
        echo "Do NOT delete formal_tuning."
        echo "######################################################################"
        exit "$RC"
    fi

    # Release cached GPU memory between carriers.
    python - <<'PY'
import gc
import torch

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
PY

  done
done


# ---------------------------------------------------------------------
# 7. Final formal carrier audit
# ---------------------------------------------------------------------
echo
echo "======================================================================"
echo "ALL 16 MATERIALIZATION IDENTITIES PROCESSED"
echo "END_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

echo
echo "========== 7. FINAL 16-CARRIER AUDIT =========="

PYTHONPATH=. python \
  experiments/utility_relation/p3_c1_b1_optuna/audit_formal_dev_carriers.py \
  --output-root "$OUTROOT" \
  --output \
  outputs/bone_p3/p3_c1_b1_optuna/formal_dev_carrier_inventory.json

RC_FINAL_AUDIT=$?

echo
echo "FINAL_AUDIT_EXIT=$RC_FINAL_AUDIT"

if [ "$RC_FINAL_AUDIT" -ne 0 ]; then
    echo "FINAL_CARRIER_AUDIT=FAIL"
    exit "$RC_FINAL_AUDIT"
fi

echo
echo "======================================================================"
echo "FORMAL DEV CARRIER R1 JOB FINISHED"
echo "Check inventory for:"
echo "expected_carrier_count = 16"
echo "complete_verified_count = 16"
echo "missing_count = 0"
echo "invalid_count = 0"
echo "ALL_16_FORMAL_DEV_CARRIERS_READY = true"
echo "======================================================================"

exit 0
