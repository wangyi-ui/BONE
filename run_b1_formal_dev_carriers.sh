#!/usr/bin/env bash

cd /root/autodl-tmp/BONE || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/bone

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

set +e

OUTROOT="outputs/bone_p3/p3_c1_b1_optuna/formal_tuning"

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

TOTAL=16
COUNT=0

echo "======================================================================"
echo "P3-C1-B1-A1 FORMAL DEV CARRIER MATERIALIZATION"
echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do

    COUNT=$((COUNT + 1))

    echo
    echo "======================================================================"
    echo "[$COUNT/$TOTAL] DATASET=$dataset DEV_SEED=$seed"
    echo "START=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================================================"

    START_TS=$(date +%s)

    PYTHONPATH=. python \
      experiments/utility_relation/p3_c1_b1_optuna/materialize_formal_dev_carrier.py \
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
      echo "Completed carriers will NOT be overwritten on resume."
      echo "######################################################################"
      exit "$RC"
    fi

    # Release cached CUDA memory between carriers.
    python - <<'PY'
import gc
import torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PY

  done
done

echo
echo "======================================================================"
echo "ALL MATERIALIZATION COMMANDS FINISHED"
echo "END_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

echo
echo "========== FINAL 16-CARRIER AUDIT =========="

PYTHONPATH=. python \
  experiments/utility_relation/p3_c1_b1_optuna/audit_formal_dev_carriers.py \
  --output-root "$OUTROOT" \
  --output-json \
  outputs/bone_p3/p3_c1_b1_optuna/formal_dev_carrier_inventory.json

RC_AUDIT=$?

echo
echo "FINAL_AUDIT_EXIT=$RC_AUDIT"
exit "$RC_AUDIT"
