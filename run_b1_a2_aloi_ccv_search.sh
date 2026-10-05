#!/usr/bin/env bash

cd /root/autodl-tmp/BONE || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/bone

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

set +e

ROOT="outputs/bone_p3/p3_c1_b1_optuna/formal_tuning"
FREEZE="experiment_freeze/bone_p3_c1_b1_a2_coverage_aware_abstention_preregistered_20261003"

DATASETS=(
  "ALOI"
  "CCV"
)

echo "======================================================================"
echo "P3-C1-B1-A2 FORMAL PARAMETER SEARCH"
echo "ALOI + CCV"
echo "A2 RUNNER ONLY"
echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

echo
echo "========== A2 FREEZE CHECK =========="

grep -qx \
  'STATUS=P3_C1_B1_A2_COVERAGE_AWARE_ABSTENTION_FROZEN' \
  "$FREEZE/STATUS.txt"

RC_FREEZE=$?

if [ "$RC_FREEZE" -ne 0 ]; then
    echo "FAIL-CLOSED: A2 freeze invalid"
    exit 10
fi

echo "A2_FREEZE=PASS"

echo
echo "========== EXACT RUNNER CHECK =========="

PYTHONPATH=. python -m \
  experiments.utility_relation.p3_c1_b1_optuna.run_formal_optuna_tuning_a2 \
  --help >/dev/null 2>&1

RC_RUNNER=$?

if [ "$RC_RUNNER" -ne 0 ]; then
    echo "FAIL-CLOSED: A2 runner unavailable"
    exit 11
fi

echo "A2_RUNNER=PASS"

echo
echo "========== GPU CHECK =========="

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

COUNT=0
TOTAL=2

for dataset in "${DATASETS[@]}"
do
    COUNT=$((COUNT + 1))

    echo
    echo "######################################################################"
    echo "[$COUNT/$TOTAL] DATASET=$dataset"
    echo "A2 RUNNER"
    echo "START=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "######################################################################"

    START_TS=$(date +%s)

    PYTHONPATH=. python -u -m \
      experiments.utility_relation.p3_c1_b1_optuna.run_formal_optuna_tuning_a2 \
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
        echo "A2 PARAMETER SEARCH FAIL-CLOSED"
        echo "FAILED_DATASET=$dataset"
        echo "EXIT_CODE=$RC"
        echo
        echo "DO NOT DELETE:"
        echo "  study_a2.db"
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
echo "ALOI + CCV A2 SEARCH RETURNED EXIT=0"
echo "END_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

exit 0
