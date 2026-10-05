#!/usr/bin/env bash

cd /root/autodl-tmp/BONE || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/bone

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

set +e

OUT="outputs/bone_p3/p3_c1_b1_final_formal"

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

SEEDS=(20 30 50)

slug_of () {
    case "$1" in
        HandWrite)   echo "handwrite" ;;
        BDGP)        echo "bdgp" ;;
        MSRC)        echo "msrc" ;;
        COIL20)      echo "coil20" ;;
        Caltech-5V)  echo "caltech_5v" ;;
        MNIST-USPS)  echo "mnist_usps" ;;
        ALOI)        echo "aloi" ;;
        CCV)         echo "ccv" ;;
        *) return 1 ;;
    esac
}

echo "======================================================================"
echo "P3-C1-B1 FINAL FORMAL EVALUATION"
echo "8 DATASETS × 3 SEEDS"
echo "RESUME-AWARE"
echo "START=$(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

PAIR=0

for dataset in "${DATASETS[@]}"
do
    slug=$(slug_of "$dataset") || exit 10

    for seed in "${SEEDS[@]}"
    do
        PAIR=$((PAIR + 1))

        RUN_ROOT="$OUT/$slug/seed$seed"
        METRICS="$RUN_ROOT/FORMAL_METRICS.json"
        COMPLETE="$RUN_ROOT/COMPLETE.json"
        SEAL="$RUN_ROOT/PRE_GT_SEAL.json"

        echo
        echo "######################################################################"
        echo "PAIR=$PAIR/24"
        echo "DATASET=$dataset"
        echo "SEED=$seed"
        echo "START=$(date '+%Y-%m-%d %H:%M:%S')"
        echo "######################################################################"

        # 已经完整评估过：不重复运行
        if [ -f "$METRICS" ]; then
            echo "FORMAL_METRICS_ALREADY_EXISTS=SKIP"
            continue
        fi

        # 如果 pre-GT 尚未完整，则先训练
        if [ ! -f "$COMPLETE" ] || [ ! -f "$SEAL" ]; then

            echo "PRE_GT_STATUS=RUN"

            PYTHONPATH=. python -u -m \
              experiments.utility_relation.p3_c1_b1_final_formal.run_final_formal \
              --dataset "$dataset" \
              --training-seed "$seed" \
              --device cuda:0 \
              --output-root "$OUT" \
              --execute

            RC_RUN=$?

            echo "PRE_GT_EXIT=$RC_RUN"

            if [ "$RC_RUN" -ne 0 ]; then
                echo "FAIL-CLOSED"
                echo "FAILED_DATASET=$dataset"
                echo "FAILED_SEED=$seed"
                echo "FAILED_STAGE=PRE_GT"
                exit "$RC_RUN"
            fi
        else
            echo "PRE_GT_ALREADY_COMPLETE=REUSE"
        fi

        # Post-seal GT reporting
        echo "POST_GT_STATUS=RUN"

        PYTHONPATH=. python -u -m \
          experiments.utility_relation.p3_c1_b1_final_formal.evaluate_final_formal \
          --dataset "$dataset" \
          --training-seed "$seed" \
          --output-root "$OUT" \
          --execute

        RC_EVAL=$?

        echo "POST_GT_EXIT=$RC_EVAL"

        if [ "$RC_EVAL" -ne 0 ]; then
            echo "FAIL-CLOSED"
            echo "FAILED_DATASET=$dataset"
            echo "FAILED_SEED=$seed"
            echo "FAILED_STAGE=POST_GT"
            exit "$RC_EVAL"
        fi

        if [ ! -f "$METRICS" ]; then
            echo "FAIL-CLOSED: FORMAL_METRICS.json NOT CREATED"
            exit 20
        fi

        echo "PAIR_COMPLETE=true"
        echo "END=$(date '+%Y-%m-%d %H:%M:%S')"

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
echo "ALL FORMAL PAIRS RETURNED EXIT=0"
echo "END=$(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

python - <<'PY'
from pathlib import Path

root = Path(
    "outputs/bone_p3/p3_c1_b1_final_formal"
)

files = sorted(root.glob("*/seed*/FORMAL_METRICS.json"))

print("FORMAL_METRICS_COUNT =", len(files))

for p in files:
    print(p)

if len(files) != 24:
    raise SystemExit(
        f"EXPECTED_24_FORMAL_METRICS_BUT_FOUND_{len(files)}"
    )

print("ALL_24_FORMAL_METRICS_PRESENT=true")
PY

