#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/reproducibility/frozen_inputs"

mkdir -p \
  "$ROOT/outputs/bone_p3/sparse_splits" \
  "$ROOT/outputs/bone_p3/corruptions" \
  "$ROOT/outputs/bone_p3/ccv_incremental"

cp "$SRC/MSRC_2perclass.npz" \
   "$ROOT/outputs/bone_p3/sparse_splits/MSRC_2perclass.npz"
cp "$SRC/MSRC_mask.npz" \
   "$ROOT/outputs/bone_p3/corruptions/MSRC_mask.npz"

for D in BDGP HandWrite Caltech-5V MNIST-USPS COIL20 ALOI
do
    cp "$SRC/${D}_2perclass.npz" \
       "$ROOT/outputs/bone_p3/sparse_splits/${D}_2perclass.npz"

    cp "$SRC/${D}_mask.npz" \
       "$ROOT/outputs/bone_p3/corruptions/${D}_mask.npz"
done

cp "$SRC/CCV_2perclass.npz" \
   "$ROOT/outputs/bone_p3/ccv_incremental/CCV_2perclass.npz"

cp "$SRC/CCV_mask.npz" \
   "$ROOT/outputs/bone_p3/ccv_incremental/CCV_corruption_mask.npz"

echo "FROZEN_INPUT_RESTORE=PASS"
