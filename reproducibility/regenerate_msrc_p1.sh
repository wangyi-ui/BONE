#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python experiments/weak_quality/materialize_bone_p1_msrc_corruption.py

echo \
"2b05b7b5adfe872c60475fe8d3d9f4959229a57ffb1eb50ff29adb015f38db87  outputs/bone_p1/msrc_snr2p5_exact2of4_seed20/bone_p1_msrc_corruption.npz" \
| sha256sum -c -

echo "MSRC_P1_REGENERATION=PASS"
