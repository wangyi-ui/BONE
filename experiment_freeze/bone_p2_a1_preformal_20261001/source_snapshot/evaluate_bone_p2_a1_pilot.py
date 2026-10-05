"""Post-seal-only evaluator for the four BONE-P2-A1 pilot arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from experiments.utility_relation.bone_p2_a0_protocol import PARENT_ARTIFACT
from experiments.utility_relation.bone_p2_a1_protocol import ARMS, DEFAULT_OUTPUT_DIR
from experiments.utility_relation.materialize_bone_p2_a0_carrier import sha256_file
from utils.method_util import get_cluster_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]


def verify_pre_gt_seal(arm_dir: Path | str) -> None:
    """Fail closed before GT unless an arm's prediction and frozen sources match its seal."""
    arm_dir = Path(arm_dir)
    seal_path = arm_dir / "pre_gt_seal.json"
    if not seal_path.is_file():
        raise ValueError("missing P2-A1 pre-GT seal")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    try:
        expected = seal["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("invalid P2-A1 pre-GT seal") from error
    targets = {
        "predictions": arm_dir / "final_predictions.npz",
        "audit": arm_dir / "pre_gt_audit.json",
        "runner_source": REPO_ROOT / "experiments/utility_relation/run_bone_p2_a1_pilot.py",
        "action_source": REPO_ROOT / "experiments/utility_relation/bone_p2_a1_relation_action.py",
        "protocol_source": REPO_ROOT / "experiments/utility_relation/bone_p2_a1_protocol.py",
        "method_bone": REPO_ROOT / "method/bone.py",
        "method_amv": REPO_ROOT / "method/amv.py",
        "carrier_npz": REPO_ROOT / "outputs/bone_p2/p2_a0_msrc_seed20/p2_a0_carrier_pre_gt.npz",
        "initial_model_state": REPO_ROOT / "outputs/bone_p2/p2_a0_msrc_seed20/carrier_model_state.pt",
    }
    for name, path in targets.items():
        if name not in expected or not path.is_file() or sha256_file(path) != expected[name]:
            raise ValueError(f"invalid P2-A1 pre-GT seal entry: {name}")


def evaluate(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Dict[str, object]:
    """Verify every arm first; only then read full GT and compute registered comparisons."""
    output_dir = Path(output_dir)
    for arm in ARMS:
        verify_pre_gt_seal(output_dir / arm)
    with np.load(PARENT_ARTIFACT, allow_pickle=False) as artifact:
        labels = np.asarray(artifact["labels"], dtype=np.int64)
    metrics: Dict[str, Dict[str, float]] = {}
    for arm in ARMS:
        with np.load(output_dir / arm / "final_predictions.npz", allow_pickle=False) as prediction_file:
            predictions = np.asarray(prediction_file["predictions"], dtype=np.int64)
        acc, nmi, purity = get_cluster_metrics(labels.copy(), predictions, nmi=False)
        metrics[arm] = {"ACC": float(acc), "NMI": float(nmi), "Purity": float(purity)}
    true, base, uniform, shuffled = metrics["TRUE_U"], metrics["BASE"], metrics["TRUE_UNIFORM"], metrics["SHUFFLE_REL"]
    delta_base_acc = true["ACC"] - base["ACC"]
    delta_uniform_acc = true["ACC"] - uniform["ACC"]
    delta_shuffle_acc = true["ACC"] - shuffled["ACC"]
    delta_base_nmi, delta_base_purity = true["NMI"] - base["NMI"], true["Purity"] - base["Purity"]
    if delta_base_acc >= .01 and delta_uniform_acc >= .01 and delta_shuffle_acc >= .01 and (delta_base_nmi > 0 or delta_base_purity > 0):
        verdict = "P2_A1_GO"
    elif delta_base_acc <= 0:
        verdict = "P2_A1_FAIL_CLOSED"
    else:
        verdict = "P2_A1_INCONCLUSIVE"
    return {"arms": metrics, "Delta_TRUE_BASE_ACC": delta_base_acc, "Delta_TRUE_UNIFORM_ACC": delta_uniform_acc,
            "Delta_TRUE_SHUFFLE_ACC": delta_shuffle_acc, "Delta_TRUE_BASE_NMI": delta_base_nmi,
            "Delta_TRUE_BASE_PURITY": delta_base_purity, "verdict": verdict}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true", help="verify all pre-GT seals and evaluate four arms")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.execute:
        raise SystemExit("refusing to read full GT without --execute")
    result = evaluate(args.output_dir)
    output = args.output_dir / "p2_a1_metrics.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
