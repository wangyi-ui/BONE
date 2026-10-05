"""Post-seal-only evaluator for the P2-B0 2x2 replacement diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from experiments.utility_relation.bone_p2_a0_protocol import PARENT_ARTIFACT
from experiments.utility_relation.bone_p2_b0_protocol import ARMS, DEFAULT_OUTPUT_DIR, NEW_ARMS, PARENT_P2_A1_DIR
from experiments.utility_relation.evaluate_bone_p2_a1_pilot import verify_pre_gt_seal
from experiments.utility_relation.materialize_bone_p2_a0_carrier import sha256_file
from experiments.utility_relation.run_bone_p2_b0_replacement import verify_parent_controls
from utils.method_util import get_cluster_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]


def verify_new_arm_seal(arm_dir: Path | str) -> None:
    """Fail closed unless a new P2-B0 prediction and its frozen dependencies all verify."""
    arm_dir = Path(arm_dir)
    seal_path = arm_dir / "pre_gt_seal.json"
    if not seal_path.is_file():
        raise ValueError("missing P2-B0 pre-GT seal")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    try:
        expected = seal["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("invalid P2-B0 pre-GT seal") from error
    targets = {
        "predictions": arm_dir / "final_predictions.npz", "audit": arm_dir / "pre_gt_audit.json",
        "runner_source": REPO_ROOT / "experiments/utility_relation/run_bone_p2_b0_replacement.py",
        "replacement_source": REPO_ROOT / "experiments/utility_relation/bone_p2_b0_replacement.py",
        "protocol_source": REPO_ROOT / "experiments/utility_relation/bone_p2_b0_protocol.py",
        "method_bone": REPO_ROOT / "method/bone.py", "method_amv": REPO_ROOT / "method/amv.py",
        "carrier_npz": REPO_ROOT / "outputs/bone_p2/p2_a0_msrc_seed20/p2_a0_carrier_pre_gt.npz",
        "initial_model_state": REPO_ROOT / "outputs/bone_p2/p2_a0_msrc_seed20/carrier_model_state.pt",
    }
    if set(expected) != set(targets):
        raise ValueError("invalid P2-B0 pre-GT seal schema")
    for name, path in targets.items():
        if not path.is_file() or sha256_file(path) != expected[name]:
            raise ValueError(f"invalid P2-B0 pre-GT seal entry: {name}")


def _metrics_for_prediction(labels: np.ndarray, path: Path) -> Dict[str, float]:
    with np.load(path, allow_pickle=False) as predictions_file:
        predictions = np.asarray(predictions_file["predictions"], dtype=np.int64)
    acc, nmi, purity = get_cluster_metrics(labels.copy(), predictions, nmi=False)
    return {"ACC": float(acc), "NMI": float(nmi), "Purity": float(purity)}


def replacement_verdict(result: Dict[str, object]) -> str:
    """Apply the preregistered replacement gate without changing any reported metric."""
    if (result["Delta_REPLACE_ADDON_ACC"] >= .01 and result["Delta_REPLACE_REG_ACC"] >= .01
            and result["Interaction_ACC"] > 0 and (result["Delta_REPLACE_ADDON_NMI"] > 0 or result["Delta_REPLACE_ADDON_Purity"] > 0)):
        return "P2_B0_REPLACEMENT_GO"
    if result["Delta_REPLACE_ADDON_ACC"] <= 0:
        return "P2_B0_REPLACEMENT_FAIL_CLOSED"
    return "P2_B0_REPLACEMENT_INCONCLUSIVE"


def evaluate(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Dict[str, object]:
    """Validate parent/new seals first, then load GT once for all four registered conditions."""
    output_dir = Path(output_dir)
    parent_metrics = verify_parent_controls()
    for source_arm in ("BASE", "TRUE_U"):
        verify_pre_gt_seal(PARENT_P2_A1_DIR / source_arm)
    for arm in NEW_ARMS:
        verify_new_arm_seal(output_dir / arm)
    with np.load(PARENT_ARTIFACT, allow_pickle=False) as artifact:
        labels = np.asarray(artifact["labels"], dtype=np.int64)
    metrics = {
        "NATIVE": _metrics_for_prediction(labels, PARENT_P2_A1_DIR / "BASE/final_predictions.npz"),
        "ADDON_TRUE_U": _metrics_for_prediction(labels, PARENT_P2_A1_DIR / "TRUE_U/final_predictions.npz"),
        "REG_ONLY": _metrics_for_prediction(labels, output_dir / "REG_ONLY/final_predictions.npz"),
        "REPLACE_TRUE_U": _metrics_for_prediction(labels, output_dir / "REPLACE_TRUE_U/final_predictions.npz"),
    }
    for arm in ("NATIVE", "ADDON_TRUE_U"):
        if metrics[arm] != parent_metrics[arm]:
            raise ValueError("frozen parent metrics do not match sealed parent predictions")
    def delta(metric: str, left: str, right: str) -> float:
        return metrics[left][metric] - metrics[right][metric]
    result: Dict[str, object] = {"arms": metrics}
    for metric in ("ACC", "NMI", "Purity"):
        result[f"Delta_REPLACE_ADDON_{metric}"] = delta(metric, "REPLACE_TRUE_U", "ADDON_TRUE_U")
        result[f"Delta_REPLACE_REG_{metric}"] = delta(metric, "REPLACE_TRUE_U", "REG_ONLY")
        result[f"Delta_REG_NATIVE_{metric}"] = delta(metric, "REG_ONLY", "NATIVE")
        result[f"Delta_ADDON_NATIVE_{metric}"] = delta(metric, "ADDON_TRUE_U", "NATIVE")
        result[f"Interaction_{metric}"] = result[f"Delta_REPLACE_REG_{metric}"] - result[f"Delta_ADDON_NATIVE_{metric}"]
    result["verdict"] = replacement_verdict(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true", help="verify seals then evaluate the P2-B0 2x2 design")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.execute:
        raise SystemExit("refusing to read GT without --execute")
    result = evaluate(args.output_dir)
    output = args.output_dir / "p2_b0_metrics.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
