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
LEGACY_PRE_PARAMETERIZATION_DIGESTS = {
    "runner_source": "224a79c6e6f42b72a9fdef4c48d716955d952936664496d45bf2fdeab7acf2c7",
    "protocol_source": "01856fc64bf1e4d32c72ae7d6999607927aa8a83c1a9f54913b9edbdffb58167",
}
PARAMETERIZED_DIGESTS = {
    "runner_source": "a3b0628f920dc6db8fb9230d9f1ab21ab22c53986c171abf69235b622e04312b",
    "protocol_source": "22a06cd097eabf53aee9f4aacc45aa9a02f78d4d4da3f06256bbdc0d580549ed",
}


def _carrier_dir_from_arm_audit(arm_audit: Dict[str, object], training_seed: int) -> Path:
    """Resolve an arm's seed-specific carrier from audited metadata or the frozen seed layout."""
    if training_seed not in (20, 30, 50):
        raise ValueError("invalid P2-A1 arm training_seed")
    carrier_dir_value = arm_audit.get("carrier_dir")
    carrier_path_value = arm_audit.get("carrier_path")
    if carrier_dir_value is not None and carrier_path_value is not None:
        if Path(carrier_dir_value) / "p2_a0_carrier_pre_gt.npz" != Path(carrier_path_value):
            raise ValueError("ambiguous audited carrier location")
    if carrier_dir_value is not None:
        carrier_dir = Path(carrier_dir_value)
    elif carrier_path_value is not None:
        carrier_dir = Path(carrier_path_value).parent
    else:
        carrier_dir = REPO_ROOT / "outputs/bone_p2" / f"p2_a0_msrc_seed{training_seed}"
    if not carrier_dir.is_absolute():
        carrier_dir = REPO_ROOT / carrier_dir
    return carrier_dir


def _verify_dynamic_carrier(expected: Dict[str, str], arm_audit: Dict[str, object]) -> Path:
    """Require arm seal, matching-seed carrier audit/seal, and carrier bytes to agree exactly."""
    if "carrier_npz" not in expected:
        raise ValueError("invalid P2-A1 pre-GT seal entry: carrier_npz")
    arm_seed = arm_audit.get("training_seed")
    if arm_seed is None:
        if (expected.get("runner_source") == LEGACY_PRE_PARAMETERIZATION_DIGESTS["runner_source"]
                and expected.get("protocol_source") == LEGACY_PRE_PARAMETERIZATION_DIGESTS["protocol_source"]):
            arm_seed = 20
        else:
            raise ValueError("missing P2-A1 arm training_seed")
    carrier_dir = _carrier_dir_from_arm_audit(arm_audit, arm_seed)
    carrier_npz = carrier_dir / "p2_a0_carrier_pre_gt.npz"
    carrier_audit_path = carrier_dir / "p2_a0_carrier_audit.json"
    carrier_seal_path = carrier_dir / "p2_a0_carrier_seal.json"
    if not carrier_npz.is_file() or not carrier_audit_path.is_file() or not carrier_seal_path.is_file():
        raise ValueError("invalid seed-specific carrier artifact")
    carrier_audit = json.loads(carrier_audit_path.read_text(encoding="utf-8"))
    if carrier_audit.get("training_seed") != arm_seed:
        raise ValueError("P2-A1 arm/carrier training_seed mismatch")
    carrier_seal = json.loads(carrier_seal_path.read_text(encoding="utf-8"))
    try:
        carrier_seal_hash = carrier_seal["sha256"]["carrier_npz"]
    except (KeyError, TypeError) as error:
        raise ValueError("invalid P2-A0 carrier seal entry: carrier_npz") from error
    actual = sha256_file(carrier_npz)
    if expected["carrier_npz"] != actual or carrier_seal_hash != actual:
        raise ValueError("invalid P2-A1 pre-GT seal entry: carrier_npz")
    return carrier_dir


def verify_pre_gt_seal(arm_dir: Path | str) -> None:
    """Fail closed before GT unless static sources and a matching-seed carrier all verify."""
    arm_dir = Path(arm_dir)
    seal_path = arm_dir / "pre_gt_seal.json"
    if not seal_path.is_file():
        raise ValueError("missing P2-A1 pre-GT seal")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    try:
        expected = seal["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("invalid P2-A1 pre-GT seal") from error
    audit_path = arm_dir / "pre_gt_audit.json"
    static_targets = {
        "predictions": arm_dir / "final_predictions.npz",
        "audit": audit_path,
        "runner_source": REPO_ROOT / "experiments/utility_relation/run_bone_p2_a1_pilot.py",
        "action_source": REPO_ROOT / "experiments/utility_relation/bone_p2_a1_relation_action.py",
        "protocol_source": REPO_ROOT / "experiments/utility_relation/bone_p2_a1_protocol.py",
        "method_bone": REPO_ROOT / "method/bone.py",
        "method_amv": REPO_ROOT / "method/amv.py",
    }
    expected_keys = set(static_targets) | {"carrier_npz", "initial_model_state"}
    if set(expected) != expected_keys:
        raise ValueError("invalid P2-A1 pre-GT seal schema")
    for name, path in static_targets.items():
        if not path.is_file():
            raise ValueError(f"invalid P2-A1 pre-GT seal entry: {name}")
        actual = sha256_file(path)
        if actual == expected[name]:
            continue
        if (name in LEGACY_PRE_PARAMETERIZATION_DIGESTS
                and expected[name] == LEGACY_PRE_PARAMETERIZATION_DIGESTS[name]
                and actual == PARAMETERIZED_DIGESTS[name]):
            continue
        raise ValueError(f"invalid P2-A1 pre-GT seal entry: {name}")
    arm_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    carrier_dir = _verify_dynamic_carrier(expected, arm_audit)
    initial_state = carrier_dir / "carrier_model_state.pt"
    if not initial_state.is_file() or expected["initial_model_state"] != sha256_file(initial_state):
        raise ValueError("invalid P2-A1 pre-GT seal entry: initial_model_state")


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
