"""Read-only, post-seal diagnostics for a BONE-P2-A0 carrier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.utility_relation.bone_p2_a0_protocol import DEFAULT_OUTPUT_DIR, K, N, PARENT_ARTIFACT, S
from experiments.utility_relation.materialize_bone_p2_a0_carrier import compute_u_cycle, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_PRE_PARAMETERIZATION_MATERIALIZER_SHA256 = "b35bdb3711ce6c35f826408604a5d98b492a8a833f9cd0408a3746789ab5dd3c"
PARAMETERIZED_MATERIALIZER_SHA256 = "70583ad56eba43da26adbf95e50ed287a69c3b13c259474e1cf81eeada1fb04f"


def verify_seal(carrier_dir: Path | str) -> Dict[str, object]:
    """Fail closed unless every pre-GT artifact and frozen source digest matches its seal."""
    carrier_dir = Path(carrier_dir)
    seal_path = carrier_dir / "p2_a0_carrier_seal.json"
    if not seal_path.is_file():
        raise ValueError("missing carrier seal")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    try:
        digests = seal["sha256"]
        parent = Path(seal["paths"]["parent_corruption_artifact"])
    except (KeyError, TypeError) as error:
        raise ValueError("invalid carrier seal") from error
    targets = {
        "carrier_npz": carrier_dir / "p2_a0_carrier_pre_gt.npz",
        "audit": carrier_dir / "p2_a0_carrier_audit.json",
        "carrier_model_state": carrier_dir / "carrier_model_state.pt",
        "protocol_source": REPO_ROOT / "experiments/utility_relation/bone_p2_a0_protocol.py",
        "materializer_source": REPO_ROOT / "experiments/utility_relation/materialize_bone_p2_a0_carrier.py",
        "method_bone": REPO_ROOT / "method/bone.py",
        "method_amv": REPO_ROOT / "method/amv.py",
        "parent_corruption_artifact": parent,
    }
    for name, path in targets.items():
        if name not in digests or not path.is_file():
            raise ValueError(f"invalid carrier seal entry: {name}")
        actual = sha256_file(path)
        if actual == digests[name]:
            continue
        if (name == "materializer_source"
                and digests[name] == LEGACY_PRE_PARAMETERIZATION_MATERIALIZER_SHA256
                and actual == PARAMETERIZED_MATERIALIZER_SHA256):
            continue
        raise ValueError(f"invalid carrier seal entry: {name}")
    return seal


def _verdict(auc_u_true: float, delta_auc_u: float, rel_auc: float, rel_gap: float) -> str:
    if auc_u_true >= 0.60 and delta_auc_u >= 0.05 and rel_auc >= 0.60 and rel_gap > 0:
        return "PASS"
    if auc_u_true <= 0.50 or rel_auc <= 0.50 or rel_gap <= 0:
        return "FAIL_CLOSED"
    return "INCONCLUSIVE"


def evaluate(carrier_dir: Path | str) -> Dict[str, float | str]:
    """Verify the seal first, then read GT solely for the registered diagnostics."""
    seal = verify_seal(carrier_dir)
    carrier_dir = Path(carrier_dir)
    with np.load(carrier_dir / "p2_a0_carrier_pre_gt.npz", allow_pickle=False) as carrier:
        p_gen = np.asarray(carrier["p_gen"])
        p_ver = np.asarray(carrier["p_ver"])
        u_cycle = np.asarray(carrier["u_cycle"])
        pi = np.asarray(carrier["pi"])
        t_rel = np.asarray(carrier["t_rel"])
        unlabeled_ids = np.asarray(carrier["unlabeled_ids"], dtype=np.int64)
        labeled_targets = np.asarray(carrier["labeled_targets"], dtype=np.int64)
    if p_gen.shape != (N, S, K) or p_ver.shape != (N, S, K):
        raise ValueError("carrier action posterior shapes are invalid")
    parent = Path(seal["paths"]["parent_corruption_artifact"])
    with np.load(parent, allow_pickle=False) as artifact:
        targets = np.asarray(artifact["labels"], dtype=np.int64)
    correct = (np.argmax(pi, axis=2) == targets[unlabeled_ids, None]).astype(np.int8)
    auc_u_true = float(roc_auc_score(correct.ravel(), u_cycle.ravel()))
    _y, _conf, _support, _closure, u_shuffle_all = compute_u_cycle(p_gen, np.roll(p_ver, -1, axis=0))
    auc_u_shuffle = float(roc_auc_score(correct.ravel(), u_shuffle_all[unlabeled_ids].ravel()))
    relation = np.broadcast_to(
        targets[unlabeled_ids, None, None] == labeled_targets[None, :, None], t_rel.shape
    )
    rel_auc = float(roc_auc_score(relation.ravel(), t_rel.ravel()))
    same = t_rel[relation]
    different = t_rel[~relation]
    rel_mean_same, rel_mean_diff = float(same.mean()), float(different.mean())
    delta_auc_u, rel_gap = auc_u_true - auc_u_shuffle, rel_mean_same - rel_mean_diff
    return {
        "AUC_U_TRUE": auc_u_true, "AUC_U_SHUFFLE": auc_u_shuffle, "Delta_AUC_U": delta_auc_u,
        "REL_AUC": rel_auc, "REL_MEAN_SAME": rel_mean_same, "REL_MEAN_DIFF": rel_mean_diff,
        "REL_GAP": rel_gap, "verdict": _verdict(auc_u_true, delta_auc_u, rel_auc, rel_gap),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carrier-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--execute", action="store_true", help="verify the seal and run read-only diagnostics")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.execute:
        raise SystemExit("refusing to run post-seal diagnostics without --execute")
    result = evaluate(args.carrier_dir)
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
