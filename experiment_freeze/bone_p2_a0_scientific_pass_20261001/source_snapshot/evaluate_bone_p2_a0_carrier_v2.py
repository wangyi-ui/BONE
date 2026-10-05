"""Read-only V2 derangement-null diagnostics for the sealed BONE-P2-A0 carrier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.utility_relation.bone_p2_a0_protocol import DEFAULT_OUTPUT_DIR, K, N, NU, S
from experiments.utility_relation.evaluate_bone_p2_a0_carrier import _verdict, verify_seal
from experiments.utility_relation.materialize_bone_p2_a0_carrier import compute_u_cycle


NULL_SEED = 20
NULL_CONTROL_VERSION = "global_derangement_v2"
ADJACENT_SHIFT_SAME_FRACTION = 203 / 210
RANDOM_SAME_CLASS_REFERENCE = 1 / 7
DEFAULT_OUTPUT_NAME = "p2_a0_postseal_metrics_v2.json"


def global_derangement(size: int = NU, seed: int = NULL_SEED) -> np.ndarray:
    """Return the first seed-determined global derangement without inspecting any data."""
    if size <= 1:
        raise ValueError("a derangement requires at least two elements")
    rng = np.random.default_rng(seed)
    identity = np.arange(size, dtype=np.int64)
    while True:
        permutation = rng.permutation(size).astype(np.int64, copy=False)
        if not np.any(permutation == identity):
            return permutation


def validate_derangement(permutation: np.ndarray, size: int = NU) -> None:
    """Fail closed unless the null permutation is an exact fixed-point-free bijection."""
    permutation = np.asarray(permutation)
    if permutation.shape != (size,) or permutation.dtype != np.int64:
        raise ValueError("null permutation must be an int64 vector of the expected length")
    if permutation.min() != 0 or permutation.max() != size - 1:
        raise ValueError("null permutation bounds are invalid")
    if not np.array_equal(np.sort(permutation), np.arange(size, dtype=np.int64)):
        raise ValueError("null permutation is not a bijection")
    if np.any(permutation == np.arange(size, dtype=np.int64)):
        raise ValueError("null permutation has fixed points")


def permutation_sha256(permutation: np.ndarray) -> str:
    """Hash the frozen int64 permutation in a platform-independent byte order."""
    validate_derangement(permutation)
    stable_bytes = np.asarray(permutation, dtype="<i8").tobytes(order="C")
    return hashlib.sha256(stable_bytes).hexdigest()


def build_null_utility(p_gen_query: np.ndarray, p_ver_query: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Break only verifier sample correspondence; generator content stays byte-for-byte intact."""
    if p_gen_query.shape != (NU, S, K) or p_ver_query.shape != (NU, S, K):
        raise ValueError(f"query posteriors must both have shape {(NU, S, K)}")
    permutation = global_derangement()
    validate_derangement(permutation)
    p_ver_null = p_ver_query[permutation].copy()
    y_gen_true = np.argmax(p_gen_query, axis=2).astype(np.int64)
    conf_gen_true = np.take_along_axis(p_gen_query, y_gen_true[..., None], axis=2)[..., 0]
    y_gen_null, conf_gen_null, support_ver_null, closure_null, u_null = compute_u_cycle(
        p_gen_query, p_ver_null
    )
    if not np.array_equal(y_gen_null, y_gen_true):
        raise ValueError("null generator labels changed")
    if not np.array_equal(conf_gen_null, conf_gen_true):
        raise ValueError("null generator confidences changed")
    return (
        permutation,
        p_ver_null,
        y_gen_true,
        conf_gen_true,
        y_gen_null,
        conf_gen_null,
        support_ver_null,
        closure_null,
        u_null,
    )


def evaluate(carrier_dir: Path | str) -> Dict[str, float | int | str]:
    """Verify the original seal before GT and calculate the registered V2 null diagnostic."""
    seal = verify_seal(carrier_dir)
    carrier_dir = Path(carrier_dir)
    with np.load(carrier_dir / "p2_a0_carrier_pre_gt.npz", allow_pickle=False) as carrier:
        p_gen = np.asarray(carrier["p_gen"])
        p_ver = np.asarray(carrier["p_ver"])
        y_gen = np.asarray(carrier["y_gen"], dtype=np.int64)
        conf_gen = np.asarray(carrier["conf_gen"])
        u_cycle = np.asarray(carrier["u_cycle"])
        pi = np.asarray(carrier["pi"])
        t_rel = np.asarray(carrier["t_rel"])
        unlabeled_ids = np.asarray(carrier["unlabeled_ids"], dtype=np.int64)
        labeled_targets = np.asarray(carrier["labeled_targets"], dtype=np.int64)
    if p_gen.shape != (N, S, K) or p_ver.shape != (N, S, K):
        raise ValueError("carrier action posterior shapes are invalid")
    if (unlabeled_ids.shape != (NU,) or u_cycle.shape != (NU, S)
            or y_gen.shape != (N, S) or conf_gen.shape != (N, S)):
        raise ValueError("carrier query index or TRUE utility shape is invalid")
    p_gen_query = p_gen[unlabeled_ids]
    p_ver_query = p_ver[unlabeled_ids]
    permutation, _p_ver_null, y_true, conf_true, y_null, conf_null, _support, _closure, u_null = (
        build_null_utility(p_gen_query, p_ver_query)
    )
    if not np.array_equal(y_true, y_gen[unlabeled_ids]) or not np.array_equal(y_null, y_gen[unlabeled_ids]):
        raise ValueError("NULL labels do not exactly match frozen TRUE labels")
    if not np.array_equal(conf_true, conf_gen[unlabeled_ids]) or not np.array_equal(conf_null, conf_gen[unlabeled_ids]):
        raise ValueError("NULL confidences do not exactly match frozen TRUE confidences")
    parent = Path(seal["paths"]["parent_corruption_artifact"])
    with np.load(parent, allow_pickle=False) as artifact:
        targets = np.asarray(artifact["labels"], dtype=np.int64)
    correct = (np.argmax(pi, axis=2) == targets[unlabeled_ids, None]).astype(np.int8)
    auc_u_true = float(roc_auc_score(correct.ravel(), u_cycle.ravel()))
    auc_u_derangement = float(roc_auc_score(correct.ravel(), u_null.ravel()))
    relation = np.broadcast_to(
        targets[unlabeled_ids, None, None] == labeled_targets[None, :, None], t_rel.shape
    )
    rel_auc = float(roc_auc_score(relation.ravel(), t_rel.ravel()))
    rel_mean_same, rel_mean_diff = float(t_rel[relation].mean()), float(t_rel[~relation].mean())
    delta_auc_u, rel_gap = auc_u_true - auc_u_derangement, rel_mean_same - rel_mean_diff
    return {
        "null_control_version": NULL_CONTROL_VERSION,
        "NULL_SEED": NULL_SEED,
        "permutation_sha256": permutation_sha256(permutation),
        "fixed_point_count": int(np.count_nonzero(permutation == np.arange(NU, dtype=np.int64))),
        "AUC_U_TRUE": auc_u_true,
        "AUC_U_DERANGEMENT": auc_u_derangement,
        "Delta_AUC_U": delta_auc_u,
        "REL_AUC": rel_auc,
        "REL_GAP": rel_gap,
        "REL_MEAN_SAME": rel_mean_same,
        "REL_MEAN_DIFF": rel_mean_diff,
        "adjacent_shift_same_fraction": ADJACENT_SHIFT_SAME_FRACTION,
        "derangement_same_class_fraction": float(
            np.mean(targets[unlabeled_ids] == targets[unlabeled_ids[permutation]])
        ),
        "random_same_class_reference": RANDOM_SAME_CLASS_REFERENCE,
        "verdict": _verdict(auc_u_true, delta_auc_u, rel_auc, rel_gap),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carrier-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true", help="verify the seal and run V2 read-only diagnostics")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.execute:
        raise SystemExit("refusing to run V2 post-seal diagnostics without --execute")
    result = evaluate(args.carrier_dir)
    output_path = args.carrier_dir / DEFAULT_OUTPUT_NAME
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
