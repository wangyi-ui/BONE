"""CPU-only, read-only diagnosis of the frozen MSRC seed11 carrier utility."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.utility_relation.p3_c1_b1_optuna.amendment_a1 import validate_class_feasibility


REQUIRED_KEYS = (
    "q_views", "p_gen", "p_ver", "U_cycle", "generator_actions",
    "verifier_actions", "sample_ids", "labeled_ids", "labeled_targets",
)
EXPECTED_ZERO_MASS_RAW_IDS = [15, 47, 114, 175]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _action_complement_pass(generator: np.ndarray, verifier: np.ndarray, view_count: int) -> bool:
    if generator.ndim != 2 or verifier.shape != generator.shape:
        return False
    all_views = set(range(view_count))
    for gen_row, ver_row in zip(generator, verifier):
        gen, ver = set(map(int, gen_row)), set(map(int, ver_row))
        if not gen or not ver or gen & ver or gen | ver != all_views:
            return False
    return True


def diagnose(input_path: str | Path) -> dict:
    """Recompute U and all zero-mass facts without loading data or ground truth."""
    input_path = Path(input_path)
    with np.load(input_path, allow_pickle=False) as archive:
        missing = set(REQUIRED_KEYS) - set(archive.files)
        if missing:
            raise ValueError(f"MISSING_CARRIER_KEYS:{sorted(missing)}")
        arrays = {name: np.asarray(archive[name]) for name in REQUIRED_KEYS}

    q_views, p_gen, p_ver = arrays["q_views"], arrays["p_gen"], arrays["p_ver"]
    saved_u = arrays["U_cycle"]
    sample_ids = arrays["sample_ids"]
    labeled_ids, labeled_targets = arrays["labeled_ids"], arrays["labeled_targets"]
    if q_views.ndim != 3 or p_gen.ndim != 3 or p_ver.shape != p_gen.shape:
        raise ValueError("INVALID_CARRIER_SHAPE")
    sample_count, view_count, cluster_count = q_views.shape
    if p_gen.shape[0] != sample_count or saved_u.shape != p_gen.shape[:2]:
        raise ValueError("INVALID_CARRIER_SHAPE")
    if sample_ids.shape != (sample_count,) or labeled_ids.ndim != 1 or labeled_targets.shape != labeled_ids.shape:
        raise ValueError("INVALID_ID_SHAPE")
    if not all(np.isfinite(array).all() for array in (q_views, p_gen, p_ver, saved_u)):
        raise ValueError("NONFINITE_CARRIER")

    y_gen = np.argmax(p_gen, axis=2)
    y_ver = np.argmax(p_ver, axis=2)
    conf_gen = np.max(p_gen, axis=2)
    support_ver = np.take_along_axis(p_ver, y_gen[..., None], axis=2)[..., 0]
    closure = y_gen == y_ver
    recomputed_u = closure.astype(float) * np.sqrt(conf_gen * support_ver)
    parity = bool(np.allclose(saved_u, recomputed_u, rtol=1e-6, atol=1e-7))
    max_abs_diff = float(np.max(np.abs(saved_u - recomputed_u)))

    utility_masses = saved_u.sum(axis=1)
    zero_positions = np.flatnonzero(utility_masses == 0)
    zero_raw_ids = sorted(int(value) for value in sample_ids[zero_positions])
    id_to_position = {int(raw_id): position for position, raw_id in enumerate(sample_ids)}
    zero_labeled_records = []
    labeled_masses = []
    for raw_id, target in zip(labeled_ids, labeled_targets):
        position = id_to_position[int(raw_id)]
        mass = float(utility_masses[position])
        labeled_masses.append(mass)
        if mass == 0:
            zero_labeled_records.append({"raw_id": int(raw_id), "class": int(target), "array_position": int(position)})

    feasibility = validate_class_feasibility(
        labeled_targets, labeled_masses, expected_class_ids=range(cluster_count)
    )
    position_175 = id_to_position[175]
    sparse_175 = labeled_targets[np.flatnonzero(labeled_ids == 175)[0]]
    raw175 = {
        "array_position": int(position_175),
        "sparse_target": int(sparse_175),
        "u_vector": [float(value) for value in saved_u[position_175]],
        "utility_mass": float(utility_masses[position_175]),
        "generator_predictions": [int(value) for value in y_gen[position_175]],
        "verifier_predictions": [int(value) for value in y_ver[position_175]],
        "closure_vector": [int(value) for value in closure[position_175]],
        "closure_count": int(np.count_nonzero(closure[position_175])),
        "conf_gen": [float(value) for value in conf_gen[position_175]],
        "support_ver": [float(value) for value in support_ver[position_175]],
    }
    sample_unique = len(set(map(int, sample_ids))) == sample_count
    action_complement = _action_complement_pass(
        arrays["generator_actions"], arrays["verifier_actions"], view_count
    )
    result = {
        "dataset": "MSRC", "dev_seed": 11,
        "input_path": str(input_path), "input_sha256": _sha256(input_path),
        "sample_count": int(sample_count), "view_count": int(view_count),
        "cluster_count": int(cluster_count), "action_count": int(p_gen.shape[1]),
        "u_formula_parity": parity, "u_max_abs_diff": max_abs_diff,
        "positive_sample_action_count": int(np.count_nonzero(saved_u > 0)),
        "zero_sample_action_count": int(np.count_nonzero(saved_u == 0)),
        "zero_mass_sample_count": len(zero_raw_ids), "zero_mass_raw_ids": zero_raw_ids,
        "labeled_anchor_count": int(len(labeled_ids)),
        "zero_mass_labeled_count": len(zero_labeled_records),
        "zero_mass_labeled_records": zero_labeled_records,
        "raw175": raw175,
        "sample_id_unique_pass": bool(sample_unique),
        "action_complement_pass": bool(action_complement),
        "class_feasibility": feasibility,
        "full_gt_loaded": False,
    }

    if (sample_count, view_count, cluster_count, p_gen.shape[1]) != (210, 4, 7, 6):
        raise RuntimeError("UNEXPECTED_MSRC_CARRIER_DIMENSIONS")
    if not parity:
        raise RuntimeError("U_FORMULA_PARITY_FAILURE")
    if not sample_unique:
        raise RuntimeError("SAMPLE_ID_MAPPING_FAILURE")
    if not action_complement:
        raise RuntimeError("ACTION_COMPLEMENT_FAILURE")
    if zero_raw_ids != EXPECTED_ZERO_MASS_RAW_IDS:
        raise RuntimeError("UNEXPECTED_ZERO_MASS_RAW_IDS")
    if zero_labeled_records != [{"raw_id": 175, "class": 5, "array_position": position_175}]:
        raise RuntimeError("UNEXPECTED_ZERO_MASS_LABELED_ANCHOR")
    if raw175["utility_mass"] != 0 or raw175["closure_count"] != 0:
        raise RuntimeError("RAW175_DIAGNOSIS_MISMATCH")
    return result


def write_diagnosis(input_path: str | Path, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"REFUSE_TO_OVERWRITE:{output_path}")
    result = diagnose(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = write_diagnosis(args.input, args.output)
    print(json.dumps({"diagnosis": str(output)}, indent=2))


if __name__ == "__main__":
    main()
