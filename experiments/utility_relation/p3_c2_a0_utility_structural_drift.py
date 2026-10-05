"""P3-C2-A0 GT-free, training-free utility--structural drift diagnostic.

This module deliberately does not import any formal evaluator. It consumes
only sealed pre-GT artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "outputs" / "bone_p3" / "p3_c1_b1_final_formal"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "bone_p3" / "p3_c2_a0_utility_structural_drift"
PREREGISTRATION_ROOT = (
    REPO_ROOT / "experiment_freeze" / "bone_p3_c2_a0_utility_structural_drift_preregistered_20261004"
)
DATASETS = (
    "HandWrite", "BDGP", "MSRC", "COIL20", "Caltech-5V", "MNIST-USPS", "ALOI", "CCV",
)
SEEDS = (20, 30, 50)
REQUIRED_CARRIER_KEYS = frozenset((
    "sample_ids", "query_sample_ids", "query_rows", "U_cycle", "query_U_cycle",
))
REQUIRED_PREDICTION_KEYS = frozenset(("sample_ids", "predictions"))
PARENT_SOURCE_SHA256 = {
    "method/bone.py": "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092",
    "method/amv.py": "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc",
    "test_cmvsc.py": "fc836c5a8e053624eee7bcf200a592b3a064a9eec71e719012eb976dd1641d8d",
    "experiments/utility_relation/run_bone_p3_c0_cross_dataset_formal.py": "330ac8999db35cae64092fcfeb0dc8dc900370996001a82d214a1eca90233f6b",
    "experiments/utility_relation/bone_p3_b0_generic_carrier.py": "5a9f3c817d0e0ea07e311f0899fdb926cef993d0f3b7edbd65faf7f83dfabc27",
    "experiments/utility_relation/p3_c1_b1_optuna/amendment_a2.py": "b3237d057e11051909c721828c3efb3d123651f3f8b7e1961694ba9f03ee46c0",
    "experiments/utility_relation/p3_c1_b1_optuna/canonical_fold_order_r3.py": "98702c3c4af819a5229b9b05da576c81a67aa1bbcf9ce0fd9e67fab122e43caf",
    "experiments/utility_relation/p3_c1_b1_final_formal/final_formal_protocol.py": "4e3d443bc9e5f21ddf92cb763832a6b3ed5bf5520a07b7113067288fd8847ffb",
    "experiments/utility_relation/p3_c1_b1_final_formal/final_formal_context.py": "77840c28319df9e4d07a548523903a859dfaf09a544e9a74cc17fc7411cf8e56",
    "experiments/utility_relation/p3_c1_b1_final_formal/run_final_formal.py": "b6abc513f81ae08d00141ea897d2c8c209840677503715fd79ab3f237284a6bd",
}
PARENT_FREEZE_SHA256 = {
    "experiment_freeze/bone_p3_c1_b1_final_formal_24run_frozen_20261004/FREEZE_SHA256.txt": "cb1964a756ef27b1670c48dfcb25ebcfa9982f1ff67d7d0c82a0455a6a9836bb",
    "experiment_freeze/bone_p3_c1_b1_a2_final_hyperparameters_20261004/FREEZE_SHA256.txt": "1798034f347b297295699d8cee5b6322025d32cf66ba46814cb09cf4ee469f33",
    "experiment_freeze/bone_p3_c1_b1_final_formal_runner_preregistered_20261004/FREEZE_SHA256.txt": "73784068c92f496a9c4904234687cfff1ce8cde0d86c8187453db7e3eb79d8cf",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_parent_integrity() -> dict:
    observed = {}
    for relative, expected in {**PARENT_SOURCE_SHA256, **PARENT_FREEZE_SHA256}.items():
        path = REPO_ROOT / relative
        if not path.is_file() or (actual := sha256_file(path)) != expected:
            raise RuntimeError(f"PARENT_INTEGRITY_FAILURE:{relative}")
        observed[relative] = actual
    return observed


def dataset_slug(dataset: str) -> str:
    if dataset not in DATASETS:
        raise ValueError("UNREGISTERED_DATASET")
    return dataset.lower().replace("-", "_")


def source_run_root(dataset: str, seed: int, source_root: str | Path = SOURCE_ROOT) -> Path:
    if seed not in SEEDS:
        raise ValueError("UNREGISTERED_SEED")
    return Path(source_root) / dataset_slug(dataset) / f"seed{seed}"


def output_run_root(dataset: str, seed: int, output_root: str | Path = OUTPUT_ROOT) -> Path:
    if seed not in SEEDS:
        raise ValueError("UNREGISTERED_SEED")
    return Path(output_root) / dataset_slug(dataset) / f"seed{seed}"


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"INVALID_JSON:{path.name}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"INVALID_JSON_OBJECT:{path.name}")
    return value


def _unique_ids(values: np.ndarray, name: str) -> np.ndarray:
    ids = np.asarray(values, dtype=np.int64)
    if ids.ndim != 1 or ids.size == 0 or np.unique(ids).size != ids.size:
        raise ValueError(f"INVALID_OR_DUPLICATE_SAMPLE_IDS:{name}")
    return ids


def _prediction_mapping(ids: np.ndarray, predictions: np.ndarray, name: str) -> dict[int, int]:
    ids = _unique_ids(ids, f"{name}_sample_ids")
    labels = np.asarray(predictions)
    if labels.ndim != 1 or labels.shape != ids.shape:
        raise ValueError(f"INVALID_PREDICTION_SHAPE:{name}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"NONINTEGER_CLUSTER_LABEL:{name}")
    return {int(sample_id): int(label) for sample_id, label in zip(ids, labels)}


def _verify_pre_gt_seal(root: Path) -> dict:
    seal_path = root / "PRE_GT_SEAL.json"
    seal = _read_json(seal_path)
    expected = seal.get("artifact_sha256")
    if seal.get("status") != "PRE_GT_SEALED" or not isinstance(expected, dict):
        raise RuntimeError("INVALID_PRE_GT_SEAL")
    resolved_root = root.resolve()
    for relative, expected_hash in expected.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise RuntimeError("INVALID_PRE_GT_SEAL_ENTRY")
        target = (root / relative).resolve()
        if resolved_root not in target.parents or not target.is_file():
            raise RuntimeError("PRE_GT_SEAL_PATH_INVALID")
        if sha256_file(target) != expected_hash:
            raise RuntimeError("PRE_GT_SEAL_HASH_MISMATCH")
    if seal.get("GT_loaded_before_seal") is not False:
        raise RuntimeError("PRE_GT_SEAL_GT_ISOLATION_FAILURE")
    return seal


def load_validated_pre_gt_pair(
    dataset: str, seed: int, source_root: str | Path = SOURCE_ROOT
) -> dict[str, Any]:
    """Load only sealed PRE-GT artifacts and prove explicit sample-ID alignment."""
    root = source_run_root(dataset, seed, source_root)
    required = (
        root / "RUN_CONFIG.json", root / "PRE_GT_AUDIT.json", root / "PRE_GT_SEAL.json",
        root / "COMPLETE.json", root / "carrier" / "carrier_pre_gt.npz",
        root / "BASE" / "pre_gt_predictions.npz", root / "OURS" / "pre_gt_predictions.npz",
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("MISSING_PRE_GT_FINAL_FORMAL_ARTIFACT")
    config, audit, complete = (
        _read_json(root / "RUN_CONFIG.json"),
        _read_json(root / "PRE_GT_AUDIT.json"),
        _read_json(root / "COMPLETE.json"),
    )
    _verify_pre_gt_seal(root)
    if (
        config.get("dataset") != dataset or config.get("training_seed") != seed
        or audit.get("dataset") != dataset or audit.get("training_seed") != seed
        or complete.get("dataset") != dataset or complete.get("training_seed") != seed
        or complete.get("status") != "COMPLETE"
    ):
        raise RuntimeError("SOURCE_RUN_IDENTITY_MISMATCH")
    isolation_fields = (
        "GT_loaded_before_seal", "full_gt_used_for_training",
        "full_gt_used_for_model_selection", "full_gt_used_for_hyperparameter_selection",
        "ACC_NMI_ARI_computed",
    )
    if any(audit.get(name) is not False for name in isolation_fields):
        raise RuntimeError("SOURCE_PRE_GT_ISOLATION_MISMATCH")
    with np.load(root / "carrier" / "carrier_pre_gt.npz", allow_pickle=False) as archive:
        if not REQUIRED_CARRIER_KEYS.issubset(archive.files):
            raise ValueError("CARRIER_SCHEMA_MISSING_REQUIRED_KEYS")
        carrier = {name: np.asarray(archive[name]) for name in REQUIRED_CARRIER_KEYS}
    arms = {}
    for arm in ("BASE", "OURS"):
        with np.load(root / arm / "pre_gt_predictions.npz", allow_pickle=False) as archive:
            if set(archive.files) != REQUIRED_PREDICTION_KEYS:
                raise ValueError(f"PREDICTION_SCHEMA_INVALID:{arm}")
            arms[arm] = {name: np.asarray(archive[name]) for name in REQUIRED_PREDICTION_KEYS}

    carrier_ids = _unique_ids(carrier["sample_ids"], "carrier")
    query_ids = _unique_ids(carrier["query_sample_ids"], "query")
    query_rows = np.asarray(carrier["query_rows"], dtype=np.int64)
    if (
        query_rows.ndim != 1 or query_rows.shape != query_ids.shape
        or np.any(query_rows < 0) or np.any(query_rows >= carrier_ids.size)
        or not np.array_equal(carrier_ids[query_rows], query_ids)
    ):
        raise ValueError("QUERY_ROW_SAMPLE_ID_ALIGNMENT_FAILURE")
    if not set(query_ids.tolist()).issubset(set(carrier_ids.tolist())):
        raise ValueError("QUERY_IDS_NOT_SUBSET_OF_CARRIER_IDS")
    utility = np.asarray(carrier["query_U_cycle"])
    full_utility = np.asarray(carrier["U_cycle"])
    if (
        utility.ndim != 2 or full_utility.ndim != 2
        or utility.shape[0] != query_ids.size or utility.shape[1] != full_utility.shape[1]
        or full_utility.shape[0] != carrier_ids.size
        or not np.isfinite(utility).all() or not np.isfinite(full_utility).all()
        or np.any(utility < 0) or np.any(full_utility < 0)
    ):
        raise ValueError("INVALID_QUERY_UTILITY")
    if not np.array_equal(utility, full_utility[query_rows]):
        raise ValueError("QUERY_UTILITY_ROW_ALIGNMENT_FAILURE")
    base_map = _prediction_mapping(arms["BASE"]["sample_ids"], arms["BASE"]["predictions"], "BASE")
    ours_map = _prediction_mapping(arms["OURS"]["sample_ids"], arms["OURS"]["predictions"], "OURS")
    base_ids, ours_ids = set(base_map), set(ours_map)
    if base_ids != ours_ids or base_ids != set(carrier_ids.tolist()):
        raise ValueError("FULL_SAMPLE_ID_SET_MISMATCH")
    try:
        base_query = np.asarray([base_map[int(value)] for value in query_ids], dtype=np.int64)
        ours_query = np.asarray([ours_map[int(value)] for value in query_ids], dtype=np.int64)
        base_full = np.asarray([base_map[int(value)] for value in carrier_ids], dtype=np.int64)
        ours_full = np.asarray([ours_map[int(value)] for value in carrier_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError("QUERY_PREDICTION_SAMPLE_ID_MISSING") from error
    return {
        "root": root, "config": config, "carrier_ids": carrier_ids, "query_ids": query_ids,
        "query_utility": utility, "base_query": base_query, "ours_query": ours_query,
        "base_full": base_full, "ours_full": ours_full,
    }


def hungarian_align_ours_to_base(base: np.ndarray, ours: np.ndarray) -> tuple[np.ndarray, dict]:
    """Map OURS IDs to BASE IDs using shared samples only, with a sentinel for extras."""
    base_values, ours_values = np.asarray(base, dtype=np.int64), np.asarray(ours, dtype=np.int64)
    if base_values.ndim != 1 or ours_values.shape != base_values.shape or base_values.size == 0:
        raise ValueError("INVALID_CLUSTER_ALIGNMENT_INPUT")
    base_labels, ours_labels = np.unique(base_values), np.unique(ours_values)
    contingency = np.zeros((base_labels.size, ours_labels.size), dtype=np.int64)
    for base_label, ours_label in zip(base_values, ours_values):
        contingency[np.searchsorted(base_labels, base_label), np.searchsorted(ours_labels, ours_label)] += 1
    rows, cols = linear_sum_assignment(contingency.max() - contingency)
    mapping = {int(ours_labels[col]): int(base_labels[row]) for row, col in zip(rows, cols)}
    sentinel = "UNMATCHED_OURS_CLUSTER"
    aligned = np.asarray([mapping.get(int(value), sentinel) for value in ours_values], dtype=object)
    unmatched = [int(value) for value in ours_labels if int(value) not in mapping]
    return aligned, {
        "base_cluster_count": int(base_labels.size),
        "ours_cluster_count": int(ours_labels.size),
        "contingency_shape": [int(base_labels.size), int(ours_labels.size)],
        "ours_to_base": {str(key): value for key, value in sorted(mapping.items())},
        "unmatched_ours_clusters": unmatched,
        "unmatched_sentinel": sentinel if unmatched else None,
        "matching_overlap": int(contingency[rows, cols].sum()),
    }


def _safe_rate(numerator: int, denominator: int) -> tuple[float | None, str | None]:
    if denominator == 0:
        return None, "EMPTY_GROUP"
    return float(numerator / denominator), None


def _safe_mean(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.mean(values))


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else float(left - right)


def _spearman(utility: np.ndarray, flips: np.ndarray) -> dict:
    values, changed = np.asarray(utility, dtype=float), np.asarray(flips, dtype=np.int64)
    if values.size == 0:
        return {"rho": None, "p_value": None, "reason": "EMPTY_GROUP"}
    if np.ptp(values) == 0 or np.ptp(changed) == 0:
        return {"rho": None, "p_value": None, "reason": "CONSTANT_INPUT"}
    rho, p_value = spearmanr(values, changed)
    if not np.isfinite(rho) or not np.isfinite(p_value):
        return {"rho": None, "p_value": None, "reason": "UNDEFINED"}
    return {"rho": float(rho), "p_value": float(p_value), "reason": None}


def _quartiles(sample_ids: np.ndarray, mean_utility: np.ndarray, flips: np.ndarray, supported: np.ndarray) -> list[dict]:
    supported_positions = np.flatnonzero(supported)
    if supported_positions.size < 4:
        raise ValueError("SUPPORTED_SAMPLE_COUNT_LT_FOUR")
    order = np.lexsort((sample_ids[supported_positions], mean_utility[supported_positions]))
    partitions = np.array_split(supported_positions[order], 4)
    results = []
    for number, positions in enumerate(partitions, start=1):
        count = int(positions.size)
        flip_count = int(flips[positions].sum())
        results.append({
            "quartile": f"Q{number}", "count": count,
            "mean_utility": float(mean_utility[positions].mean()),
            "flip_count": flip_count, "flip_rate": float(flip_count / count),
            "sample_ids_ascending_utility_tiebreak": [int(value) for value in sample_ids[positions]],
        })
    return results


def _normalized_entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    k_effective = probabilities.size
    if k_effective <= 1:
        return 0.0
    return float(-(positive * np.log(positive)).sum() / np.log(k_effective))


def _occupancy(base: np.ndarray, ours_aligned: np.ndarray) -> dict:
    base_labels = [int(value) for value in np.unique(base)]
    sentinel = "UNMATCHED_OURS_CLUSTER"
    has_sentinel = bool(np.any(ours_aligned == sentinel))
    support: list[object] = [*base_labels, *([sentinel] if has_sentinel else [])]
    base_proportions = np.asarray([np.mean(base == label) for label in support], dtype=float)
    ours_proportions = np.asarray([np.mean(ours_aligned == label) for label in support], dtype=float)
    base_native_proportions = np.asarray(
        [np.mean(base == label) for label in base_labels], dtype=float
    )
    return {
        "occupancy_tv_query": float(0.5 * np.abs(base_proportions - ours_proportions).sum()),
        "base_query_entropy_norm": _normalized_entropy(base_native_proportions),
        "ours_query_entropy_norm": _normalized_entropy(ours_proportions),
        "occupancy_support": [str(value) for value in support],
        "base_query_proportions": [float(value) for value in base_proportions],
        "ours_aligned_query_proportions": [float(value) for value in ours_proportions],
    }


def compute_diagnostic(loaded: dict[str, Any]) -> dict:
    """Compute the preregistered descriptive diagnostic from already aligned arrays."""
    query_ids = loaded["query_ids"]
    utility = np.asarray(loaded["query_utility"], dtype=float)
    mean_utility = utility.mean(axis=1)
    utility_mass = utility.sum(axis=1)
    supported = utility_mass > 0
    abstained = utility_mass == 0
    if not np.array_equal(supported | abstained, np.ones(query_ids.size, dtype=bool)):
        raise RuntimeError("UTILITY_SUPPORT_PARTITION_FAILURE")
    full_aligned, full_mapping = hungarian_align_ours_to_base(loaded["base_full"], loaded["ours_full"])
    query_aligned, query_mapping = hungarian_align_ours_to_base(loaded["base_query"], loaded["ours_query"])
    # The mapping is defined on the full shared sample population, as preregistered.
    mapping = {int(key): value for key, value in full_mapping["ours_to_base"].items()}
    sentinel = full_mapping["unmatched_sentinel"] or "UNMATCHED_OURS_CLUSTER"
    ours_query_full_mapping = np.asarray(
        [mapping.get(int(value), sentinel) for value in loaded["ours_query"]], dtype=object
    )
    flips = np.asarray(loaded["base_query"], dtype=object) != ours_query_full_mapping
    full_flips = np.asarray(loaded["base_full"], dtype=object) != full_aligned
    supported_count, abstained_count = int(supported.sum()), int(abstained.sum())
    flip_count = int(flips.sum())
    supported_flip_count = int(flips[supported].sum())
    abstained_flip_count = int(flips[abstained].sum())
    supported_rate, supported_reason = _safe_rate(supported_flip_count, supported_count)
    abstained_rate, abstained_reason = _safe_rate(abstained_flip_count, abstained_count)
    flip_supported_fraction, flip_fraction_reason = _safe_rate(supported_flip_count, flip_count)
    flip_abstained_fraction, _ = _safe_rate(abstained_flip_count, flip_count)
    mean_flip, mean_stable = _safe_mean(mean_utility[flips]), _safe_mean(mean_utility[~flips])
    mass_flip, mass_stable = _safe_mean(utility_mass[flips]), _safe_mean(utility_mass[~flips])
    occupancy = _occupancy(loaded["base_query"], ours_query_full_mapping)
    return {
        "N": int(loaded["carrier_ids"].size), "Nq": int(query_ids.size), "S": int(utility.shape[1]),
        "sample_id_alignment_pass": True,
        "utility_supported_count": supported_count, "utility_abstained_count": abstained_count,
        "utility_coverage": float(supported_count / query_ids.size),
        "query_flip_count": flip_count, "query_flip_rate": float(flip_count / query_ids.size),
        "full_flip_count": int(full_flips.sum()), "full_flip_rate": float(full_flips.mean()),
        "supported_flip_count": supported_flip_count, "supported_flip_rate": supported_rate,
        "supported_flip_rate_reason": supported_reason,
        "abstained_flip_count": abstained_flip_count, "abstained_flip_rate": abstained_rate,
        "abstained_flip_rate_reason": abstained_reason,
        "flip_supported_fraction": flip_supported_fraction,
        "flip_abstained_fraction": flip_abstained_fraction,
        "flip_fraction_reason": flip_fraction_reason,
        "mean_utility_flip": mean_flip, "mean_utility_stable": mean_stable,
        "delta_mean_utility_flip_minus_stable": _difference(mean_flip, mean_stable),
        "utility_mass_flip": mass_flip, "utility_mass_stable": mass_stable,
        "delta_mass_flip_minus_stable": _difference(mass_flip, mass_stable),
        "supported_utility_rank_quartiles": _quartiles(query_ids, mean_utility, flips, supported),
        "spearman_all": _spearman(mean_utility, flips),
        "spearman_supported": _spearman(mean_utility[supported], flips[supported]),
        "agreement_nmi_full": float(normalized_mutual_info_score(loaded["base_full"], loaded["ours_full"])),
        "agreement_ari_full": float(adjusted_rand_score(loaded["base_full"], loaded["ours_full"])),
        "agreement_nmi_query": float(normalized_mutual_info_score(loaded["base_query"], loaded["ours_query"])),
        "agreement_ari_query": float(adjusted_rand_score(loaded["base_query"], loaded["ours_query"])),
        "base_cluster_count_full": full_mapping["base_cluster_count"],
        "ours_cluster_count_full": full_mapping["ours_cluster_count"],
        "base_cluster_count_query": query_mapping["base_cluster_count"],
        "ours_cluster_count_query": query_mapping["ours_cluster_count"],
        "occupancy_tv_query": occupancy["occupancy_tv_query"],
        "base_query_entropy_norm": occupancy["base_query_entropy_norm"],
        "ours_query_entropy_norm": occupancy["ours_query_entropy_norm"],
        "occupancy_audit": occupancy,
        "hungarian_mapping_audit": full_mapping,
        "cluster_alignment_uses_gt": False,
    }


def _verify_preregistration(root: str | Path = PREREGISTRATION_ROOT) -> None:
    verify_parent_integrity()
    status = Path(root) / "STATUS.txt"
    manifest = Path(root) / "new_source_sha256.txt"
    if not status.is_file() or not manifest.is_file():
        raise RuntimeError("P3_C2_A0_PREREGISTRATION_MISSING")
    text = status.read_text(encoding="utf-8")
    if "STATUS=P3_C2_A0_UTILITY_STRUCTURAL_DRIFT_PREREGISTERED" not in text:
        raise RuntimeError("P3_C2_A0_PREREGISTRATION_STATUS_INVALID")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError("P3_C2_A0_PREREGISTERED_SOURCE_HASH_MISMATCH")


def schema_preflight(
    source_root: str | Path = SOURCE_ROOT,
) -> dict:
    """Read schemas and seals only; do not compute diagnostic quantities."""
    verify_parent_integrity()
    records, unresolved = [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            try:
                loaded = load_validated_pre_gt_pair(dataset, seed, source_root)
                record = {
                    "dataset": dataset, "training_seed": seed, "resolvable": True,
                    "N": int(loaded["carrier_ids"].size), "Nq": int(loaded["query_ids"].size),
                    "S": int(loaded["query_utility"].shape[1]),
                    "sample_id_alignment_pass": True,
                    "source_run_root": str(loaded["root"]),
                }
            except Exception as error:
                record = {
                    "dataset": dataset, "training_seed": seed, "resolvable": False,
                    "error": f"{type(error).__name__}:{error}",
                }
                unresolved.append(record)
            records.append(record)
    return {
        "STATUS": "PASS" if not unresolved else "FAIL",
        "INTENDED_PAIR_COUNT": 24, "RESOLVABLE_PAIR_COUNT": len(records) - len(unresolved),
        "UNRESOLVED_PAIR_COUNT": len(unresolved), "ALL_24_SCHEMA_PREFLIGHT_PASS": not unresolved,
        "FULL_GT_READ": False, "FORMAL_METRICS_READ": False, "MODEL_TRAINING_RUN": False,
        "records": records,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_one(
    dataset: str, seed: int, source_root: str | Path = SOURCE_ROOT,
    output_root: str | Path = OUTPUT_ROOT, preregistration_root: str | Path = PREREGISTRATION_ROOT,
) -> dict:
    _verify_preregistration(preregistration_root)
    target = output_run_root(dataset, seed, output_root) / "DIAGNOSTIC.json"
    if target.is_file():
        existing = _read_json(target)
        if existing.get("dataset") != dataset or existing.get("training_seed") != seed:
            raise RuntimeError("EXISTING_DIAGNOSTIC_IDENTITY_MISMATCH")
        return existing
    loaded = load_validated_pre_gt_pair(dataset, seed, source_root)
    result = compute_diagnostic(loaded)
    result.update({
        "dataset": dataset, "training_seed": seed, "source_run_root": str(loaded["root"]),
        "semantic_lr": loaded["config"]["semantic_lr"],
        "semantic_exposure": loaded["config"]["semantic_exposure"],
        "full_gt_read": False, "formal_metrics_read": False, "training_run": False,
        "reporting_only": True,
    })
    _write_json(target, result)
    return result


SUMMARY_FIELDS = (
    "utility_coverage", "query_flip_rate", "supported_flip_rate", "abstained_flip_rate",
    "flip_abstained_fraction", "delta_mean_utility_flip_minus_stable", "agreement_nmi_query",
    "agreement_ari_query", "occupancy_tv_query",
)


def _row_for_summary(result: dict) -> dict:
    row = {"dataset": result["dataset"], "training_seed": result["training_seed"]}
    for key in SUMMARY_FIELDS:
        row[key] = result[key]
    for quartile in result["supported_utility_rank_quartiles"]:
        row[f"{quartile['quartile']}_flip_rate"] = quartile["flip_rate"]
    row["spearman_supported_rho"] = result["spearman_supported"]["rho"]
    return row


def _population_summary(values: Iterable[float | None]) -> tuple[float | None, float | None]:
    usable = np.asarray([value for value in values if value is not None], dtype=float)
    if usable.size == 0:
        return None, None
    return float(usable.mean()), float(usable.std())


def summarize_all(output_root: str | Path = OUTPUT_ROOT) -> dict:
    results = []
    for dataset in DATASETS:
        for seed in SEEDS:
            path = output_run_root(dataset, seed, output_root) / "DIAGNOSTIC.json"
            if not path.is_file():
                raise FileNotFoundError("DIAGNOSTIC_MISSING_FOR_SUMMARY")
            result = _read_json(path)
            if result.get("dataset") != dataset or result.get("training_seed") != seed:
                raise RuntimeError("DIAGNOSTIC_SUMMARY_IDENTITY_MISMATCH")
            results.append(result)
    per_seed = [_row_for_summary(result) for result in results]
    fields = tuple(per_seed[0])
    summary = {}
    for dataset in DATASETS:
        rows = [row for row in per_seed if row["dataset"] == dataset]
        values = {"dataset": dataset, "seed_count": len(rows)}
        for field in fields:
            if field in ("dataset", "training_seed"):
                continue
            mean, std = _population_summary(row[field] for row in rows)
            values[f"{field}_mean"] = mean
            values[f"{field}_std_population"] = std
        summary[dataset] = values
    summary_root = Path(output_root) / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    with (summary_root / "per_seed_diagnostic.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(per_seed)
    summary_rows = list(summary.values())
    summary_fields = tuple(summary_rows[0])
    with (summary_root / "three_seed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields); writer.writeheader(); writer.writerows(summary_rows)
    payload = {
        "status": "DESCRIPTIVE_ONLY", "pair_count": len(results), "dataset_count": len(summary),
        "full_gt_read": False, "formal_metrics_read": False, "training_run": False,
        "parameter_selection": False, "datasets": summary,
    }
    _write_json(summary_root / "three_seed_summary.json", payload)
    return payload


def run_all(
    source_root: str | Path = SOURCE_ROOT, output_root: str | Path = OUTPUT_ROOT,
    preregistration_root: str | Path = PREREGISTRATION_ROOT,
) -> dict:
    preflight = schema_preflight(source_root)
    if preflight["ALL_24_SCHEMA_PREFLIGHT_PASS"] is not True:
        raise RuntimeError("SCHEMA_PREFLIGHT_FAILED")
    for dataset in DATASETS:
        for seed in SEEDS:
            run_one(dataset, seed, source_root, output_root, preregistration_root)
    return summarize_all(output_root)


def format_short_table(summary: dict) -> str:
    headers = ("DATASET", "U_COVER", "FLIP_ALL", "FLIP_U_POS", "FLIP_U_ZERO", "FLIP_ZERO_SHARE", "Q1_FLIP", "Q4_FLIP", "RHO_U_FLIP", "AGR_NMI", "AGR_ARI", "OCC_TV")
    lines = [" ".join(headers)]
    mapping = (
        "utility_coverage_mean", "query_flip_rate_mean", "supported_flip_rate_mean",
        "abstained_flip_rate_mean", "flip_abstained_fraction_mean", "Q1_flip_rate_mean",
        "Q4_flip_rate_mean", "spearman_supported_rho_mean", "agreement_nmi_query_mean",
        "agreement_ari_query_mean", "occupancy_tv_query_mean",
    )
    for dataset in DATASETS:
        row = summary["datasets"][dataset]
        values = [dataset] + ["null" if row[key] is None else f"{row[key]:.6f}" for key in mapping]
        lines.append(" ".join(values))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--schema-preflight", action="store_true")
    parser.add_argument("--execute-all", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.schema_preflight:
        print(json.dumps(schema_preflight(args.source_root), indent=2, sort_keys=True))
        return
    if not args.execute_all:
        raise SystemExit("use --schema-preflight or --execute-all")
    summary = run_all(args.source_root, args.output_root)
    print(format_short_table(summary))


if __name__ == "__main__":
    main()
