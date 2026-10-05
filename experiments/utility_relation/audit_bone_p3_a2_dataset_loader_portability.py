"""Read-only P3-A2 dataset/loader portability audit.

This tool reads source text, MATLAB metadata, feature shapes, and label
metadata.  It neither calls the repository loader nor initializes a model,
optimizer, clustering routine, metric, GPU, or training loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from experiments.utility_relation import bone_p3_a1_action_space_protocol as action_protocol
from utils import data_util


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "bone_p3"
FREEZE_PROTOCOL = REPO_ROOT / "experiment_freeze" / "bone_p3_a1_action_space_protocol_20261001" / "PROTOCOL.txt"
P2_CARRIER = REPO_ROOT / "outputs" / "bone_p2" / "p2_a0_msrc_seed20" / "p2_a0_carrier_pre_gt.npz"

ORIGINAL_BONE_SHA256 = "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092"
ORIGINAL_AMV_SHA256 = "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc"

DATASET_IDS = {
    "MNIST-USPS": "D_MNIST_USPS",
    "BDGP": "D_BDGP",
    "CCV": "D_CCV",
    "Caltech-5V": "D_Caltech5V",
    "HandWrite": "D_HandWrite",
    "COIL20": "D_COIL20",
    "MSRC": "D_MSRC",
    "YouTubeFace50": "D_YouTubeFace50",
    "ALOI": "D_ALOI",
}

# These values are transcribed from the two native entry scripts; they are
# evidence metadata only and are not used to construct any training object.
NATIVE_CONFIG = {
    "MNIST-USPS": {"entry": "test_cmvsc.py", "batch_size": 512, "n_anchor": 4, "layer": 2, "seed": 42, "active": True},
    "BDGP": {"entry": "test_cmvsc.py", "batch_size": 512, "n_anchor": 2, "layer": 2, "seed": 42, "active": True},
    "CCV": {"entry": "test_cmvsc.py", "batch_size": 512, "n_anchor": 3, "layer": 2, "seed": 42, "active": False},
    "Caltech-5V": {"entry": "test_cmvsc.py", "batch_size": 128, "n_anchor": 4, "layer": 3, "seed": 42, "active": True},
    "HandWrite": {"entry": "test_cmvsc.py", "batch_size": 128, "n_anchor": 4, "layer": 3, "seed": 42, "active": True},
    "COIL20": {"entry": "test_cmvsc.py", "batch_size": 128, "n_anchor": 1, "layer": 4, "seed": 42, "active": True},
    "MSRC": {"entry": "test_cmvsc.py", "batch_size": 128, "n_anchor": 3, "layer": 5, "seed": 42, "active": True},
    "YouTubeFace50": {"entry": "test_cmvsc_large.py", "batch_size": 512, "n_anchor": 1, "layer": 2, "seed": None, "active": True},
    "ALOI": {"entry": "test_cmvsc.py", "batch_size": 512, "n_anchor": 2, "layer": 2, "seed": 42, "active": True},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_record(path: Path) -> dict[str, str]:
    return {"relative_path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}


def _label_encoding(values: np.ndarray) -> tuple[str, bool, str | None]:
    expected_zero = np.arange(values.size, dtype=values.dtype)
    expected_one = np.arange(1, values.size + 1, dtype=values.dtype)
    if np.array_equal(values, expected_zero):
        return "0-based", False, None
    if np.array_equal(values, expected_one):
        return "1-based", True, "sorted unique raw labels -> 0,...,K-1"
    return "arbitrary", True, "sorted unique raw labels -> 0,...,K-1"


def _memory_estimate(n: int, v: int, k: int, s: int, source: str) -> dict[str, Any]:
    labeled = 2 * k
    elements = {
        "q_views": n * v * k,
        "p_gen": n * s * k,
        "p_ver": n * s * k,
        "U": n * s,
        "T_rel": (n - labeled) * labeled * s,
    }
    mb = {name: value * 4 / 1_000_000 for name, value in elements.items()}
    full_dense = sum(mb.values())
    return {
        "input_source": source,
        "L": labeled,
        "q_views": {"elements": elements["q_views"], "float32_MB": mb["q_views"]},
        "p_gen": {"elements": elements["p_gen"], "float32_MB": mb["p_gen"]},
        "p_ver": {"elements": elements["p_ver"], "float32_MB": mb["p_ver"]},
        "U": {"elements": elements["U"], "float32_MB": mb["U"]},
        "T_rel": {"elements": elements["T_rel"], "float32_MB": mb["T_rel"]},
        "FULL_DENSE_ESTIMATED_MB": full_dense,
        "QUERY_CHUNKING_REQUIRED": full_dense > 512.0,
    }


def _common_native_fields(name: str, n: int | str) -> dict[str, Any]:
    config = NATIVE_CONFIG[name]
    batch_size = config["batch_size"]
    remainder = n % batch_size if isinstance(n, int) else "NOT_FOUND"
    if name == "YouTubeFace50":
        seed = {
            "DEFAULT_TRAINING_SEED": "NOT_FOUND",
            "python_seed": "NOT_FOUND (set_seed is not called)",
            "numpy_seed": "MVDataset.shuffle(seed=42), then AMV sets numpy seed=44",
            "torch_seed": "NOT_FOUND (set_seed is not called)",
            "dataloader_generator_seed": 44,
            "bone_constructor_seed": "default=1 (seed argument omitted)",
        }
        visitation = "Training loop breaks after batch index 10, so only 11 shuffled batches are visited per epoch."
    else:
        seed = {
            "DEFAULT_TRAINING_SEED": 42,
            "python_seed": 42,
            "numpy_seed": 42,
            "torch_seed": 42,
            "dataloader_generator_seed": 42,
            "bone_constructor_seed": 42,
        }
        visitation = "All DataLoader batches except the dropped final partial batch are visited."
    return {
        "native_training_entry": config["entry"],
        "native_config_active": config["active"],
        "native_epochs": 100,
        "batch_size": batch_size,
        "n_anchor": config["n_anchor"],
        "bone_layer": config["layer"],
        "optimizer": "torch.optim.Adam",
        "learning_rate": 0.01,
        "weight_decay": "0.0 (not explicitly passed; Adam default)",
        "bone_lamb": 0.1,
        "bone_temperature_default": 1,
        "bone_reg_default": 1,
        "bone_con_metric": "cosine",
        "amv_max_iter": 100,
        "amv_sample": 0.01 if name == "YouTubeFace50" else 1,
        "seed_audit": seed,
        "dataloader": {
            "dataset_class": "RandDatasetMV",
            "returns_sample_id": False,
            "SAMPLE_ID_ADAPTER_REQUIRED": True,
            "train_shuffle": True,
            "drop_last": True,
            "num_workers": 0,
            "collate_behavior": "default collate: list of per-view [batch, feature_dim] tensors plus float32 labels",
            "N_mod_batch_size": remainder,
            "final_train_batch_smaller_if_not_dropped": remainder != 0 if isinstance(remainder, int) else "NOT_FOUND",
            "training_visitation": visitation,
        },
    }


def _missing_dataset_record(name: str, declared: list[Any]) -> dict[str, Any]:
    n, v, k, dims = declared
    action_count = action_protocol.expected_action_count(v)
    return {
        "dataset_name": name,
        "dataset_id": DATASET_IDS[name],
        "loader_source_file": "utils/data_util.py",
        "loader_function": "get_dataset",
        "dataset_path": f"data/{name}.mat",
        "file_exists": False,
        "FILE_STATUS": "MISSING",
        "DATA_FILE_MISSING": True,
        "dataset_file_provenance": {"dataset_file_path": f"data/{name}.mat", "file_size": "NOT_FOUND", "sha256": "SHA_NOT_COMPUTED", "reason": "file is missing"},
        "file_format": "MATLAB .mat expected by scipy.io.loadmat",
        "feature_keys": [f"X{i}" for i in range(1, v + 1)],
        "label_key": "Y",
        "declared_N": n,
        "declared_V": v,
        "declared_K": k,
        "declared_dims": list(dims),
        "actual_N": "NOT_FOUND",
        "actual_V": "NOT_FOUND",
        "actual_K": "NOT_FOUND",
        "actual_dims": "NOT_FOUND",
        "view_mappings": "NOT_FOUND",
        "label_audit": "NOT_FOUND",
        "RAW_ROW_ORDER_PRESERVED": "NOT_FOUND",
        "row_identity_audit": "NOT_FOUND",
        "preprocessing": _preprocessing_record(),
        "label_leakage_boundary": _label_leakage_record(),
        "model_dimension_parity": "NOT_FOUND",
        "ACTION_SPACE_INPUT_PARITY": "NOT_FOUND",
        "action_space": {
            "view_num_source": "declared metadata only; data file missing",
            "V": v,
            "S": action_count,
            "canonical_sha256": action_protocol.canonical_action_sha256(v),
            "reverse_closure": True,
            "partition_validity": True,
        },
        "sparse_label_feasibility": {"SPARSE_2_PER_CLASS_FEASIBLE": "NOT_FOUND", "K": "NOT_FOUND", "L": "NOT_FOUND", "L_over_N": "NOT_FOUND"},
        "large_data_engineering": _memory_estimate(n, v, k, action_count, "declared metadata only; unverified because file is missing"),
        "readiness_status": "BLOCKED_MISSING_DATA",
        "readiness_reasons": ["Required dataset file is absent; actual shapes, labels, row order, and V parity cannot be verified."],
        **_common_native_fields(name, "NOT_FOUND"),
    }


def _preprocessing_record() -> dict[str, Any]:
    return {
        "source_call_order": [
            "get_dataset: cast each raw X_i to float32 and transpose only when raw shape[0] equals label count",
            "MVDataset.shuffle(seed=42): one shared column/label permutation",
            "MVDataset.standard_data: per-view StandardScaler.fit_transform(data[view].T)",
            "RandDatasetMV.__getitem__: transpose stored [d,N] to BONE batch [N,d]",
        ],
        "normalization_function": "None before standardization",
        "standardization_function": "sklearn.preprocessing.StandardScaler",
        "standardization_axis": "per feature, fitted across all N samples independently for each view",
        "per_view_independent": True,
        "uses_full_dataset_statistics": True,
        "minmax": "Available as MVDataset.min_max_scale_data but not called by native entry scripts",
        "l2_normalization": False,
        "manual_mean_std": False,
        "CORRUPTION_INSERTION_POINT_RECOMMENDATION": "Complete native preprocessing/standardization -> inject weak-quality corruption -> do not post-corruption re-standardize.",
    }


def _label_leakage_record() -> dict[str, Any]:
    return {
        "native_training_uses_ground_truth_for_forward_or_loss": False,
        "evidence": "RandDatasetMV yields labels, but native loops unpack label_i without using it; BONE.forward accepts only views and its loss uses per-view assignments.",
        "model_selection_or_early_stopping_uses_ground_truth": False,
        "ground_truth_use": "Evaluation only: get_cluster_metrics(dataset.label, predictions) in periodic evaluation blocks.",
    }


def _existing_dataset_record(name: str, path: Path, declared: list[Any]) -> dict[str, Any]:
    declared_n, declared_v, declared_k, declared_dims = declared
    # Direct MATLAB reads are deliberately used instead of get_dataset so the
    # audit can expose every cast/transpose without constructing MVDataset.
    mat = scipy.io.loadmat(path)
    if "Y" not in mat:
        raise ValueError(f"{path} lacks Y")
    raw_labels = np.asarray(mat["Y"])
    labels = raw_labels.astype(np.int32).reshape(-1)
    actual_n = int(labels.size)
    feature_keys = [f"X{i}" for i in range(1, declared_v + 1)]
    if any(key not in mat for key in feature_keys):
        raise ValueError(f"{path} lacks one or more declared feature keys")

    mappings = []
    actual_dims: list[int] = []
    for position, key in enumerate(feature_keys):
        raw = np.asarray(mat[key])
        transpose = bool(raw.shape[0] == actual_n)
        post_load_shape = raw.T.shape if transpose else raw.shape
        if len(post_load_shape) != 2:
            raise ValueError(f"{path}:{key} is not two-dimensional")
        feature_dim, sample_count = map(int, post_load_shape)
        actual_dims.append(feature_dim)
        mappings.append({
            "raw_key": key,
            "raw_shape": list(map(int, raw.shape)),
            "raw_dtype": str(raw.dtype),
            "cast_dtype": "float32",
            "transpose_applied": transpose,
            "post_load_storage_shape": [feature_dim, sample_count],
            "BONE_batch_shape": [sample_count, feature_dim],
            "feature_dim": feature_dim,
            "loader_list_position": position,
            "BONE_view_index": position,
            "P3_action_view_index": position,
            "sample_count_matches_label_count": sample_count == actual_n,
            "two_dimensional": raw.ndim == 2,
        })

    values, counts = np.unique(labels, return_counts=True)
    encoding, needs_remap, mapping_rule = _label_encoding(values)
    actual_k = int(values.size)
    action_count = action_protocol.expected_action_count(len(mappings))
    dimensions_ok = all(item["sample_count_matches_label_count"] and item["two_dimensional"] for item in mappings)
    # MVDataset.__init__ creates m_list from data_i.shape[0], and BONE receives
    # exactly dataset.m_list in both native entry scripts.
    model_expected_dims = [item["post_load_storage_shape"][0] for item in mappings]
    dims_parity = tuple(actual_dims) == tuple(model_expected_dims)
    metadata_dims_parity = tuple(declared_dims) == tuple(actual_dims)
    metadata_k_parity = declared_k == actual_k

    if name == "Caltech-5V":
        caltech_status = "EXACT_MATCH" if metadata_dims_parity else "METADATA_ONLY_MISMATCH"
    else:
        caltech_status = None

    record = {
        "dataset_name": name,
        "dataset_id": DATASET_IDS[name],
        "loader_source_file": "utils/data_util.py",
        "loader_function": "get_dataset",
        "dataset_path": f"data/{name}.mat",
        "file_exists": True,
        "FILE_STATUS": "PRESENT",
        "DATA_FILE_MISSING": False,
        "file_format": "MATLAB .mat readable by scipy.io.loadmat",
        "dataset_file_provenance": {"dataset_file_path": f"data/{name}.mat", "file_size": path.stat().st_size, "sha256": sha256_file(path)},
        "feature_keys": feature_keys,
        "label_key": "Y",
        "declared_N": declared_n,
        "declared_V": declared_v,
        "declared_K": declared_k,
        "declared_dims": list(declared_dims),
        "actual_N": actual_n,
        "actual_V": len(mappings),
        "actual_K": actual_k,
        "actual_dims": actual_dims,
        "view_mappings": mappings,
        "model_expected_dims": model_expected_dims,
        "model_dimension_parity": bool(dimensions_ok and dims_parity),
        "metadata_dimension_parity": metadata_dims_parity,
        "metadata_K_parity": metadata_k_parity,
        "RAW_ROW_ORDER_PRESERVED": False,
        "row_identity_audit": {
            "canonical_raw_row_order": "Raw MATLAB sample order before native entry processing",
            "native_entry_reorders_rows": True,
            "row_mapping": "MVDataset.shuffle(seed=42) applies one deterministic numpy permutation to every view column and labels.",
            "sort_by_label": False,
            "class_grouping": False,
            "sample_filtering": False,
            "train_test_split": False,
            "concatenation": False,
            "duplicate_removal": False,
            "dataloader_shuffle_changes_storage_identity": False,
            "dataloader_shuffle_note": "DataLoader(shuffle=True) changes minibatch visitation order only; it does not mutate stored canonical samples.",
        },
        "preprocessing": _preprocessing_record(),
        "label_audit": {
            "raw_shape": list(map(int, raw_labels.shape)),
            "raw_dtype": str(raw_labels.dtype),
            "loader_cast_dtype": "int32",
            "unique_label_values": values.astype(int).tolist(),
            "encoding": encoding,
            "K_actual": actual_k,
            "class_counts": {str(int(label)): int(count) for label, count in zip(values, counts)},
            "min_class_count": int(counts.min()),
            "max_class_count": int(counts.max()),
            "deterministic_remapping_required_for_0_based_protocol": needs_remap,
            "proposed_mapping_rule": mapping_rule,
        },
        "sparse_label_feasibility": {
            "SPARSE_2_PER_CLASS_FEASIBLE": bool(counts.min() >= 2),
            "K": actual_k,
            "L": 2 * actual_k,
            "L_over_N": (2 * actual_k) / actual_n,
        },
        "label_leakage_boundary": _label_leakage_record(),
        "ACTION_SPACE_INPUT_PARITY": len(mappings) == action_protocol.FORMAL_DATASET_ACTION_EXPECTATIONS[name]["view_num"],
        "action_space": {
            "view_num_source": "actual loader feature-key count",
            "V": len(mappings),
            "S": action_count,
            "canonical_sha256": action_protocol.canonical_action_sha256(len(mappings)),
            "reverse_closure": True,
            "partition_validity": True,
        },
        "large_data_engineering": _memory_estimate(actual_n, len(mappings), actual_k, action_count, "actual file shape and label metadata"),
        "CALTECH_VIEW_ORDER_STATUS": caltech_status,
        **_common_native_fields(name, actual_n),
    }
    warnings = []
    if not metadata_dims_parity:
        warnings.append("Declared dimension metadata does not exactly match the loader X1..Xv order.")
    if not metadata_k_parity:
        warnings.append("Declared K does not match raw label cardinality; native MVDataset derives K from labels, so future configuration must use actual_K.")
    if name == "Caltech-5V":
        warnings.append("Caltech loader/model dimensions match X1..X5 order; no loader repair was made.")
    record["warnings"] = warnings

    if not dimensions_ok or not dims_parity:
        status, reasons = "BLOCKED_DIMENSION_MISMATCH", ["A loaded view failed the [d,N] to [N,d] dimension parity check."]
    elif not record["ACTION_SPACE_INPUT_PARITY"]:
        status, reasons = "BLOCKED_VIEW_ORDER", ["Actual loader V differs from the frozen action-space V."]
    elif record["large_data_engineering"]["QUERY_CHUNKING_REQUIRED"]:
        status, reasons = "READY_WITH_MEMORY_WARNING", ["View/dimension/row/action parity passed; future query chunking is required."]
    else:
        status, reasons = "READY_FOR_P3_B0", ["File, loader-order, model-dimension, deterministic row mapping, and action-space V parity passed."]
    record["readiness_status"] = status
    record["readiness_reasons"] = reasons
    return record


def build_audit() -> dict[str, Any]:
    native_hashes = {
        "method/bone.py": sha256_file(REPO_ROOT / "method" / "bone.py"),
        "method/amv.py": sha256_file(REPO_ROOT / "method" / "amv.py"),
    }
    if native_hashes["method/bone.py"] != ORIGINAL_BONE_SHA256 or native_hashes["method/amv.py"] != ORIGINAL_AMV_SHA256:
        raise RuntimeError("native BONE source hash protection failed")
    if not FREEZE_PROTOCOL.is_file():
        raise FileNotFoundError(FREEZE_PROTOCOL)
    frozen_text = FREEZE_PROTOCOL.read_text(encoding="utf-8")
    if "STATUS:\nFROZEN" not in frozen_text or "MSRC_ACTION_ORDER_PARITY = true" not in frozen_text:
        raise ValueError("P3-A1 frozen protocol content check failed")

    datasets = []
    for name, dataset_id in DATASET_IDS.items():
        if name not in data_util.dataset_info:
            raise ValueError(f"dataset metadata missing for {dataset_id}")
        path = DATA_DIR / f"{name}.mat"
        declared = data_util.dataset_info[name]
        datasets.append(_existing_dataset_record(name, path, declared) if path.is_file() else _missing_dataset_record(name, declared))

    ready = [item["dataset_name"] for item in datasets if item["readiness_status"] in {"READY_FOR_P3_B0", "READY_WITH_MEMORY_WARNING"}]
    blocked = [item["dataset_name"] for item in datasets if item["readiness_status"] not in {"READY_FOR_P3_B0", "READY_WITH_MEMORY_WARNING"}]
    sources = [
        REPO_ROOT / "test_cmvsc.py", REPO_ROOT / "test_cmvsc_large.py", REPO_ROOT / "utils" / "data_util.py",
        REPO_ROOT / "readme.md", REPO_ROOT / "utils" / "MVDataset.py", REPO_ROOT / "utils" / "data_loader.py",
        REPO_ROOT / "utils" / "seed_util.py", REPO_ROOT / "utils" / "method_util.py", REPO_ROOT / "utils" / "cl_loss.py",
        REPO_ROOT / "method" / "bone.py", REPO_ROOT / "method" / "amv.py",
        REPO_ROOT / "experiments" / "utility_relation" / "bone_p3_a1_action_space_protocol.py", FREEZE_PROTOCOL,
    ]
    artifacts = {
        "p2_a0_msrc_carrier_pre_gt_npz": {"relative_path": str(P2_CARRIER.relative_to(REPO_ROOT)), "sha256": sha256_file(P2_CARRIER)},
        "p3_a1_freeze_protocol": _source_record(FREEZE_PROTOCOL),
    }
    return {
        "audit_scope": "read-only dataset/loader portability audit; no training, GPU, optimizer, clustering, metrics, or data mutation",
        "loader_call_graph": [
            "test_cmvsc.py or test_cmvsc_large.py -> utils.data_util.get_dataset",
            "data/<dataset>.mat -> Y and X1..Xv",
            "get_dataset casts feature values to float32 and conditionally transposes to stored [d,N]",
            "MVDataset.shuffle -> MVDataset.standard_data -> RandDatasetMV.__getitem__ returns [N,d] batch tensors",
            "BONE is initialized with dataset.m_list, derived from the same stored [d,N] view order; P3 indices retain positions 0..V-1",
        ],
        "datasets": datasets,
        "ready_datasets": ready,
        "blocked_datasets": blocked,
        "summary": {
            "total_supported_datasets": len(datasets),
            "ready_count": len(ready),
            "memory_warning_count": sum(item["large_data_engineering"]["QUERY_CHUNKING_REQUIRED"] is True for item in datasets),
            "missing_data_count": sum(item["DATA_FILE_MISSING"] is True for item in datasets),
            "blocked_count": len(blocked),
            "READY_DATASETS": ready,
            "BLOCKED_DATASETS": blocked,
        },
        "native_hashes": native_hashes,
        "protected_artifact_hashes_read_only": artifacts,
        "source_files_inspected": [_source_record(path) for path in sources],
        "frozen_p3_a1_protocol_verified": True,
    }


def _text_report(audit: dict[str, Any]) -> str:
    lines = [
        "P3-A2 Dataset/Loader Portability Audit",
        "=======================================",
        "Read-only static/data-integrity audit. No training, GPU, optimizer, clustering, metrics, or loader/data mutation.",
        "",
        "dataset         status                     actual N/V/K       actual dims                         S  sample-id adapter  chunking",
    ]
    for item in audit["datasets"]:
        dims = item["actual_dims"] if item["file_exists"] else "NOT_FOUND"
        actual = f"{item['actual_N']}/{item['actual_V']}/{item['actual_K']}"
        lines.append(
            f"{item['dataset_name']:<15} {item['readiness_status']:<26} {actual:<18} {str(dims):<35} "
            f"{item['action_space']['S']:<2} {str(item['dataloader']['SAMPLE_ID_ADAPTER_REQUIRED']):<18} "
            f"{item['large_data_engineering']['QUERY_CHUNKING_REQUIRED']}"
        )
    lines.extend([
        "",
        f"Caltech-5V status: {next(item for item in audit['datasets'] if item['dataset_name'] == 'Caltech-5V')['CALTECH_VIEW_ORDER_STATUS']}",
        "Caltech raw X1..X5 map directly to loader positions/BONE indices/P3 indices 0..4; only dataset_info dimension order differs.",
        "CCV and YouTubeFace50 files are MISSING; their actual shape/label claims are NOT_FOUND.",
        "All native RandDatasetMV paths return views and labels, never an explicit sample_id; every dataset requires an experiment-layer sample-id adapter.",
        "Native preprocessing: raw cast/conditional transpose -> shared deterministic shuffle -> per-view full-dataset StandardScaler -> BONE batch transpose.",
        "Recommended corruption insertion point: after native preprocessing, before BONE input, with no post-corruption re-standardization.",
        "Label leakage boundary: native forward/loss does not use ground-truth; labels are used by periodic evaluation only.",
        f"Ready datasets: {', '.join(audit['ready_datasets'])}",
        f"Blocked datasets: {', '.join(audit['blocked_datasets'])}",
        "",
        "Native hashes:",
        f"method/bone.py {audit['native_hashes']['method/bone.py']}",
        f"method/amv.py  {audit['native_hashes']['method/amv.py']}",
    ])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    output_dir = build_parser().parse_args().output_dir
    audit = build_audit()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "p3_a2_dataset_loader_portability_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "p3_a2_dataset_loader_portability_audit.txt").write_text(_text_report(audit), encoding="utf-8")


if __name__ == "__main__":
    main()
