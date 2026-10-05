"""CPU-only offline materializer for P3-A3 deterministic sparse-label splits.

This module reads raw MATLAB labels solely to freeze sparse supervision.  It
does not load features, call native loaders, initialize models/optimizers, run
metrics, use an RNG, or perform training.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import scipy.io

from experiments.utility_relation.bone_p3_a3_sparse_label_protocol import (
    LABELS_PER_CLASS,
    build_sparse_label_split,
    canonical_split_payload,
    split_content_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "bone_p3" / "sparse_splits"
READY_DATASETS = ("MNIST-USPS", "BDGP", "Caltech-5V", "HandWrite", "COIL20", "MSRC", "ALOI")
BLOCKED_DATASETS = ("CCV", "YouTubeFace50")
EXPECTED_FRACTIONS = {
    "MNIST-USPS": 20 / 5000, "BDGP": 10 / 2500, "Caltech-5V": 14 / 1400,
    "HandWrite": 20 / 2000, "COIL20": 40 / 1440, "MSRC": 14 / 210, "ALOI": 200 / 10800,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw_labels(dataset_name: str, data_dir: Path = DATA_DIR) -> np.ndarray:
    """Read only Y from a raw .mat file; feature matrices are not requested."""
    path = data_dir / f"{dataset_name}.mat"
    if not path.is_file():
        raise FileNotFoundError(path)
    mat = scipy.io.loadmat(path, variable_names=["Y"])
    if "Y" not in mat:
        raise ValueError(f"{path} lacks Y")
    return np.asarray(mat["Y"]).reshape(-1)


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = BytesIO()
    np.save(stream, np.ascontiguousarray(array, dtype=np.int64), allow_pickle=False)
    return stream.getvalue()


def write_deterministic_npz(path: Path, labeled_ids: np.ndarray, labeled_targets: np.ndarray) -> None:
    """Write a stable two-array NPZ with fixed ZIP metadata and member order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, array in (("labeled_ids.npy", labeled_ids), ("labeled_targets.npy", labeled_targets)):
            info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(array), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def materialize_dataset(dataset_name: str, output_dir: Path = DEFAULT_OUTPUT_DIR, data_dir: Path = DATA_DIR) -> dict[str, object]:
    """Materialize one READY dataset's split and return its JSON audit record."""
    if dataset_name not in READY_DATASETS:
        raise ValueError(f"{dataset_name} is not a READY materialization target")
    dataset_path = data_dir / f"{dataset_name}.mat"
    raw_labels = load_raw_labels(dataset_name, data_dir)
    split = build_sparse_label_split(dataset_name, raw_labels)
    if dataset_name == "ALOI" and split.k != 100:
        raise ValueError(f"ALOI requires actual K=100, got {split.k}")
    payload = canonical_split_payload(split, raw_labels.size, sha256_file(dataset_path))
    expected_fraction = EXPECTED_FRACTIONS[dataset_name]
    if not np.isclose(payload["label_fraction"], expected_fraction, rtol=0.0, atol=1e-15):
        raise ValueError("label fraction does not match the pre-registered value")
    npz_path = output_dir / f"{dataset_name}_2perclass.npz"
    write_deterministic_npz(npz_path, split.labeled_ids, split.labeled_targets)
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != {"labeled_ids", "labeled_targets"}:
            raise ValueError("split NPZ schema is not minimal")
        if not np.array_equal(archive["labeled_ids"], split.labeled_ids) or not np.array_equal(archive["labeled_targets"], split.labeled_targets):
            raise ValueError("split NPZ content validation failed")
    payload["npz_path"] = str(npz_path.resolve().relative_to(REPO_ROOT))
    payload["npz_sha256"] = sha256_file(npz_path)
    payload["split_content_sha256"] = split_content_sha256(split)
    return payload


def materialize_all(output_dir: Path = DEFAULT_OUTPUT_DIR, data_dir: Path = DATA_DIR) -> dict[str, object]:
    """Materialize exactly the seven approved READY datasets and record blocks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [materialize_dataset(name, output_dir, data_dir) for name in READY_DATASETS]
    blocked = [
        {"dataset_name": name, "status": "BLOCKED_MISSING_DATA", "npz_materialized": False,
         "reason": "raw .mat file is absent; no split is guessed or generated"}
        for name in BLOCKED_DATASETS
    ]
    audit = {
        "protocol": "BONE_P3_A3_V1 deterministic hash-stratified sparse labels, with MSRC legacy exception",
        "labels_per_class": LABELS_PER_CLASS,
        "records": records,
        "blocked": blocked,
        "full_gt_leakage_boundary": "Full GT was read only for offline split materialization. Training must receive only labeled_ids/labeled_targets; full GT is post-seal evaluator only.",
        "no_rng": True,
        "no_training": True,
        "no_gpu": True,
    }
    audit_path = output_dir / "sparse_split_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    audit = materialize_all(build_parser().parse_args().output_dir)
    print(json.dumps({"materialized": [record["dataset_name"] for record in audit["records"]], "blocked": [record["dataset_name"] for record in audit["blocked"]]}, indent=2))


if __name__ == "__main__":
    main()
