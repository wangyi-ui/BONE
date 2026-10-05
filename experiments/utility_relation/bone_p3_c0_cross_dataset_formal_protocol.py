"""Frozen constants and invariant helpers for the P3-C0 formal protocol.

This module defines no new learning component.  It records the cross-dataset
schedule and resolves the already-frozen P3-A3/P3-A4 artifacts used by the
opt-in runner.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "bone_p3" / "p3_c0_formal"
FORMAL_DATASETS = ("Caltech-5V", "BDGP", "HandWrite", "COIL20", "MNIST-USPS", "ALOI", "CCV")
FORMAL_SEEDS = (20, 30, 50)
CORRUPTION_SEED, SNR_DB = 20, 2.5
CARRIER_EPOCHS, REFINEMENT_EPOCHS = 100, 20
SEMANTIC_LR, NATIVE_LR = 1e-4, 1e-2
SEMANTIC_BATCH_SIZE, QUERY_CHUNK_SIZE = 128, 256
RAW_ID_SEMANTICS = "RAW_FILE_ROW_ID"
ENGINEERING_ONLY, FORMAL_RESULT = False, True
NATIVE_HASHES = {
    "method/bone.py": "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092",
    "method/amv.py": "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc",
}
PARENT_FREEZE_HASHES = {
    "p3_a1_source": "5564354b41f78917b1ad4f98ae8737fe997d3726902e5e2a97e1ec024e20d61a",
    "p3_a3_source": "07571e80de1917e7358e8d614a3fb8d159b8ee635331f2f4c83e709acc387e7a",
    "p3_a4_source": "1865741c1ac3f2f65ffb9c325dc30a7d34978f96adc1be0ad000b45ed899b3c9",
    "p3_b0_source": "5a9f3c817d0e0ea07e311f0899fdb926cef993d0f3b7edbd65faf7f83dfabc27",
    "p3_b1_runner": "d686d59a2a7000eecb2626413b0640edb95a4d25de0db2c8834fa402cddef771",
}
PARENT_FREEZE_PATHS = {
    "p3_a1_source": REPO_ROOT / "experiments/utility_relation/bone_p3_a1_action_space_protocol.py",
    "p3_a3_source": REPO_ROOT / "experiments/utility_relation/bone_p3_a3_sparse_label_protocol.py",
    "p3_a4_source": REPO_ROOT / "experiments/utility_relation/bone_p3_a4_corruption_protocol.py",
    "p3_b0_source": REPO_ROOT / "experiments/utility_relation/bone_p3_b0_generic_carrier.py",
    "p3_b1_runner": REPO_ROOT / "experiments/utility_relation/run_bone_p3_b1_caltech_engineering_smoke.py",
}


@dataclass(frozen=True)
class DatasetSpec:
    n: int
    v: int
    dims: tuple[int, ...]
    batch_size: int
    n_anchor: int
    layer: int


DATASET_SPECS = {
    "Caltech-5V": DatasetSpec(1400, 5, (40, 254, 1984, 512, 928), 128, 4, 3),
    "BDGP": DatasetSpec(2500, 2, (1750, 79), 512, 2, 2),
    "HandWrite": DatasetSpec(2000, 6, (216, 76, 64, 6, 240, 47), 128, 4, 3),
    "COIL20": DatasetSpec(1440, 3, (1024, 3304, 6750), 128, 1, 4),
    "MNIST-USPS": DatasetSpec(5000, 2, (784, 784), 512, 4, 2),
    "ALOI": DatasetSpec(10800, 4, (77, 13, 64, 125), 512, 2, 2),
    "CCV": DatasetSpec(6773, 3, (5000, 5000, 4000), 512, 3, 2),
}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_dict_sha256(state_dict: dict) -> str:
    """Canonical state content hash independent of torch serialization bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sha256_file_from_array(values: np.ndarray) -> str:
    """Stable digest for audited integer sample/query schedules."""
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def dataset_slug(dataset_name: str) -> str:
    if dataset_name not in FORMAL_DATASETS:
        raise ValueError(f"unsupported P3-C0 dataset: {dataset_name}")
    return dataset_name.lower().replace("-", "_")


def artifact_paths(dataset_name: str) -> tuple[Path, Path]:
    """Resolve frozen A3 split and A4 mask; CCV has its approved addendum path."""
    if dataset_name not in FORMAL_DATASETS:
        raise ValueError(f"unsupported P3-C0 dataset: {dataset_name}")
    if dataset_name == "CCV":
        root = REPO_ROOT / "outputs" / "bone_p3" / "ccv_incremental"
        return root / "CCV_2perclass.npz", root / "CCV_corruption_mask.npz"
    root = REPO_ROOT / "outputs" / "bone_p3"
    return root / "sparse_splits" / f"{dataset_name}_2perclass.npz", root / "corruptions" / f"{dataset_name}_mask.npz"


def semantic_orders(query_count: int, training_seed: int) -> np.ndarray:
    """Fixed [20,N-L] query permutations; no label or performance input exists."""
    if training_seed not in FORMAL_SEEDS:
        raise ValueError("unregistered formal training seed")
    rng = np.random.default_rng(training_seed)
    return np.stack([rng.permutation(query_count).astype(np.int64) for _ in range(REFINEMENT_EPOCHS)])


def verify_native_hashes() -> dict[str, str]:
    observed = {name: sha256_file(REPO_ROOT / name) for name in NATIVE_HASHES}
    if observed != NATIVE_HASHES:
        raise RuntimeError(f"native source hash mismatch: {observed}")
    return observed


def verify_parent_freeze_hashes() -> dict[str, str]:
    """Fail closed if a P3-A1/A3/A4/B0/B1 frozen parent source drifts."""
    observed = {name: sha256_file(path) for name, path in PARENT_FREEZE_PATHS.items()}
    if observed != PARENT_FREEZE_HASHES:
        raise RuntimeError(f"frozen parent source hash mismatch: {observed}")
    return observed


def expected_shapes(dataset_name: str, k: int) -> dict[str, tuple[int, ...]]:
    spec = DATASET_SPECS[dataset_name]
    labeled = 2 * k
    # S is provided by frozen P3-A1 based solely on V.
    from experiments.utility_relation.bone_p3_a1_action_space_protocol import expected_action_count
    s = expected_action_count(spec.v)
    return {"q_views": (spec.n, spec.v, k), "p_gen": (spec.n, s, k), "p_ver": (spec.n, s, k),
            "U_cycle": (spec.n, s), "Pi": (spec.n - labeled, s, k), "T_rel": (spec.n - labeled, labeled, s)}
