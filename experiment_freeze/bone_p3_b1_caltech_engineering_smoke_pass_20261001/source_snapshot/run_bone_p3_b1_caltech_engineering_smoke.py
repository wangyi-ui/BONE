"""P3-B1 Caltech-5V engineering-only, pre-GT BONE carrier smoke.

The command is deliberately opt-in (``--execute``).  It transports raw file
row IDs through the native Caltech preprocessing/training path, uses only the
frozen P3-A3/P3-A4 artifacts, and writes a sealed pre-GT carrier.  It never
loads Caltech's ``Y`` field and never computes clustering metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import scipy.io
import torch
from torch import optim
from torch.utils.data import DataLoader

from experiments.utility_relation import bone_p3_a1_action_space_protocol as action_protocol
from experiments.utility_relation import bone_p3_a4_corruption_protocol as corruption_protocol
from experiments.utility_relation import bone_p3_b0_generic_carrier as carrier
from experiments.utility_relation import materialize_bone_p3_b0_carrier as b0_materializer
from method.amv import amv
from method.bone import BONE
from utils.seed_util import set_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "Caltech-5V"
DATA_PATH = REPO_ROOT / "data" / f"{DATASET_NAME}.mat"
OUTPUT_DIR = REPO_ROOT / "outputs" / "bone_p3" / "p3_b1_caltech_engineering_smoke_seed20"
SPARSE_SPLIT = REPO_ROOT / "outputs" / "bone_p3" / "sparse_splits" / f"{DATASET_NAME}_2perclass.npz"
CORRUPTION_MASK = REPO_ROOT / "outputs" / "bone_p3" / "corruptions" / f"{DATASET_NAME}_mask.npz"

N, V, K = 1400, 5, 7
FEATURE_DIMS = (40, 254, 1984, 512, 928)
TRAINING_SEED = 20
ENGINEERING_EPOCHS = 2
BATCH_SIZE, N_ANCHOR, LAYER = 128, 4, 3
LR, LAMBDA = 1e-2, 1e-1
QUERY_CHUNK_SIZE = 256
NATIVE_BONE_PATH = REPO_ROOT / "method" / "bone.py"
NATIVE_AMV_PATH = REPO_ROOT / "method" / "amv.py"
P3_A1_SOURCE = REPO_ROOT / "experiments" / "utility_relation" / "bone_p3_a1_action_space_protocol.py"
P3_B0_SOURCE = REPO_ROOT / "experiments" / "utility_relation" / "bone_p3_b0_generic_carrier.py"
EXPECTED_NATIVE_HASHES = {
    "method/bone.py": "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092",
    "method/amv.py": "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc",
}


def sha256_file(path: Path | str) -> str:
    """Return a content hash, failing if a required frozen input is absent."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_hashes() -> dict[str, str]:
    return {"method/bone.py": sha256_file(NATIVE_BONE_PATH), "method/amv.py": sha256_file(NATIVE_AMV_PATH)}


def verify_native_hash_protection() -> dict[str, str]:
    """Require native BONE/AMV sources to remain byte-identical to P3's freeze."""
    observed = native_hashes()
    if observed != EXPECTED_NATIVE_HASHES:
        raise RuntimeError(f"native source hash protection failed: {observed}")
    return observed


def frozen_input_hashes() -> dict[str, str]:
    return {
        "caltech_data_source": sha256_file(DATA_PATH),
        "p3_a1_protocol": sha256_file(P3_A1_SOURCE),
        "p3_a3_split": sha256_file(SPARSE_SPLIT),
        "p3_a4_mask": sha256_file(CORRUPTION_MASK),
        "p3_b0_source": sha256_file(P3_B0_SOURCE),
    }


def load_caltech_raw_views(data_path: Path = DATA_PATH) -> list[np.ndarray]:
    """Read only X1..X5 in raw file-row order; the label field is not requested."""
    keys = tuple(f"X{index}" for index in range(1, V + 1))
    mat = scipy.io.loadmat(data_path, variable_names=keys)
    views: list[np.ndarray] = []
    for key, dim in zip(keys, FEATURE_DIMS):
        if key not in mat:
            raise ValueError(f"Caltech source is missing {key}")
        stored = np.asarray(mat[key])
        if stored.shape != (dim, N):
            raise ValueError(f"{key} expected {(dim, N)}, found {stored.shape}")
        view = np.ascontiguousarray(stored.T, dtype=np.float32)
        if view.shape != (N, dim) or not np.isfinite(view).all():
            raise ValueError(f"{key} is not a finite raw [N, D] view")
        views.append(view)
    return views


def load_and_audit_frozen_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load frozen A3/A4 artifacts and enforce all Caltech-specific invariants."""
    labeled_ids, labeled_targets, split_path = b0_materializer.load_frozen_sparse_split(DATASET_NAME)
    mask, raw_row_ids, mask_path = b0_materializer.load_frozen_corruption_mask(DATASET_NAME)
    if split_path != SPARSE_SPLIT or mask_path != CORRUPTION_MASK:
        raise RuntimeError("P3-B1 must use the canonical frozen Caltech artifact locations")
    if mask.shape != (N, V) or not np.array_equal(raw_row_ids, np.arange(N, dtype=np.int64)):
        raise ValueError("frozen Caltech mask does not cover exactly raw IDs 0..1399")
    corruption_protocol.validate_balanced_half_view_mask(mask, DATASET_NAME)
    row_values, row_counts = np.unique(mask.sum(axis=1), return_counts=True)
    row_histogram = {str(int(value)): int(count) for value, count in zip(row_values, row_counts)}
    if row_histogram != {"2": 700, "3": 700} or not np.array_equal(mask.sum(axis=0), np.full(V, 700)):
        raise ValueError("frozen Caltech mask violates required row/per-view counts")
    if labeled_ids.shape != (14,) or labeled_targets.shape != (14,):
        raise ValueError("frozen Caltech sparse split must contain L=14")
    if np.unique(labeled_ids).size != 14 or np.any(labeled_ids < 0) or np.any(labeled_ids >= N):
        raise ValueError("frozen Caltech sparse IDs must be unique raw rows")
    if not np.array_equal(np.bincount(labeled_targets, minlength=K), np.full(K, 2)):
        raise ValueError("frozen Caltech sparse split must have exactly two labels per class")
    return labeled_ids, labeled_targets, mask, {
        "sparse_split_path": str(split_path),
        "corruption_mask_path": str(mask_path),
        "row_histogram": row_histogram,
        "per_view_counts": mask.sum(axis=0).astype(int).tolist(),
        "L": int(labeled_ids.size),
        "labels_per_class": 2,
        "raw_id_semantics": carrier.RAW_ID_SEMANTICS,
    }


def verify_sample_id_adapter(adapter: carrier.SampleIdAdapter, labeled_ids: np.ndarray) -> dict[str, Any]:
    """Check canonical IDs and their one-to-one training-row recovery."""
    ids = adapter.sample_ids
    if ids.shape != (N,) or np.unique(ids).size != N or not np.array_equal(np.sort(ids), np.arange(N, dtype=np.int64)):
        raise ValueError("sample IDs must be exactly the complete raw-row set 0..1399")
    anchor_rows = adapter.row_positions(labeled_ids)
    if np.unique(anchor_rows).size != labeled_ids.size or not np.array_equal(ids[anchor_rows], labeled_ids):
        raise ValueError("each sparse anchor raw ID must map to exactly one training row")
    return {"unique_raw_ids": int(np.unique(ids).size), "exact_raw_id_set_0_to_1399": True,
            "anchor_rows": anchor_rows.astype(int).tolist(), "anchor_mapping_exact_once": True}


def audit_dataloader_identity(dataset: carrier.RawIdDataset, adapter: carrier.SampleIdAdapter, views: Sequence[np.ndarray]) -> dict[str, Any]:
    """Verify an unshuffled audit pass and every returned feature/ID pairing."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)
    observed: list[np.ndarray] = []
    for data_batch, raw_ids in loader:
        raw = raw_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        positions = adapter.row_positions(raw)
        for view_index, batch_view in enumerate(data_batch):
            if not np.array_equal(batch_view.detach().cpu().numpy(), views[view_index][positions]):
                raise ValueError("DataLoader feature rows do not recover their declared raw IDs")
        observed.append(raw.copy())
    ordered = np.concatenate(observed)
    if not np.array_equal(ordered, adapter.sample_ids):
        raise ValueError("DataLoader identity audit changed native training order")
    return {"audit_loader_drop_last": False, "audit_loader_observed_ids": int(ordered.size),
            "dataloader_feature_to_raw_id_mapping_exact": True}


def _native_container(views: Sequence[np.ndarray]) -> SimpleNamespace:
    return SimpleNamespace(data=[view.T for view in views], m_list=list(FEATURE_DIMS), v=V, k=K)


def train_engineering_carrier(
    views: Sequence[np.ndarray], adapter: carrier.SampleIdAdapter, device: torch.device
) -> tuple[BONE, list[dict[str, Any]], dict[str, Any]]:
    """Run exactly two native-semantic epochs, retaining IDs alongside each batch."""
    set_seed(TRAINING_SEED)
    native_data = _native_container(views)
    anchors = amv(native_data, m=N_ANCHOR, ret_anchor=True, max_iter=100, seed=TRAINING_SEED,
                  info=True, info_iter=False)
    raw_dataset = carrier.RawIdDataset([view.T for view in views], adapter)
    loader = DataLoader(raw_dataset, batch_size=BATCH_SIZE, generator=torch.Generator().manual_seed(TRAINING_SEED),
                        num_workers=0, shuffle=True, drop_last=True)
    model = BONE(m_list=list(FEATURE_DIMS), k=K, a=N_ANCHOR, v=V, anchor_init=anchors,
                 layer=LAYER, lamb=LAMBDA, seed=TRAINING_SEED, device=device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    epoch_audit: list[dict[str, Any]] = []
    for epoch in range(ENGINEERING_EPOCHS):
        model.train()
        losses: list[float] = []
        seen: list[np.ndarray] = []
        for data_batch, raw_ids in loader:
            ids = raw_ids.detach().cpu().numpy().astype(np.int64, copy=False)
            positions = adapter.row_positions(ids)
            for view_index, batch_view in enumerate(data_batch):
                expected = views[view_index][positions]
                if not np.array_equal(batch_view.detach().cpu().numpy(), expected):
                    raise ValueError("shuffled training batch lost raw sample identity")
            # RawIdDataset intentionally keeps audit data on CPU.  This is the
            # sole device adapter; BONE/Adam semantics remain native.
            data_batch = [batch_view.to(device=device, dtype=torch.float32) for batch_view in data_batch]
            optimizer.zero_grad()
            result = model(data_batch)
            loss = result["loss"]
            if loss is None or not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("BONE training loss is non-finite")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            seen.append(ids.copy())
        if len(losses) != N // BATCH_SIZE or not np.isfinite(losses).all():
            raise RuntimeError("native engineering epoch did not complete with finite losses")
        emitted = np.concatenate(seen)
        if np.unique(emitted).size != emitted.size or np.any(emitted < 0) or np.any(emitted >= N):
            raise ValueError("shuffled native DataLoader emitted duplicate or invalid raw IDs")
        epoch_audit.append({"epoch": epoch + 1, "completed": True, "losses_finite": True,
                            "batch_count": len(losses), "loss_min": float(np.min(losses)),
                            "loss_max": float(np.max(losses)), "shuffled_ids_recoverable": True,
                            "emitted_unique_ids": int(np.unique(emitted).size)})
    return model, epoch_audit, {"batch_size": BATCH_SIZE, "drop_last": True, "num_workers": 0,
                                 "shuffle": True, "generator_seed": TRAINING_SEED}


def _validate_carrier_shapes(payload: dict[str, np.ndarray]) -> dict[str, list[int]]:
    expected = {"q_views": (N, V, K), "p_gen": (N, 20, K), "p_ver": (N, 20, K),
                "U_cycle": (N, 20), "Pi": (N - 14, 20, K), "T_rel": (N - 14, 14, 20)}
    for name, shape in expected.items():
        values = payload[name]
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"carrier {name} expected finite shape {shape}, found {values.shape}")
    if payload["generator_actions"].shape[0] != 20 or payload["verifier_actions"].shape[0] != 20:
        raise ValueError("P3-A1 action count must be S=20")
    return {name: list(shape) for name, shape in expected.items()}


def prepare_cuda_memory_stats() -> None:
    """Select CUDA:0 then use PyTorch's current-device memory-stat APIs."""
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_cuda_memory_stats() -> dict[str, int]:
    """Read peak statistics from the current CUDA device without a device argument."""
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def run_engineering_smoke(output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    """Execute the opt-in CUDA smoke and seal its pre-GT carrier artifacts."""
    if not torch.cuda.is_available():
        raise RuntimeError("FAIL-CLOSED: CUDA device 0 is required for P3-B1")
    native_before = verify_native_hash_protection()
    frozen_before = frozen_input_hashes()
    labeled_ids, labeled_targets, raw_mask, frozen_audit = load_and_audit_frozen_inputs()
    raw_views = load_caltech_raw_views()
    training_views, adapter = carrier.preprocess_with_raw_ids(raw_views, raw_mask)
    sample_id_audit = verify_sample_id_adapter(adapter, labeled_ids)
    raw_dataset = carrier.RawIdDataset([view.T for view in training_views], adapter)
    dataloader_audit = audit_dataloader_identity(raw_dataset, adapter, training_views)
    device = torch.device("cuda:0")
    prepare_cuda_memory_stats()
    model, epoch_audit, loader_config = train_engineering_carrier(training_views, adapter, device)
    model.eval()
    q_views = carrier.extract_q_views(model, training_views, device=device)
    carrier.validate_q_views(q_views, n=N, v=V, k=K)
    payload = carrier.build_generic_carrier(q_views, adapter.sample_ids, labeled_ids, labeled_targets,
                                            query_chunk_size=QUERY_CHUNK_SIZE)
    tensor_shapes = _validate_carrier_shapes(payload)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "caltech_p3_b1_carrier_model_state.pt"
    carrier_path = output_dir / "caltech_p3_b1_pre_gt_carrier.npz"
    audit_path = output_dir / "caltech_p3_b1_pre_gt_audit.json"
    torch.save(model.state_dict(), state_path)
    post_state_hash = sha256_file(state_path)
    np.savez_compressed(carrier_path, **payload)
    native_after = verify_native_hash_protection()
    frozen_after = frozen_input_hashes()
    if native_before != native_after or frozen_before != frozen_after:
        raise RuntimeError("FAIL-CLOSED: native or frozen artifact hash changed during smoke")
    audit: dict[str, Any] = {
        "status": "PASS", "ENGINEERING_ONLY": True, "FORMAL_RESULT": False,
        "GT_loaded_before_seal": False, "full_gt_evaluation_performed": False,
        "dataset": {"name": DATASET_NAME, "N": N, "V": V, "K": K, "dims": list(FEATURE_DIMS)},
        "training": {"seed": TRAINING_SEED, "native_epochs": ENGINEERING_EPOCHS, "all_losses_finite": True, "native_config": {
            "batch_size": BATCH_SIZE, "n_anchor": N_ANCHOR, "layer": LAYER, "lr": LR, "lamb": LAMBDA},
            "epoch_audit": epoch_audit, "dataloader": loader_config},
        "device": {"requested": "cuda:0", "name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
                   "cuda_version": torch.version.cuda, **peak_cuda_memory_stats()},
        "sample_id_audit": {**sample_id_audit, **dataloader_audit},
        "corruption_audit": {**frozen_audit, "snr_db": 2.5, "corruption_seed": 20,
                               "standardized_before_corruption": True, "post_corruption_restandardization": False},
        "carrier": {"query_chunk_size": QUERY_CHUNK_SIZE, "action_count": 20, "tensor_shapes": tensor_shapes,
                    "q_finite": True, "u_cycle_finite": True, "t_rel_finite": True,
                    "query_count": int(payload["query_sample_ids"].size)},
        "hashes": {**frozen_after, "p3_b0_model_state": post_state_hash, "native_before": native_before,
                   "native_after": native_after, "frozen_before_equals_after": True},
        "artifacts": {"carrier": str(carrier_path), "model_state": str(state_path)},
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"carrier": carrier_path, "audit": audit_path, "model_state": state_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true", help="run the two-epoch CUDA engineering smoke")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.execute:
        parser = build_parser()
        parser.error("refusing to launch P3-B1 GPU smoke without --execute")
    paths = run_engineering_smoke(args.output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
