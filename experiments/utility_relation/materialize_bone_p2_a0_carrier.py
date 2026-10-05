"""Materialize the frozen, pre-GT BONE-P2-A0 utility/relation carrier.

No work occurs at import time.  The native CORRUPT_BASE replay is protected by
--execute so tests and inspection cannot start the 100-epoch operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from experiments.utility_relation.bone_p2_a0_protocol import (
    CORRUPTION_SEED,
    DEFAULT_OUTPUT_DIR,
    EXPECTED_TARGETS,
    FEATURE_DIMS,
    GENERATOR_SUBSETS,
    K,
    L,
    LABELED_IDS,
    N,
    NU,
    PARENT_ARTIFACT,
    S,
    TARGET_SNR_DB,
    TRAINING_SEED,
    V,
    VERIFIER_SUBSETS,
    action_subsets,
    unlabeled_complement,
    validate_action_subsets,
    validate_sparse_targets,
)
from method.amv import amv
from method.bone import BONE
from utils.data_loader import RandDatasetMV
from utils.seed_util import set_seed


EPOCHS = 100
BATCH_SIZE = 128
N_ANCHOR = 3
LAYER = 5
LAMBDA = 0.1
LR = 0.01
DEVICE = torch.device("cpu")
VALID_TRAINING_SEEDS = (20, 30, 50)
REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_corrupted_views(artifact_path: Path | str) -> list[np.ndarray]:
    """Read only the frozen, already-standardized corrupted feature arrays."""
    with np.load(artifact_path, allow_pickle=False) as archive:
        views = [np.ascontiguousarray(archive[f"x_corrupt_v{view}"]) for view in range(V)]
    for view, values in enumerate(views):
        if values.shape != (N, FEATURE_DIMS[view]) or not np.isfinite(values).all():
            raise ValueError(f"invalid x_corrupt_v{view} in parent artifact")
    return views


def _native_data(views: Sequence[np.ndarray]) -> SimpleNamespace:
    return SimpleNamespace(data=[view.T for view in views], m_list=list(FEATURE_DIMS), v=V, k=K)


def replay_corrupt_base(views: Sequence[np.ndarray], training_seed: int = TRAINING_SEED) -> BONE:
    """Replay the fixed corrupted P1 input with the requested paired training seed."""
    if training_seed not in VALID_TRAINING_SEEDS:
        raise ValueError(f"training_seed must be one of {VALID_TRAINING_SEEDS}")
    set_seed(training_seed)
    data = _native_data(views)
    anchors = amv(data, m=N_ANCHOR, ret_anchor=True, max_iter=100, seed=training_seed,
                  info=True, info_iter=False)
    dummy_labels = np.zeros(N, dtype=np.float32)
    loader = DataLoader(
        RandDatasetMV(data.data, dummy_labels, device=DEVICE),
        batch_size=BATCH_SIZE,
        generator=torch.Generator().manual_seed(training_seed),
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )
    model = BONE(m_list=list(FEATURE_DIMS), k=K, a=N_ANCHOR, v=V, anchor_init=anchors,
                 layer=LAYER, lamb=LAMBDA, seed=training_seed, device=DEVICE)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        model.train()
        for batch, _dummy in loader:
            optimizer.zero_grad()
            output = model(batch)
            output["loss"].backward()
            optimizer.step()
    return model.eval()


def extract_q_views(model: BONE, views: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Capture BONE's four per-view posteriors without changing its public forward API."""
    observed: list[np.ndarray] = []

    def capture(_module, _inputs, output) -> None:
        observed.append(output.detach().cpu().numpy().copy())

    handle = model.label_fc.register_forward_hook(capture)
    try:
        full_views = [torch.from_numpy(view).to(device=DEVICE, dtype=torch.float32) for view in views]
        with torch.no_grad():
            fused_labels = model(full_views, test=True)["label"].detach().cpu().numpy()
    finally:
        handle.remove()
    if len(observed) != V or any(item.shape != (N, K) for item in observed):
        raise ValueError(f"label_fc hook must collect exactly {V} arrays shaped {(N, K)}")
    q_views = np.stack(observed, axis=1)
    if not np.isfinite(q_views).all() or np.any(q_views < 0):
        raise ValueError("per-view posterior contains non-finite or negative values")
    if not np.allclose(q_views.sum(axis=2), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("per-view posterior rows must sum to one")
    hook_labels = np.argmax(q_views.mean(axis=1), axis=1)
    if not np.array_equal(hook_labels, fused_labels):
        raise ValueError("hook-derived fused labels do not exactly match BONE output")
    return q_views, fused_labels


def compute_action_posteriors(q_views: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Average frozen posterior distributions over each generator/verifier side."""
    generator_subsets, verifier_subsets = action_subsets()
    validate_action_subsets(generator_subsets, verifier_subsets)
    if q_views.shape != (N, V, K):
        raise ValueError(f"q_views must have shape {(N, V, K)}")
    p_gen = q_views[:, generator_subsets, :].mean(axis=2)
    p_ver = q_views[:, verifier_subsets, :].mean(axis=2)
    return p_gen, p_ver


def compute_u_cycle(p_gen: np.ndarray, p_ver: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute frozen action validity from directional generation and complement support."""
    if p_gen.shape != p_ver.shape or p_gen.ndim != 3 or p_gen.shape[1:] != (S, K):
        raise ValueError("p_gen and p_ver must have equal shape [n, 6, 7]")
    y_gen = np.argmax(p_gen, axis=2).astype(np.int64)
    conf_gen = np.take_along_axis(p_gen, y_gen[..., None], axis=2)[..., 0]
    support_ver = np.take_along_axis(p_ver, y_gen[..., None], axis=2)[..., 0]
    closure = (np.argmax(p_ver, axis=2) == y_gen).astype(np.int8)
    u_cycle = closure * np.sqrt(conf_gen * support_ver)
    if not np.isfinite(u_cycle).all() or np.any(u_cycle < 0) or np.any(u_cycle > 1):
        raise ValueError("utility must be finite and within [0, 1]")
    return y_gen, conf_gen, support_ver, closure, u_cycle


def sparse_targets_from_parent(artifact_path: Path | str) -> np.ndarray:
    """Read the parent labels only at the frozen 14-row parity-check positions."""
    with np.load(artifact_path, allow_pickle=False) as archive:
        observed = np.asarray(archive["labels"][LABELED_IDS], dtype=np.int64)
    validate_sparse_targets(observed)
    return observed


def build_semantic_relation(
    p_gen: np.ndarray, labeled_ids: np.ndarray, labeled_targets: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build WHAT-only relation scores from sparse anchor signatures, never utilities."""
    labeled_ids = np.asarray(labeled_ids, dtype=np.int64)
    labeled_targets = np.asarray(labeled_targets, dtype=np.int64)
    unlabeled_ids = unlabeled_complement(labeled_ids)
    validate_sparse_targets(labeled_targets)
    if p_gen.shape != (N, S, K):
        raise ValueError(f"p_gen must have shape {(N, S, K)}")
    anchor_signature = p_gen[labeled_ids].mean(axis=1)
    class_signature = np.empty((K, K), dtype=p_gen.dtype)
    for class_id in range(K):
        members = anchor_signature[labeled_targets == class_id]
        if members.shape != (2, K):
            raise ValueError("each class must have exactly two sparse anchors")
        class_signature[class_id] = members.mean(axis=0)
    energy = np.einsum("usk,ck->usc", p_gen[unlabeled_ids], class_signature)
    denominator = energy.sum(axis=2)
    if np.any(denominator <= 0) or not np.isfinite(denominator).all():
        raise ValueError("semantic relation denominator must be finite and strictly positive")
    pi = energy / denominator[..., None]
    t_rel = np.take(pi, labeled_targets, axis=2).transpose(0, 2, 1)
    if not np.isfinite(pi).all() or not np.isfinite(t_rel).all():
        raise ValueError("semantic relation contains non-finite values")
    if np.any(pi < 0) or np.any(t_rel < 0) or np.any(pi > 1) or np.any(t_rel > 1):
        raise ValueError("semantic relation scores must be within [0, 1]")
    if not np.allclose(pi.sum(axis=2), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("semantic distributions must sum to one")
    return unlabeled_ids, anchor_signature, class_signature, pi, t_rel


def _audit(payload: Dict[str, np.ndarray], artifact_path: Path, training_seed: int) -> Dict[str, object]:
    u_cycle = payload["u_cycle"]
    pi = payload["pi"]
    all_tensors = {name: values for name, values in payload.items() if isinstance(values, np.ndarray)}
    return {
        "N": N, "V": V, "K": K, "L": L, "Nu": NU, "S": S,
        "training_seed": training_seed,
        "corruption_seed": CORRUPTION_SEED,
        "target_snr_db": TARGET_SNR_DB,
        "parent_corruption_artifact_sha256": sha256_file(artifact_path),
        "sparse_split_parity": True,
        "q_extraction_parity": True,
        "tensor_shapes": {name: list(values.shape) for name, values in all_tensors.items()},
        "u_min": float(u_cycle.min()), "u_max": float(u_cycle.max()), "u_mean": float(u_cycle.mean()),
        "u_nonzero_fraction": float(np.count_nonzero(u_cycle) / u_cycle.size),
        "utility_mass_per_action": [float(value) for value in u_cycle.sum(axis=0)],
        "pi_normalization_max_error": float(np.abs(pi.sum(axis=2) - 1.0).max()),
        "nan_inf_audit": {name: bool(np.isfinite(values).all()) for name, values in all_tensors.items()},
        "labels_outside_labeled_ids_used_before_seal": False,
        "post_corruption_standardization": False,
    }


def _write_seal(output_dir: Path, artifact_path: Path, files: Dict[str, Path]) -> Path:
    seal = {
        "sha256": {
            "carrier_npz": sha256_file(files["carrier"]),
            "audit": sha256_file(files["audit"]),
            "carrier_model_state": sha256_file(files["state"]),
            "protocol_source": sha256_file(REPO_ROOT / "experiments/utility_relation/bone_p2_a0_protocol.py"),
            "materializer_source": sha256_file(Path(__file__)),
            "method_bone": sha256_file(REPO_ROOT / "method/bone.py"),
            "method_amv": sha256_file(REPO_ROOT / "method/amv.py"),
            "parent_corruption_artifact": sha256_file(artifact_path),
        },
        "paths": {"parent_corruption_artifact": str(artifact_path)},
    }
    seal_path = output_dir / "p2_a0_carrier_seal.json"
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return seal_path


def materialize(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR, training_seed: int = TRAINING_SEED
) -> Dict[str, Path]:
    """Replay the fixed parent artifact with one approved training seed and write its carrier."""
    if training_seed not in VALID_TRAINING_SEEDS:
        raise ValueError(f"training_seed must be one of {VALID_TRAINING_SEEDS}")
    artifact_path, output_dir = PARENT_ARTIFACT, Path(output_dir)
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    views = load_corrupted_views(artifact_path)
    labeled_targets = sparse_targets_from_parent(artifact_path)
    model = replay_corrupt_base(views, training_seed)
    q_views, _fused_labels = extract_q_views(model, views)
    p_gen, p_ver = compute_action_posteriors(q_views)
    y_gen, conf_gen, support_ver, closure, u_cycle_all = compute_u_cycle(p_gen, p_ver)
    if np.any(u_cycle_all.sum(axis=0) <= 0):
        raise ValueError("every action must retain positive utility mass")
    unlabeled_ids, anchor_signature, class_signature, pi, t_rel = build_semantic_relation(
        p_gen, LABELED_IDS, labeled_targets
    )
    generator_subsets, verifier_subsets = action_subsets()
    payload = {
        "sample_ids": np.arange(N, dtype=np.int64), "labeled_ids": LABELED_IDS,
        "labeled_targets": labeled_targets, "unlabeled_ids": unlabeled_ids,
        "generator_subsets": generator_subsets, "verifier_subsets": verifier_subsets,
        "q_views": q_views, "p_gen": p_gen, "p_ver": p_ver, "y_gen": y_gen,
        "conf_gen": conf_gen, "support_ver": support_ver, "closure": closure,
        "u_cycle_all": u_cycle_all, "u_cycle": u_cycle_all[unlabeled_ids],
        "anchor_signature": anchor_signature, "class_signature": class_signature,
        "pi": pi, "t_rel": t_rel,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "carrier": output_dir / "p2_a0_carrier_pre_gt.npz",
        "audit": output_dir / "p2_a0_carrier_audit.json",
        "state": output_dir / "carrier_model_state.pt",
    }
    np.savez_compressed(files["carrier"], **payload)
    files["audit"].write_text(json.dumps(_audit(payload, artifact_path, training_seed), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save(model.state_dict(), files["state"])
    files["seal"] = _write_seal(output_dir, artifact_path, files)
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--training-seed", type=int, choices=VALID_TRAINING_SEEDS, default=TRAINING_SEED)
    parser.add_argument("--execute", action="store_true", help="replay the 100-epoch CORRUPT_BASE and materialize")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.execute:
        raise SystemExit("refusing to start carrier materialization without --execute")
    files = materialize(args.output_dir, args.training_seed)
    print(json.dumps({name: str(path) for name, path in files.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
