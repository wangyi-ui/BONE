"""Protected P2-A1 four-arm pilot runner and zero-action identity smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Sequence

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from experiments.utility_relation.bone_p2_a0_protocol import FEATURE_DIMS, LABELED_IDS, PARENT_ARTIFACT
from experiments.utility_relation.bone_p2_a1_protocol import (
    ACTIVE_SEMANTIC_ARMS, ARMS, CARRIER_DIR, DEFAULT_OUTPUT_DIR, LAMBDA, LAYER, N_ANCHOR,
    NATIVE_BATCH_SIZE, NATIVE_LR, REFINEMENT_EPOCHS, SEMANTIC_LR, TRAINING_SEED, array_sha256,
    frozen_derangement, semantic_orders,
)
from experiments.utility_relation.bone_p2_a1_relation_action import (
    detached_frozen, semantic_epoch, zero_action_semantic_graph,
)
from experiments.utility_relation.evaluate_bone_p2_a0_carrier import verify_seal
from experiments.utility_relation.materialize_bone_p2_a0_carrier import load_corrupted_views, sha256_file
from method.bone import BONE
from utils.data_loader import RandDatasetMV
from utils.seed_util import set_seed


DEVICE = torch.device("cpu")
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FrozenCarrier:
    unlabeled_ids: np.ndarray
    labeled_ids: np.ndarray
    labeled_targets: np.ndarray
    u_cycle: np.ndarray
    t_rel: np.ndarray


def model_state_sha256(model: torch.nn.Module) -> str:
    """Canonical content hash for a loaded model state, independent of torch serialization."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_state(path: Path) -> dict:
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def instantiate_initial_model(carrier_dir: Path | str = CARRIER_DIR) -> BONE:
    """Build a fresh BONE instance and load the one frozen P2-A0 state."""
    state_path = Path(carrier_dir) / "carrier_model_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    set_seed(TRAINING_SEED)
    model = BONE(m_list=list(FEATURE_DIMS), k=7, a=N_ANCHOR, v=4, anchor_init=None,
                 layer=LAYER, lamb=LAMBDA, seed=TRAINING_SEED, device=DEVICE)
    model.load_state_dict(_load_state(state_path), strict=True)
    return model.to(DEVICE)


def load_frozen_carrier(carrier_dir: Path | str = CARRIER_DIR) -> FrozenCarrier:
    """Validate P2-A0 first, then read only frozen pre-GT tensors required by Phase A."""
    carrier_dir = Path(carrier_dir)
    verify_seal(carrier_dir)
    with np.load(carrier_dir / "p2_a0_carrier_pre_gt.npz", allow_pickle=False) as carrier:
        unlabeled_ids = np.asarray(carrier["unlabeled_ids"], dtype=np.int64)
        labeled_ids = np.asarray(carrier["labeled_ids"], dtype=np.int64)
        labeled_targets = np.asarray(carrier["labeled_targets"], dtype=np.int64)
        u_cycle = np.asarray(carrier["u_cycle"], dtype=np.float32)
        t_rel = np.asarray(carrier["t_rel"], dtype=np.float32)
    if unlabeled_ids.shape != (196,) or labeled_ids.shape != (14,) or labeled_targets.shape != (14,):
        raise ValueError("frozen carrier ID shapes are invalid")
    if u_cycle.shape != (196, 6) or t_rel.shape != (196, 14, 6):
        raise ValueError("frozen carrier U/T shapes are invalid")
    if not np.isfinite(u_cycle).all() or not np.isfinite(t_rel).all():
        raise ValueError("frozen carrier U/T contains non-finite values")
    return FrozenCarrier(unlabeled_ids, labeled_ids, labeled_targets, u_cycle, t_rel)


def arm_frozen_tensors(arm: str, carrier: FrozenCarrier) -> tuple[np.ndarray, np.ndarray]:
    """Choose frozen WHAT/WHETHER tensors; BASE has no semantic Phase A."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    if arm == "BASE":
        raise ValueError("BASE does not have semantic action tensors")
    relation = carrier.t_rel
    utility = carrier.u_cycle
    if arm == "TRUE_UNIFORM":
        utility = np.ones_like(utility)
    elif arm == "SHUFFLE_REL":
        relation = relation[frozen_derangement()].copy()
    return relation.copy(), utility.copy()


def _native_container(views: Sequence[np.ndarray]) -> SimpleNamespace:
    return SimpleNamespace(data=[view.T for view in views], m_list=list(FEATURE_DIMS), v=4, k=7)


class _AuditedRandDatasetMV(RandDatasetMV):
    """Preserve P1 dummy labels while exposing canonical IDs solely for sample-order audit."""
    def __getitem__(self, item: int):
        data, label = super().__getitem__(item)
        return data, label, item


def _native_loader(views: Sequence[np.ndarray]) -> DataLoader:
    """P1-native loader configuration with fixed dummy labels and audit-only sample IDs."""
    return DataLoader(
        _AuditedRandDatasetMV(_native_container(views).data, np.zeros(210, dtype=np.float32), device=DEVICE),
        batch_size=NATIVE_BATCH_SIZE,
        generator=torch.Generator().manual_seed(TRAINING_SEED),
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )


def _native_epoch(model: BONE, optimizer: optim.Optimizer, loader: DataLoader) -> str:
    """One P1-native BONE consolidation epoch, with an audit hash of observed sample order."""
    observed_ids: list[np.ndarray] = []
    model.train()
    for data_batch, _dummy_labels, audit_ids in loader:
        observed_ids.append(audit_ids.detach().cpu().numpy().astype(np.int64))
        optimizer.zero_grad(set_to_none=True)
        output = model(data_batch)
        output["loss"].backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("native gradients were not cleared before the next semantic Phase A")
    return array_sha256(np.concatenate(observed_ids))


def _full_views(views: Sequence[np.ndarray]) -> list[torch.Tensor]:
    return [torch.from_numpy(view).to(device=DEVICE, dtype=torch.float32) for view in views]


def _write_pre_gt_seal(arm_dir: Path, files: Dict[str, Path]) -> Path:
    seal = {
        "sha256": {
            "predictions": sha256_file(files["predictions"]),
            "audit": sha256_file(files["audit"]),
            "runner_source": sha256_file(Path(__file__)),
            "action_source": sha256_file(REPO_ROOT / "experiments/utility_relation/bone_p2_a1_relation_action.py"),
            "protocol_source": sha256_file(REPO_ROOT / "experiments/utility_relation/bone_p2_a1_protocol.py"),
            "method_bone": sha256_file(REPO_ROOT / "method/bone.py"),
            "method_amv": sha256_file(REPO_ROOT / "method/amv.py"),
            "carrier_npz": sha256_file(CARRIER_DIR / "p2_a0_carrier_pre_gt.npz"),
            "initial_model_state": sha256_file(CARRIER_DIR / "carrier_model_state.pt"),
        }
    }
    path = arm_dir / "pre_gt_seal.json"
    path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_arm(
    arm: str,
    carrier: FrozenCarrier,
    views: Sequence[np.ndarray],
    orders: np.ndarray,
    output_dir: Path,
) -> Dict[str, object]:
    """Independently run one paired arm from the same loaded initial state."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    model = instantiate_initial_model()
    initial_hash = model_state_sha256(model)
    native_optimizer = optim.Adam(model.parameters(), lr=NATIVE_LR)
    semantic_optimizer = None
    relation_targets = action_weights = None
    if arm in ACTIVE_SEMANTIC_ARMS:
        relation, utility = arm_frozen_tensors(arm, carrier)
        relation_targets = detached_frozen(relation, DEVICE)
        action_weights = detached_frozen(utility, DEVICE)
        semantic_optimizer = optim.Adam(model.parameters(), lr=SEMANTIC_LR)
        if semantic_optimizer is native_optimizer:
            raise RuntimeError("semantic and native optimizers must be independent objects")
    loader = _native_loader(views)
    full_views = _full_views(views)
    unlabeled_ids = torch.as_tensor(carrier.unlabeled_ids, dtype=torch.long, device=DEVICE)
    labeled_ids = torch.as_tensor(carrier.labeled_ids, dtype=torch.long, device=DEVICE)
    native_order_hashes: list[str] = []
    semantic_steps: list[int] = []
    for epoch in range(REFINEMENT_EPOCHS):
        if semantic_optimizer is not None:
            semantic_steps.append(semantic_epoch(
                model, semantic_optimizer, full_views, unlabeled_ids, labeled_ids,
                relation_targets, action_weights, orders[epoch],
            ))
        native_order_hashes.append(_native_epoch(model, native_optimizer, loader))
    model.eval()
    with torch.no_grad():
        predictions = model(full_views, test=True)["label"].cpu().numpy().astype(np.int64)
    arm_dir = output_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    files = {"predictions": arm_dir / "final_predictions.npz", "audit": arm_dir / "pre_gt_audit.json"}
    np.savez_compressed(files["predictions"], sample_ids=np.arange(210, dtype=np.int64), predictions=predictions)
    audit = {
        "arm": arm,
        "initial_model_hash": initial_hash,
        "native_optimizer_reinitialized": True,
        "semantic_optimizer_created": semantic_optimizer is not None,
        "semantic_optimizer_steps_per_epoch": semantic_steps,
        "semantic_order_sha256": array_sha256(orders) if semantic_optimizer is not None else None,
        "native_order_hashes": native_order_hashes,
        "semantic_to_native_grad_leak": False,
        "native_to_next_semantic_grad_leak": False,
        "u_cycle_frozen_detached": action_weights is None or not action_weights.requires_grad,
        "t_rel_frozen_detached": relation_targets is None or not relation_targets.requires_grad,
        "full_gt_used_before_seal": False,
    }
    files["audit"].write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files["seal"] = _write_pre_gt_seal(arm_dir, files)
    return {"initial_model_hash": initial_hash, "files": files, "audit": audit}


def run_pilot(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Dict[str, object]:
    """Run all four formal paired arms; caller must explicitly pass --execute on the CLI."""
    carrier = load_frozen_carrier()
    views = load_corrupted_views(PARENT_ARTIFACT)
    orders = semantic_orders()
    output_dir = Path(output_dir)
    results = {arm: run_arm(arm, carrier, views, orders, output_dir) for arm in ARMS}
    hashes = {result["initial_model_hash"] for result in results.values()}
    if len(hashes) != 1:
        raise RuntimeError("four arms did not begin from exactly the same model state")
    active_orders = {result["audit"]["semantic_order_sha256"] for arm, result in results.items() if arm in ACTIVE_SEMANTIC_ARMS}
    if len(active_orders) != 1:
        raise RuntimeError("active arms did not share exactly one semantic query order")
    native_orders = {tuple(result["audit"]["native_order_hashes"]) for result in results.values()}
    if len(native_orders) != 1:
        raise RuntimeError("arms did not share exact native sample orders")
    return results


def zero_action_smoke(carrier_dir: Path | str = CARRIER_DIR) -> Dict[str, object]:
    """Permitted identity smoke: one zero-scaled semantic graph, no native Phase B or writes."""
    carrier = load_frozen_carrier(carrier_dir)
    model = instantiate_initial_model(carrier_dir)
    before = model_state_sha256(model)
    views = _full_views(load_corrupted_views(PARENT_ARTIFACT))
    relation, utility = arm_frozen_tensors("TRUE_U", carrier)
    semantic_optimizer = optim.Adam(model.parameters(), lr=SEMANTIC_LR)
    zero_action_semantic_graph(
        model, semantic_optimizer, views,
        torch.as_tensor(carrier.unlabeled_ids, dtype=torch.long, device=DEVICE),
        torch.as_tensor(carrier.labeled_ids, dtype=torch.long, device=DEVICE),
        detached_frozen(relation, DEVICE), detached_frozen(utility, DEVICE),
    )
    after = model_state_sha256(model)
    if before != after:
        raise RuntimeError("zero-action smoke is not exact model identity")
    return {"initial_model_hash": before, "final_model_hash": after, "exact_identity": True}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true", help="run all four formal P2-A1 arms")
    parser.add_argument("--zero-action-smoke", action="store_true", help="run only the zero-scaled semantic identity smoke")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.zero_action_smoke:
        print(json.dumps(zero_action_smoke(), indent=2, sort_keys=True))
        return
    if not args.execute:
        raise SystemExit("refusing to start formal P2-A1 arms without --execute")
    print(json.dumps({arm: result["initial_model_hash"] for arm, result in run_pilot(args.output_dir).items()}, indent=2))


if __name__ == "__main__":
    main()
