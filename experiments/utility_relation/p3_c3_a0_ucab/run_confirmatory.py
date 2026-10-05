"""Pre-GT-only UCAB dataset-validation runner; it never loads full labels or metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import optim

from experiments.utility_relation import bone_p3_b0_generic_carrier as b0
from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as c0_protocol
from experiments.utility_relation import run_bone_p3_c0_cross_dataset_formal as c0
from method.amv import amv
from utils.seed_util import set_seed

from .budget import build_budget
from .semantic_epoch import run_semantic_epoch
from . import validation_io as io


def _semantic_orders(query_count: int, training_seed: int) -> np.ndarray:
    """The frozen C0 permutation recipe, extended only to preregistered new seeds."""
    rng = np.random.default_rng(training_seed)
    return np.stack([rng.permutation(query_count).astype(np.int64) for _ in range(c0_protocol.REFINEMENT_EPOCHS)])


def _run_config(dataset: str, training_seed: int, output_root: str | Path, device: str) -> dict:
    io.validate_dataset_seed(dataset, training_seed)
    if device != "cuda:0":
        raise ValueError("FROZEN_VALIDATION_DEVICE_MUST_BE_CUDA_0")
    semantic = io.semantic_config(dataset)
    return {
        "stage": "P3-C3-A0-UCAB-DATASET-VALIDATION",
        "dataset": dataset,
        "training_seed": training_seed,
        "seed_role": io.seed_role(training_seed),
        "arms": list(io.arms_for_seed(training_seed)),
        "device": device,
        "semantic_optimizer": semantic["optimizer"],
        "semantic_lr": semantic["semantic_lr"],
        "semantic_exposure": semantic["semantic_exposure"],
        "carrier_epochs": c0_protocol.CARRIER_EPOCHS,
        "refinement_epochs": c0_protocol.REFINEMENT_EPOCHS,
        "native_optimizer": "Adam",
        "native_lr": c0_protocol.NATIVE_LR,
        "full_gt_read_by_training": False,
        "formal_metrics_computed": False,
        "hyperparameter_cli_override_allowed": False,
        "seed_replacement_allowed": False,
        "output_root": str(Path(output_root).resolve()),
    }


def _validate_carrier_payload(payload: dict[str, np.ndarray], adapter: b0.SampleIdAdapter, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {"query_rows", "anchor_rows", "T_rel", "U_cycle", "query_U_cycle"}
    if not required.issubset(payload):
        raise RuntimeError("CARRIER_PAYLOAD_SCHEMA_MISMATCH")
    query_rows = np.asarray(payload["query_rows"], dtype=np.int64)
    anchor_rows = np.asarray(payload["anchor_rows"], dtype=np.int64)
    t_rel = np.asarray(payload["T_rel"])
    full_u_cycle = np.asarray(payload["U_cycle"])
    query_u_cycle = np.asarray(payload["query_U_cycle"])
    if query_rows.ndim != 1 or anchor_rows.ndim != 1 or np.unique(query_rows).size != query_rows.size or np.unique(anchor_rows).size != anchor_rows.size:
        raise RuntimeError("CARRIER_ROW_SCHEMA_MISMATCH")
    if np.intersect1d(query_rows, anchor_rows).size or np.any(query_rows < 0) or np.any(query_rows >= adapter.sample_ids.size):
        raise RuntimeError("QUERY_ANCHOR_ROW_ALIGNMENT_MISMATCH")
    if (
        full_u_cycle.ndim != 2
        or query_u_cycle.ndim != 2
        or full_u_cycle.shape[0] != adapter.sample_ids.size
        or query_u_cycle.shape[0] != query_rows.size
        or full_u_cycle.shape[1] != query_u_cycle.shape[1]
        or t_rel.shape != (query_rows.size, anchor_rows.size, query_u_cycle.shape[1])
    ):
        raise RuntimeError("T_REL_U_CYCLE_QUERY_ALIGNMENT_MISMATCH")
    if not np.array_equal(query_u_cycle, full_u_cycle[query_rows]):
        raise RuntimeError("QUERY_U_CYCLE_PROJECTION_MISMATCH")
    if (
        not np.isfinite(t_rel).all()
        or not np.isfinite(full_u_cycle).all()
        or not np.isfinite(query_u_cycle).all()
        or np.any(full_u_cycle < 0)
        or np.any(full_u_cycle > 1)
        or np.any(query_u_cycle < 0)
        or np.any(query_u_cycle > 1)
    ):
        raise RuntimeError("CARRIER_UTILITY_INVALID")
    query_ids = np.asarray(adapter.sample_ids[query_rows], dtype=np.int64)
    if np.unique(query_ids).size != query_ids.size or k < 2:
        raise RuntimeError("RAW_QUERY_SAMPLE_ID_ALIGNMENT_MISMATCH")
    return query_rows, anchor_rows, t_rel, query_u_cycle


def execute_pre_gt_training(config: dict, root: Path) -> dict:
    """Real GPU backend. It intentionally has no full-GT or metric dependency."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_DATASET_VALIDATION")
    torch.cuda.set_device(0)
    device, dataset, seed = torch.device("cuda:0"), config["dataset"], config["training_seed"]
    spec = c0_protocol.DATASET_SPECS[dataset]
    labeled_ids, labeled_targets, corruption_mask, frozen_input_hashes = c0._load_frozen_inputs(dataset)
    raw_views = c0.load_raw_views(dataset)
    views, adapter = b0.preprocess_with_raw_ids(raw_views, corruption_mask)
    if not np.array_equal(np.sort(adapter.sample_ids), np.arange(spec.n)):
        raise RuntimeError("RAW_SAMPLE_ID_SET_MISMATCH")
    if np.unique(adapter.row_positions(labeled_ids)).size != labeled_ids.size:
        raise RuntimeError("SPARSE_LABEL_ROW_MAPPING_MISMATCH")
    k = int(labeled_targets.max()) + 1

    carrier_dir = root / "carrier"
    carrier_dir.mkdir()
    set_seed(seed)
    anchors = amv(c0._container(views, k), m=spec.n_anchor, ret_anchor=True, max_iter=100, seed=seed, info=True, info_iter=False)
    carrier = c0._new_model(views, k, spec, seed, device, anchors)
    carrier_optimizer = optim.Adam(carrier.parameters(), lr=c0_protocol.NATIVE_LR)
    carrier_loader = c0._native_loader(views, adapter, seed, spec.batch_size)
    carrier_orders, carrier_losses = [], []
    for _ in range(c0_protocol.CARRIER_EPOCHS):
        loss, order_hash = c0._native_epoch(carrier, carrier_optimizer, carrier_loader, device)
        carrier_losses.append(loss); carrier_orders.append(order_hash)
    if not np.isfinite(carrier_losses).all():
        raise FloatingPointError("NONFINITE_CARRIER_LOSS")
    state_path = carrier_dir / "carrier_model_state.pt"
    state_file_sha = c0._save_state(carrier, state_path)
    carrier_state_sha = c0._state_hash(carrier)
    payload = b0.build_generic_carrier(b0.extract_q_views(carrier, views, device=device), adapter.sample_ids, labeled_ids, labeled_targets)
    query_rows, anchor_rows, t_rel_np, u_cycle_np = _validate_carrier_payload(payload, adapter, k)
    bundle_path = carrier_dir / "carrier_pre_gt.npz"
    np.savez_compressed(bundle_path, **payload)
    carrier_audit = {
        "dataset": dataset, "training_seed": seed, "carrier_epochs": c0_protocol.CARRIER_EPOCHS,
        "native_optimizer": "Adam", "native_lr": c0_protocol.NATIVE_LR,
        "carrier_state_sha256": carrier_state_sha, "carrier_state_file_sha256": state_file_sha,
        "carrier_native_order_hashes": carrier_orders, "sparse_label_sha256": frozen_input_hashes["p3_a3_split"],
        "corruption_mask_sha256": frozen_input_hashes["p3_a4_mask"], "GT_loaded_before_seal": False,
    }
    io.write_json(carrier_dir / "carrier_audit.json", carrier_audit)

    arms = tuple(config["arms"])
    branches = {}
    for arm in arms:
        directory = root / arm
        directory.mkdir()
        model = c0._new_model(views, k, spec, seed, device)
        model.load_state_dict(c0._load_state(state_path, device), strict=True)
        branches[arm] = {"model": model, "native": optim.Adam(model.parameters(), lr=c0_protocol.NATIVE_LR), "semantic": optim.Adam(model.parameters(), lr=config["semantic_lr"]), "loader": c0._native_loader(views, adapter, seed, spec.batch_size), "initial": c0._state_hash(model), "native_orders": [], "semantic_steps": 0}
    if set(branch["initial"] for branch in branches.values()) != {carrier_state_sha}:
        raise RuntimeError("ALL_ARM_INITIAL_STATE_PARITY_FAILED")
    full_views = [torch.from_numpy(view).to(device=device, dtype=torch.float32) for view in views]
    query_rows_t = torch.as_tensor(query_rows, dtype=torch.long, device=device)
    anchor_rows_t = torch.as_tensor(anchor_rows, dtype=torch.long, device=device)
    query_ids_t = torch.as_tensor(adapter.sample_ids[query_rows], dtype=torch.long, device=device)
    t_rel = torch.as_tensor(t_rel_np.copy(), dtype=torch.float32, device=device).detach()
    u_cycle = torch.as_tensor(u_cycle_np.copy(), dtype=torch.float32, device=device).detach()
    alpha, budget, _ = build_budget("TRUE_BUDGET", u_cycle, query_ids_t)
    detach_contract = {"U_cycle_detached": not u_cycle.requires_grad, "T_rel_detached": not t_rel.requires_grad, "alpha_detached": not alpha.requires_grad, "budget_detached": not budget.requires_grad, "anchor_posterior_detached": True}
    if not all(detach_contract.values()):
        raise RuntimeError("DETACH_CONTRACT_FAILED")
    orders = _semantic_orders(query_rows.size, seed)
    for epoch in range(c0_protocol.REFINEMENT_EPOCHS):
        for arm, branch in branches.items():
            if epoch < config["semantic_exposure"]:
                loss = run_semantic_epoch(branch["model"], branch["semantic"], full_views, query_rows_t, anchor_rows_t, t_rel, u_cycle, orders[epoch], k, arm, query_ids=query_ids_t)
                if not np.isfinite(loss):
                    raise FloatingPointError(f"NONFINITE_SEMANTIC_LOSS:{arm}")
                branch["semantic_steps"] += 1
            loss, order_hash = c0._native_epoch(branch["model"], branch["native"], branch["loader"], device)
            if not np.isfinite(loss):
                raise FloatingPointError(f"NONFINITE_NATIVE_LOSS:{arm}")
            branch["native_orders"].append(order_hash)
    native_orders = {arm: branch["native_orders"] for arm, branch in branches.items()}
    if len({tuple(value) for value in native_orders.values()}) != 1:
        raise RuntimeError("NATIVE_ORDER_PARITY_FAILED")
    if any(branch["semantic_steps"] != config["semantic_exposure"] for branch in branches.values()):
        raise RuntimeError("SEMANTIC_STEP_COUNT_PARITY_FAILED")
    artifacts = ["carrier/carrier_model_state.pt", "carrier/carrier_pre_gt.npz", "carrier/carrier_audit.json"]
    for arm, branch in branches.items():
        c0._save_state(branch["model"], root / arm / "final_model_state.pt")
        np.savez_compressed(root / arm / "pre_gt_predictions.npz", sample_ids=adapter.sample_ids, predictions=c0._predict(branch["model"], views, device))
        artifacts.extend((f"{arm}/final_model_state.pt", f"{arm}/pre_gt_predictions.npz"))
    branches_marker = {"status": "BRANCHES_COMPLETE", "dataset": dataset, "training_seed": seed, "arms": list(arms), "initial_state_sha256": {arm: branch["initial"] for arm, branch in branches.items()}, "carrier_state_sha256": carrier_state_sha, "semantic_schedule_sha256": c0_protocol.sha256_file_from_array(orders), "native_order_hashes": native_orders, "semantic_step_count": {arm: branch["semantic_steps"] for arm, branch in branches.items()}}
    io.write_json(root / "BRANCHES_COMPLETE.json", branches_marker)
    artifacts.append("BRANCHES_COMPLETE.json")
    audit = {"stage": config["stage"], "dataset": dataset, "training_seed": seed, "seed_role": config["seed_role"], "arms": list(arms), "carrier_epochs": c0_protocol.CARRIER_EPOCHS, "refinement_epochs": c0_protocol.REFINEMENT_EPOCHS, "semantic_optimizer": "Adam", "semantic_lr": config["semantic_lr"], "semantic_exposure": config["semantic_exposure"], "native_optimizer": "Adam", "native_lr": c0_protocol.NATIVE_LR, "carrier_state_sha256": carrier_state_sha, "initial_state_sha256": branches_marker["initial_state_sha256"], "semantic_schedule_sha256": branches_marker["semantic_schedule_sha256"], "native_order_hashes": native_orders, "semantic_step_count": branches_marker["semantic_step_count"], "sparse_label_sha256": frozen_input_hashes["p3_a3_split"], "corruption_mask_sha256": frozen_input_hashes["p3_a4_mask"], "T_rel_shape": list(t_rel.shape), "U_cycle_shape": list(u_cycle.shape), "query_count": int(query_rows.size), "anchor_count": int(anchor_rows.size), "action_count": int(u_cycle.shape[1]), "view_count": len(views), "detach_contract": detach_contract, "source_hashes": io.validation_source_hashes(), "GT_loaded_before_seal": False, "full_gt_used_for_training": False, "full_gt_used_for_selection": False, "formal_metrics_computed": False}
    return {"artifacts": artifacts, "audit": audit}


def run_confirmatory(dataset: str, training_seed: int, output_root: str | Path = io.OUTPUT_ROOT, device: str = "cuda:0", backend: Callable[[dict, Path], dict] | None = None) -> dict:
    """Run or verify exactly one preregistered pre-GT dataset/seed cell."""
    io.verify_parent_freezes()
    config = _run_config(dataset, training_seed, output_root, device)
    root = io.run_root(output_root, dataset, training_seed)
    if root.exists():
        if (root / "PRE_GT_COMPLETE.json").is_file():
            verified = io.verify_pre_gt_seal(root)
            if verified["seal"].get("run_config") != config:
                raise RuntimeError("COMPLETE_RUN_CONFIG_MISMATCH")
            return {"status": "ALREADY_COMPLETE_VERIFIED", "root": str(root)}
        raise RuntimeError("OUTPUT_ROOT_EXISTS_OR_PARTIAL_FAIL_CLOSED")
    root.mkdir(parents=True, exist_ok=False)
    io.write_json(root / "RUN_CONFIG.json", config)
    result = (backend or execute_pre_gt_training)(config, root)
    if not isinstance(result, dict) or not isinstance(result.get("artifacts"), list) or not isinstance(result.get("audit"), dict):
        raise RuntimeError("PRE_GT_BACKEND_CONTRACT_INVALID")
    audit = result["audit"]
    false_fields = ("GT_loaded_before_seal", "full_gt_used_for_training", "full_gt_used_for_selection", "formal_metrics_computed")
    if audit.get("dataset") != dataset or audit.get("training_seed") != training_seed or tuple(audit.get("arms", ())) != io.arms_for_seed(training_seed) or any(audit.get(field) is not False for field in false_fields):
        raise RuntimeError("PRE_GT_AUDIT_CONTRACT_INVALID")
    io.write_json(root / "seal/pre_gt_audit.json", audit)
    required = io.required_artifacts(training_seed)
    seal = {"status": "PRE_GT_SEALED", "dataset": dataset, "training_seed": training_seed, "seed_role": config["seed_role"], "run_config": config, "artifact_sha256": io.hash_relative_files(root, required), "source_sha256": io.validation_source_hashes(), "frozen_parent_hashes": io.verify_parent_freezes(), "GT_loaded_before_seal": False, "full_gt_used_for_training": False, "full_gt_used_for_selection": False, "formal_metrics_computed": False}
    seal_path = root / "seal/pre_gt_seal.json"
    io.write_json(seal_path, seal)
    io.write_json(root / "PRE_GT_COMPLETE.json", {"status": "COMPLETE", "dataset": dataset, "training_seed": training_seed, "pre_gt_seal_sha256": io.sha256_file(seal_path), "formal_metrics_computed": False})
    io.verify_pre_gt_seal(root)
    return {"status": "COMPLETE", "root": str(root)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("ALOI", "CCV"))
    parser.add_argument("--training-seed", required=True, type=int, choices=(20, 30, 50, 140, 150, 170))
    parser.add_argument("--output-root", type=Path, default=io.OUTPUT_ROOT)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.execute:
        raise SystemExit("refusing to launch formal pre-GT training without --execute")
    print(run_confirmatory(args.dataset, args.training_seed, args.output_root, args.device))


if __name__ == "__main__":
    main()
