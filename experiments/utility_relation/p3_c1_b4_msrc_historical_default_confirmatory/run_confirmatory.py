"""Pre-GT-only B4 runner: paired BASE vs P2-A1B historical default."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import optim

from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as c0_protocol
from experiments.utility_relation import run_bone_p3_c0_cross_dataset_formal as c0
from experiments.utility_relation.p3_c1_b1_final_formal import run_final_formal as final_runner
from experiments.utility_relation.p3_c1_b1_optuna import msrc_adapter

from . import audit, context, protocol


def _hash_relative(root: Path, relatives: list[str]) -> dict[str, str]:
    result = {}
    for relative in relatives:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"B4_SEALED_ARTIFACT_MISSING:{relative}")
        result[relative] = protocol.sha256_file(path)
    return result


def _semantic_orders(query_count: int, seed: int) -> np.ndarray:
    if seed not in protocol.CONFIRMATORY_SEEDS:
        raise ValueError("B4_UNREGISTERED_CONFIRMATORY_SEED")
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(query_count).astype(np.int64) for _ in range(20)])


def _branches(resolved, seed: int, device: torch.device, root: Path, state_path: Path, arrays: dict, carrier_sha: str) -> dict:
    marker_path = root / "BRANCHES_COMPLETE.json"
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if _hash_relative(root, list(marker["artifact_sha256"])) != marker["artifact_sha256"]:
            raise RuntimeError("B4_BRANCH_SEAL_INVALID")
        return marker
    dirs = {arm: root / arm for arm in protocol.ARMS}
    if any(path.exists() and any(path.iterdir()) for path in dirs.values()):
        raise RuntimeError("B4_UNSEALED_PARTIAL_BRANCH_FAIL_CLOSED")
    for path in dirs.values(): path.mkdir(parents=True, exist_ok=True)
    models = {arm: final_runner._load_branch(resolved, seed, device, state_path) for arm in protocol.ARMS}
    initial = {arm: c0._state_hash(model) for arm, model in models.items()}
    if len(set((*initial.values(), carrier_sha))) != 1:
        raise RuntimeError("B4_SHARED_INITIAL_STATE_FAILURE")
    native_lr = float(msrc_adapter.p2_carrier.LR)
    if native_lr != protocol.NATIVE_LR:
        raise RuntimeError("B4_HISTORICAL_NATIVE_LR_MISMATCH")
    base_optimizer = optim.Adam(models["BASE"].parameters(), lr=native_lr)
    historical_native_optimizer = optim.Adam(models["OURS_HISTORICAL_DEFAULT"].parameters(), lr=native_lr)
    semantic_optimizer = optim.Adam(models["OURS_HISTORICAL_DEFAULT"].parameters(), lr=protocol.HISTORICAL_DEFAULT_SEMANTIC_LR)
    base_loader = c0._native_loader(resolved.views, resolved.adapter, seed, resolved.spec.batch_size)
    historical_loader = c0._native_loader(resolved.views, resolved.adapter, seed, resolved.spec.batch_size)
    full_views = [torch.from_numpy(view).to(device=device, dtype=torch.float32) for view in resolved.views]
    query_rows = torch.as_tensor(arrays["query_rows"], dtype=torch.long, device=device)
    anchor_rows = torch.as_tensor(arrays["anchor_rows"], dtype=torch.long, device=device)
    t_rel = torch.as_tensor(arrays["T_rel"].copy(), dtype=torch.float32, device=device).detach()
    u_cycle = torch.as_tensor(arrays["query_U_cycle"].copy(), dtype=torch.float32, device=device).detach()
    orders = _semantic_orders(query_rows.numel(), seed)
    base_orders, historical_orders, semantic_losses, semantic_steps = [], [], [], []
    for epoch in range(protocol.REFINEMENT_EPOCHS):
        semantic_losses.append(c0._semantic_epoch(models["OURS_HISTORICAL_DEFAULT"], semantic_optimizer, full_views, query_rows, anchor_rows, t_rel, u_cycle, orders[epoch], resolved.k))
        semantic_steps.append(1)  # _semantic_epoch is the frozen one-step implementation.
        historical_loss, historical_order = c0._native_epoch(models["OURS_HISTORICAL_DEFAULT"], historical_native_optimizer, historical_loader, device)
        base_loss, base_order = c0._native_epoch(models["BASE"], base_optimizer, base_loader, device)
        if not np.isfinite((historical_loss, base_loss, semantic_losses[-1])).all():
            raise FloatingPointError("B4_NONFINITE_REFINEMENT_LOSS")
        base_orders.append(base_order); historical_orders.append(historical_order)
    if base_orders != historical_orders or semantic_steps != [1] * 20:
        raise RuntimeError("B4_PAIRED_SCHEDULE_PARITY_FAILURE")
    artifacts = []
    for arm in protocol.ARMS:
        c0._save_state(models[arm], dirs[arm] / "final_model_state.pt")
        np.savez_compressed(dirs[arm] / "pre_gt_predictions.npz", sample_ids=resolved.adapter.sample_ids, predictions=c0._predict(models[arm], resolved.views, device))
        artifacts.extend([f"{arm}/final_model_state.pt", f"{arm}/pre_gt_predictions.npz"])
    marker = {
        "status": "BRANCHES_COMPLETE", "dataset": protocol.DATASET, "training_seed": seed,
        "initial_state_sha256": initial, "carrier_state_sha256": carrier_sha,
        "native_optimizer": "Adam", "semantic_optimizer": protocol.SEMANTIC_OPTIMIZER,
        "native_lr": native_lr, "semantic_lr": protocol.HISTORICAL_DEFAULT_SEMANTIC_LR,
        "base_semantic_steps": 0, "historical_default_semantic_steps": 20,
        "semantic_optimizer_steps_per_epoch": semantic_steps,
        "base_native_epochs": 20, "historical_default_native_epochs": 20,
        "native_order_hashes": base_orders,
        "semantic_schedule_sha256": c0_protocol.sha256_file_from_array(orders),
        "artifact_sha256": _hash_relative(root, artifacts),
    }
    protocol.write_json(marker_path, marker)
    return marker


def execute_pre_gt_training(config: dict, root: Path) -> dict:
    if config["device"] != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("B4_CUDA_0_REQUIRED")
    torch.cuda.set_device(0)
    resolved = context.resolve_training_inputs()
    state_path, arrays, carrier_sha, feasibility = final_runner._materialize_or_resume_carrier(resolved, config["training_seed"], torch.device("cuda:0"), root, config["query_chunk_size"])
    branches = _branches(resolved, config["training_seed"], torch.device("cuda:0"), root, state_path, arrays, carrier_sha)
    initial = branches["initial_state_sha256"]
    return {"artifacts": ["carrier/carrier_model_state.pt", "carrier/carrier_pre_gt.npz", "carrier/CARRIER_COMPLETE.json", "BASE/final_model_state.pt", "BASE/pre_gt_predictions.npz", "OURS_HISTORICAL_DEFAULT/final_model_state.pt", "OURS_HISTORICAL_DEFAULT/pre_gt_predictions.npz", "BRANCHES_COMPLETE.json"], "audit": {
        "dataset": protocol.DATASET, "training_seed": config["training_seed"], "arms": list(protocol.ARMS),
        "carrier_state_sha256": carrier_sha, "base_initial_state_sha256": initial["BASE"],
        "historical_default_initial_state_sha256": initial["OURS_HISTORICAL_DEFAULT"],
        "corruption_artifact_sha256": config["corruption_artifact_sha256"], "sparse_split_sha256": config["sparse_split_sha256"],
        "carrier_pre_gt_sha256": protocol.sha256_file(root / "carrier/carrier_pre_gt.npz"),
        "semantic_lr": protocol.HISTORICAL_DEFAULT_SEMANTIC_LR, "semantic_exposure": 20,
        "semantic_optimizer": protocol.SEMANTIC_OPTIMIZER, "semantic_optimizer_steps_per_epoch": branches["semantic_optimizer_steps_per_epoch"],
        "base_semantic_steps": 0, "historical_default_semantic_steps": 20,
        "A2_FEASIBILITY_PRESERVED": feasibility["A2_FEASIBILITY"] == "PASS",
        "METHOD_CHANGED": False, "OPTIMIZER_CHANGED": False, "U_cycle_changed": False,
        "T_rel_changed": False, "Pi_changed": False, "relation_primitive_changed": False,
        "semantic_loss_changed": False, "action_budget_changed": False,
        "corruption_protocol_changed": False, "sparse_label_protocol_changed": False,
        "GT_loaded_before_seal": False, "full_gt_used_for_training": False,
        "full_gt_used_for_selection": False, "ACC_NMI_ARI_computed": False,
        "b1_b2_result_used_for_parameter_selection": False, "optimizer_search": False, "parameter_search": False,
    }}


def _config(seed: int, output_root: str | Path, query_chunk_size: int, preregistration_freeze: str | Path) -> tuple[dict, Path]:
    if seed not in protocol.CONFIRMATORY_SEEDS: raise ValueError("B4_UNREGISTERED_CONFIRMATORY_SEED")
    prereg = audit.verify_preregistration(preregistration_freeze)
    parent = audit.verify_parent_integrity()
    root = protocol.run_root(output_root, seed)
    return {"stage": "P3-C1-B4-MSRC-HISTORICAL-DEFAULT-CONFIRMATORY", "dataset": protocol.DATASET, "training_seed": seed, "arms": list(protocol.ARMS), "device": "cuda:0", "output_root": str(Path(output_root).resolve()), "run_root": str(root.resolve()), "preregistration_freeze": str(Path(preregistration_freeze).resolve()), "query_chunk_size": query_chunk_size, "carrier_epochs": 100, "refinement_epochs": 20, "semantic_lr": protocol.HISTORICAL_DEFAULT_SEMANTIC_LR, "semantic_exposure": 20, "semantic_optimizer": "Adam", "preregistration_status_sha256": prereg["status_sha256"], "corruption_artifact_sha256": parent["corruption_artifact_sha256"], "sparse_split_sha256": parent["sparse_split_sha256"], "parameter_cli_override_allowed": False, "parameter_retuning_allowed": False}, root


def verify_complete_run(root: str | Path) -> dict:
    root = Path(root); seal_path = root / "PRE_GT_SEAL.json"; complete_path = root / "COMPLETE.json"
    if not seal_path.is_file() or not complete_path.is_file(): raise FileNotFoundError("B4_COMPLETE_OR_SEAL_MISSING")
    seal = json.loads(seal_path.read_text(encoding="utf-8")); complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("pre_gt_seal_sha256") != protocol.sha256_file(seal_path): raise RuntimeError("B4_COMPLETE_INVALID")
    if _hash_relative(root, list(seal.get("artifact_sha256", {}))) != seal.get("artifact_sha256"): raise RuntimeError("B4_PRE_GT_SEAL_INVALID")
    if seal.get("source_sha256") != protocol.source_hashes(): raise RuntimeError("B4_SOURCE_SEAL_INVALID")
    config = json.loads((root / "RUN_CONFIG.json").read_text(encoding="utf-8")); audit.verify_preregistration(config["preregistration_freeze"]); audit_row = json.loads((root / "PRE_GT_AUDIT.json").read_text(encoding="utf-8"))
    if audit_row["base_initial_state_sha256"] != audit_row["historical_default_initial_state_sha256"] or audit_row["base_initial_state_sha256"] != audit_row["carrier_state_sha256"]: raise RuntimeError("B4_INITIAL_PARITY_INVALID")
    if audit_row["semantic_optimizer_steps_per_epoch"] != [1] * 20 or audit_row["base_semantic_steps"] != 0: raise RuntimeError("B4_SEMANTIC_STEP_AUDIT_INVALID")
    if audit_row.get("corruption_artifact_sha256") != config["corruption_artifact_sha256"] or audit_row.get("sparse_split_sha256") != config["sparse_split_sha256"] or not isinstance(audit_row.get("carrier_pre_gt_sha256"), str): raise RuntimeError("B4_INPUT_PROVENANCE_AUDIT_INVALID")
    if any(audit_row.get(k) is not False for k in ("GT_loaded_before_seal", "full_gt_used_for_training", "full_gt_used_for_selection", "ACC_NMI_ARI_computed", "b1_b2_result_used_for_parameter_selection")): raise RuntimeError("B4_GT_OR_SELECTION_FIREWALL_INVALID")
    return {"config": config, "audit": audit_row, "seal": seal}


def run_confirmatory(seed: int, output_root: str | Path = protocol.OUTPUT_ROOT, query_chunk_size: int = protocol.QUERY_CHUNK_SIZE, preregistration_freeze: str | Path = protocol.PREREGISTERED_FREEZE, backend=None) -> dict:
    config, root = _config(seed, output_root, query_chunk_size, preregistration_freeze)
    if (root / "COMPLETE.json").is_file():
        verify_complete_run(root); return {"status": "ALREADY_COMPLETE_VERIFIED", "root": str(root)}
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "RUN_CONFIG.json"
    if config_path.is_file() and json.loads(config_path.read_text()) != config: raise RuntimeError("B4_PARTIAL_CONFIG_MISMATCH")
    if not config_path.is_file(): protocol.write_json(config_path, config)
    result = (backend or execute_pre_gt_training)(config, root)
    if not isinstance(result.get("artifacts"), list) or not isinstance(result.get("audit"), dict): raise RuntimeError("B4_BACKEND_CONTRACT_INVALID")
    audit_row = result["audit"]
    required_false = ("GT_loaded_before_seal", "full_gt_used_for_training", "full_gt_used_for_selection", "ACC_NMI_ARI_computed", "b1_b2_result_used_for_parameter_selection", "optimizer_search", "parameter_search")
    if audit_row.get("semantic_optimizer") != "Adam" or audit_row.get("semantic_optimizer_steps_per_epoch") != [1] * 20 or audit_row.get("base_semantic_steps") != 0 or audit_row.get("historical_default_semantic_steps") != 20 or audit_row.get("corruption_artifact_sha256") != config["corruption_artifact_sha256"] or audit_row.get("sparse_split_sha256") != config["sparse_split_sha256"] or not isinstance(audit_row.get("carrier_pre_gt_sha256"), str) or any(audit_row.get(name) is not False for name in required_false): raise RuntimeError("B4_HISTORICAL_CONTRACT_INVALID")
    protocol.write_json(root / "PRE_GT_AUDIT.json", audit_row)
    seal = {"status": "PRE_GT_SEALED", "dataset": protocol.DATASET, "training_seed": seed, "artifact_sha256": _hash_relative(root, ["RUN_CONFIG.json", "PRE_GT_AUDIT.json", *result["artifacts"]]), "source_sha256": protocol.source_hashes(), "GT_loaded_before_seal": False}
    protocol.write_json(root / "PRE_GT_SEAL.json", seal)
    protocol.write_json(root / "COMPLETE.json", {"status": "COMPLETE", "pre_gt_seal_sha256": protocol.sha256_file(root / "PRE_GT_SEAL.json"), "metrics_computed": False})
    verify_complete_run(root)
    return {"status": "COMPLETE", "root": str(root)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-seed", choices=protocol.CONFIRMATORY_SEEDS, type=int)
    parser.add_argument("--output-root", type=Path, default=protocol.OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.execute or args.training_seed is None: raise SystemExit("refusing to train without --execute and exact seed")
    print(json.dumps(run_confirmatory(args.training_seed, args.output_root), indent=2))


if __name__ == "__main__": main()
