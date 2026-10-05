"""One-seed, three-arm pre-GT runner for P3-C1-B3; this module never reads Y."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import optim

from experiments.utility_relation import (
    bone_p3_c0_cross_dataset_formal_protocol as c0_protocol,
)
from experiments.utility_relation import run_bone_p3_c0_cross_dataset_formal as c0
from experiments.utility_relation.p3_c1_b1_final_formal import (
    run_final_formal as final_runner,
)
from experiments.utility_relation.p3_c1_b1_optuna import msrc_adapter

from . import context, protocol


def _hash_relative_files(root: Path, relative_paths: list[str]) -> dict[str, str]:
    observed = {}
    resolved_root = root.resolve()
    for relative in relative_paths:
        path = (root / relative).resolve()
        if resolved_root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"B3_SEALED_ARTIFACT_MISSING:{relative}")
        observed[relative] = protocol.sha256_file(path)
    return observed


def _source_hashes() -> dict[str, str]:
    return protocol.source_hashes()


def _verify_stage_marker(root: Path, marker_name: str) -> dict:
    path = root / marker_name
    if not path.is_file():
        raise FileNotFoundError(f"B3_STAGE_MARKER_MISSING:{marker_name}")
    marker = json.loads(path.read_text(encoding="utf-8"))
    expected = marker.get("artifact_sha256")
    if not isinstance(expected, dict) or _hash_relative_files(root, list(expected)) != expected:
        raise RuntimeError(f"B3_STAGE_MARKER_HASH_MISMATCH:{marker_name}")
    return marker


def _semantic_orders(query_count: int, training_seed: int) -> np.ndarray:
    """Exact frozen final-formal algorithm, extended only to preregistered seeds."""
    if training_seed not in protocol.CONFIRMATORY_SEEDS:
        raise ValueError("B3_UNREGISTERED_CONFIRMATORY_SEED")
    rng = np.random.default_rng(training_seed)
    return np.stack(
        [
            rng.permutation(query_count).astype(np.int64)
            for _ in range(protocol.REFINEMENT_EPOCHS)
        ]
    )


def _materialize_or_resume_branches(
    resolved: context.ResolvedInputs,
    training_seed: int,
    device: torch.device,
    root: Path,
    state_path: Path,
    arrays: dict[str, np.ndarray],
    carrier_state_sha256: str,
) -> dict:
    marker_name = "BRANCHES_COMPLETE.json"
    if (root / marker_name).is_file():
        return _verify_stage_marker(root, marker_name)
    directories = {arm: root / arm for arm in protocol.ARMS}
    if any(path.exists() and any(path.iterdir()) for path in directories.values()):
        raise RuntimeError("B3_UNSEALED_PARTIAL_BRANCH_FAIL_CLOSED")
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    models = {
        arm: final_runner._load_branch(resolved, training_seed, device, state_path)
        for arm in protocol.ARMS
    }
    initial = {arm: c0._state_hash(model) for arm, model in models.items()}
    if len(set((*initial.values(), carrier_state_sha256))) != 1:
        raise RuntimeError("B3_SHARED_CARRIER_INITIALIZATION_PARITY_FAILURE")

    native_lr = float(msrc_adapter.p2_carrier.LR)
    native_optimizers = {
        arm: optim.Adam(model.parameters(), lr=native_lr)
        for arm, model in models.items()
    }
    semantic_optimizers = {
        arm: optim.Adam(
            models[arm].parameters(),
            lr=float(protocol.ARM_PARAMETERS[arm]["semantic_lr"]),
        )
        for arm in ("OURS_B1", "OURS_B2")
    }
    loaders = {
        arm: c0._native_loader(
            resolved.views, resolved.adapter, training_seed, resolved.spec.batch_size
        )
        for arm in protocol.ARMS
    }
    full_views = [
        torch.from_numpy(view).to(device=device, dtype=torch.float32)
        for view in resolved.views
    ]
    query_rows = torch.as_tensor(
        arrays["query_rows"], dtype=torch.long, device=device
    )
    anchor_rows = torch.as_tensor(
        arrays["anchor_rows"], dtype=torch.long, device=device
    )
    t_rel = torch.as_tensor(
        arrays["T_rel"].copy(), dtype=torch.float32, device=device
    ).detach()
    u_cycle = torch.as_tensor(
        arrays["query_U_cycle"].copy(), dtype=torch.float32, device=device
    ).detach()
    orders = _semantic_orders(query_rows.numel(), training_seed)
    semantic_losses = {"OURS_B1": [], "OURS_B2": []}
    native_order_hashes = {arm: [] for arm in protocol.ARMS}
    native_losses = {arm: [] for arm in protocol.ARMS}

    for epoch in range(protocol.REFINEMENT_EPOCHS):
        if epoch == 0:
            for arm in ("OURS_B1", "OURS_B2"):
                semantic_losses[arm].append(
                    c0._semantic_epoch(
                        models[arm],
                        semantic_optimizers[arm],
                        full_views,
                        query_rows,
                        anchor_rows,
                        t_rel,
                        u_cycle,
                        orders[epoch],
                        resolved.k,
                    )
                )
        for arm in protocol.ARMS:
            loss, order_hash = c0._native_epoch(
                models[arm], native_optimizers[arm], loaders[arm], device
            )
            native_losses[arm].append(loss)
            native_order_hashes[arm].append(order_hash)
    if any(
        native_order_hashes[arm] != native_order_hashes["BASE"]
        for arm in protocol.ARMS
    ):
        raise RuntimeError("B3_THREE_ARM_NATIVE_ORDER_PARITY_FAILURE")
    if any(len(values) != protocol.REFINEMENT_EPOCHS for values in native_losses.values()):
        raise RuntimeError("B3_REFINEMENT_EPOCH_COUNT_FAILURE")
    if any(len(values) != 1 for values in semantic_losses.values()):
        raise RuntimeError("B3_SEMANTIC_EXPOSURE_FAILURE")
    finite = [value for values in native_losses.values() for value in values]
    finite.extend(value for values in semantic_losses.values() for value in values)
    if not np.isfinite(finite).all():
        raise FloatingPointError("B3_NONFINITE_REFINEMENT_LOSS")

    artifact_paths = []
    for arm in protocol.ARMS:
        directory = directories[arm]
        c0._save_state(models[arm], directory / "final_model_state.pt")
        np.savez_compressed(
            directory / "pre_gt_predictions.npz",
            sample_ids=resolved.adapter.sample_ids,
            predictions=c0._predict(models[arm], resolved.views, device),
        )
        artifact_paths.extend(
            [f"{arm}/final_model_state.pt", f"{arm}/pre_gt_predictions.npz"]
        )
    marker = {
        "status": "BRANCHES_COMPLETE",
        "dataset": protocol.DATASET,
        "training_seed": training_seed,
        "arms": list(protocol.ARMS),
        "initial_state_sha256": initial,
        "carrier_state_sha256": carrier_state_sha256,
        "native_optimizer": "torch.optim.Adam",
        "semantic_optimizer": "torch.optim.Adam",
        "native_lr": native_lr,
        "arm_parameters": protocol.ARM_PARAMETERS,
        "base_native_epochs": protocol.REFINEMENT_EPOCHS,
        "b1_native_epochs": protocol.REFINEMENT_EPOCHS,
        "b2_native_epochs": protocol.REFINEMENT_EPOCHS,
        "b1_semantic_steps": 1,
        "b2_semantic_steps": 1,
        "b1_native_only_epochs": 19,
        "b2_native_only_epochs": 19,
        "semantic_then_native_epochs": 1,
        "native_order_hashes": native_order_hashes["BASE"],
        "semantic_schedule_sha256": c0_protocol.sha256_file_from_array(orders),
        "artifact_sha256": _hash_relative_files(root, artifact_paths),
    }
    protocol.write_json(root / marker_name, marker)
    return marker


def execute_pre_gt_training(config: dict, root: Path) -> dict:
    if config["device"] != "cuda:0":
        raise ValueError("B3_FROZEN_DEVICE_MUST_BE_CUDA_0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_B3_CONFIRMATORY_EXECUTION")
    torch.cuda.set_device(0)
    device = torch.device(config["device"])
    resolved = context.resolve_training_inputs(config["dataset"])
    state_path, arrays, carrier_state, a2_audit = final_runner._materialize_or_resume_carrier(
        resolved,
        config["training_seed"],
        device,
        root,
        config["query_chunk_size"],
    )
    branches = _materialize_or_resume_branches(
        resolved,
        config["training_seed"],
        device,
        root,
        state_path,
        arrays,
        carrier_state,
    )
    initial = branches["initial_state_sha256"]
    return {
        "artifacts": [
            "carrier/carrier_model_state.pt",
            "carrier/carrier_pre_gt.npz",
            "carrier/CARRIER_COMPLETE.json",
            *[
                name
                for arm in protocol.ARMS
                for name in (
                    f"{arm}/final_model_state.pt",
                    f"{arm}/pre_gt_predictions.npz",
                )
            ],
            "BRANCHES_COMPLETE.json",
        ],
        "audit": {
            "dataset": protocol.DATASET,
            "training_seed": config["training_seed"],
            "arms": list(protocol.ARMS),
            "arm_parameters": protocol.ARM_PARAMETERS,
            "dataset_adapter": resolved.route,
            "carrier_epochs": protocol.CARRIER_EPOCHS,
            "refinement_epochs": protocol.REFINEMENT_EPOCHS,
            "carrier_state_sha256": carrier_state,
            "initial_BASE_state_sha256": initial["BASE"],
            "initial_B1_state_sha256": initial["OURS_B1"],
            "initial_B2_state_sha256": initial["OURS_B2"],
            "ALL_ARMS_SHARED_CARRIER": len(set((*initial.values(), carrier_state))) == 1,
            "B1_B2_SAME_INITIALIZATION": initial["OURS_B1"] == initial["OURS_B2"],
            "SAME_MSRC_INPUT": True,
            "SAME_WEAK_QUALITY_REALIZATION": True,
            "SAME_CORRUPTION": True,
            "SAME_SPARSE_LABELS": True,
            "SAME_NATIVE_ORDER": True,
            "A2_FEASIBILITY_PRESERVED": a2_audit["A2_FEASIBILITY"] == "PASS",
            "a2_feasibility": a2_audit,
            "r3_numeric_canonicalization": resolved.r3_audit,
            "input_hashes": resolved.input_hashes,
            "METHOD_CHANGED": False,
            "OPTIMIZER_CHANGED": False,
            "U_cycle_changed": False,
            "T_rel_changed": False,
            "Pi_changed": False,
            "WHAT_changed": False,
            "relation_primitive_changed": False,
            "semantic_loss_changed": False,
            "action_budget_changed": False,
            "corruption_protocol_changed": False,
            "sparse_label_protocol_changed": False,
            "GT_loaded_before_seal": False,
            "full_gt_used_for_training": False,
            "full_gt_used_for_selection": False,
            "ACC_NMI_ARI_computed": False,
            "formal_20_30_50_results_read": False,
        },
    }


def _build_run_config(
    training_seed: int,
    device: str,
    output_root: str | Path,
    preregistration_freeze: str | Path,
    query_chunk_size: int,
) -> tuple[dict, Path]:
    if training_seed not in protocol.CONFIRMATORY_SEEDS:
        raise ValueError("B3_UNREGISTERED_CONFIRMATORY_SEED")
    if isinstance(query_chunk_size, bool) or query_chunk_size < 1:
        raise ValueError("B3_INVALID_QUERY_CHUNK_SIZE")
    prereg = protocol.verify_preregistration(preregistration_freeze)
    root = protocol.run_root(output_root, training_seed)
    return {
        "stage": "P3-C1-B3-MSRC-B2-CONFIRMATORY",
        "dataset": protocol.DATASET,
        "training_seed": training_seed,
        "arms": list(protocol.ARMS),
        "arm_parameters": protocol.ARM_PARAMETERS,
        "device": device,
        "output_root": str(Path(output_root).resolve()),
        "run_root": str(root.resolve()),
        "preregistration_freeze": str(Path(preregistration_freeze).resolve()),
        "dataset_adapter": protocol.final_protocol.MSRC_ROUTE,
        "query_chunk_size": query_chunk_size,
        "carrier_epochs": protocol.CARRIER_EPOCHS,
        "refinement_epochs": protocol.REFINEMENT_EPOCHS,
        "preregistration_status_sha256": prereg["preregistration_status_sha256"],
        "preregistration_protocol_sha256": prereg["preregistration_protocol_sha256"],
        "hyperparameter_cli_override_allowed": False,
        "parameter_retuning_allowed": False,
        "full_gt_read_by_training": False,
    }, root


def verify_complete_run(root: str | Path) -> dict:
    root = Path(root)
    complete_path, seal_path = root / "COMPLETE.json", root / "PRE_GT_SEAL.json"
    if not complete_path.is_file() or not seal_path.is_file():
        raise FileNotFoundError("B3_COMPLETE_OR_PRE_GT_SEAL_MISSING")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if (
        complete.get("status") != "COMPLETE"
        or complete.get("pre_gt_seal_sha256") != protocol.sha256_file(seal_path)
    ):
        raise RuntimeError("B3_COMPLETE_MARKER_INVALID")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    expected = seal.get("artifact_sha256")
    if not isinstance(expected, dict) or _hash_relative_files(root, list(expected)) != expected:
        raise RuntimeError("B3_PRE_GT_ARTIFACT_SEAL_INVALID")
    if seal.get("runner_source_sha256") != _source_hashes():
        raise RuntimeError("B3_RUNNER_SOURCE_SEAL_INVALID")
    config = json.loads((root / "RUN_CONFIG.json").read_text(encoding="utf-8"))
    prereg = protocol.verify_preregistration(config["preregistration_freeze"])
    if seal.get("preregistration_status_sha256") != prereg["preregistration_status_sha256"]:
        raise RuntimeError("B3_PREREGISTRATION_SEAL_INVALID")
    audit = json.loads((root / "PRE_GT_AUDIT.json").read_text(encoding="utf-8"))
    false_fields = (
        "GT_loaded_before_seal",
        "full_gt_used_for_training",
        "full_gt_used_for_selection",
        "ACC_NMI_ARI_computed",
        "formal_20_30_50_results_read",
    )
    if any(audit.get(name) is not False for name in false_fields):
        raise RuntimeError("B3_GT_ISOLATION_INVALID")
    hashes = (
        audit.get("carrier_state_sha256"),
        audit.get("initial_BASE_state_sha256"),
        audit.get("initial_B1_state_sha256"),
        audit.get("initial_B2_state_sha256"),
    )
    if len(set(hashes)) != 1:
        raise RuntimeError("B3_SHARED_INITIAL_STATE_INVALID")
    return {"config": config, "audit": audit, "seal": seal, "complete": complete}


def run_confirmatory(
    training_seed: int,
    device: str = "cuda:0",
    output_root: str | Path = protocol.OUTPUT_ROOT,
    preregistration_freeze: str | Path = protocol.PREREGISTERED_FREEZE,
    query_chunk_size: int = protocol.QUERY_CHUNK_SIZE,
    backend: Callable[[dict, Path], dict] | None = None,
) -> dict:
    protocol.verify_parent_integrity()
    config, root = _build_run_config(
        training_seed, device, output_root, preregistration_freeze, query_chunk_size
    )
    if (root / "COMPLETE.json").is_file():
        verified = verify_complete_run(root)
        if verified["config"] != config:
            raise RuntimeError("B3_COMPLETE_RUN_CONFIG_MISMATCH")
        return {"status": "ALREADY_COMPLETE_VERIFIED", "root": str(root)}
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "RUN_CONFIG.json"
    if config_path.is_file():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise RuntimeError("B3_PARTIAL_RUN_CONFIG_MISMATCH")
    else:
        protocol.write_json(config_path, config)
    result = (backend or execute_pre_gt_training)(config, root)
    artifacts, audit = result.get("artifacts"), result.get("audit")
    required_false = (
        "GT_loaded_before_seal",
        "full_gt_used_for_training",
        "full_gt_used_for_selection",
        "ACC_NMI_ARI_computed",
        "formal_20_30_50_results_read",
    )
    if (
        not isinstance(artifacts, list)
        or not artifacts
        or not isinstance(audit, dict)
        or audit.get("dataset") != protocol.DATASET
        or audit.get("training_seed") != training_seed
        or audit.get("arms") != list(protocol.ARMS)
        or audit.get("arm_parameters") != protocol.ARM_PARAMETERS
        or any(audit.get(name) is not False for name in required_false)
    ):
        raise RuntimeError("B3_PRE_GT_BACKEND_CONTRACT_INVALID")
    initial = (
        audit.get("carrier_state_sha256"),
        audit.get("initial_BASE_state_sha256"),
        audit.get("initial_B1_state_sha256"),
        audit.get("initial_B2_state_sha256"),
    )
    if len(set(initial)) != 1:
        raise RuntimeError("B3_PRE_GT_INITIALIZATION_PARITY_FAILURE")
    protocol.write_json(root / "PRE_GT_AUDIT.json", audit)
    sealed_artifacts = ["RUN_CONFIG.json", "PRE_GT_AUDIT.json", *artifacts]
    seal = {
        "status": "PRE_GT_SEALED",
        "dataset": protocol.DATASET,
        "training_seed": training_seed,
        "arms": list(protocol.ARMS),
        "artifact_sha256": _hash_relative_files(root, sealed_artifacts),
        "runner_source_sha256": _source_hashes(),
        "preregistration_status_sha256": config["preregistration_status_sha256"],
        "parent_integrity": protocol.verify_parent_integrity(),
        "GT_loaded_before_seal": False,
    }
    protocol.write_json(root / "PRE_GT_SEAL.json", seal)
    complete = {
        "status": "COMPLETE",
        "dataset": protocol.DATASET,
        "training_seed": training_seed,
        "run_count": len(protocol.ARMS),
        "pre_gt_seal_sha256": protocol.sha256_file(root / "PRE_GT_SEAL.json"),
        "metrics_computed": False,
        "parameter_retuning_allowed": False,
    }
    protocol.write_json(root / "COMPLETE.json", complete)
    verify_complete_run(root)
    return {"status": "COMPLETE", "root": str(root)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-seed", type=int, choices=protocol.CONFIRMATORY_SEEDS
    )
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    parser.add_argument("--output-root", type=Path, default=protocol.OUTPUT_ROOT)
    parser.add_argument("--preregister", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.preregister:
        if args.execute or args.training_seed is not None:
            raise SystemExit("preregistration must be a separate pre-run action")
        print(protocol.preregister())
        return
    if not args.execute or args.training_seed is None:
        raise SystemExit("refusing to launch B3 GPU training without --execute and seed")
    print(
        json.dumps(
            run_confirmatory(args.training_seed, output_root=args.output_root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
