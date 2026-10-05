"""Single-dataset/single-seed final formal runner; this module never reads Y."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import optim

from experiments.utility_relation import bone_p3_b0_generic_carrier as b0
from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as c0_protocol
from experiments.utility_relation import run_bone_p3_c0_cross_dataset_formal as c0
from experiments.utility_relation.p3_c1_b1_optuna import msrc_adapter
from method.amv import amv
from utils.seed_util import set_seed

from . import final_formal_context as context
from . import final_formal_protocol as protocol


def _source_paths() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parent
    return tuple(root / name for name in (
        "__init__.py",
        "final_formal_protocol.py",
        "final_formal_context.py",
        "run_final_formal.py",
        "evaluate_final_formal.py",
        "summarize_final_formal.py",
    ))


def _hash_relative_files(root: Path, relative_paths: list[str]) -> dict[str, str]:
    result = {}
    resolved_root = root.resolve()
    for relative in relative_paths:
        path = (root / relative).resolve()
        if resolved_root not in path.parents:
            raise ValueError("SEALED_ARTIFACT_PATH_ESCAPE")
        if not path.is_file():
            raise FileNotFoundError(f"SEALED_ARTIFACT_MISSING:{relative}")
        result[relative] = protocol.sha256_file(path)
    return result


def _verify_stage_marker(root: Path, marker_name: str) -> dict:
    marker_path = root / marker_name
    if not marker_path.is_file():
        raise FileNotFoundError(f"STAGE_MARKER_MISSING:{marker_name}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = marker.get("artifact_sha256")
    if not isinstance(expected, dict) or _hash_relative_files(root, list(expected)) != expected:
        raise RuntimeError(f"STAGE_MARKER_HASH_MISMATCH:{marker_name}")
    return marker


def _carrier_container(resolved: context.ResolvedInputs):
    if resolved.dataset == "MSRC":
        return msrc_adapter.p2_carrier._native_data(resolved.views)
    return c0._container(resolved.views, resolved.k)


def _materialize_or_resume_carrier(
    resolved: context.ResolvedInputs,
    training_seed: int,
    device: torch.device,
    root: Path,
    query_chunk_size: int,
) -> tuple[Path, dict[str, np.ndarray], str, dict]:
    carrier_dir = root / "carrier"
    marker_relative = "carrier/CARRIER_COMPLETE.json"
    state_path = carrier_dir / "carrier_model_state.pt"
    arrays_path = carrier_dir / "carrier_pre_gt.npz"
    if (root / marker_relative).is_file():
        marker = _verify_stage_marker(root, marker_relative)
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        return state_path, arrays, marker["carrier_state_sha256"], marker["a2_feasibility"]
    if carrier_dir.exists() and any(carrier_dir.iterdir()):
        raise RuntimeError("UNSEALED_PARTIAL_CARRIER_FAIL_CLOSED")
    carrier_dir.mkdir(parents=True, exist_ok=True)

    set_seed(training_seed)
    anchors = amv(
        _carrier_container(resolved),
        m=resolved.spec.n_anchor,
        ret_anchor=True,
        max_iter=100,
        seed=training_seed,
        info=True,
        info_iter=False,
    )
    model = c0._new_model(
        resolved.views,
        resolved.k,
        resolved.spec,
        training_seed,
        device,
        anchors,
    )
    native_lr = (
        float(msrc_adapter.p2_carrier.LR)
        if resolved.dataset == "MSRC"
        else c0_protocol.NATIVE_LR
    )
    optimizer = optim.Adam(model.parameters(), lr=native_lr)
    loader = c0._native_loader(
        resolved.views, resolved.adapter, training_seed, resolved.spec.batch_size
    )
    losses, order_hashes = [], []
    for epoch in range(protocol.CARRIER_EPOCHS):
        loss, order_hash = c0._native_epoch(model, optimizer, loader, device)
        losses.append(loss)
        order_hashes.append(order_hash)
        if (epoch + 1) % 10 == 0:
            print(
                f"{resolved.dataset} carrier epoch={epoch + 1}/{protocol.CARRIER_EPOCHS} loss={loss:.8f}",
                flush=True,
            )
    if len(losses) != protocol.CARRIER_EPOCHS or not np.isfinite(losses).all():
        raise RuntimeError("FINAL_FORMAL_CARRIER_EPOCH_FAILURE")
    state_file_sha = c0._save_state(model, state_path)
    state_sha = c0._state_hash(model)
    q_views = b0.extract_q_views(model, resolved.views, device=device)
    arrays = b0.build_generic_carrier(
        q_views,
        resolved.adapter.sample_ids,
        resolved.labeled_ids,
        resolved.labeled_targets,
        query_chunk_size=query_chunk_size,
    )
    a2_audit = context.validate_a2_for_carrier(resolved, arrays)
    np.savez_compressed(arrays_path, **arrays)
    marker = {
        "status": "CARRIER_COMPLETE",
        "dataset": resolved.dataset,
        "training_seed": training_seed,
        "carrier_epochs": protocol.CARRIER_EPOCHS,
        "native_lr": native_lr,
        "carrier_state_sha256": state_sha,
        "carrier_state_file_sha256": state_file_sha,
        "native_order_hashes": order_hashes,
        "a2_feasibility": a2_audit,
        "GT_loaded": False,
        "artifact_sha256": _hash_relative_files(
            root,
            ["carrier/carrier_model_state.pt", "carrier/carrier_pre_gt.npz"],
        ),
    }
    protocol.write_json(root / marker_relative, marker)
    return state_path, arrays, state_sha, a2_audit


def _load_branch(resolved: context.ResolvedInputs, seed: int, device: torch.device, state_path: Path):
    model = c0._new_model(resolved.views, resolved.k, resolved.spec, seed, device)
    model.load_state_dict(c0._load_state(state_path, device), strict=True)
    return model


def _materialize_or_resume_branches(
    resolved: context.ResolvedInputs,
    training_seed: int,
    semantic_lr: float,
    semantic_exposure: int,
    device: torch.device,
    root: Path,
    state_path: Path,
    arrays: dict[str, np.ndarray],
    carrier_state_sha: str,
) -> dict:
    marker_relative = "BRANCHES_COMPLETE.json"
    if (root / marker_relative).is_file():
        return _verify_stage_marker(root, marker_relative)
    base_dir, ours_dir = root / "BASE", root / "OURS"
    if any(path.exists() and any(path.iterdir()) for path in (base_dir, ours_dir)):
        raise RuntimeError("UNSEALED_PARTIAL_BRANCH_FAIL_CLOSED")
    base_dir.mkdir(parents=True, exist_ok=True)
    ours_dir.mkdir(parents=True, exist_ok=True)

    base_model = _load_branch(resolved, training_seed, device, state_path)
    ours_model = _load_branch(resolved, training_seed, device, state_path)
    initial = {"BASE": c0._state_hash(base_model), "OURS": c0._state_hash(ours_model)}
    if len(set((*initial.values(), carrier_state_sha))) != 1:
        raise RuntimeError("BASE_OURS_SHARED_CARRIER_PARITY_FAILURE")
    native_lr = (
        float(msrc_adapter.p2_carrier.LR)
        if resolved.dataset == "MSRC"
        else c0_protocol.NATIVE_LR
    )
    base_optimizer = optim.Adam(base_model.parameters(), lr=native_lr)
    ours_native_optimizer = optim.Adam(ours_model.parameters(), lr=native_lr)
    semantic_optimizer = optim.Adam(ours_model.parameters(), lr=semantic_lr)
    base_loader = c0._native_loader(
        resolved.views, resolved.adapter, training_seed, resolved.spec.batch_size
    )
    ours_loader = c0._native_loader(
        resolved.views, resolved.adapter, training_seed, resolved.spec.batch_size
    )
    full_views = [
        torch.from_numpy(view).to(device=device, dtype=torch.float32)
        for view in resolved.views
    ]
    query_rows = torch.as_tensor(arrays["query_rows"], dtype=torch.long, device=device)
    anchor_rows = torch.as_tensor(arrays["anchor_rows"], dtype=torch.long, device=device)
    t_rel = torch.as_tensor(arrays["T_rel"].copy(), dtype=torch.float32, device=device).detach()
    u_cycle = torch.as_tensor(
        arrays["query_U_cycle"].copy(), dtype=torch.float32, device=device
    ).detach()
    orders = c0_protocol.semantic_orders(query_rows.numel(), training_seed)
    base_orders, ours_orders, semantic_losses = [], [], []
    for epoch in range(protocol.REFINEMENT_EPOCHS):
        if epoch < semantic_exposure:
            semantic_losses.append(
                c0._semantic_epoch(
                    ours_model,
                    semantic_optimizer,
                    full_views,
                    query_rows,
                    anchor_rows,
                    t_rel,
                    u_cycle,
                    orders[epoch],
                    resolved.k,
                )
            )
        ours_loss, ours_order = c0._native_epoch(
            ours_model, ours_native_optimizer, ours_loader, device
        )
        base_loss, base_order = c0._native_epoch(
            base_model, base_optimizer, base_loader, device
        )
        if not np.isfinite((ours_loss, base_loss)).all():
            raise FloatingPointError("NONFINITE_FINAL_REFINEMENT_LOSS")
        ours_orders.append(ours_order)
        base_orders.append(base_order)
    if base_orders != ours_orders or len(semantic_losses) != semantic_exposure:
        raise RuntimeError("FINAL_BASE_OURS_SCHEDULE_PARITY_FAILURE")
    if not np.isfinite(semantic_losses).all():
        raise FloatingPointError("NONFINITE_FINAL_SEMANTIC_LOSS")

    artifact_paths = []
    for arm, model, directory in (
        ("BASE", base_model, base_dir),
        ("OURS", ours_model, ours_dir),
    ):
        c0._save_state(model, directory / "final_model_state.pt")
        np.savez_compressed(
            directory / "pre_gt_predictions.npz",
            sample_ids=resolved.adapter.sample_ids,
            predictions=c0._predict(model, resolved.views, device),
        )
        artifact_paths.extend(
            [f"{arm}/final_model_state.pt", f"{arm}/pre_gt_predictions.npz"]
        )
    marker = {
        "status": "BRANCHES_COMPLETE",
        "dataset": resolved.dataset,
        "training_seed": training_seed,
        "initial_state_sha256": initial,
        "base_native_epochs": protocol.REFINEMENT_EPOCHS,
        "ours_native_epochs": protocol.REFINEMENT_EPOCHS,
        "ours_semantic_steps": semantic_exposure,
        "ours_native_only_epochs": protocol.REFINEMENT_EPOCHS - semantic_exposure,
        "semantic_then_native_epochs": semantic_exposure,
        "semantic_lr": semantic_lr,
        "native_lr": native_lr,
        "native_order_hashes": base_orders,
        "semantic_schedule_sha256": c0_protocol.sha256_file_from_array(orders),
        "artifact_sha256": _hash_relative_files(root, artifact_paths),
    }
    protocol.write_json(root / marker_relative, marker)
    return marker


def execute_pre_gt_training(config: dict, root: Path) -> dict:
    """Real opt-in backend. Tests replace this function; preflight never calls it."""
    if config["device"] != "cuda:0":
        raise ValueError("FROZEN_FINAL_FORMAL_DEVICE_MUST_BE_CUDA_0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_FINAL_FORMAL_EXECUTION")
    torch.cuda.set_device(0)
    device = torch.device(config["device"])
    resolved = context.resolve_training_inputs(config["dataset"])
    if resolved.route != config["dataset_adapter"]:
        raise RuntimeError("DATASET_ROUTE_DRIFT")
    state_path, arrays, carrier_state, a2_audit = _materialize_or_resume_carrier(
        resolved,
        config["training_seed"],
        device,
        root,
        config["query_chunk_size"],
    )
    branches = _materialize_or_resume_branches(
        resolved,
        config["training_seed"],
        config["semantic_lr"],
        config["semantic_exposure"],
        device,
        root,
        state_path,
        arrays,
        carrier_state,
    )
    artifacts = [
        "carrier/carrier_model_state.pt",
        "carrier/carrier_pre_gt.npz",
        "carrier/CARRIER_COMPLETE.json",
        "BASE/final_model_state.pt",
        "BASE/pre_gt_predictions.npz",
        "OURS/final_model_state.pt",
        "OURS/pre_gt_predictions.npz",
        "BRANCHES_COMPLETE.json",
    ]
    return {
        "artifacts": artifacts,
        "audit": {
            "dataset": resolved.dataset,
            "training_seed": config["training_seed"],
            "dataset_adapter": resolved.route,
            "semantic_lr": config["semantic_lr"],
            "semantic_exposure": config["semantic_exposure"],
            "final_parameter_freeze_sha256": config["final_parameter_freeze_sha256"],
            "carrier_epochs": protocol.CARRIER_EPOCHS,
            "refinement_epochs": protocol.REFINEMENT_EPOCHS,
            "BASE_definition": "shared carrier plus 20 native-only refinement epochs",
            "OURS_definition": "shared carrier; first E epochs semantic then native, remaining 20-E native only",
            "BASE_OURS_same_dataset_seed_corruption": True,
            "carrier_state_sha256": carrier_state,
            "initial_BASE_state_sha256": branches["initial_state_sha256"]["BASE"],
            "initial_OURS_state_sha256": branches["initial_state_sha256"]["OURS"],
            "A2_FEASIBILITY_PRESERVED": a2_audit["A2_FEASIBILITY"] == "PASS",
            "a2_feasibility": a2_audit,
            "r3_numeric_canonicalization": resolved.r3_audit,
            "input_hashes": resolved.input_hashes,
            "U_cycle_changed": False,
            "T_rel_changed": False,
            "WHAT_changed": False,
            "GT_loaded_before_seal": False,
            "full_gt_used_for_training": False,
            "full_gt_used_for_model_selection": False,
            "full_gt_used_for_hyperparameter_selection": False,
            "ACC_NMI_ARI_computed": False,
        },
    }


def _build_run_config(
    dataset: str,
    training_seed: int,
    device: str,
    output_root: str | Path,
    freeze_dir: str | Path,
    query_chunk_size: int,
) -> tuple[dict, Path]:
    if training_seed not in protocol.FORMAL_SEEDS:
        raise ValueError("UNREGISTERED_FINAL_FORMAL_SEED")
    if isinstance(query_chunk_size, bool) or query_chunk_size < 1:
        raise ValueError("INVALID_QUERY_CHUNK_SIZE")
    binding = protocol.resolve_binding(dataset, freeze_dir)
    route = protocol.MSRC_ROUTE if dataset == "MSRC" else protocol.GENERIC_ROUTE
    root = protocol.run_root(output_root, dataset, training_seed)
    config = {
        "stage": "P3-C1-B1-FINAL-FORMAL",
        "dataset": dataset,
        "training_seed": training_seed,
        "device": device,
        "output_root": str(Path(output_root).resolve()),
        "run_root": str(root.resolve()),
        "dataset_adapter": route,
        "semantic_lr": binding.semantic_lr,
        "semantic_exposure": binding.semantic_exposure,
        "final_parameter_freeze_sha256": binding.final_parameter_freeze_sha256,
        "final_parameter_source": str(
            (Path(freeze_dir) / "FINAL_HYPERPARAMETERS.json").resolve()
        ),
        "query_chunk_size": query_chunk_size,
        "carrier_epochs": protocol.CARRIER_EPOCHS,
        "refinement_epochs": protocol.REFINEMENT_EPOCHS,
        "hyperparameter_cli_override_allowed": False,
        "parameter_retuning_allowed": False,
        "full_gt_read_by_training": False,
    }
    return config, root


def verify_complete_run(root: str | Path) -> dict:
    root = Path(root)
    complete_path, seal_path = root / "COMPLETE.json", root / "PRE_GT_SEAL.json"
    if not complete_path.is_file() or not seal_path.is_file():
        raise FileNotFoundError("FINAL_FORMAL_COMPLETE_OR_SEAL_MISSING")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("status") != "COMPLETE" or complete.get("pre_gt_seal_sha256") != protocol.sha256_file(seal_path):
        raise RuntimeError("FINAL_FORMAL_COMPLETE_INVALID")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    expected_artifacts = seal.get("artifact_sha256")
    if not isinstance(expected_artifacts, dict) or _hash_relative_files(root, list(expected_artifacts)) != expected_artifacts:
        raise RuntimeError("FINAL_FORMAL_ARTIFACT_SEAL_INVALID")
    if seal.get("parent_source_sha256") != protocol.verify_parent_sources():
        raise RuntimeError("FINAL_FORMAL_PARENT_SEAL_INVALID")
    observed_sources = {
        str(path.relative_to(protocol.REPO_ROOT)): protocol.sha256_file(path)
        for path in _source_paths()
        if path.is_file()
    }
    if seal.get("runner_source_sha256") != observed_sources:
        raise RuntimeError("FINAL_FORMAL_RUNNER_SOURCE_SEAL_INVALID")
    config = json.loads((root / "RUN_CONFIG.json").read_text(encoding="utf-8"))
    binding = protocol.resolve_binding(
        config["dataset"], Path(config["final_parameter_source"]).parent
    )
    if (
        config["training_seed"] not in protocol.FORMAL_SEEDS
        or config["semantic_lr"] != binding.semantic_lr
        or config["semantic_exposure"] != binding.semantic_exposure
        or config["final_parameter_freeze_sha256"] != binding.final_parameter_freeze_sha256
    ):
        raise RuntimeError("FINAL_FORMAL_CONFIG_BINDING_INVALID")
    audit = json.loads((root / "PRE_GT_AUDIT.json").read_text(encoding="utf-8"))
    false_fields = (
        "GT_loaded_before_seal",
        "full_gt_used_for_training",
        "full_gt_used_for_model_selection",
        "full_gt_used_for_hyperparameter_selection",
        "ACC_NMI_ARI_computed",
    )
    if any(audit.get(name) is not False for name in false_fields):
        raise RuntimeError("FINAL_FORMAL_GT_ISOLATION_INVALID")
    return {"config": config, "audit": audit, "seal": seal, "complete": complete}


def run_final_formal(
    dataset: str,
    training_seed: int,
    device: str = "cuda:0",
    output_root: str | Path = protocol.DEFAULT_OUTPUT_ROOT,
    freeze_dir: str | Path = protocol.DEFAULT_FREEZE,
    query_chunk_size: int = protocol.QUERY_CHUNK_SIZE,
    backend: Callable[[dict, Path], dict] | None = None,
) -> dict:
    """Execute or safely resume one pre-GT run; never overwrite a sealed result."""
    protocol.verify_parent_sources()
    config, root = _build_run_config(
        dataset, training_seed, device, output_root, freeze_dir, query_chunk_size
    )
    if (root / "COMPLETE.json").is_file():
        verified = verify_complete_run(root)
        if verified["config"] != config:
            raise RuntimeError("COMPLETE_RUN_CONFIG_MISMATCH")
        return {"status": "ALREADY_COMPLETE_VERIFIED", "root": str(root)}
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "RUN_CONFIG.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError("PARTIAL_RUN_CONFIG_MISMATCH")
    else:
        protocol.write_json(config_path, config)

    result = (backend or execute_pre_gt_training)(config, root)
    artifacts = result.get("artifacts")
    audit = result.get("audit")
    if not isinstance(artifacts, list) or not artifacts or not isinstance(audit, dict):
        raise RuntimeError("TRAINING_BACKEND_CONTRACT_INVALID")
    required_false = (
        "GT_loaded_before_seal",
        "full_gt_used_for_training",
        "full_gt_used_for_model_selection",
        "full_gt_used_for_hyperparameter_selection",
        "ACC_NMI_ARI_computed",
    )
    if (
        audit.get("dataset") != dataset
        or audit.get("training_seed") != training_seed
        or audit.get("semantic_lr") != config["semantic_lr"]
        or audit.get("semantic_exposure") != config["semantic_exposure"]
        or audit.get("final_parameter_freeze_sha256") != config["final_parameter_freeze_sha256"]
        or any(audit.get(name) is not False for name in required_false)
    ):
        raise RuntimeError("PRE_GT_AUDIT_CONTRACT_INVALID")
    protocol.write_json(root / "PRE_GT_AUDIT.json", audit)
    sealed_artifacts = ["RUN_CONFIG.json", "PRE_GT_AUDIT.json", *artifacts]
    runner_sources = {
        str(path.relative_to(protocol.REPO_ROOT)): protocol.sha256_file(path)
        for path in _source_paths()
        if path.is_file()
    }
    seal = {
        "status": "PRE_GT_SEALED",
        "dataset": dataset,
        "training_seed": training_seed,
        "final_parameter_freeze_sha256": config["final_parameter_freeze_sha256"],
        "artifact_sha256": _hash_relative_files(root, sealed_artifacts),
        "runner_source_sha256": runner_sources,
        "parent_source_sha256": protocol.verify_parent_sources(),
        "GT_loaded_before_seal": False,
    }
    protocol.write_json(root / "PRE_GT_SEAL.json", seal)
    complete = {
        "status": "COMPLETE",
        "dataset": dataset,
        "training_seed": training_seed,
        "pre_gt_seal_sha256": protocol.sha256_file(root / "PRE_GT_SEAL.json"),
        "formal_metrics_computed": False,
        "parameter_retuning_allowed": False,
    }
    protocol.write_json(root / "COMPLETE.json", complete)
    verify_complete_run(root)
    return {"status": "COMPLETE", "root": str(root)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=protocol.DATASETS, required=True)
    parser.add_argument(
        "--training-seed", type=int, choices=protocol.FORMAL_SEEDS, required=True
    )
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    parser.add_argument("--output-root", type=Path, default=protocol.DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--final-hyperparameter-freeze", type=Path, default=protocol.DEFAULT_FREEZE
    )
    parser.add_argument(
        "--query-chunk-size", type=int, default=protocol.QUERY_CHUNK_SIZE
    )
    parser.add_argument(
        "--execute", action="store_true", help="run one real final formal pair"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.execute:
        raise SystemExit("refusing to launch final formal GPU training without --execute")
    result = run_final_formal(
        dataset=args.dataset,
        training_seed=args.training_seed,
        device=args.device,
        output_root=args.output_root,
        freeze_dir=args.final_hyperparameter_freeze,
        query_chunk_size=args.query_chunk_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

