"""Opt-in CCV seed20 UCAB engineering smoke; this is not a formal experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import optim

from experiments.utility_relation import bone_p3_b0_generic_carrier as b0
from experiments.utility_relation import run_bone_p3_c0_cross_dataset_formal as legacy

from . import protocol as ucab_protocol
from .budget import build_budget, validate_u_cycle
from .semantic_epoch import run_semantic_epoch


ROOT = Path(__file__).resolve().parents[3]
DEFINITION_FREEZE = ROOT / "experiment_freeze/bone_p3_c3_a0_ucab_definition_preregistered_20261004"
CONFIRMATORY_FREEZE = ROOT / "experiment_freeze/bone_p3_c3_a0_ucab_confirmatory_preregistered_20261004"
CPU_IMPLEMENTATION_FREEZE = ROOT / "experiment_freeze/bone_p3_c3_a0_ucab_cpu_implementation_pass_20261004"
SMOKE_DATASET = "CCV"
SMOKE_SEED = 20
SMOKE_SEMANTIC_LR = ucab_protocol.DATASET_CONFIG[SMOKE_DATASET]["semantic_lr"]
SCIENCE_ARMS = ("CURRENT", "TRUE_BUDGET", "MEAN_BUDGET", "SHUFFLE_BUDGET")
DIRECT_LEGACY_REFERENCE = "DIRECT_LEGACY_REFERENCE"
CURRENT_PARAMETER_TOLERANCE = 1e-7


class CountingAdam(optim.Adam):
    """Frozen Adam configuration plus a smoke-only exact step counter."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.step_count = 0

    def step(self, closure: Callable[[], float] | None = None):
        self.step_count += 1
        return super().step(closure)


@dataclass(frozen=True)
class SmokeSources:
    carrier_state: Path
    pre_gt_bundle: Path
    carrier_audit: Path
    pre_gt_seal: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_manifest(directory: Path) -> None:
    manifest = directory / "FREEZE_MANIFEST.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(f"FREEZE_MANIFEST_MISSING:{manifest}")
    for line in manifest.read_text().splitlines():
        if line.strip():
            expected, raw_name = line.split(maxsplit=1)
            candidate = directory / raw_name.lstrip(" *")
            if not candidate.is_file() or _sha256(candidate) != expected:
                raise RuntimeError(f"FREEZE_MANIFEST_MISMATCH:{candidate}")


def _verify_hash_list(path: Path) -> None:
    for line in path.read_text().splitlines():
        if line.strip():
            expected, raw_name = line.split(maxsplit=1)
            candidate = Path(raw_name.strip())
            if not candidate.is_file() or _sha256(candidate) != expected:
                raise RuntimeError(f"FROZEN_SOURCE_HASH_MISMATCH:{candidate}")


def verify_smoke_parent_freezes() -> None:
    """Fail closed on any preregistration, CPU implementation, or source drift."""
    for directory in (DEFINITION_FREEZE, CONFIRMATORY_FREEZE, CPU_IMPLEMENTATION_FREEZE):
        _verify_manifest(directory)
    _verify_hash_list(CPU_IMPLEMENTATION_FREEZE / "IMPLEMENTATION_SOURCE_SHA256.txt")
    _verify_hash_list(CPU_IMPLEMENTATION_FREEZE / "LEGACY_SOURCE_SHA256.txt")
    if _sha256(ucab_protocol.LEGACY_SEMANTIC_CORE) != ucab_protocol.LEGACY_SEMANTIC_CORE_SHA256:
        raise RuntimeError("LEGACY_SEMANTIC_CORE_HASH_MISMATCH")
    ucab_protocol.verify_preregistered_protocol()


def validate_fixed_config(dataset: str, training_seed: int) -> None:
    """The runner exposes no free dataset/seed choice and rejects all alternatives."""
    if dataset != SMOKE_DATASET:
        raise ValueError("ENGINEERING_SMOKE_DATASET_MUST_BE_CCV")
    if training_seed != SMOKE_SEED:
        raise ValueError("ENGINEERING_SMOKE_TRAINING_SEED_MUST_BE_20")


def validate_source_paths(sources: SmokeSources) -> None:
    for name, path in vars(sources).items():
        if not isinstance(path, Path) or not path.is_file():
            raise FileNotFoundError(f"SMOKE_SOURCE_MISSING:{name}:{path}")


def prepare_output_dir(output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"SMOKE_OUTPUT_ALREADY_EXISTS:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"INVALID_{label}:{path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"INVALID_{label}:{path}")
    return value


def _verify_source_provenance(sources: SmokeSources) -> dict[str, str]:
    audit = _load_json(sources.carrier_audit, "CARRIER_AUDIT")
    seal = _load_json(sources.pre_gt_seal, "PRE_GT_SEAL")
    hashes = seal.get("sha256")
    if not isinstance(hashes, dict):
        raise RuntimeError("PRE_GT_SEAL_SHA256_MISSING")
    expected_state = hashes.get("carrier_state")
    expected_bundle = hashes.get("carrier_npz")
    if expected_state != _sha256(sources.carrier_state) or expected_bundle != _sha256(sources.pre_gt_bundle):
        raise RuntimeError("CARRIER_OR_BUNDLE_SEAL_MISMATCH")
    if audit.get("training_seed") != SMOKE_SEED or audit.get("GT_loaded_before_seal") is not False:
        raise RuntimeError("CARRIER_AUDIT_PROVENANCE_MISMATCH")
    if audit.get("carrier_file_sha256") != _sha256(sources.carrier_state):
        raise RuntimeError("CARRIER_AUDIT_STATE_HASH_MISMATCH")
    return {
        "carrier_state": _sha256(sources.carrier_state),
        "pre_gt_bundle": _sha256(sources.pre_gt_bundle),
        "carrier_audit": _sha256(sources.carrier_audit),
        "pre_gt_seal": _sha256(sources.pre_gt_seal),
    }


def _load_smoke_inputs(sources: SmokeSources, device: torch.device) -> tuple:
    """Restore pre-GT state and aligned tensors; raw full labels are never requested."""
    labels, targets, mask, parent_hashes = legacy._load_frozen_inputs(SMOKE_DATASET)
    raw_views = legacy.load_raw_views(SMOKE_DATASET)
    views, adapter = b0.preprocess_with_raw_ids(raw_views, mask)
    with np.load(sources.pre_gt_bundle, allow_pickle=False) as archive:
        required = {"sample_ids", "query_sample_ids", "anchor_sample_ids", "query_rows", "anchor_rows", "T_rel", "query_U_cycle", "labeled_ids", "labeled_targets"}
        if not required.issubset(archive.files):
            raise RuntimeError("PRE_GT_BUNDLE_SCHEMA_MISMATCH")
        payload = {name: np.asarray(archive[name]).copy() for name in required}
    if not np.array_equal(payload["sample_ids"], adapter.sample_ids):
        raise RuntimeError("BUNDLE_ADAPTER_SAMPLE_ID_MISMATCH")
    if not np.array_equal(payload["labeled_ids"], labels) or not np.array_equal(payload["labeled_targets"], targets):
        raise RuntimeError("BUNDLE_SPARSE_LABEL_MISMATCH")
    query_rows = payload["query_rows"].astype(np.int64, copy=False)
    anchor_rows = payload["anchor_rows"].astype(np.int64, copy=False)
    query_ids = payload["query_sample_ids"].astype(np.int64, copy=False)
    anchor_ids = payload["anchor_sample_ids"].astype(np.int64, copy=False)
    t_rel, u_cycle = payload["T_rel"], payload["query_U_cycle"]
    if query_rows.ndim != 1 or anchor_rows.ndim != 1 or query_ids.shape != query_rows.shape or anchor_ids.shape != anchor_rows.shape:
        raise RuntimeError("BUNDLE_ROW_ID_SHAPE_MISMATCH")
    if np.unique(query_ids).size != query_ids.size or not np.array_equal(payload["sample_ids"][query_rows], query_ids):
        raise RuntimeError("QUERY_ID_ALIGNMENT_MISMATCH")
    if np.unique(anchor_ids).size != anchor_ids.size or not np.array_equal(payload["sample_ids"][anchor_rows], anchor_ids):
        raise RuntimeError("ANCHOR_ID_ALIGNMENT_MISMATCH")
    if t_rel.shape != (query_rows.size, anchor_rows.size, u_cycle.shape[1]) or not np.isfinite(t_rel).all() or not np.isfinite(u_cycle).all():
        raise RuntimeError("T_REL_U_CYCLE_ALIGNMENT_MISMATCH")
    if np.any(u_cycle < 0) or np.any(u_cycle > 1):
        raise RuntimeError("U_CYCLE_RANGE_MISMATCH")
    k = int(targets.max()) + 1
    spec = legacy.protocol.DATASET_SPECS[SMOKE_DATASET]
    full_views = [torch.from_numpy(view).to(device=device, dtype=torch.float32) for view in views]
    tensors = {
        "query_rows": torch.as_tensor(query_rows, device=device),
        "anchor_rows": torch.as_tensor(anchor_rows, device=device),
        "query_ids": torch.as_tensor(query_ids, device=device),
        "t_rel": torch.as_tensor(t_rel, dtype=torch.float32, device=device).detach(),
        "u_cycle": torch.as_tensor(u_cycle, dtype=torch.float32, device=device).detach(),
    }
    validate_u_cycle(tensors["u_cycle"])
    if tensors["t_rel"].requires_grad or tensors["u_cycle"].requires_grad:
        raise RuntimeError("FROZEN_SEMANTIC_TENSOR_NOT_DETACHED")
    return views, adapter, full_views, tensors, k, spec, parent_hashes


def _restore_model(views: list[np.ndarray], k: int, spec, device: torch.device, carrier_state: Path) -> torch.nn.Module:
    model = legacy._new_model(views, k, spec, SMOKE_SEED, device)
    model.load_state_dict(legacy._load_state(carrier_state, device), strict=True)
    return model


def _parameter_changed(before: dict[str, torch.Tensor], model: torch.nn.Module) -> bool:
    return any(not torch.equal(value, model.state_dict()[name].detach().cpu()) for name, value in before.items())


def _run_with_anchor_detach_probe(model: torch.nn.Module, anchor_count: int, action: Callable[[], float]) -> tuple[float, bool]:
    observed: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        output.retain_grad()
        observed.append(output)

    handle = model.label_fc.register_forward_hook(capture)
    try:
        loss = action()
    finally:
        handle.remove()
    anchor_detached = bool(observed) and all(
        posterior.grad is not None and bool(torch.count_nonzero(posterior.grad[-anchor_count:]) == 0) for posterior in observed
    )
    return loss, anchor_detached


def _max_state_difference(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    if left.keys() != right.keys():
        raise RuntimeError("CURRENT_PARITY_STATE_KEYS_MISMATCH")
    return max(float((left[name] - right[name]).abs().max().item()) for name in left) if left else 0.0


def _semantic_branch(model: torch.nn.Module, arm: str, inputs: dict, order: np.ndarray) -> tuple[dict, bool]:
    optimizer = CountingAdam(model.parameters(), lr=SMOKE_SEMANTIC_LR)
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.cuda.reset_peak_memory_stats()
    loss, anchor_detached = _run_with_anchor_detach_probe(
        model,
        int(inputs["anchor_rows"].numel()),
        lambda: run_semantic_epoch(
            model, optimizer, inputs["full_views"], inputs["query_rows"], inputs["anchor_rows"], inputs["t_rel"],
            inputs["u_cycle"], order, inputs["k"], arm, query_ids=inputs["query_ids"],
        ),
    )
    if optimizer.step_count != 1 or not np.isfinite(loss) or not _parameter_changed(initial, model):
        raise RuntimeError(f"SEMANTIC_SMOKE_GATE_FAILED:{arm}")
    return {
        "initial_state_hash": legacy.protocol.state_dict_sha256(initial),
        "semantic_loss": loss,
        "semantic_optimizer_step_count": optimizer.step_count,
        "semantic_parameter_update_occurred": True,
        "post_semantic_state_hash": legacy._state_hash(model),
        "semantic_optimizer": "Adam",
        "semantic_lr": SMOKE_SEMANTIC_LR,
    }, anchor_detached


def run_engineering_smoke(sources: SmokeSources, output_dir: Path) -> dict:
    """Execute exactly one CCV seed20 semantic/native smoke epoch per science arm."""
    validate_fixed_config(SMOKE_DATASET, SMOKE_SEED)
    validate_source_paths(sources)
    verify_smoke_parent_freezes()
    if not torch.cuda.is_available():
        raise RuntimeError("ENGINEERING_SMOKE_REQUIRES_CUDA")
    created = False
    try:
        source_hashes = _verify_source_provenance(sources)
        output_dir = prepare_output_dir(output_dir)
        created = True
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        views, adapter, full_views, tensors, k, spec, parent_hashes = _load_smoke_inputs(sources, device)
        tensors["full_views"], tensors["k"] = full_views, k
        order = legacy.protocol.semantic_orders(tensors["query_rows"].numel(), SMOKE_SEED)[0]
        order_hash = legacy.protocol.sha256_file_from_array(order)
        alpha, budget, _ = build_budget("TRUE_BUDGET", tensors["u_cycle"], tensors["query_ids"])
        detach = {
            "U_CYCLE_DETACHED": not tensors["u_cycle"].requires_grad,
            "T_REL_DETACHED": not tensors["t_rel"].requires_grad,
            "ALPHA_DETACHED": not alpha.requires_grad,
            "BUDGET_DETACHED": not budget.requires_grad,
        }
        if not all(detach.values()):
            raise RuntimeError("UTILITY_DETACH_CONTRACT_FAILED")
        branches = {name: _restore_model(views, k, spec, device, sources.carrier_state) for name in (DIRECT_LEGACY_REFERENCE,) + SCIENCE_ARMS}
        source_state_hash = _load_json(sources.carrier_audit, "CARRIER_AUDIT")["carrier_state_sha256"]
        initial_hashes = {name: legacy._state_hash(model) for name, model in branches.items()}
        if set(initial_hashes.values()) != {source_state_hash}:
            raise RuntimeError("INITIAL_STATE_PARITY_FAILED")
        direct_optimizer = CountingAdam(branches[DIRECT_LEGACY_REFERENCE].parameters(), lr=SMOKE_SEMANTIC_LR)
        direct_initial = {name: value.detach().cpu().clone() for name, value in branches[DIRECT_LEGACY_REFERENCE].state_dict().items()}
        torch.cuda.reset_peak_memory_stats()
        direct_loss, direct_anchor_detached = _run_with_anchor_detach_probe(
            branches[DIRECT_LEGACY_REFERENCE], int(tensors["anchor_rows"].numel()),
            lambda: legacy._semantic_epoch(branches[DIRECT_LEGACY_REFERENCE], direct_optimizer, full_views, tensors["query_rows"], tensors["anchor_rows"], tensors["t_rel"], tensors["u_cycle"], order, k),
        )
        if direct_optimizer.step_count != 1 or not np.isfinite(direct_loss) or not _parameter_changed(direct_initial, branches[DIRECT_LEGACY_REFERENCE]):
            raise RuntimeError("DIRECT_LEGACY_REFERENCE_GATE_FAILED")
        direct_state = {name: value.detach().cpu().clone() for name, value in branches[DIRECT_LEGACY_REFERENCE].state_dict().items()}
        direct_reference = {
            "initial_state_hash": legacy.protocol.state_dict_sha256(direct_initial), "semantic_loss": direct_loss,
            "semantic_optimizer_step_count": direct_optimizer.step_count, "semantic_parameter_update_occurred": True,
            "post_semantic_state_hash": legacy._state_hash(branches[DIRECT_LEGACY_REFERENCE]), "semantic_optimizer": "Adam",
            "semantic_lr": SMOKE_SEMANTIC_LR, "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        arms: dict[str, dict] = {}
        anchor_results = [direct_anchor_detached]
        for arm in SCIENCE_ARMS:
            semantic, anchor_detached = _semantic_branch(branches[arm], arm, tensors, order)
            anchor_results.append(anchor_detached)
            if arm == "CURRENT":
                wrapper_state = {name: value.detach().cpu().clone() for name, value in branches[arm].state_dict().items()}
                loss_abs_diff = abs(direct_loss - semantic["semantic_loss"])
                max_parameter_abs_diff = _max_state_difference(direct_state, wrapper_state)
                if loss_abs_diff > CURRENT_PARAMETER_TOLERANCE or max_parameter_abs_diff > CURRENT_PARAMETER_TOLERANCE:
                    raise RuntimeError("CURRENT_WRAPPER_PARITY_FAILED")
                semantic.update({
                    "direct_legacy_loss": direct_loss,
                    "wrapper_current_loss": semantic["semantic_loss"],
                    "loss_abs_diff": loss_abs_diff,
                    "max_parameter_abs_diff": max_parameter_abs_diff,
                    "CURRENT_PARITY_PASS": True,
                })
            native_optimizer = optim.Adam(branches[arm].parameters(), lr=legacy.protocol.NATIVE_LR)
            native_loader = legacy._native_loader(views, adapter, SMOKE_SEED, spec.batch_size)
            native_loss, native_order_hash = legacy._native_epoch(branches[arm], native_optimizer, native_loader, device)
            if not np.isfinite(native_loss):
                raise RuntimeError(f"NATIVE_SMOKE_LOSS_NONFINITE:{arm}")
            semantic.update({
                "native_loss": native_loss, "native_order_hash": native_order_hash, "post_native_state_hash": legacy._state_hash(branches[arm]),
                "native_optimizer": "Adam", "native_lr": legacy.protocol.NATIVE_LR,
                "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            })
            arms[arm] = semantic
        if len({arms[arm]["native_order_hash"] for arm in SCIENCE_ARMS}) != 1:
            raise RuntimeError("NATIVE_ORDER_PARITY_FAILED")
        anchor_pass = all(anchor_results)
        if not anchor_pass:
            raise RuntimeError("ANCHOR_POSTERIOR_DETACH_NOT_VERIFIED")
        audit = {
            "stage": ucab_protocol.STAGE, "dataset": SMOKE_DATASET, "training_seed": SMOKE_SEED, "engineering_only": True,
            "source_carrier_paths": {name: str(path) for name, path in vars(sources).items()}, "source_carrier_hashes": source_hashes,
            "source_parent_hashes": parent_hashes, "initial_state_parity": True, "semantic_order_sha256": order_hash,
            "query_count": int(tensors["query_rows"].numel()), "anchor_count": int(tensors["anchor_rows"].numel()),
            "action_count": int(tensors["u_cycle"].shape[1]), "view_count": len(full_views), "arms": arms,
            "direct_legacy_reference": direct_reference,
            **detach, "ANCHOR_POSTERIOR_DETACHED": True,
            "FULL_GT_LOADED": False, "FULL_GT_USED": False, "ACC_COMPUTED": False, "NMI_COMPUTED": False, "ARI_COMPUTED": False,
            "METRICS_IMPORTED": False, "PREDICTION_PATH_USED": False,
            "FORMAL_RESULT": False, "METHOD_SELECTION_ALLOWED": False, "HYPERPARAMETER_SELECTION_ALLOWED": False, "CONFIRMATORY_EVIDENCE": False,
        }
        (output_dir / "engineering_smoke_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        status = "\n".join((
            "STATUS=P3_C3_A0_UCAB_ENGINEERING_SMOKE_PASS", "DATASET=CCV", "TRAINING_SEED=20", "ENGINEERING_ONLY=true",
            "CURRENT_PARITY_PASS=true", "INITIAL_STATE_PARITY=true", "SEMANTIC_STEP_COUNT_PASS=true", "SEMANTIC_PARAMETER_UPDATE_PASS=true",
            "NATIVE_ORDER_PARITY_PASS=true", "DETACH_CONTRACT_PASS=true", "FULL_GT_LOADED=false", "FORMAL_RESULT=false", "CONFIRMATORY_EVIDENCE=false", "",
        ))
        (output_dir / "ENGINEERING_SMOKE_STATUS.txt").write_text(status)
        return audit
    except Exception as error:
        if created:
            (output_dir / "ENGINEERING_SMOKE_FAILED.txt").write_text(f"FAIL_CLOSED:{type(error).__name__}:{error}\n")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carrier-state", required=True, type=Path)
    parser.add_argument("--pre-gt-bundle", required=True, type=Path)
    parser.add_argument("--carrier-audit", required=True, type=Path)
    parser.add_argument("--pre-gt-seal", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sources = SmokeSources(args.carrier_state, args.pre_gt_bundle, args.carrier_audit, args.pre_gt_seal)
    print(json.dumps(run_engineering_smoke(sources, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
