"""Materialize one cached 20-native-epoch BASE reference per dev seed."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Callable, Mapping

import numpy as np
import torch
from torch import optim

from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as c0_protocol
from experiments.utility_relation import run_bone_p3_c0_cross_dataset_formal as c0

from . import amendment_a1, objective_a1, protocol, trial_runner_a1
from .formal_tuning_context import CarrierContext, FormalTuningContext
from .parent_integrity import sha256_file


@dataclass(frozen=True)
class BaseReference:
    dataset: str
    dev_seed: int
    path: Path
    carrier_state_sha256: str
    carrier_pre_gt_sha256: str
    native_epochs: int
    J_BASE_A: float
    J_BASE_B: float
    fold_audit: Mapping[str, dict]
    native_order_hashes: tuple[str, ...]
    protocol_fingerprint: str


def base_reference_directory(formal_tuning_root: str | Path, dataset: str, dev_seed: int) -> Path:
    from .formal_carrier_materializer import DATASET_SLUGS
    if dataset not in protocol.DATASETS or dev_seed not in protocol.DEV_SEEDS:
        raise ValueError("INVALID_BASE_REFERENCE_IDENTITY")
    return Path(formal_tuning_root) / DATASET_SLUGS[dataset] / "base_reference" / f"dev_seed{dev_seed}"


def base_reference_path(formal_tuning_root: str | Path, dataset: str, dev_seed: int) -> Path:
    return base_reference_directory(formal_tuning_root, dataset, dev_seed) / "base_reference.json"


def _json_hash(path: Path) -> str:
    return sha256_file(path)


def _load_model(carrier: CarrierContext, device: torch.device):
    model = c0._new_model(carrier.views, carrier.k, carrier.spec, carrier.dev_seed, device)
    model.load_state_dict(c0._load_state(carrier.state_path, device), strict=True)
    return model


def _run_real_base(carrier: CarrierContext, device_name: str) -> dict:
    if device_name != "cuda:0":
        raise ValueError("FORMAL_BASE_REQUIRES_CUDA_0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device(device_name)
    model = _load_model(carrier, device)
    optimizer = optim.Adam(model.parameters(), lr=(
        getattr(carrier.spec, "native_lr", None) or
        (c0_protocol.NATIVE_LR if carrier.dataset != "MSRC" else 1e-2)
    ))
    # MSRC's frozen native LR is imported from its adapter rather than guessed.
    if carrier.dataset == "MSRC":
        from . import msrc_adapter
        optimizer = optim.Adam(model.parameters(), lr=msrc_adapter.p2_carrier.LR)
    loader = c0._native_loader(carrier.views, carrier.adapter, carrier.dev_seed, carrier.spec.batch_size)
    order_hashes, losses = [], []
    for _ in range(protocol.REFINEMENT_HORIZON):
        loss, order_hash = c0._native_epoch(model, optimizer, loader, device)
        if not np.isfinite(loss):
            raise FloatingPointError("NONFINITE_BASE_NATIVE_LOSS")
        losses.append(float(loss)); order_hashes.append(order_hash)
    return {"model": model, "losses": losses, "native_order_hashes": order_hashes}


def _evaluate_base(carrier: CarrierContext, model, device_name: str) -> tuple[dict, dict]:
    device = torch.device(device_name)
    fold_payloads = trial_runner_a1.build_fold_semantics_a1(
        carrier.arrays, carrier.adapter, carrier.folds
    )
    scores, audits = {}, {}
    # Importing through trial_runner keeps the frozen posterior helper in one place.
    from . import trial_runner
    for fold in protocol.FOLDS:
        payload = fold_payloads[fold]
        pi = trial_runner._validation_pi(
            model, carrier.views, device, payload["validation_rows"], payload["class_signature"]
        )
        score, audit = objective_a1.validation_j_with_audit(
            payload["U_validation"], pi, payload["validation_targets"]
        )
        scores[fold], audits[fold] = float(score), audit
    return scores, audits


def _validate_reference_payload(payload: Mapping, context: FormalTuningContext, seed: int) -> None:
    required = {"dataset", "dev_seed", "carrier_state_sha256", "carrier_pre_gt_sha256",
                "native_epochs", "J_BASE_A", "J_BASE_B", "folds", "full_gt_loaded",
                "protocol_fingerprint", "native_order_hashes"}
    if not required.issubset(payload):
        raise ValueError("BASE_REFERENCE_SCHEMA_MISMATCH")
    carrier = context.carrier(seed)
    if payload["dataset"] != context.dataset or payload["dev_seed"] != seed:
        raise ValueError("BASE_REFERENCE_IDENTITY_MISMATCH")
    if payload["carrier_state_sha256"] != carrier.state_file_sha256 or payload["carrier_pre_gt_sha256"] != carrier.pre_gt_sha256:
        raise ValueError("BASE_REFERENCE_CARRIER_HASH_MISMATCH")
    if payload["native_epochs"] != protocol.REFINEMENT_HORIZON or payload["full_gt_loaded"] is not False:
        raise ValueError("BASE_REFERENCE_PROTOCOL_MISMATCH")
    if payload["protocol_fingerprint"] != context.protocol_fingerprint:
        raise ValueError("BASE_REFERENCE_FINGERPRINT_MISMATCH")
    for fold in protocol.FOLDS:
        record = payload["folds"].get(fold)
        if not isinstance(record, Mapping) or not np.isfinite(float(record["J_BASE"])):
            raise ValueError("BASE_REFERENCE_FOLD_SCHEMA_MISMATCH")
        if record.get("full_gt_loaded") is not False:
            raise ValueError("BASE_REFERENCE_GT_FLAG_MISMATCH")
    if len(payload["native_order_hashes"]) != protocol.REFINEMENT_HORIZON:
        raise ValueError("BASE_REFERENCE_NATIVE_SCHEDULE_MISMATCH")


def load_verified_base_reference(context: FormalTuningContext, dev_seed: int) -> BaseReference:
    path = base_reference_path(context.formal_tuning_root, context.dataset, dev_seed)
    marker_path = path.parent / "COMPLETE.json"
    if not path.is_file() or not marker_path.is_file():
        raise FileNotFoundError("BASE_REFERENCE_MISSING")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "COMPLETE" or marker.get("base_reference_sha256") != sha256_file(path):
        raise ValueError("BASE_REFERENCE_HASH_MISMATCH")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_reference_payload(payload, context, dev_seed)
    return BaseReference(
        dataset=context.dataset, dev_seed=dev_seed, path=path,
        carrier_state_sha256=payload["carrier_state_sha256"],
        carrier_pre_gt_sha256=payload["carrier_pre_gt_sha256"],
        native_epochs=payload["native_epochs"],
        J_BASE_A=float(payload["folds"]["A"]["J_BASE"]),
        J_BASE_B=float(payload["folds"]["B"]["J_BASE"]),
        fold_audit=payload["folds"],
        native_order_hashes=tuple(payload["native_order_hashes"]),
        protocol_fingerprint=payload["protocol_fingerprint"],
    )


def build_base_reference(
    context: FormalTuningContext,
    dev_seed: int,
    device: str = "cuda:0",
    replace_incomplete: bool = False,
    runner: Callable[[CarrierContext, str], dict] | None = None,
) -> BaseReference:
    """Build exactly one BASE state for a seed, with hash-checked cache reuse."""
    carrier = context.carrier(dev_seed)
    path = base_reference_path(context.formal_tuning_root, context.dataset, dev_seed)
    directory = path.parent
    marker_path = directory / "COMPLETE.json"
    if path.is_file() or marker_path.is_file():
        if path.is_file() and marker_path.is_file():
            return load_verified_base_reference(context, dev_seed)
        if not replace_incomplete:
            raise RuntimeError("INCOMPLETE_BASE_REFERENCE")
    if directory.exists() and any(directory.iterdir()) and not replace_incomplete:
        raise RuntimeError("INCOMPLETE_BASE_REFERENCE")
    if directory.exists() and replace_incomplete:
        for child in directory.iterdir():
            if child.is_dir() or child.is_file():
                if child.is_dir(): shutil.rmtree(child)
                else: child.unlink()
    directory.mkdir(parents=True, exist_ok=True)
    run = (runner or _run_real_base)(carrier, device)
    model = run.get("model")
    if model is None:
        raise ValueError("BASE_RUNNER_MUST_RETURN_MODEL")
    scores, audits = _evaluate_base(carrier, model, device)
    native_order_hashes = tuple(run.get("native_order_hashes", ()))
    if len(native_order_hashes) != protocol.REFINEMENT_HORIZON:
        raise ValueError("BASE_NATIVE_EPOCH_COUNT_MISMATCH")
    payload = {
        "dataset": context.dataset, "dev_seed": dev_seed,
        "carrier_state_sha256": carrier.state_file_sha256,
        "carrier_pre_gt_sha256": carrier.pre_gt_sha256,
        "native_epochs": protocol.REFINEMENT_HORIZON,
        "J_BASE_A": scores["A"], "J_BASE_B": scores["B"],
        "folds": {
            fold: {"J_BASE": scores[fold], **audits[fold], "full_gt_loaded": False}
            for fold in protocol.FOLDS
        },
        "full_gt_loaded": False,
        "native_order_hashes": list(native_order_hashes),
        "native_schedule_provenance": "frozen_c0_native_loader_and_epoch",
        "protocol_fingerprint": context.protocol_fingerprint,
        "amendment_id": amendment_a1.AMENDMENT_ID,
        "optuna_study_created": False,
    }
    staging = directory / f".base_reference_staging_{uuid.uuid4().hex}.json"
    staging.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, path)
    marker_staging = directory / f".base_reference_complete_{uuid.uuid4().hex}.json"
    marker_staging.write_text(json.dumps({
        "status": "COMPLETE", "dataset": context.dataset, "dev_seed": dev_seed,
        "base_reference_sha256": sha256_file(path), "full_gt_loaded": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(marker_staging, marker_path)
    return load_verified_base_reference(context, dev_seed)


# Compatibility aliases used by tests and callers.
materialize_base_reference = build_base_reference
verify_base_reference = load_verified_base_reference
