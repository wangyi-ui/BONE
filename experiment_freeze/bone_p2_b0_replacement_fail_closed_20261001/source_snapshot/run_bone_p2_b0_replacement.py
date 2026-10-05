"""Protected seed20 runner for P2-B0 REG_ONLY and REPLACE_TRUE_U arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from torch import optim

from experiments.utility_relation.bone_p2_a0_protocol import PARENT_ARTIFACT
from experiments.utility_relation.bone_p2_a1_relation_action import detached_frozen, semantic_epoch
from experiments.utility_relation.bone_p2_a1_protocol import array_sha256, semantic_orders
from experiments.utility_relation.evaluate_bone_p2_a1_pilot import verify_pre_gt_seal
from experiments.utility_relation.materialize_bone_p2_a0_carrier import load_corrupted_views, sha256_file
from experiments.utility_relation.run_bone_p2_a1_pilot import (
    DEVICE, _full_views, _native_loader, instantiate_initial_model, load_frozen_carrier, model_state_sha256,
)
from experiments.utility_relation.bone_p2_b0_protocol import (
    ARMS, CARRIER_DIR, DEFAULT_OUTPUT_DIR, FROZEN_INITIAL_MODEL_SHA256, NEW_ARMS, PARENT_ARM_SOURCES,
    PARENT_P2_A1_DIR, REFINEMENT_EPOCHS, SEMANTIC_LR, NATIVE_LR, TRAINING_SEED,
)
from experiments.utility_relation.bone_p2_b0_replacement import replacement_native_epoch


REPO_ROOT = Path(__file__).resolve().parents[2]


def verify_parent_controls() -> Dict[str, Dict[str, float]]:
    """Bind the frozen P2-A1 BASE and TRUE_U controls without retraining either one."""
    for source_arm in PARENT_ARM_SOURCES.values():
        verify_pre_gt_seal(PARENT_P2_A1_DIR / source_arm)
    metrics_path = PARENT_P2_A1_DIR / "p2_a1_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    frozen_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    try:
        return {
            "NATIVE": frozen_metrics["arms"]["BASE"],
            "ADDON_TRUE_U": frozen_metrics["arms"]["TRUE_U"],
        }
    except (KeyError, TypeError) as error:
        raise ValueError("invalid frozen P2-A1 parent metrics") from error


def _write_pre_gt_seal(arm_dir: Path, files: Dict[str, Path]) -> Path:
    seal = {"sha256": {
        "predictions": sha256_file(files["predictions"]),
        "audit": sha256_file(files["audit"]),
        "runner_source": sha256_file(Path(__file__)),
        "replacement_source": sha256_file(REPO_ROOT / "experiments/utility_relation/bone_p2_b0_replacement.py"),
        "protocol_source": sha256_file(REPO_ROOT / "experiments/utility_relation/bone_p2_b0_protocol.py"),
        "method_bone": sha256_file(REPO_ROOT / "method/bone.py"),
        "method_amv": sha256_file(REPO_ROOT / "method/amv.py"),
        "carrier_npz": sha256_file(CARRIER_DIR / "p2_a0_carrier_pre_gt.npz"),
        "initial_model_state": sha256_file(CARRIER_DIR / "carrier_model_state.pt"),
    }}
    path = arm_dir / "pre_gt_seal.json"
    path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_new_arm(arm: str, output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Dict[str, object]:
    """Train one new replacement arm only; formal execution is guarded by this module's CLI."""
    if arm not in NEW_ARMS:
        raise ValueError("only REG_ONLY and REPLACE_TRUE_U are new P2-B0 arms")
    carrier = load_frozen_carrier(CARRIER_DIR, TRAINING_SEED)
    model = instantiate_initial_model(CARRIER_DIR, TRAINING_SEED)
    initial_hash = model_state_sha256(model)
    if initial_hash != FROZEN_INITIAL_MODEL_SHA256:
        raise RuntimeError("replacement arm initial state does not match frozen P2-A1 seed20 state")
    native_optimizer = optim.Adam(model.parameters(), lr=NATIVE_LR)
    semantic_optimizer = None
    relation_targets = action_weights = None
    if arm == "REPLACE_TRUE_U":
        relation_targets = detached_frozen(carrier.t_rel, DEVICE)
        action_weights = detached_frozen(carrier.u_cycle, DEVICE)
        semantic_optimizer = optim.Adam(model.parameters(), lr=SEMANTIC_LR)
        if semantic_optimizer is native_optimizer:
            raise RuntimeError("semantic and native optimizers must be independent")
    views = load_corrupted_views(PARENT_ARTIFACT)
    full_views = _full_views(views)
    loader = _native_loader(views, TRAINING_SEED)
    unlabeled_ids = torch.as_tensor(carrier.unlabeled_ids, dtype=torch.long, device=DEVICE)
    labeled_ids = torch.as_tensor(carrier.labeled_ids, dtype=torch.long, device=DEVICE)
    orders = semantic_orders(TRAINING_SEED)
    native_order_hashes, semantic_steps = [], []
    for epoch in range(REFINEMENT_EPOCHS):
        if semantic_optimizer is not None:
            semantic_steps.append(semantic_epoch(
                model, semantic_optimizer, full_views, unlabeled_ids, labeled_ids,
                relation_targets, action_weights, orders[epoch],
            ))
        native_order_hashes.append(replacement_native_epoch(model, native_optimizer, loader))
    model.eval()
    with torch.no_grad():
        predictions = model(full_views, test=True)["label"].cpu().numpy().astype(np.int64)
    arm_dir = Path(output_dir) / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    files = {"predictions": arm_dir / "final_predictions.npz", "audit": arm_dir / "pre_gt_audit.json"}
    np.savez_compressed(files["predictions"], sample_ids=np.arange(210, dtype=np.int64), predictions=predictions)
    audit = {
        "arm": arm, "training_seed": TRAINING_SEED, "initial_model_hash": initial_hash,
        "native_optimizer_reinitialized": True, "semantic_optimizer_created": semantic_optimizer is not None,
        "semantic_optimizer_steps_per_epoch": semantic_steps,
        "semantic_order_sha256": array_sha256(orders) if semantic_optimizer is not None else None,
        "native_order_hashes": native_order_hashes, "phase_b": "model.reg * con_reg_loss(assign_mat)",
        "con_inter_view_call_count": 0, "u_cycle_frozen_detached": action_weights is None or not action_weights.requires_grad,
        "t_rel_frozen_detached": relation_targets is None or not relation_targets.requires_grad,
        "full_gt_used_before_seal": False,
    }
    files["audit"].write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files["seal"] = _write_pre_gt_seal(arm_dir, files)
    return {"initial_model_hash": initial_hash, "files": files, "audit": audit}


def run_replacement_pair(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Dict[str, object]:
    """Run only the two newly registered replacement arms after parent-control validation."""
    verify_parent_controls()
    results = {arm: run_new_arm(arm, output_dir) for arm in NEW_ARMS}
    if len({item["initial_model_hash"] for item in results.values()}) != 1:
        raise RuntimeError("replacement arms did not begin from the same state")
    if tuple(results["REG_ONLY"]["audit"]["native_order_hashes"]) != tuple(results["REPLACE_TRUE_U"]["audit"]["native_order_hashes"]):
        raise RuntimeError("replacement arms did not share native sample order")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true", help="train REG_ONLY and REPLACE_TRUE_U only")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.execute:
        raise SystemExit("refusing to start formal P2-B0 arms without --execute")
    print(json.dumps({arm: item["initial_model_hash"] for arm, item in run_replacement_pair(args.output_dir).items()}, indent=2))


if __name__ == "__main__":
    main()
