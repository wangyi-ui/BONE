"""
P3-C1-B0 candidate validation runner.

Training process NEVER loads full GT.

Validation configuration:
    semantic_lr = 1e-5
    refinement_budget = 5

BASE:
    frozen carrier -> 5 native epochs

OURS:
    frozen carrier ->
        5 x [semantic Phase A -> native Phase B]

BASE and OURS:
    - same carrier
    - same native learning rate
    - same number of native epochs
    - same native minibatch schedule
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import optim

from experiments.utility_relation import (
    bone_p3_b0_generic_carrier as b0,
)
from experiments.utility_relation import (
    bone_p3_c0_cross_dataset_formal_protocol as c0_protocol,
)
from experiments.utility_relation import (
    run_bone_p3_c0_cross_dataset_formal as c0,
)
from experiments.utility_relation import (
    bone_p3_c1_b0_candidate_protocol as candidate,
)
from utils.seed_util import set_seed


RUNNER_SOURCE = Path(__file__)
PROTOCOL_SOURCE = Path(candidate.__file__)


def root_for(
    dataset: str,
    seed: int,
) -> Path:

    if dataset == candidate.TUNING_DATASET:
        return (
            candidate.OUTPUT_ROOT
            / "_parity_aloi"
            / f"seed{seed}"
        )

    return (
        candidate.OUTPUT_ROOT
        / c0_protocol.dataset_slug(dataset)
        / f"seed{seed}"
    )


def formal_root(
    dataset: str,
    seed: int,
) -> Path:

    return (
        c0_protocol.OUTPUT_ROOT
        / c0_protocol.dataset_slug(dataset)
        / f"seed{seed}"
    )


def save_prediction(
    path: Path,
    sample_ids: np.ndarray,
    predictions: np.ndarray,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        sample_ids=np.asarray(
            sample_ids,
            dtype=np.int64,
        ),
        predictions=np.asarray(
            predictions,
            dtype=np.int64,
        ),
    )


def run(
    dataset: str,
    seed: int,
    parity_calibration: bool,
) -> dict:

    allowed = (
        dataset
        in candidate.VALIDATION_DATASETS
    )

    if dataset == candidate.TUNING_DATASET:
        if not parity_calibration:
            raise RuntimeError(
                "ALOI may only be used with "
                "--parity-calibration"
            )
    elif not allowed:
        raise ValueError(
            "unregistered dataset"
        )

    if seed not in candidate.SEEDS:
        raise ValueError(
            "unregistered seed"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required"
        )

    c0_protocol.verify_native_hashes()
    c0_protocol.verify_parent_freeze_hashes()

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    spec = c0_protocol.DATASET_SPECS[
        dataset
    ]

    # =========================================================
    # Frozen weak-quality / sparse-label inputs.
    # No GT is loaded.
    # =========================================================

    labels, targets, mask, parent_hashes = (
        c0._load_frozen_inputs(
            dataset
        )
    )

    raw_views = c0.load_raw_views(
        dataset
    )

    views, adapter = (
        b0.preprocess_with_raw_ids(
            raw_views,
            mask,
        )
    )

    if not np.array_equal(
        np.sort(adapter.sample_ids),
        np.arange(spec.n),
    ):
        raise RuntimeError(
            "raw sample-ID mismatch"
        )

    k = int(
        targets.max()
    ) + 1

    # =========================================================
    # Reuse frozen P3-C0 carrier.
    # Never retrain 100 carrier epochs.
    # =========================================================

    frozen_root = formal_root(
        dataset,
        seed,
    )

    carrier_state_path = (
        frozen_root
        / "carrier"
        / "carrier_model_state.pt"
    )

    carrier_npz_path = (
        frozen_root
        / "carrier"
        / "carrier_pre_gt.npz"
    )

    carrier_audit_path = (
        frozen_root
        / "carrier"
        / "carrier_pre_gt_audit.json"
    )

    for path in (
        carrier_state_path,
        carrier_npz_path,
        carrier_audit_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    carrier_audit = json.loads(
        carrier_audit_path.read_text()
    )

    if carrier_audit.get(
        "GT_loaded_before_seal"
    ) is not False:
        raise RuntimeError(
            "invalid frozen carrier GT gate"
        )

    expected_carrier_hash = (
        carrier_audit[
            "carrier_state_sha256"
        ]
    )

    with np.load(
        carrier_npz_path,
        allow_pickle=False,
    ) as d:

        query_rows_np = np.asarray(
            d["query_rows"],
            dtype=np.int64,
        )

        anchor_rows_np = np.asarray(
            d["anchor_rows"],
            dtype=np.int64,
        )

        # [N_query, L, S]
        t_rel_np = np.asarray(
            d["T_rel"],
            dtype=np.float32,
        )

        # [N_query, S]
        u_cycle_np = np.asarray(
            d["query_U_cycle"],
            dtype=np.float32,
        )

        carrier_sample_ids = np.asarray(
            d["sample_ids"],
            dtype=np.int64,
        )

    if not np.array_equal(
        carrier_sample_ids,
        adapter.sample_ids,
    ):
        raise RuntimeError(
            "carrier/sample-ID mismatch"
        )

    # =========================================================
    # BASE / OURS exact shared initialization.
    # =========================================================

    set_seed(seed)

    def load_branch():

        model = c0._new_model(
            views,
            k,
            spec,
            seed,
            device,
        )

        model.load_state_dict(
            c0._load_state(
                carrier_state_path,
                device,
            ),
            strict=True,
        )

        return model

    base_model = load_branch()
    ours_model = load_branch()

    base_initial_hash = (
        c0._state_hash(
            base_model
        )
    )

    ours_initial_hash = (
        c0._state_hash(
            ours_model
        )
    )

    if not (
        base_initial_hash
        == ours_initial_hash
        == expected_carrier_hash
    ):
        raise RuntimeError(
            "shared carrier parity failure"
        )

    # =========================================================
    # Optimizers.
    #
    # Candidate semantic LR = 1e-5.
    # Native LR remains frozen P3-C0.
    # =========================================================

    base_optimizer = optim.Adam(
        base_model.parameters(),
        lr=c0_protocol.NATIVE_LR,
    )

    ours_native_optimizer = optim.Adam(
        ours_model.parameters(),
        lr=c0_protocol.NATIVE_LR,
    )

    semantic_optimizer = optim.Adam(
        ours_model.parameters(),
        lr=candidate.SEMANTIC_LR,
    )

    base_loader = c0._native_loader(
        views,
        adapter,
        seed,
        spec.batch_size,
    )

    ours_loader = c0._native_loader(
        views,
        adapter,
        seed,
        spec.batch_size,
    )

    # list[V], each tensor [N, D_v]
    full_views = [
        torch.from_numpy(view).to(
            device=device,
            dtype=torch.float32,
        )
        for view in views
    ]

    # [N_query]
    query_rows = torch.as_tensor(
        query_rows_np,
        dtype=torch.long,
        device=device,
    )

    # [L]
    anchor_rows = torch.as_tensor(
        anchor_rows_np,
        dtype=torch.long,
        device=device,
    )

    # [N_query, L, S], stop-gradient.
    t_rel = torch.as_tensor(
        t_rel_np,
        dtype=torch.float32,
        device=device,
    ).detach()

    # [N_query, S], stop-gradient.
    u_cycle = torch.as_tensor(
        u_cycle_np,
        dtype=torch.float32,
        device=device,
    ).detach()

    all_orders = c0_protocol.semantic_orders(
        query_rows.numel(),
        seed,
    )

    semantic_orders = all_orders[
        :candidate.REFINEMENT_BUDGET
    ]

    if len(semantic_orders) != 5:
        raise RuntimeError(
            "candidate semantic budget mismatch"
        )

    base_orders = []
    ours_orders = []

    semantic_losses = []
    base_native_losses = []
    ours_native_losses = []

    # =========================================================
    # Candidate protocol: exactly five paired refinements.
    # =========================================================

    for epoch, semantic_order in enumerate(
        semantic_orders,
        start=1,
    ):

        semantic_loss = c0._semantic_epoch(
            ours_model,
            semantic_optimizer,
            full_views,
            query_rows,
            anchor_rows,
            t_rel,
            u_cycle,
            semantic_order,
            k,
        )

        ours_loss, ours_order = (
            c0._native_epoch(
                ours_model,
                ours_native_optimizer,
                ours_loader,
                device,
            )
        )

        base_loss, base_order = (
            c0._native_epoch(
                base_model,
                base_optimizer,
                base_loader,
                device,
            )
        )

        if ours_order != base_order:
            raise RuntimeError(
                "paired native schedule mismatch"
            )

        if not np.isfinite(
            (
                semantic_loss,
                ours_loss,
                base_loss,
            )
        ).all():
            raise FloatingPointError(
                "non-finite loss"
            )

        semantic_losses.append(
            float(semantic_loss)
        )

        ours_native_losses.append(
            float(ours_loss)
        )

        base_native_losses.append(
            float(base_loss)
        )

        ours_orders.append(
            ours_order
        )

        base_orders.append(
            base_order
        )

        print(
            f"{dataset} seed={seed} "
            f"epoch={epoch}/5 "
            f"semantic={semantic_loss:.8f} "
            f"ours_native={ours_loss:.8f} "
            f"base_native={base_loss:.8f}"
        )

    if ours_orders != base_orders:
        raise RuntimeError(
            "native order parity failed"
        )

    # =========================================================
    # Persist PRE-GT artifacts.
    # =========================================================

    root = root_for(
        dataset,
        seed,
    )

    base_dir = root / "base"
    ours_dir = root / "ours"
    seal_dir = root / "seal"

    for path in (
        base_dir,
        ours_dir,
        seal_dir,
    ):
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    base_state_path = (
        base_dir
        / "final_model_state.pt"
    )

    ours_state_path = (
        ours_dir
        / "final_model_state.pt"
    )

    c0._save_state(
        base_model,
        base_state_path,
    )

    c0._save_state(
        ours_model,
        ours_state_path,
    )

    base_pred = c0._predict(
        base_model,
        views,
        device,
    )

    ours_pred = c0._predict(
        ours_model,
        views,
        device,
    )

    base_pred_path = (
        base_dir
        / "pre_gt_predictions.npz"
    )

    ours_pred_path = (
        ours_dir
        / "pre_gt_predictions.npz"
    )

    save_prediction(
        base_pred_path,
        adapter.sample_ids,
        base_pred,
    )

    save_prediction(
        ours_pred_path,
        adapter.sample_ids,
        ours_pred,
    )

    is_validation = (
        dataset
        in candidate.VALIDATION_DATASETS
    )

    audit = {
        "experiment":
            "P3-C1-B0_candidate_validation",

        "dataset":
            dataset,

        "training_seed":
            seed,

        "dataset_role":
            (
                "held_out_validation"
                if is_validation
                else "ALOI_parity_only"
            ),

        "semantic_lr":
            candidate.SEMANTIC_LR,

        "refinement_budget":
            candidate.REFINEMENT_BUDGET,

        "native_lr":
            c0_protocol.NATIVE_LR,

        "base_native_epochs":
            5,

        "ours_native_epochs":
            5,

        "ours_semantic_epochs":
            5,

        "carrier_retrained":
            False,

        "frozen_U_cycle":
            True,

        "frozen_T_rel":
            True,

        "paired_native_schedule":
            True,

        "GT_loaded_before_seal":
            False,

        "full_gt_used_for_training":
            False,

        "full_gt_used_for_model_selection":
            False,

        # ALOI selected the candidate.
        # Held-out datasets DID NOT.
        "full_gt_used_for_hyperparameter_selection":
            (
                not is_validation
            ),

        "hyperparameters_selected_on":
            "ALOI_only",

        "no_dataset_specific_retuning":
            True,

        "initial_BASE_state_sha256":
            base_initial_hash,

        "initial_OURS_state_sha256":
            ours_initial_hash,

        "carrier_state_sha256":
            expected_carrier_hash,

        "parent_hashes":
            parent_hashes,

        "semantic_loss_first":
            semantic_losses[0],

        "semantic_loss_last":
            semantic_losses[-1],

        "base_native_loss_first":
            base_native_losses[0],

        "base_native_loss_last":
            base_native_losses[-1],

        "ours_native_loss_first":
            ours_native_losses[0],

        "ours_native_loss_last":
            ours_native_losses[-1],
    }

    audit_path = (
        seal_dir
        / "pre_gt_audit.json"
    )

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    targets_to_seal = {
        "carrier_state":
            carrier_state_path,

        "carrier_npz":
            carrier_npz_path,

        "base_state":
            base_state_path,

        "ours_state":
            ours_state_path,

        "base_predictions":
            base_pred_path,

        "ours_predictions":
            ours_pred_path,

        "audit":
            audit_path,

        "runner_source":
            RUNNER_SOURCE,

        "protocol_source":
            PROTOCOL_SOURCE,
    }

    seal = {
        "dataset":
            dataset,

        "training_seed":
            seed,

        "sha256": {
            name:
                c0_protocol.sha256_file(path)
            for name, path
            in targets_to_seal.items()
        },
    }

    seal_path = (
        seal_dir
        / "pre_gt_seal.json"
    )

    seal_path.write_text(
        json.dumps(
            seal,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print(
        "PRE_GT_SEAL_PASS"
    )

    print(
        "output =",
        root,
    )

    return {
        "root":
            root,

        "seal":
            seal_path,
    }


def main():

    parser = argparse.ArgumentParser()

    choices = (
        candidate.VALIDATION_DATASETS
        + (
            candidate.TUNING_DATASET,
        )
    )

    parser.add_argument(
        "--dataset",
        choices=choices,
        required=True,
    )

    parser.add_argument(
        "--training-seed",
        type=int,
        choices=candidate.SEEDS,
        required=True,
    )

    parser.add_argument(
        "--parity-calibration",
        action="store_true",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.execute:
        raise SystemExit(
            "Add --execute"
        )

    run(
        args.dataset,
        args.training_seed,
        args.parity_calibration,
    )


if __name__ == "__main__":
    main()
