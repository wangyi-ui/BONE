"""
P3-C1-A0 Semantic-LR Sensitivity Diagnostic
============================================

Purpose
-------
Test whether the frozen U_cycle / T_rel mechanism is sensitive to
semantic optimization magnitude.

ONLY semantic learning rate is changed.

Frozen:
    - carrier state
    - U_cycle
    - T_rel
    - sparse labels
    - corruption
    - native LR
    - refinement epochs
    - BONE architecture/configuration
    - native minibatch schedule
    - semantic schedule

This is NOT a formal result.

GT is loaded only AFTER training, for explicit hyperparameter diagnosis.
Therefore outputs are tagged:
    full_gt_used_for_training = False
    full_gt_used_for_hyperparameter_selection = True
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch import optim

from experiments.utility_relation import (
    bone_p3_b0_generic_carrier as b0,
)
from experiments.utility_relation import (
    bone_p3_c0_cross_dataset_formal_protocol as protocol,
)
from experiments.utility_relation import (
    run_bone_p3_c0_cross_dataset_formal as c0,
)
from experiments.utility_relation import (
    evaluate_bone_p3_c0_cross_dataset_formal as evaluator,
)
from utils.seed_util import set_seed


FORMAL_ROOT = Path(
    "outputs/bone_p3/p3_c0_formal/aloi"
)

OUTPUT_ROOT = Path(
    "outputs/bone_p3/p3_c1_a0_semantic_lr/aloi"
)

ALLOWED_LRS = (
    1e-5,
    3e-5,
    1e-4,
)


def lr_slug(lr: float) -> str:
    mapping = {
        1e-5: "lr1e-5",
        3e-5: "lr3e-5",
        1e-4: "lr1e-4",
    }

    for value, name in mapping.items():
        if abs(lr - value) < 1e-15:
            return name

    raise ValueError(
        f"unregistered semantic LR: {lr}"
    )


def load_gt(n: int) -> np.ndarray:
    """
    Post-training diagnostic only.

    Returns:
        y_raw: [N]
    """

    y_raw = np.asarray(
        scipy.io.loadmat(
            protocol.DATA_ROOT / "ALOI.mat",
            variable_names=("Y",),
        )["Y"]
    ).reshape(-1).astype(
        np.int64
    )

    if y_raw.shape != (n,):
        raise RuntimeError(
            "ALOI GT length mismatch"
        )

    return y_raw


def run_one(
    seed: int,
    semantic_lr: float,
) -> dict:

    if seed not in (20, 30, 50):
        raise ValueError(
            "unregistered seed"
        )

    if not any(
        abs(
            semantic_lr - x
        ) < 1e-15
        for x in ALLOWED_LRS
    ):
        raise ValueError(
            "unregistered semantic LR"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required"
        )

    protocol.verify_native_hashes()
    protocol.verify_parent_freeze_hashes()

    torch.cuda.set_device(0)

    device = torch.device(
        "cuda:0"
    )

    spec = protocol.DATASET_SPECS[
        "ALOI"
    ]

    # =========================================================
    # Frozen P3-C0 input construction.
    # =========================================================

    labels, targets, mask, parent_hashes = (
        c0._load_frozen_inputs(
            "ALOI"
        )
    )

    raw_views = c0.load_raw_views(
        "ALOI"
    )

    views, adapter = (
        b0.preprocess_with_raw_ids(
            raw_views,
            mask,
        )
    )

    k = int(
        targets.max()
    ) + 1

    if not np.array_equal(
        np.sort(
            adapter.sample_ids
        ),
        np.arange(
            spec.n
        ),
    ):
        raise RuntimeError(
            "sample-ID adapter mismatch"
        )

    # =========================================================
    # Frozen carrier from original formal P3-C0.
    # No 100-epoch carrier retraining.
    # =========================================================

    formal_seed_root = (
        FORMAL_ROOT
        / f"seed{seed}"
    )

    carrier_state_path = (
        formal_seed_root
        / "carrier"
        / "carrier_model_state.pt"
    )

    carrier_npz_path = (
        formal_seed_root
        / "carrier"
        / "carrier_pre_gt.npz"
    )

    carrier_audit_path = (
        formal_seed_root
        / "carrier"
        / "carrier_pre_gt_audit.json"
    )

    formal_metrics_path = (
        formal_seed_root
        / "formal_metrics.json"
    )

    for path in (
        carrier_state_path,
        carrier_npz_path,
        carrier_audit_path,
        formal_metrics_path,
    ):
        if not path.exists():
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

    carrier_hash = carrier_audit[
        "carrier_state_sha256"
    ]

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

        t_rel_np = np.asarray(
            d["T_rel"],
            dtype=np.float32,
        )

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
    # Build OURS from exact same frozen carrier state.
    # =========================================================

    set_seed(seed)

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

    initial_state_hash = (
        c0._state_hash(
            model
        )
    )

    if initial_state_hash != carrier_hash:
        raise RuntimeError(
            "initial carrier parity failure"
        )

    # =========================================================
    # Optimizers.
    #
    # ONLY THIS LR CHANGES:
    # semantic_lr
    #
    # native LR remains exact frozen P3-C0 value.
    # =========================================================

    semantic_optimizer = optim.Adam(
        model.parameters(),
        lr=semantic_lr,
    )

    native_optimizer = optim.Adam(
        model.parameters(),
        lr=protocol.NATIVE_LR,
    )

    native_loader = c0._native_loader(
        views,
        adapter,
        seed,
        spec.batch_size,
    )

    # full_views:
    # list[V] of [N, D_v]
    full_views = [
        torch.from_numpy(
            view
        ).to(
            device=device,
            dtype=torch.float32,
        )
        for view in views
    ]

    # query_rows:
    # [N_query]
    query_rows = torch.as_tensor(
        query_rows_np,
        dtype=torch.long,
        device=device,
    )

    # anchor_rows:
    # [L]
    anchor_rows = torch.as_tensor(
        anchor_rows_np,
        dtype=torch.long,
        device=device,
    )

    # T_rel:
    # [N_query, L, S]
    # Frozen / stop-gradient.
    t_rel = torch.as_tensor(
        t_rel_np,
        dtype=torch.float32,
        device=device,
    ).detach()

    # U_cycle:
    # [N_query, S]
    # Frozen / stop-gradient.
    u_cycle = torch.as_tensor(
        u_cycle_np,
        dtype=torch.float32,
        device=device,
    ).detach()

    semantic_orders = protocol.semantic_orders(
        query_rows.numel(),
        seed,
    )

    if len(
        semantic_orders
    ) != protocol.REFINEMENT_EPOCHS:
        raise RuntimeError(
            "semantic schedule length mismatch"
        )

    semantic_losses = []
    native_losses = []
    native_orders = []

    # =========================================================
    # Exact alternating refinement:
    #
    # Semantic Phase A
    #       ->
    # Native Phase B
    #
    # 20 epochs.
    # =========================================================

    for epoch_index, order in enumerate(
        semantic_orders,
        start=1,
    ):

        semantic_loss = c0._semantic_epoch(
            model,
            semantic_optimizer,
            full_views,
            query_rows,
            anchor_rows,
            t_rel,
            u_cycle,
            order,
            k,
        )

        native_loss, native_order = (
            c0._native_epoch(
                model,
                native_optimizer,
                native_loader,
                device,
            )
        )

        if not np.isfinite(
            semantic_loss
        ):
            raise FloatingPointError(
                "non-finite semantic loss"
            )

        if not np.isfinite(
            native_loss
        ):
            raise FloatingPointError(
                "non-finite native loss"
            )

        semantic_losses.append(
            float(
                semantic_loss
            )
        )

        native_losses.append(
            float(
                native_loss
            )
        )

        native_orders.append(
            native_order
        )

        print(
            f"seed={seed} "
            f"lr={semantic_lr:.1e} "
            f"epoch={epoch_index:02d}/"
            f"{protocol.REFINEMENT_EPOCHS} "
            f"semantic_loss={semantic_loss:.8f} "
            f"native_loss={native_loss:.8f}"
        )

    # =========================================================
    # Save PRE-GT prediction first.
    # =========================================================

    pred = c0._predict(
        model,
        views,
        device,
    )

    pred = np.asarray(
        pred,
        dtype=np.int64,
    ).reshape(-1)

    if pred.shape != (
        spec.n,
    ):
        raise RuntimeError(
            "prediction shape mismatch"
        )

    slug = lr_slug(
        semantic_lr
    )

    out_dir = (
        OUTPUT_ROOT
        / slug
        / f"seed{seed}"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_path = (
        out_dir
        / "pre_gt_predictions.npz"
    )

    np.savez_compressed(
        prediction_path,
        sample_ids=adapter.sample_ids,
        prediction=pred,
    )

    audit = {
        "experiment":
            "P3-C1-A0_semantic_lr_diagnostic",

        "dataset":
            "ALOI",

        "training_seed":
            seed,

        "semantic_lr":
            semantic_lr,

        "native_lr":
            protocol.NATIVE_LR,

        "refinement_epochs":
            protocol.REFINEMENT_EPOCHS,

        "carrier_retrained":
            False,

        "frozen_U_cycle":
            True,

        "frozen_T_rel":
            True,

        "initial_carrier_state_sha256":
            initial_state_hash,

        "expected_carrier_state_sha256":
            carrier_hash,

        "GT_loaded_before_training":
            False,

        "full_gt_used_for_training":
            False,

        # Explicitly TRUE:
        # this phase is hyperparameter diagnosis.
        "full_gt_used_for_hyperparameter_selection":
            True,

        "FORMAL_RESULT":
            False,

        "DIAGNOSTIC_ONLY":
            True,

        "parent_hashes":
            parent_hashes,
    }

    audit_path = (
        out_dir
        / "diagnostic_audit.json"
    )

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    # =========================================================
    # POST-TRAINING GT evaluation.
    #
    # GT is not involved in any gradient / update.
    # =========================================================

    y_raw = load_gt(
        spec.n
    )

    metrics = evaluator.compute_metrics(
        y_raw[
            adapter.sample_ids
        ],
        pred,
    )

    formal = json.loads(
        formal_metrics_path.read_text()
    )

    base_metrics = formal[
        "BASE"
    ]

    delta_vs_base = {
        metric:
            float(
                metrics[metric]
                - base_metrics[metric]
            )
        for metric in (
            "ACC",
            "NMI",
            "Purity",
            "ARI",
        )
    }

    # For lr=1e-4 only:
    # this runner must exactly reproduce frozen P3-C0 OURS.
    original_parity = None

    if abs(
        semantic_lr
        - 1e-4
    ) < 1e-15:

        parity_errors = {
            metric:
                abs(
                    metrics[metric]
                    - formal["OURS"][metric]
                )
            for metric in (
                "ACC",
                "NMI",
                "Purity",
                "ARI",
            )
        }

        original_parity = {
            "errors":
                parity_errors,

            "pass":
                all(
                    value <= 1e-10
                    for value
                    in parity_errors.values()
                ),
        }

        if not original_parity[
            "pass"
        ]:
            raise RuntimeError(
                "1e-4 frozen formal parity FAILED"
            )

    result = {
        "experiment":
            "P3-C1-A0_semantic_lr_diagnostic",

        "dataset":
            "ALOI",

        "training_seed":
            seed,

        "semantic_lr":
            semantic_lr,

        "DIAGNOSTIC_ONLY":
            True,

        "GT_USED_FOR_HYPERPARAMETER_SELECTION":
            True,

        "OURS":
            metrics,

        "FROZEN_BASE":
            base_metrics,

        "delta_OURS_minus_BASE":
            delta_vs_base,

        "original_1e-4_parity":
            original_parity,

        "semantic_loss_first":
            semantic_losses[0],

        "semantic_loss_last":
            semantic_losses[-1],

        "native_loss_first":
            native_losses[0],

        "native_loss_last":
            native_losses[-1],

        "native_order_count":
            len(
                native_orders
            ),
    }

    result_path = (
        out_dir
        / "diagnostic_metrics.json"
    )

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("=" * 100)
    print(
        "P3-C1-A0 SEMANTIC-LR RESULT"
    )
    print("=" * 100)

    print(
        "seed        =",
        seed,
    )

    print(
        "semantic_lr =",
        semantic_lr,
    )

    print(
        "OURS        =",
        metrics,
    )

    print(
        "FROZEN BASE =",
        base_metrics,
    )

    print(
        "DELTA       =",
        delta_vs_base,
    )

    if original_parity is not None:
        print(
            "1e-4 FORMAL PARITY =",
            (
                "PASS"
                if original_parity["pass"]
                else "FAIL"
            ),
        )

    print()
    print(
        "P3_C1_A0_SEMANTIC_LR_RUN_PASS"
    )

    return result


def summarize() -> None:

    candidate_lrs = (
        1e-5,
        3e-5,
    )

    metrics = (
        "ACC",
        "NMI",
        "Purity",
        "ARI",
    )

    all_rows = {}

    for lr in candidate_lrs:

        slug = lr_slug(
            lr
        )

        rows = []

        for seed in (
            20,
            30,
            50,
        ):

            path = (
                OUTPUT_ROOT
                / slug
                / f"seed{seed}"
                / "diagnostic_metrics.json"
            )

            if not path.exists():
                raise FileNotFoundError(
                    path
                )

            rows.append(
                json.loads(
                    path.read_text()
                )
            )

        all_rows[
            str(lr)
        ] = rows

    print()
    print("=" * 110)
    print(
        "P3-C1-A0 SEMANTIC-LR THREE-SEED SUMMARY"
    )
    print("=" * 110)

    summary = {}

    for lr in candidate_lrs:

        key = str(
            lr
        )

        rows = all_rows[
            key
        ]

        print()
        print(
            f"========== semantic_lr={lr:.1e} =========="
        )

        summary[
            key
        ] = {}

        for metric in metrics:

            values = np.asarray(
                [
                    row[
                        "delta_OURS_minus_BASE"
                    ][metric]
                    for row in rows
                ],
                dtype=np.float64,
            )

            info = {
                "mean_delta":
                    float(
                        values.mean()
                    ),

                "std_delta":
                    float(
                        values.std()
                    ),

                "worst_seed_delta":
                    float(
                        values.min()
                    ),

                "positive_seed_count":
                    int(
                        np.sum(
                            values > 0
                        )
                    ),

                "seed20":
                    float(
                        values[0]
                    ),

                "seed30":
                    float(
                        values[1]
                    ),

                "seed50":
                    float(
                        values[2]
                    ),
            }

            summary[
                key
            ][metric] = info

            print(
                f"{metric:7s} "
                f"mean={info['mean_delta']:+.8f} "
                f"std={info['std_delta']:.8f} "
                f"worst={info['worst_seed_delta']:+.8f} "
                f"positive={info['positive_seed_count']}/3 "
                f"[20={info['seed20']:+.8f}, "
                f"30={info['seed30']:+.8f}, "
                f"50={info['seed50']:+.8f}]"
            )

    # Frozen P3-C0 reference.
    frozen_rows = []

    for seed in (
        20,
        30,
        50,
    ):

        path = (
            FORMAL_ROOT
            / f"seed{seed}"
            / "formal_metrics.json"
        )

        d = json.loads(
            path.read_text()
        )

        frozen_rows.append(
            d
        )

    print()
    print(
        "========== FROZEN semantic_lr=1e-4 REFERENCE =========="
    )

    frozen_summary = {}

    for metric in metrics:

        values = np.asarray(
            [
                row[
                    "delta_OURS_minus_BASE"
                ][metric]
                for row in frozen_rows
            ],
            dtype=np.float64,
        )

        info = {
            "mean_delta":
                float(
                    values.mean()
                ),

            "std_delta":
                float(
                    values.std()
                ),

            "worst_seed_delta":
                float(
                    values.min()
                ),

            "positive_seed_count":
                int(
                    np.sum(
                        values > 0
                    )
                ),

            "seed20":
                float(
                    values[0]
                ),

            "seed30":
                float(
                    values[1]
                ),

            "seed50":
                float(
                    values[2]
                ),
        }

        frozen_summary[
            metric
        ] = info

        print(
            f"{metric:7s} "
            f"mean={info['mean_delta']:+.8f} "
            f"std={info['std_delta']:.8f} "
            f"worst={info['worst_seed_delta']:+.8f} "
            f"positive={info['positive_seed_count']}/3 "
            f"[20={info['seed20']:+.8f}, "
            f"30={info['seed30']:+.8f}, "
            f"50={info['seed50']:+.8f}]"
        )

    output = {
        "experiment":
            "P3-C1-A0_semantic_lr_diagnostic",

        "DIAGNOSTIC_ONLY":
            True,

        "GT_USED_FOR_HYPERPARAMETER_SELECTION":
            True,

        "candidate_summary":
            summary,

        "frozen_1e-4_reference":
            frozen_summary,

        "interpretation_rule": (
            "No automatic winner is selected. "
            "Assess whether lowering semantic LR improves "
            "three-seed stability and mean/worst-seed deltas "
            "without relying on a single favorable seed."
        ),
    }

    summary_path = (
        OUTPUT_ROOT
        / "semantic_lr_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print(
        "P3_C1_A0_SEMANTIC_LR_SUMMARY_COMPLETE"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training-seed",
        type=int,
        choices=(
            20,
            30,
            50,
        ),
    )

    parser.add_argument(
        "--semantic-lr",
        type=float,
        choices=ALLOWED_LRS,
    )

    parser.add_argument(
        "--summarize",
        action="store_true",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.execute:
        raise SystemExit(
            "Add --execute."
        )

    if args.summarize:

        summarize()
        return

    if (
        args.training_seed is None
        or args.semantic_lr is None
    ):
        raise SystemExit(
            "--training-seed and "
            "--semantic-lr are required"
        )

    run_one(
        args.training_seed,
        args.semantic_lr,
    )


if __name__ == "__main__":
    main()
