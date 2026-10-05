"""
P3-C1-A1 Refinement-Budget Sensitivity Diagnostic
==================================================

Scientific question:
    With semantic_lr fixed at 1e-5, is instability caused by
    excessive repeated semantic intervention?

ONLY refinement budget changes:
    E in {5, 10, 20}

For every E:
    BASE:
        E native epochs

    OURS:
        E x [Semantic Phase A -> Native Phase B]

Therefore BASE and OURS always have exactly the same number
of native structural epochs.

Frozen:
    - carrier state
    - U_cycle
    - T_rel
    - semantic_lr = 1e-5
    - native_lr
    - corruption
    - sparse labels
    - BONE architecture/config
    - semantic schedule prefix
    - paired native minibatch schedule

This is DIAGNOSTIC / HYPERPARAMETER ANALYSIS only.
Full GT is loaded only after training.
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

C1_A0_ROOT = Path(
    "outputs/bone_p3/p3_c1_a0_semantic_lr/aloi"
)

OUTPUT_ROOT = Path(
    "outputs/bone_p3/p3_c1_a1_refinement_budget/aloi"
)

SEMANTIC_LR = 1e-5

ALLOWED_BUDGETS = (
    5,
    10,
    20,
)


def load_gt(n: int) -> np.ndarray:
    """
    Post-training diagnostic only.

    Returns
    -------
    y_raw : [N]
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


def evaluate_model(
    model,
    views,
    adapter,
    y_raw,
    device,
):
    """
    Prediction:
        [N]

    GT:
        [N]
    """

    pred = np.asarray(
        c0._predict(
            model,
            views,
            device,
        ),
        dtype=np.int64,
    ).reshape(-1)

    if pred.shape != adapter.sample_ids.shape:
        raise RuntimeError(
            "prediction/sample-ID mismatch"
        )

    metrics = evaluator.compute_metrics(
        y_raw[
            adapter.sample_ids
        ],
        pred,
    )

    return metrics, pred


def run_one(
    seed: int,
    budget: int,
) -> dict:

    if seed not in (
        20,
        30,
        50,
    ):
        raise ValueError(
            "unregistered seed"
        )

    if budget not in ALLOWED_BUDGETS:
        raise ValueError(
            "unregistered refinement budget"
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
    # Load exact frozen carrier.
    # No carrier retraining.
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
    # BASE / OURS start from EXACT SAME carrier.
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
        == carrier_hash
    ):
        raise RuntimeError(
            "shared carrier initialization mismatch"
        )

    # =========================================================
    # Exact optimizer definitions.
    # =========================================================

    base_optimizer = optim.Adam(
        base_model.parameters(),
        lr=protocol.NATIVE_LR,
    )

    ours_native_optimizer = optim.Adam(
        ours_model.parameters(),
        lr=protocol.NATIVE_LR,
    )

    semantic_optimizer = optim.Adam(
        ours_model.parameters(),
        lr=SEMANTIC_LR,
    )

    # Two independent loaders with identical seeded generators.
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

    # full_views:
    # list[V] where each view is [N, D_v]
    full_views = [
        torch.from_numpy(
            view
        ).to(
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

    # [N_query, L, S]
    # Frozen / stop-gradient.
    t_rel = torch.as_tensor(
        t_rel_np,
        dtype=torch.float32,
        device=device,
    ).detach()

    # [N_query, S]
    # Frozen / stop-gradient.
    u_cycle = torch.as_tensor(
        u_cycle_np,
        dtype=torch.float32,
        device=device,
    ).detach()

    all_semantic_orders = protocol.semantic_orders(
        query_rows.numel(),
        seed,
    )

    if len(
        all_semantic_orders
    ) != protocol.REFINEMENT_EPOCHS:
        raise RuntimeError(
            "unexpected semantic schedule length"
        )

    # Prefix only:
    # E=5 uses first 5 frozen schedules,
    # E=10 uses first 10,
    # E=20 uses all 20.
    semantic_orders = (
        all_semantic_orders[
            :budget
        ]
    )

    semantic_losses = []
    ours_native_losses = []
    base_native_losses = []

    ours_native_orders = []
    base_native_orders = []

    # =========================================================
    # Paired budget experiment.
    #
    # OURS:
    #   Semantic -> Native
    #
    # BASE:
    #   Native
    #
    # Same native budget and minibatch order.
    # =========================================================

    for epoch_index, semantic_order in enumerate(
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

        ours_native_loss, ours_order = (
            c0._native_epoch(
                ours_model,
                ours_native_optimizer,
                ours_loader,
                device,
            )
        )

        base_native_loss, base_order = (
            c0._native_epoch(
                base_model,
                base_optimizer,
                base_loader,
                device,
            )
        )

        if ours_order != base_order:
            raise RuntimeError(
                "paired native minibatch order mismatch"
            )

        if not np.isfinite(
            (
                semantic_loss,
                ours_native_loss,
                base_native_loss,
            )
        ).all():
            raise FloatingPointError(
                "non-finite refinement loss"
            )

        semantic_losses.append(
            float(
                semantic_loss
            )
        )

        ours_native_losses.append(
            float(
                ours_native_loss
            )
        )

        base_native_losses.append(
            float(
                base_native_loss
            )
        )

        ours_native_orders.append(
            ours_order
        )

        base_native_orders.append(
            base_order
        )

        print(
            f"seed={seed} "
            f"budget={budget} "
            f"epoch={epoch_index:02d}/{budget} "
            f"semantic_loss={semantic_loss:.8f} "
            f"ours_native={ours_native_loss:.8f} "
            f"base_native={base_native_loss:.8f}"
        )

    if ours_native_orders != base_native_orders:
        raise RuntimeError(
            "BASE/OURS native schedule mismatch"
        )

    # =========================================================
    # PRE-GT predictions.
    # =========================================================

    base_pred = np.asarray(
        c0._predict(
            base_model,
            views,
            device,
        ),
        dtype=np.int64,
    ).reshape(-1)

    ours_pred = np.asarray(
        c0._predict(
            ours_model,
            views,
            device,
        ),
        dtype=np.int64,
    ).reshape(-1)

    if (
        base_pred.shape != (spec.n,)
        or ours_pred.shape != (spec.n,)
    ):
        raise RuntimeError(
            "prediction shape mismatch"
        )

    out_dir = (
        OUTPUT_ROOT
        / f"budget{budget}"
        / f"seed{seed}"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        out_dir
        / "pre_gt_predictions.npz",
        sample_ids=adapter.sample_ids,
        base_prediction=base_pred,
        ours_prediction=ours_pred,
    )

    audit = {
        "experiment":
            "P3-C1-A1_refinement_budget",

        "dataset":
            "ALOI",

        "training_seed":
            seed,

        "semantic_lr":
            SEMANTIC_LR,

        "native_lr":
            protocol.NATIVE_LR,

        "refinement_budget":
            budget,

        "base_native_epochs":
            budget,

        "ours_native_epochs":
            budget,

        "ours_semantic_epochs":
            budget,

        "carrier_retrained":
            False,

        "frozen_U_cycle":
            True,

        "frozen_T_rel":
            True,

        "paired_native_schedule":
            True,

        "initial_BASE_state_sha256":
            base_initial_hash,

        "initial_OURS_state_sha256":
            ours_initial_hash,

        "expected_carrier_state_sha256":
            carrier_hash,

        "GT_loaded_before_training":
            False,

        "full_gt_used_for_training":
            False,

        "full_gt_used_for_hyperparameter_selection":
            True,

        "FORMAL_RESULT":
            False,

        "DIAGNOSTIC_ONLY":
            True,

        "parent_hashes":
            parent_hashes,
    }

    (
        out_dir
        / "diagnostic_audit.json"
    ).write_text(
        json.dumps(
            audit,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    # =========================================================
    # Post-training GT diagnostic.
    # =========================================================

    y_raw = load_gt(
        spec.n
    )

    base_metrics = evaluator.compute_metrics(
        y_raw[
            adapter.sample_ids
        ],
        base_pred,
    )

    ours_metrics = evaluator.compute_metrics(
        y_raw[
            adapter.sample_ids
        ],
        ours_pred,
    )

    delta = {
        metric:
            float(
                ours_metrics[metric]
                - base_metrics[metric]
            )
        for metric in (
            "ACC",
            "NMI",
            "Purity",
            "ARI",
        )
    }

    # =========================================================
    # Budget=20 exact parity gate.
    #
    # BASE -> frozen P3-C0 BASE20
    # OURS -> P3-C1-A0 lr=1e-5 OURS20
    # =========================================================

    parity = None

    if budget == 20:

        formal = json.loads(
            formal_metrics_path.read_text()
        )

        c1_a0_path = (
            C1_A0_ROOT
            / "lr1e-5"
            / f"seed{seed}"
            / "diagnostic_metrics.json"
        )

        if not c1_a0_path.exists():
            raise FileNotFoundError(
                c1_a0_path
            )

        c1_a0 = json.loads(
            c1_a0_path.read_text()
        )

        errors = {}

        for metric in (
            "ACC",
            "NMI",
            "Purity",
            "ARI",
        ):

            errors[
                f"BASE_{metric}"
            ] = abs(
                base_metrics[metric]
                - formal["BASE"][metric]
            )

            errors[
                f"OURS_{metric}"
            ] = abs(
                ours_metrics[metric]
                - c1_a0["OURS"][metric]
            )

        parity = {
            "errors":
                errors,

            "pass":
                all(
                    value <= 1e-10
                    for value
                    in errors.values()
                ),
        }

        if not parity[
            "pass"
        ]:
            raise RuntimeError(
                "budget20 parity FAILED"
            )

    result = {
        "experiment":
            "P3-C1-A1_refinement_budget",

        "dataset":
            "ALOI",

        "training_seed":
            seed,

        "semantic_lr":
            SEMANTIC_LR,

        "refinement_budget":
            budget,

        "DIAGNOSTIC_ONLY":
            True,

        "GT_USED_FOR_HYPERPARAMETER_SELECTION":
            True,

        "BASE":
            base_metrics,

        "OURS":
            ours_metrics,

        "delta_OURS_minus_BASE":
            delta,

        "budget20_reference_parity":
            parity,

        "semantic_loss_first":
            semantic_losses[0],

        "semantic_loss_last":
            semantic_losses[-1],

        "ours_native_loss_first":
            ours_native_losses[0],

        "ours_native_loss_last":
            ours_native_losses[-1],

        "base_native_loss_first":
            base_native_losses[0],

        "base_native_loss_last":
            base_native_losses[-1],
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
    print("=" * 105)
    print(
        "P3-C1-A1 REFINEMENT-BUDGET RESULT"
    )
    print("=" * 105)

    print(
        "seed            =",
        seed,
    )

    print(
        "semantic_lr     =",
        SEMANTIC_LR,
    )

    print(
        "budget          =",
        budget,
    )

    print(
        "BASE            =",
        base_metrics,
    )

    print(
        "OURS            =",
        ours_metrics,
    )

    print(
        "DELTA           =",
        delta,
    )

    if parity is not None:

        print(
            "BUDGET20 PARITY =",
            (
                "PASS"
                if parity["pass"]
                else "FAIL"
            ),
        )

    print()
    print(
        "P3_C1_A1_REFINEMENT_BUDGET_RUN_PASS"
    )

    return result


def load_budget20_reference(
    seed: int,
) -> dict:
    """
    Budget20 reference:
        BASE = original frozen P3-C0 BASE20
        OURS = P3-C1-A0 semantic_lr=1e-5 OURS20

    No rerun required for summary.
    """

    formal_path = (
        FORMAL_ROOT
        / f"seed{seed}"
        / "formal_metrics.json"
    )

    ours_path = (
        C1_A0_ROOT
        / "lr1e-5"
        / f"seed{seed}"
        / "diagnostic_metrics.json"
    )

    if (
        not formal_path.exists()
        or not ours_path.exists()
    ):
        raise FileNotFoundError(
            f"missing budget20 reference for seed{seed}"
        )

    formal = json.loads(
        formal_path.read_text()
    )

    ours = json.loads(
        ours_path.read_text()
    )

    base_metrics = formal[
        "BASE"
    ]

    ours_metrics = ours[
        "OURS"
    ]

    delta = {
        metric:
            float(
                ours_metrics[metric]
                - base_metrics[metric]
            )
        for metric in (
            "ACC",
            "NMI",
            "Purity",
            "ARI",
        )
    }

    return {
        "BASE":
            base_metrics,

        "OURS":
            ours_metrics,

        "delta_OURS_minus_BASE":
            delta,
    }


def summarize() -> None:

    metrics = (
        "ACC",
        "NMI",
        "Purity",
        "ARI",
    )

    budgets = (
        5,
        10,
        20,
    )

    all_rows = {}

    for budget in budgets:

        rows = []

        for seed in (
            20,
            30,
            50,
        ):

            if budget == 20:

                row = (
                    load_budget20_reference(
                        seed
                    )
                )

            else:

                path = (
                    OUTPUT_ROOT
                    / f"budget{budget}"
                    / f"seed{seed}"
                    / "diagnostic_metrics.json"
                )

                if not path.exists():
                    raise FileNotFoundError(
                        path
                    )

                row = json.loads(
                    path.read_text()
                )

            rows.append(
                row
            )

        all_rows[
            budget
        ] = rows

    print()
    print("=" * 115)
    print(
        "P3-C1-A1 REFINEMENT-BUDGET THREE-SEED SUMMARY"
    )
    print("=" * 115)

    summary = {}

    for budget in budgets:

        print()
        print(
            f"========== refinement_budget={budget} =========="
        )

        summary[
            str(budget)
        ] = {}

        rows = all_rows[
            budget
        ]

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
                str(budget)
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

    output = {
        "experiment":
            "P3-C1-A1_refinement_budget",

        "semantic_lr":
            SEMANTIC_LR,

        "DIAGNOSTIC_ONLY":
            True,

        "GT_USED_FOR_HYPERPARAMETER_SELECTION":
            True,

        "summary":
            summary,

        "selection_note": (
            "No automatic budget winner is selected. "
            "Compare three-seed mean, worst-seed delta, "
            "positive-seed count, and cross-metric consistency."
        ),
    }

    (
        OUTPUT_ROOT
        / "refinement_budget_summary.json"
    ).write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print(
        "P3_C1_A1_REFINEMENT_BUDGET_SUMMARY_COMPLETE"
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
        "--budget",
        type=int,
        choices=ALLOWED_BUDGETS,
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
        or args.budget is None
    ):
        raise SystemExit(
            "--training-seed and --budget required"
        )

    run_one(
        args.training_seed,
        args.budget,
    )


if __name__ == "__main__":
    main()
