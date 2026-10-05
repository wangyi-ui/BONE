"""
P3-D0-B4 Update Compatibility Diagnostic
===================================

Goal
----
Replay the frozen P3-C0 refinement process from the already trained
carrier and localize WHEN semantic action becomes beneficial/harmful.

This is a diagnostic only:
    - no carrier retraining
    - no hyperparameter tuning
    - no early-stop selection
    - no method modification
    - frozen U_cycle / T_rel
    - exact P3-C0 optimizer and schedule semantics

For each refinement epoch t:

    PRE_t
        OURS state before semantic Phase A

    SEM_t
        OURS state immediately after semantic Phase A

    NATIVE_t
        OURS state after subsequent native Phase B

    BASE_t
        paired BASE after its native epoch

We report post-seal GT metrics only to diagnose dynamics.
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
    "outputs/bone_p3/p3_d0_b4_update_compatibility/aloi"
)


def predict_metrics(
    model,
    views,
    sample_ids,
    y_raw,
    device,
):
    """
    views:
        list[V] of [N, D_v]

    predictions:
        [N]
    """

    pred = c0._predict(
        model,
        views,
        device,
    )

    pred = np.asarray(
        pred,
        dtype=np.int64,
    ).reshape(-1)

    if pred.shape != sample_ids.shape:
        raise RuntimeError(
            "prediction/sample-ID mismatch"
        )

    return evaluator.compute_metrics(
        y_raw[sample_ids],
        pred,
    ), pred


def state_distance(
    before: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, float]:
    """
    Parameter displacement for one phase.

    Each parameter tensor keeps its original shape.
    Only detached differences are accumulated.
    """

    sq = 0.0
    abs_sum = 0.0
    count = 0

    with torch.no_grad():

        for name, value in model.state_dict().items():

            if name not in before:
                raise RuntimeError(
                    f"missing state tensor: {name}"
                )

            x = value.detach().float().cpu()
            y = before[name].detach().float().cpu()

            if x.shape != y.shape:
                raise RuntimeError(
                    f"state shape mismatch: {name}"
                )

            diff = x - y

            sq += float(
                torch.sum(
                    diff * diff
                ).item()
            )

            abs_sum += float(
                torch.sum(
                    torch.abs(diff)
                ).item()
            )

            count += int(
                diff.numel()
            )

    return {
        "parameter_l2_displacement":
            float(
                np.sqrt(sq)
            ),

        "parameter_mean_abs_displacement":
            float(
                abs_sum
                / max(count, 1)
            ),
    }


def snapshot_state(
    model: torch.nn.Module,
):
    return {
        k: v.detach().cpu().clone()
        for k, v
        in model.state_dict().items()
    }


def metric_delta(
    after,
    before,
):
    return {
        key:
            float(
                after[key]
                - before[key]
            )
        for key in before
    }


def prediction_flip_rate(
    after,
    before,
):
    return float(
        np.mean(
            np.asarray(after)
            != np.asarray(before)
        )
    )


def parameter_group(name: str) -> str:
    if name.startswith("label_fc."):
        return "label_fc"
    if name.startswith("fc_Z."):
        return "fc_Z"
    if name.startswith("anchor_list."):
        return "anchor_list"
    return "other"


def update_vectors(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    """
    Convert parameter state differences into flat update vectors.

    Each returned vector is CPU float64.
    """
    grouped: dict[str, list[np.ndarray]] = {
        "all": [],
        "label_fc": [],
        "fc_Z": [],
        "anchor_list": [],
        "other": [],
    }

    for name in before:

        if name not in after:
            raise RuntimeError(
                f"missing parameter after update: {name}"
            )

        a = before[name].detach().cpu().numpy()
        b = after[name].detach().cpu().numpy()

        if a.shape != b.shape:
            raise RuntimeError(
                f"parameter shape mismatch: {name}"
            )

        d = (
            b.astype(np.float64, copy=False)
            - a.astype(np.float64, copy=False)
        ).reshape(-1)

        group = parameter_group(name)

        grouped["all"].append(d)
        grouped[group].append(d)

    result = {}

    for group, chunks in grouped.items():

        if len(chunks) == 0:
            result[group] = np.zeros(
                0,
                dtype=np.float64,
            )
        else:
            result[group] = np.concatenate(
                chunks,
                axis=0,
            )

    return result


def cosine_and_projection(
    semantic_vec: np.ndarray,
    native_vec: np.ndarray,
) -> dict[str, float]:
    """
    Compare actual semantic and native parameter updates.

    cosine:
        directional compatibility.

    native_on_semantic_projection:
        scalar projection of native update onto semantic direction.

        <N,S> / ||S||^2

        > 0 : reinforces semantic direction
        < 0 : cancels semantic direction
    """

    if semantic_vec.shape != native_vec.shape:
        raise RuntimeError(
            "update vector shape mismatch"
        )

    ns = float(
        np.linalg.norm(
            semantic_vec
        )
    )

    nn = float(
        np.linalg.norm(
            native_vec
        )
    )

    dot = float(
        np.dot(
            semantic_vec,
            native_vec,
        )
    )

    cosine = (
        dot / (ns * nn)
        if ns > 0 and nn > 0
        else float("nan")
    )

    projection = (
        dot / (ns * ns)
        if ns > 0
        else float("nan")
    )

    return {
        "semantic_update_norm":
            ns,

        "native_update_norm":
            nn,

        "dot_product":
            dot,

        "cosine":
            float(cosine),

        "native_on_semantic_projection":
            float(projection),
    }


def run(seed: int):

    if not torch.cuda.is_available():
        raise RuntimeError(
            "P3-D0-B3 requires CUDA."
        )

    torch.cuda.set_device(0)

    device = torch.device(
        "cuda:0"
    )

    protocol.verify_native_hashes()
    protocol.verify_parent_freeze_hashes()

    formal_root = (
        FORMAL_ROOT
        / f"seed{seed}"
    )

    carrier_state_path = (
        formal_root
        / "carrier/carrier_model_state.pt"
    )

    carrier_npz_path = (
        formal_root
        / "carrier/carrier_pre_gt.npz"
    )

    formal_audit_path = (
        formal_root
        / "seal/pre_gt_audit.json"
    )

    formal_metrics_path = (
        formal_root
        / "formal_metrics.json"
    )

    for p in (
        carrier_state_path,
        carrier_npz_path,
        formal_audit_path,
        formal_metrics_path,
    ):
        if not p.exists():
            raise FileNotFoundError(
                p
            )

    audit = json.loads(
        formal_audit_path.read_text()
    )

    required_false = (
        "GT_loaded_before_seal",
        "full_gt_used_for_training",
        "full_gt_used_for_model_selection",
        "full_gt_used_for_hyperparameter_selection",
    )

    if any(
        audit.get(key) is not False
        for key in required_false
    ):
        raise RuntimeError(
            "invalid formal pre-GT isolation"
        )

    # ---------------------------------------------------------
    # Reconstruct EXACT P3-C0 corrupted training views.
    # ---------------------------------------------------------

    spec = protocol.DATASET_SPECS[
        "ALOI"
    ]

    labels, targets, mask, _ = (
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

    # ---------------------------------------------------------
    # Frozen U/T carrier.
    # ---------------------------------------------------------

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
            "carrier/sample adapter mismatch"
        )

    # ---------------------------------------------------------
    # Rebuild BASE / OURS from EXACT frozen carrier state.
    # ---------------------------------------------------------

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

    set_seed(seed)

    base_model = load_branch()
    ours_model = load_branch()

    carrier_hash = audit[
        "carrier_state_sha256"
    ]

    if (
        c0._state_hash(base_model)
        != carrier_hash
        or
        c0._state_hash(ours_model)
        != carrier_hash
    ):
        raise RuntimeError(
            "carrier replay state mismatch"
        )

    # Same optimizer-reset semantics as P3-C0.
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
        lr=protocol.SEMANTIC_LR,
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

    full_views = [
        torch.from_numpy(view).to(
            device=device,
            dtype=torch.float32,
        )
        for view in views
    ]

    query_rows = torch.as_tensor(
        query_rows_np,
        dtype=torch.long,
        device=device,
    )

    anchor_rows = torch.as_tensor(
        anchor_rows_np,
        dtype=torch.long,
        device=device,
    )

    # Important:
    # U/T are frozen and detached.
    t_rel = torch.as_tensor(
        t_rel_np,
        dtype=torch.float32,
        device=device,
    ).detach()

    u_cycle = torch.as_tensor(
        u_cycle_np,
        dtype=torch.float32,
        device=device,
    ).detach()

    # ---------------------------------------------------------
    # GT is loaded only in this post-seal diagnostic.
    # It is never fed into training.
    # ---------------------------------------------------------

    y_raw = np.asarray(
        scipy.io.loadmat(
            protocol.DATA_ROOT / "ALOI.mat",
            variable_names=("Y",),
        )["Y"]
    ).reshape(-1).astype(
        np.int64
    )

    if (
        np.unique(y_raw).size
        != k
    ):
        raise RuntimeError(
            "unexpected ALOI class count"
        )

    # ---------------------------------------------------------
    # Epoch 0 / shared carrier metrics.
    # ---------------------------------------------------------

    carrier_metrics, carrier_pred = (
        predict_metrics(
            ours_model,
            views,
            adapter.sample_ids,
            y_raw,
            device,
        )
    )

    print()
    print("=" * 105)
    print(
        f"ALOI seed{seed} ACTION DYNAMICS"
    )
    print("=" * 105)

    print(
        "carrier metrics =",
        carrier_metrics,
    )

    rows = []

    semantic_orders = (
        protocol.semantic_orders(
            query_rows.numel(),
            seed,
        )
    )

    if len(
        semantic_orders
    ) != protocol.REFINEMENT_EPOCHS:
        raise RuntimeError(
            "semantic schedule length mismatch"
        )

    previous_ours_metrics = (
        carrier_metrics
    )

    previous_ours_pred = (
        carrier_pred
    )

    # ---------------------------------------------------------
    # Exact 20-epoch replay.
    # ---------------------------------------------------------

    for epoch_index, order in enumerate(
        semantic_orders,
        start=1,
    ):

        # =====================================================
        # PRE_t
        # =====================================================

        pre_metrics = dict(
            previous_ours_metrics
        )

        pre_pred = (
            previous_ours_pred.copy()
        )

        pre_state = snapshot_state(
            ours_model
        )

        # =====================================================
        # Semantic Phase A
        # =====================================================

        semantic_loss = (
            c0._semantic_epoch(
                ours_model,
                semantic_optimizer,
                full_views,
                query_rows,
                anchor_rows,
                t_rel,
                u_cycle,
                order,
                k,
            )
        )

        sem_displacement = (
            state_distance(
                pre_state,
                ours_model,
            )
        )

        semantic_after_state = snapshot_state(
            ours_model
        )

        sem_metrics, sem_pred = (
            predict_metrics(
                ours_model,
                views,
                adapter.sample_ids,
                y_raw,
                device,
            )
        )

        semantic_metric_delta = (
            metric_delta(
                sem_metrics,
                pre_metrics,
            )
        )

        semantic_flip_rate = (
            prediction_flip_rate(
                sem_pred,
                pre_pred,
            )
        )

        # =====================================================
        # Native Phase B
        # =====================================================

        native_start_state = (
            snapshot_state(
                ours_model
            )
        )

        ours_native_loss, ours_order = (
            c0._native_epoch(
                ours_model,
                ours_native_optimizer,
                ours_loader,
                device,
            )
        )

        native_displacement = (
            state_distance(
                native_start_state,
                ours_model,
            )
        )

        native_after_state = snapshot_state(
            ours_model
        )

        semantic_updates = update_vectors(
            pre_state,
            semantic_after_state,
        )

        native_updates = update_vectors(
            native_start_state,
            native_after_state,
        )

        update_compatibility = {}

        for group in (
            "all",
            "label_fc",
            "fc_Z",
            "anchor_list",
        ):
            update_compatibility[group] = (
                cosine_and_projection(
                    semantic_updates[group],
                    native_updates[group],
                )
            )

        ours_metrics, ours_pred = (
            predict_metrics(
                ours_model,
                views,
                adapter.sample_ids,
                y_raw,
                device,
            )
        )

        native_metric_delta = (
            metric_delta(
                ours_metrics,
                sem_metrics,
            )
        )

        native_flip_rate = (
            prediction_flip_rate(
                ours_pred,
                sem_pred,
            )
        )

        # =====================================================
        # Paired BASE native step
        # =====================================================

        base_native_loss, base_order = (
            c0._native_epoch(
                base_model,
                base_optimizer,
                base_loader,
                device,
            )
        )

        if (
            ours_order
            != base_order
        ):
            raise RuntimeError(
                "paired native minibatch order mismatch"
            )

        base_metrics, base_pred = (
            predict_metrics(
                base_model,
                views,
                adapter.sample_ids,
                y_raw,
                device,
            )
        )

        cumulative_method_delta = (
            metric_delta(
                ours_metrics,
                base_metrics,
            )
        )

        row = {
            "epoch":
                epoch_index,

            "semantic_loss":
                float(
                    semantic_loss
                ),

            "ours_native_loss":
                float(
                    ours_native_loss
                ),

            "base_native_loss":
                float(
                    base_native_loss
                ),

            "PRE":
                pre_metrics,

            "AFTER_SEMANTIC":
                sem_metrics,

            "AFTER_NATIVE":
                ours_metrics,

            "BASE_AFTER_NATIVE":
                base_metrics,

            "semantic_immediate_delta":
                semantic_metric_delta,

            "native_recovery_delta":
                native_metric_delta,

            "cumulative_OURS_minus_BASE":
                cumulative_method_delta,

            "semantic_prediction_flip_rate":
                semantic_flip_rate,

            "native_prediction_flip_rate":
                native_flip_rate,

            "semantic_parameter_displacement":
                sem_displacement,

            "native_parameter_displacement":
                native_displacement,

            "update_compatibility":
                update_compatibility,
        }

        rows.append(
            row
        )

        print()
        print(
            f"epoch {epoch_index:02d}"
        )

        print(
            "  semantic ΔACC =",
            semantic_metric_delta[
                "ACC"
            ],
        )

        print(
            "  native recovery ΔACC =",
            native_metric_delta[
                "ACC"
            ],
        )

        print(
            "  cumulative OURS-BASE ACC =",
            cumulative_method_delta[
                "ACC"
            ],
        )

        print(
            "  semantic flip rate =",
            semantic_flip_rate,
        )

        print(
            "  semantic L2 displacement =",
            sem_displacement[
                "parameter_l2_displacement"
            ],
        )

        print(
            "  update cosine ALL =",
            update_compatibility[
                "all"
            ]["cosine"],
        )

        print(
            "  update cosine label_fc =",
            update_compatibility[
                "label_fc"
            ]["cosine"],
        )

        print(
            "  update cosine fc_Z =",
            update_compatibility[
                "fc_Z"
            ]["cosine"],
        )

        print(
            "  update cosine anchor_list =",
            update_compatibility[
                "anchor_list"
            ]["cosine"],
        )

        previous_ours_metrics = (
            ours_metrics
        )

        previous_ours_pred = (
            ours_pred
        )

    # ---------------------------------------------------------
    # Formal final parity.
    # ---------------------------------------------------------

    formal = json.loads(
        formal_metrics_path.read_text()
    )

    final_base = rows[-1][
        "BASE_AFTER_NATIVE"
    ]

    final_ours = rows[-1][
        "AFTER_NATIVE"
    ]

    for metric in (
        "ACC",
        "NMI",
        "Purity",
        "ARI",
    ):

        if abs(
            final_base[metric]
            - formal["BASE"][metric]
        ) > 1e-10:
            raise RuntimeError(
                f"BASE final metric parity failed: {metric}"
            )

        if abs(
            final_ours[metric]
            - formal["OURS"][metric]
        ) > 1e-10:
            raise RuntimeError(
                f"OURS final metric parity failed: {metric}"
            )

    output = {
        "experiment":
            "P3-D0-B3_action_dynamics",

        "dataset":
            "ALOI",

        "training_seed":
            seed,

        "DIAGNOSTIC_ONLY":
            True,

        "NO_MODEL_SELECTION":
            True,

        "NO_HYPERPARAMETER_SELECTION":
            True,

        "frozen_U_T":
            True,

        "carrier_metrics":
            carrier_metrics,

        "formal_final_metrics":
            formal,

        "epochs":
            rows,
    }

    out_dir = (
        OUTPUT_ROOT
        / f"seed{seed}"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        out_dir
        / "action_dynamics.json"
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("=" * 105)
    print(
        "FINAL PARITY = PASS"
    )

    print(
        "formal OURS ACC =",
        formal[
            "OURS"
        ]["ACC"],
    )

    print(
        "replayed OURS ACC =",
        final_ours[
            "ACC"
        ],
    )

    print(
        "formal BASE ACC =",
        formal[
            "BASE"
        ]["ACC"],
    )

    print(
        "replayed BASE ACC =",
        final_base[
            "ACC"
        ],
    )

    print()
    print(
        "P3_D0_B4_UPDATE_COMPATIBILITY_PASS"
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
        required=True,
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

    run(
        args.training_seed
    )


if __name__ == "__main__":
    main()
