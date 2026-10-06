"""
P4-B2 unseen-seed UASES confirmatory pre-GT runner.

For each preregistered dataset/seed cell:

1. train a fresh 100-epoch carrier using the unseen training seed;
2. construct frozen carrier U_cycle / T_rel;
3. fork identical BASE / OURS branches;
4. run the frozen UASES rule;
5. freeze each arm's first legal stopping decision;
6. continue only as long as required to obtain:
       - both stopping decisions;
       - the preregistered fixed-H comparator;
7. seal predictions before any GT access.

No full GT is loaded by this module.

Tensor contracts
----------------
q_views:
    [N,V,K]

U_cycle:
    [N,S]

query_U_cycle:
    [Q,S]

T_rel:
    [Q,L,S]

prediction:
    [N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    p4_b0_early_stopping as es,
)
from experiments.utility_relation import (
    run_bone_p3_c0_cross_dataset_formal as c0,
)
from experiments.utility_relation import (
    run_bone_p4_b1_uases_shadow_pilot as b1_runner,
)
from experiments.utility_relation.p3_c1_b1_optuna import (
    msrc_adapter,
)


ROOT = Path("/root/autodl-tmp/BONE")

PREREG_ROOT = (
    ROOT
    / "experiment_freeze/"
      "bone_p4_b2_a1_uases_unseen_seed_confirmatory_preregistered_20261006"
)

PROTOCOL_PATH = (
    PREREG_ROOT
    / "PROTOCOL.json"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "outputs/bone_p4/"
      "p4_b2_uases_confirmatory"
)

CONFIRMATORY_SEEDS = (
    113,
    127,
    139,
)

DATASETS = (
    "MSRC",
    "HandWrite",
    "MNIST-USPS",
)

ARMS = (
    "BASE",
    "OURS",
)


@dataclass(frozen=True)
class CellConfig:
    dataset: str
    training_seed: int
    semantic_lr: float
    semantic_exposure: int
    fixed_h: int


def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def write_json(
    path: Path,
    payload: Any,
) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def load_protocol() -> dict[str, Any]:
    d = json.loads(
        PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    if d["stage"] != "P4-B2-A1":
        raise RuntimeError(
            "INVALID_CONFIRMATORY_PROTOCOL"
        )

    if d["training_seeds"] != [
        113,
        127,
        139,
    ]:
        raise RuntimeError(
            "CONFIRMATORY_SEED_PROTOCOL_MISMATCH"
        )

    if d["datasets"] != [
        "MSRC",
        "HandWrite",
        "MNIST-USPS",
    ]:
        raise RuntimeError(
            "CONFIRMATORY_DATASET_PROTOCOL_MISMATCH"
        )

    u = d["uases"]

    if (
        int(u["checkpoint_interval"])
        != es.CHECKPOINT_INTERVAL
        or int(
            u["H_max_safety_cap"]
        )
        != es.H_MAX
        or int(
            u["patience_checkpoints"]
        )
        != es.PATIENCE_CHECKPOINTS
        or abs(
            float(
                u["improvement_epsilon"]
            )
            - es.IMPROVEMENT_EPSILON
        )
        > 1e-15
    ):
        raise RuntimeError(
            "UASES_PROTOCOL_MISMATCH"
        )

    return d


def cell_config(
    dataset: str,
    seed: int,
) -> CellConfig:

    if dataset not in DATASETS:
        raise ValueError(
            "UNREGISTERED_CONFIRMATORY_DATASET"
        )

    if seed not in CONFIRMATORY_SEEDS:
        raise ValueError(
            "UNREGISTERED_CONFIRMATORY_SEED"
        )

    protocol = load_protocol()

    config = (
        protocol[
            "dataset_configs"
        ][dataset]
    )

    return CellConfig(
        dataset=dataset,
        training_seed=int(seed),
        semantic_lr=float(
            config[
                "semantic_lr"
            ]
        ),
        semantic_exposure=int(
            config[
                "semantic_exposure"
            ]
        ),
        fixed_h=int(
            config[
                "fixed_h"
            ]
        ),
    )


def dataset_slug(
    dataset: str,
) -> str:
    return (
        dataset.lower()
        .replace("-", "_")
    )


def load_sparse_supervision(
    dataset: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns sparse supervision only.

    labeled_ids:
        [L]

    labeled_targets:
        [L]
    """

    if dataset == "MSRC":
        (
            _views,
            labeled_ids,
            labeled_targets,
            _split_path,
            _mask_path,
        ) = (
            msrc_adapter
            ._load_frozen_msrc_inputs()
        )

    else:
        (
            labeled_ids,
            labeled_targets,
            _mask,
            _hashes,
        ) = (
            c0._load_frozen_inputs(
                dataset
            )
        )

    labeled_ids = np.asarray(
        labeled_ids,
        dtype=np.int64,
    )

    labeled_targets = np.asarray(
        labeled_targets,
        dtype=np.int64,
    )

    if (
        labeled_ids.ndim != 1
        or labeled_targets.shape
        != labeled_ids.shape
    ):
        raise RuntimeError(
            "SPARSE_SUPERVISION_SHAPE_FAILURE"
        )

    return (
        labeled_ids,
        labeled_targets,
    )


def build_fresh_carrier(
    config: CellConfig,
    resolved: b1_runner.ResolvedInputs,
    device: torch.device,
    carrier_dir: Path,
) -> tuple[
    Path,
    str,
    dict[str, np.ndarray],
    list[str],
]:
    """
    Train fresh unseen-seed carrier.

    This is the only carrier construction path in P4-B2.
    """

    dataset = config.dataset
    seed = config.training_seed

    labeled_ids, labeled_targets = (
        load_sparse_supervision(
            dataset
        )
    )

    if (
        np.unique(
            resolved.adapter.row_positions(
                labeled_ids
            )
        ).size
        != labeled_ids.size
    ):
        raise RuntimeError(
            "SPARSE_LABEL_ROW_MAPPING_FAILURE"
        )

    carrier_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    c0.set_seed(
        seed
    )

    if dataset == "MSRC":
        native_data = (
            msrc_adapter
            .p2_carrier
            ._native_data(
                resolved.views
            )
        )

        anchors = c0.amv(
            native_data,
            m=resolved.spec.n_anchor,
            ret_anchor=True,
            max_iter=100,
            seed=seed,
            info=True,
            info_iter=False,
        )

    else:
        anchors = c0.amv(
            c0._container(
                resolved.views,
                resolved.k,
            ),
            m=resolved.spec.n_anchor,
            ret_anchor=True,
            max_iter=100,
            seed=seed,
            info=True,
            info_iter=False,
        )

    model = c0._new_model(
        resolved.views,
        resolved.k,
        resolved.spec,
        seed,
        device,
        anchors,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=resolved.native_lr,
    )

    loader = c0._native_loader(
        resolved.views,
        resolved.adapter,
        seed,
        resolved.spec.batch_size,
    )

    carrier_losses = []
    carrier_orders = []

    for epoch in range(
        1,
        c0_protocol.CARRIER_EPOCHS
        + 1,
    ):
        loss, order_hash = (
            c0._native_epoch(
                model,
                optimizer,
                loader,
                device,
            )
        )

        if not np.isfinite(
            loss
        ):
            raise FloatingPointError(
                "NONFINITE_CARRIER_LOSS"
            )

        carrier_losses.append(
            float(loss)
        )

        carrier_orders.append(
            order_hash
        )

        if (
            epoch % 20 == 0
            or epoch
            == c0_protocol.CARRIER_EPOCHS
        ):
            print(
                f"[{dataset} seed={seed}] "
                f"carrier epoch="
                f"{epoch}/"
                f"{c0_protocol.CARRIER_EPOCHS} "
                f"loss={loss:.8f}",
                flush=True,
            )

    if len(
        carrier_losses
    ) != c0_protocol.CARRIER_EPOCHS:
        raise RuntimeError(
            "CARRIER_EPOCH_COUNT_FAILURE"
        )

    state_path = (
        carrier_dir
        / "carrier_model_state.pt"
    )

    c0._save_state(
        model,
        state_path,
    )

    state_hash = (
        c0._state_hash(
            model
        )
    )

    # q_views: [N,V,K]
    q_views = (
        b0.extract_q_views(
            model,
            resolved.views,
            device=device,
        )
    )

    # Build frozen U/T artifact from q + sparse labels only.
    payload = (
        b0.build_generic_carrier(
            q_views,
            resolved.adapter.sample_ids,
            labeled_ids,
            labeled_targets,
        )
    )

    required = {
        "sample_ids",
        "query_rows",
        "anchor_rows",
        "q_views",
        "U_cycle",
        "query_U_cycle",
        "T_rel",
    }

    if not required.issubset(
        payload
    ):
        raise RuntimeError(
            "CARRIER_PAYLOAD_SCHEMA_FAILURE"
        )

    if not np.array_equal(
        payload["sample_ids"],
        resolved.adapter.sample_ids,
    ):
        raise RuntimeError(
            "CARRIER_SAMPLE_ID_FAILURE"
        )

    if (
        payload["q_views"].shape
        != (
            resolved.spec.n,
            resolved.spec.v,
            resolved.k,
        )
    ):
        raise RuntimeError(
            "CARRIER_Q_SHAPE_FAILURE"
        )

    if (
        not np.isfinite(
            payload["U_cycle"]
        ).all()
        or float(
            np.asarray(
                payload[
                    "U_cycle"
                ]
            ).sum()
        )
        <= 0.0
    ):
        raise RuntimeError(
            "INVALID_CARRIER_U_CYCLE"
        )

    carrier_npz = (
        carrier_dir
        / "carrier_pre_gt.npz"
    )

    np.savez_compressed(
        carrier_npz,
        **payload,
    )

    write_json(
        carrier_dir
        / "carrier_audit.json",
        {
            "stage":
                "P4-B2",

            "dataset":
                dataset,

            "training_seed":
                seed,

            "seed_role":
                "UNSEEN_CONFIRMATORY",

            "carrier_epochs":
                c0_protocol.CARRIER_EPOCHS,

            "native_lr":
                resolved.native_lr,

            "carrier_state_sha256":
                state_hash,

            "carrier_state_file_sha256":
                sha256_file(
                    state_path
                ),

            "carrier_native_order_hashes":
                carrier_orders,

            "U_cycle_shape":
                list(
                    payload[
                        "U_cycle"
                    ].shape
                ),

            "GT_loaded":
                False,

            "full_gt_used_for_training":
                False,

            "formal_metrics_computed":
                False,
        },
    )

    return (
        state_path,
        state_hash,
        payload,
        carrier_orders,
    )


def load_branch(
    resolved: b1_runner.ResolvedInputs,
    config: CellConfig,
    state_path: Path,
    device: torch.device,
):
    model = c0._new_model(
        resolved.views,
        resolved.k,
        resolved.spec,
        config.training_seed,
        device,
    )

    model.load_state_dict(
        c0._load_state(
            state_path,
            device,
        ),
        strict=True,
    )

    return model


def should_end_cell(
    h: int,
    fixed_h: int,
    base_controller: b1_runner.VirtualStopController,
    ours_controller: b1_runner.VirtualStopController,
) -> bool:
    """
    Confirmatory prospective termination rule.

    Cell can end only when:
      BASE stop frozen,
      OURS stop frozen,
      fixed-H checkpoint materialized.
    """

    return bool(
        base_controller.frozen
        and ours_controller.frozen
        and h >= fixed_h
    )



def confirmatory_semantic_orders(
    query_count: int,
    seed: int,
    exposure: int,
) -> tuple[np.ndarray, ...]:
    """
    Extend the exact frozen P3-C0 semantic-order RNG stream
    to preregistered unseen confirmatory seeds.

    One RNG stream is initialized once:

        rng = np.random.default_rng(training_seed)

    and semantic orders are drawn sequentially:

        order_e = rng.permutation(Q)

    Tensor/index contract
    ---------------------
    each order:
        [Q]

    No GT, U_cycle, T_rel, predictions, losses, or metrics
    enter schedule generation.
    """

    query_count = int(
        query_count
    )

    seed = int(
        seed
    )

    exposure = int(
        exposure
    )

    if query_count <= 0:
        raise ValueError(
            "query_count must be positive"
        )

    if exposure <= 0:
        raise ValueError(
            "exposure must be positive"
        )

    rng = np.random.default_rng(
        seed
    )

    return tuple(
        np.asarray(
            rng.permutation(
                query_count
            ),
            dtype=np.int64,
        )
        for _ in range(
            exposure
        )
    )



def run_cell(
    dataset: str,
    seed: int,
    output_root: Path = (
        DEFAULT_OUTPUT_ROOT
    ),
) -> Path:

    config = cell_config(
        dataset,
        seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA_REQUIRED_FOR_P4_B2"
        )

    torch.cuda.set_device(0)

    device = torch.device(
        "cuda:0"
    )

    resolved = (
        b1_runner.resolve_inputs(
            dataset
        )
    )

    root = (
        Path(output_root)
        / dataset_slug(
            dataset
        )
        / f"seed{seed}"
    )

    if root.exists():
        raise RuntimeError(
            "OUTPUT_ALREADY_EXISTS_FAIL_CLOSED"
        )

    root.mkdir(
        parents=True,
        exist_ok=False,
    )

    carrier_dir = (
        root
        / "carrier"
    )

    (
        state_path,
        carrier_state_hash,
        payload,
        carrier_orders,
    ) = build_fresh_carrier(
        config,
        resolved,
        device,
        carrier_dir,
    )

    # U0: [N,S], immutable NumPy stop-gradient.
    u0 = es.freeze_utility(
        payload["U_cycle"]
    )

    models = {
        arm:
            load_branch(
                resolved,
                config,
                state_path,
                device,
            )
        for arm in ARMS
    }

    initial_hashes = {
        arm:
            c0._state_hash(
                models[arm]
            )
        for arm in ARMS
    }

    if (
        initial_hashes["BASE"]
        != carrier_state_hash
        or initial_hashes["OURS"]
        != carrier_state_hash
    ):
        raise RuntimeError(
            "BASE_OURS_SHARED_CARRIER_FAILURE"
        )

    # H0 q_views: [N,V,K]
    q0 = {
        arm:
            b0.extract_q_views(
                models[arm],
                resolved.views,
                device=device,
            )
        for arm in ARMS
    }

    for arm in ARMS:
        if not np.allclose(
            q0[arm],
            payload["q_views"],
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"{arm}_H0_Q_PARITY_FAILURE"
            )

    native_optimizers = {
        arm:
            optim.Adam(
                models[arm].parameters(),
                lr=resolved.native_lr,
            )
        for arm in ARMS
    }

    native_loaders = {
        arm:
            c0._native_loader(
                resolved.views,
                resolved.adapter,
                seed,
                resolved.spec.batch_size,
            )
        for arm in ARMS
    }

    semantic_optimizer = (
        optim.Adam(
            models["OURS"].parameters(),
            lr=config.semantic_lr,
        )
    )

    # full_views[v]: [N,D_v]
    full_views = [
        torch.from_numpy(
            view
        ).to(
            device=device,
            dtype=torch.float32,
        )
        for view in resolved.views
    ]

    # query_rows: [Q]
    query_rows = torch.as_tensor(
        payload["query_rows"],
        dtype=torch.long,
        device=device,
    )

    # anchor_rows: [L]
    anchor_rows = torch.as_tensor(
        payload["anchor_rows"],
        dtype=torch.long,
        device=device,
    )

    # T_rel: [Q,L,S]
    t_rel = torch.as_tensor(
        np.asarray(
            payload["T_rel"]
        ).copy(),
        dtype=torch.float32,
        device=device,
    ).detach()

    # query_u: [Q,S]
    query_u = torch.as_tensor(
        np.asarray(
            payload[
                "query_U_cycle"
            ]
        ).copy(),
        dtype=torch.float32,
        device=device,
    ).detach()

    if (
        t_rel.requires_grad
        or query_u.requires_grad
    ):
        raise RuntimeError(
            "DETACH_CONTRACT_FAILURE"
        )

    semantic_orders = (
        confirmatory_semantic_orders(
            query_rows.numel(),
            seed,
            config.semantic_exposure,
        )
    )

    if (
        len(semantic_orders)
        != config.semantic_exposure
    ):
        raise RuntimeError(
            "SEMANTIC_ORDER_COUNT_FAILURE"
        )

    predictions: dict[
        str,
        dict[int, np.ndarray],
    ] = {
        arm: {
            0:
                es.hard_prediction(
                    q0[arm]
                )
        }
        for arm in ARMS
    }

    previous_prediction = {
        arm:
            predictions[
                arm
            ][0].copy()
        for arm in ARMS
    }

    controllers = {
        arm:
            b1_runner
            .VirtualStopController()
        for arm in ARMS
    }

    trajectories = {
        arm: []
        for arm in ARMS
    }

    native_orders = {
        arm: []
        for arm in ARMS
    }

    semantic_losses = []

    end_h = None

    for epoch in range(
        1,
        es.H_MAX + 1,
    ):

        # OURS Phase A: semantic relation purification.
        if (
            epoch
            <= config.semantic_exposure
        ):
            semantic_loss = (
                c0._semantic_epoch(
                    models["OURS"],
                    semantic_optimizer,
                    full_views,
                    query_rows,
                    anchor_rows,
                    t_rel,
                    query_u,
                    semantic_orders[
                        epoch - 1
                    ],
                    resolved.k,
                )
            )

            if not np.isfinite(
                semantic_loss
            ):
                raise FloatingPointError(
                    "NONFINITE_SEMANTIC_LOSS"
                )

            semantic_losses.append(
                float(
                    semantic_loss
                )
            )

        # BASE / OURS always remain paired until whole-cell termination.
        for arm in ARMS:
            loss, order_hash = (
                c0._native_epoch(
                    models[arm],
                    native_optimizers[arm],
                    native_loaders[arm],
                    device,
                )
            )

            if not np.isfinite(
                loss
            ):
                raise FloatingPointError(
                    f"NONFINITE_NATIVE_LOSS:{arm}"
                )

            native_orders[
                arm
            ].append(
                order_hash
            )

        if (
            native_orders["BASE"][-1]
            != native_orders["OURS"][-1]
        ):
            raise RuntimeError(
                "NATIVE_ORDER_PARITY_FAILURE"
            )

        if (
            epoch
            % es.CHECKPOINT_INTERVAL
            != 0
        ):
            continue

        print(
            f"\n[{dataset} seed={seed}] H={epoch}",
            flush=True,
        )

        for arm in ARMS:

            # q_views: [N,V,K]
            q_views = (
                b0.extract_q_views(
                    models[arm],
                    resolved.views,
                    device=device,
                )
            )

            prediction = (
                es.hard_prediction(
                    q_views
                )
            )

            predictions[
                arm
            ][epoch] = (
                prediction.copy()
            )

            controller = (
                controllers[arm]
            )

            if not controller.frozen:

                result = (
                    es.evaluate_checkpoint(
                        q_views,
                        u0,
                        previous_prediction[
                            arm
                        ],
                    )
                )

                control = (
                    controller.observe(
                        epoch,
                        result[
                            "Q_stop"
                        ],
                    )
                )

                previous_prediction[
                    arm
                ] = (
                    result[
                        "prediction"
                    ].copy()
                )

                record = {
                    "H":
                        epoch,

                    "S_cv":
                        float(
                            result["S_cv"]
                        ),

                    "S_bal":
                        float(
                            result["S_bal"]
                        ),

                    "S_stab":
                        float(
                            result["S_stab"]
                        ),

                    "Q_stop":
                        float(
                            result["Q_stop"]
                        ),

                    "state_updated":
                        True,

                    "selected_h":
                        controller.selected_h,

                    "stop_h":
                        controller.stop_h,
                }

            else:

                record = {
                    "H":
                        epoch,

                    "S_cv":
                        None,

                    "S_bal":
                        None,

                    "S_stab":
                        None,

                    "Q_stop":
                        None,

                    "state_updated":
                        False,

                    "selected_h":
                        controller.selected_h,

                    "stop_h":
                        controller.stop_h,
                }

            trajectories[
                arm
            ].append(
                record
            )

            print(
                f"{arm:4s} "
                f"updated={record['state_updated']} "
                f"H*={record['selected_h']} "
                f"stop={record['stop_h']}",
                flush=True,
            )

        if should_end_cell(
            epoch,
            config.fixed_h,
            controllers["BASE"],
            controllers["OURS"],
        ):
            end_h = epoch
            break

    if end_h is None:
        raise RuntimeError(
            "CONFIRMATORY_CELL_DID_NOT_TERMINATE"
        )

    if end_h > es.H_MAX:
        raise RuntimeError(
            "CONFIRMATORY_HMAX_FAILURE"
        )

    if len(
        semantic_losses
    ) != config.semantic_exposure:
        raise RuntimeError(
            "SEMANTIC_EXPOSURE_FAILURE"
        )

    if (
        native_orders["BASE"]
        != native_orders["OURS"]
    ):
        raise RuntimeError(
            "GLOBAL_NATIVE_ORDER_PARITY_FAILURE"
        )

    for arm in ARMS:
        if not controllers[
            arm
        ].frozen:
            raise RuntimeError(
                f"STOP_NOT_FROZEN:{arm}"
            )

        selected_h = (
            controllers[
                arm
            ].selected_h
        )

        if selected_h not in (
            predictions[
                arm
            ]
        ):
            raise RuntimeError(
                f"SELECTED_PREDICTION_MISSING:{arm}"
            )

        if config.fixed_h not in (
            predictions[
                arm
            ]
        ):
            raise RuntimeError(
                f"FIXED_PREDICTION_MISSING:{arm}"
            )

    # --------------------------------------------------------
    # PRE-GT artifacts.
    # --------------------------------------------------------

    artifact_paths = []

    for arm in ARMS:

        arm_dir = (
            root
            / arm
        )

        arm_dir.mkdir(
            exist_ok=False
        )

        selected_h = int(
            controllers[
                arm
            ].selected_h
        )

        selected_path = (
            arm_dir
            / "selected_pre_gt_predictions.npz"
        )

        np.savez_compressed(
            selected_path,
            sample_ids=
                resolved.adapter.sample_ids.copy(),
            predictions=
                predictions[
                    arm
                ][
                    selected_h
                ].copy(),
            selected_h=
                np.asarray(
                    [selected_h],
                    dtype=np.int64,
                ),
        )

        artifact_paths.append(
            selected_path
        )

        fixed_path = (
            arm_dir
            / "fixed_h_pre_gt_predictions.npz"
        )

        np.savez_compressed(
            fixed_path,
            sample_ids=
                resolved.adapter.sample_ids.copy(),
            predictions=
                predictions[
                    arm
                ][
                    config.fixed_h
                ].copy(),
            fixed_h=
                np.asarray(
                    [config.fixed_h],
                    dtype=np.int64,
                ),
        )

        artifact_paths.append(
            fixed_path
        )

    trajectory_path = (
        root
        / "UASES_CONFIRMATORY_TRAJECTORY.json"
    )

    write_json(
        trajectory_path,
        {
            "stage":
                "P4-B2",

            "dataset":
                dataset,

            "training_seed":
                seed,

            "seed_role":
                "UNSEEN_CONFIRMATORY",

            "fixed_h":
                config.fixed_h,

            "semantic_lr":
                config.semantic_lr,

            "semantic_exposure":
                config.semantic_exposure,

            "cell_end_h":
                end_h,

            "H_max":
                es.H_MAX,

            "full_H100_shadow":
                False,

            "BASE": {
                "selected_h":
                    controllers[
                        "BASE"
                    ].selected_h,

                "stop_h":
                    controllers[
                        "BASE"
                    ].stop_h,

                "trajectory":
                    trajectories[
                        "BASE"
                    ],
            },

            "OURS": {
                "selected_h":
                    controllers[
                        "OURS"
                    ].selected_h,

                "stop_h":
                    controllers[
                        "OURS"
                    ].stop_h,

                "trajectory":
                    trajectories[
                        "OURS"
                    ],
            },

            "GT_loaded":
                False,
        },
    )

    artifact_paths.append(
        trajectory_path
    )

    audit_path = (
        root
        / "PRE_GT_AUDIT.json"
    )

    write_json(
        audit_path,
        {
            "stage":
                "P4-B2",

            "dataset":
                dataset,

            "training_seed":
                seed,

            "seed_role":
                "UNSEEN_CONFIRMATORY",

            "carrier_reused":
                False,

            "fresh_carrier_trained":
                True,

            "carrier_epochs":
                c0_protocol.CARRIER_EPOCHS,

            "carrier_state_sha256":
                carrier_state_hash,

            "carrier_native_order_hashes":
                carrier_orders,

            "BASE_OURS_initial_state_parity":
                (
                    initial_hashes[
                        "BASE"
                    ]
                    ==
                    initial_hashes[
                        "OURS"
                    ]
                    ==
                    carrier_state_hash
                ),

            "BASE_OURS_native_order_parity":
                (
                    native_orders[
                        "BASE"
                    ]
                    ==
                    native_orders[
                        "OURS"
                    ]
                ),

            "cell_end_h":
                end_h,

            "fixed_h":
                config.fixed_h,

            "BASE_selected_h":
                controllers[
                    "BASE"
                ].selected_h,

            "BASE_stop_h":
                controllers[
                    "BASE"
                ].stop_h,

            "OURS_selected_h":
                controllers[
                    "OURS"
                ].selected_h,

            "OURS_stop_h":
                controllers[
                    "OURS"
                ].stop_h,

            "U0_shape":
                list(
                    u0.shape
                ),

            "U0_writeable":
                bool(
                    u0.flags.writeable
                ),

            "same_U0_for_BASE_and_OURS":
                True,

            "U0_recomputed_during_refinement":
                False,

            "same_stopping_rule_for_BASE_and_OURS":
                True,

            "semantic_step_count":
                len(
                    semantic_losses
                ),

            "GT_loaded_before_seal":
                False,

            "full_gt_used_for_training":
                False,

            "full_gt_used_for_stopping":
                False,

            "full_gt_used_for_selection":
                False,

            "ACC_NMI_ARI_computed":
                False,

            "runner_source_sha256":
                sha256_file(
                    Path(__file__)
                ),

            "uases_source_sha256":
                sha256_file(
                    Path(
                        es.__file__
                    )
                ),
        },
    )

    artifact_paths.append(
        audit_path
    )

    seal_path = (
        root
        / "PRE_GT_SEAL.json"
    )

    write_json(
        seal_path,
        {
            "stage":
                "P4-B2",

            "dataset":
                dataset,

            "training_seed":
                seed,

            "selected_h_frozen":
                True,

            "GT_loaded_before_seal":
                False,

            "artifacts": {
                str(
                    path.relative_to(
                        root
                    )
                ):
                    sha256_file(
                        path
                    )
                for path
                in artifact_paths
            },
        },
    )

    write_json(
        root
        / "CELL_COMPLETE.json",
        {
            "stage":
                "P4-B2",

            "dataset":
                dataset,

            "training_seed":
                seed,

            "PRE_GT_COMPLETE":
                True,

            "GT_READ":
                False,

            "PRE_GT_SEAL_SHA256":
                sha256_file(
                    seal_path
                ),
        },
    )

    print()
    print(
        "CONFIRMATORY_CELL_PRE_GT=PASS"
    )

    print(
        "DATASET =",
        dataset
    )

    print(
        "SEED =",
        seed
    )

    print(
        "CELL_END_H =",
        end_h
    )

    print(
        "BASE_SELECTED_H =",
        controllers[
            "BASE"
        ].selected_h
    )

    print(
        "BASE_STOP_H =",
        controllers[
            "BASE"
        ].stop_h
    )

    print(
        "OURS_SELECTED_H =",
        controllers[
            "OURS"
        ].selected_h
    )

    print(
        "OURS_STOP_H =",
        controllers[
            "OURS"
        ].stop_h
    )

    print(
        "GT_READ=false"
    )

    return root


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS,
    )

    parser.add_argument(
        "--training-seed",
        required=True,
        type=int,
        choices=CONFIRMATORY_SEEDS,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--execute",
        action="store_true",
    )

    return parser


def main():
    args = (
        build_parser()
        .parse_args()
    )

    cell_config(
        args.dataset,
        args.training_seed,
    )

    if not args.execute:
        raise SystemExit(
            "refusing to launch P4-B2 confirmatory GPU training without --execute"
        )

    run_cell(
        args.dataset,
        args.training_seed,
        args.output_root,
    )


if __name__ == "__main__":
    main()
