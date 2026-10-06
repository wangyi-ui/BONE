"""
P5-A2 fixed-configuration held-out-seed confirmatory runner.

No full GT is loaded in this module.

For each preregistered dataset/seed cell:

1. load the frozen weak-quality realization and sparse labels;
2. train a fresh 100-epoch carrier;
3. construct frozen U_cycle and T_rel;
4. fork identical BASE and OURS branches;
5. OURS receives semantic relation action for frozen E_s epochs;
6. both branches receive paired native structural consolidation
   for exactly frozen H epochs;
7. freeze predictions and seals before GT evaluation.

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

query_rows:
    [Q]

anchor_rows:
    [L]
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
    run_bone_p3_c0_cross_dataset_formal as c0,
)

from experiments.utility_relation import (
    run_bone_p4_b2_uases_confirmatory as p4b2,
)

from experiments.utility_relation.p3_c1_b1_optuna import (
    msrc_adapter,
)


ROOT = Path(
    "/root/autodl-tmp/BONE"
)

PREREG_ROOT = (
    ROOT
    / "experiment_freeze/"
      "bone_p5_a1_fixed_config_unseen_seed_confirmatory_preregistered_20261006"
)

PROTOCOL_PATH = (
    PREREG_ROOT
    / "PROTOCOL.json"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "outputs/bone_p5/"
      "p5_a2_fixed_config_heldout_pre_gt"
)


DATASETS = (
    "MSRC",
    "BDGP",
    "HandWrite",
    "Caltech-5V",
    "MNIST-USPS",
    "COIL20",
    "ALOI",
    "CCV",
)

SEEDS = (
    149,
    163,
    179,
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
    refinement_horizon: int


@dataclass
class Inputs:
    dataset: str
    views: list[np.ndarray]
    adapter: Any
    spec: Any
    labeled_ids: np.ndarray
    labeled_targets: np.ndarray
    k: int
    native_lr: float
    provenance: dict[str, Any]


def sha256_file(
    path: Path,
) -> str:

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

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


def dataset_slug(
    dataset: str,
) -> str:

    return (
        dataset.lower()
        .replace("-", "_")
    )


def load_protocol() -> dict[str, Any]:

    d = json.loads(
        PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    if d["stage"] != "P5-A1":
        raise RuntimeError(
            "INVALID_P5_A1_PROTOCOL"
        )

    if d["training_seeds"] != [
        149,
        163,
        179,
    ]:
        raise RuntimeError(
            "P5_SEED_PROTOCOL_FAILURE"
        )

    if tuple(
        d["datasets"]
    ) != DATASETS:
        raise RuntimeError(
            "P5_DATASET_PROTOCOL_FAILURE"
        )

    return d


def cell_config(
    dataset: str,
    seed: int,
) -> CellConfig:

    if dataset not in DATASETS:
        raise ValueError(
            "UNREGISTERED_P5_DATASET"
        )

    if seed not in SEEDS:
        raise ValueError(
            "UNREGISTERED_P5_SEED"
        )

    protocol = load_protocol()

    cfg = (
        protocol[
            "configs"
        ][dataset]
    )

    return CellConfig(
        dataset=dataset,
        training_seed=int(seed),
        semantic_lr=float(
            cfg[
                "semantic_lr"
            ]
        ),
        semantic_exposure=int(
            cfg[
                "semantic_exposure"
            ]
        ),
        refinement_horizon=int(
            cfg[
                "refinement_horizon"
            ]
        ),
    )


def load_inputs(
    dataset: str,
) -> Inputs:
    """
    Load frozen weak-quality data and sparse supervision only.
    Full clustering GT is never loaded here.
    """

    if dataset == "MSRC":

        (
            views,
            labeled_ids,
            labeled_targets,
            split_path,
            mask_path,
        ) = (
            msrc_adapter
            ._load_frozen_msrc_inputs()
        )

        adapter = (
            b0.SampleIdAdapter(
                np.arange(
                    msrc_adapter.MSRC_SPEC.n,
                    dtype=np.int64,
                )
            )
        )

        spec = (
            msrc_adapter.MSRC_SPEC
        )

        native_lr = float(
            msrc_adapter
            .p2_carrier
            .LR
        )

        provenance = {
            "sparse_split":
                str(split_path),

            "corruption_mask":
                str(mask_path),
        }

    else:

        (
            labeled_ids,
            labeled_targets,
            corruption_mask,
            parent_hashes,
        ) = (
            c0._load_frozen_inputs(
                dataset
            )
        )

        raw_views = (
            c0.load_raw_views(
                dataset
            )
        )

        views, adapter = (
            b0.preprocess_with_raw_ids(
                raw_views,
                corruption_mask,
            )
        )

        spec = (
            c0_protocol
            .DATASET_SPECS[
                dataset
            ]
        )

        native_lr = float(
            c0_protocol.NATIVE_LR
        )

        provenance = {
            key:
                str(value)
            for key, value
            in parent_hashes.items()
        }

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
            "SPARSE_LABEL_SHAPE_FAILURE"
        )

    if not np.array_equal(
        np.sort(
            adapter.sample_ids
        ),
        np.arange(
            spec.n,
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            "RAW_SAMPLE_ID_SET_FAILURE"
        )

    if (
        np.unique(
            adapter.row_positions(
                labeled_ids
            )
        ).size
        != labeled_ids.size
    ):
        raise RuntimeError(
            "SPARSE_LABEL_ROW_MAPPING_FAILURE"
        )

    k = int(
        labeled_targets.max()
    ) + 1

    return Inputs(
        dataset=dataset,
        views=[
            np.ascontiguousarray(
                view,
                dtype=np.float32,
            )
            for view in views
        ],
        adapter=adapter,
        spec=spec,
        labeled_ids=labeled_ids,
        labeled_targets=labeled_targets,
        k=k,
        native_lr=native_lr,
        provenance=provenance,
    )


def new_model_from_state(
    inputs: Inputs,
    seed: int,
    state_path: Path,
    device: torch.device,
):

    model = c0._new_model(
        inputs.views,
        inputs.k,
        inputs.spec,
        seed,
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


def materialize_carrier(
    config: CellConfig,
    inputs: Inputs,
    device: torch.device,
    root: Path,
) -> tuple[
    Path,
    str,
    dict[str, np.ndarray],
]:

    carrier_dir = (
        root
        / "carrier"
    )

    carrier_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    c0.set_seed(
        config.training_seed
    )

    if config.dataset == "MSRC":

        native_data = (
            msrc_adapter
            .p2_carrier
            ._native_data(
                inputs.views
            )
        )

        anchors = c0.amv(
            native_data,
            m=inputs.spec.n_anchor,
            ret_anchor=True,
            max_iter=100,
            seed=config.training_seed,
            info=True,
            info_iter=False,
        )

    else:

        anchors = c0.amv(
            c0._container(
                inputs.views,
                inputs.k,
            ),
            m=inputs.spec.n_anchor,
            ret_anchor=True,
            max_iter=100,
            seed=config.training_seed,
            info=True,
            info_iter=False,
        )

    model = c0._new_model(
        inputs.views,
        inputs.k,
        inputs.spec,
        config.training_seed,
        device,
        anchors,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=inputs.native_lr,
    )

    loader = c0._native_loader(
        inputs.views,
        inputs.adapter,
        config.training_seed,
        inputs.spec.batch_size,
    )

    losses = []
    order_hashes = []

    for epoch in range(
        1,
        101,
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

        losses.append(
            float(loss)
        )

        order_hashes.append(
            order_hash
        )

        if (
            epoch % 20 == 0
            or epoch == 100
        ):

            print(
                f"[{config.dataset} "
                f"seed={config.training_seed}] "
                f"carrier={epoch}/100 "
                f"loss={loss:.8f}",
                flush=True,
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

    # [N,V,K]
    q_views = (
        b0.extract_q_views(
            model,
            inputs.views,
            device=device,
        )
    )

    payload = (
        b0.build_generic_carrier(
            q_views,
            inputs.adapter.sample_ids,
            inputs.labeled_ids,
            inputs.labeled_targets,
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
            "P5_CARRIER_SCHEMA_FAILURE"
        )

    if (
        payload[
            "q_views"
        ].shape
        != (
            inputs.spec.n,
            inputs.spec.v,
            inputs.k,
        )
    ):
        raise RuntimeError(
            "P5_Q_VIEW_SHAPE_FAILURE"
        )

    for name in (
        "U_cycle",
        "query_U_cycle",
        "T_rel",
    ):

        if not np.isfinite(
            payload[name]
        ).all():

            raise RuntimeError(
                f"NONFINITE_CARRIER_ARRAY:{name}"
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
        / "CARRIER_AUDIT.json",
        {
            "stage":
                "P5-A2",

            "dataset":
                config.dataset,

            "training_seed":
                config.training_seed,

            "seed_role":
                "HELD_OUT_CONFIRMATORY",

            "carrier_epochs":
                100,

            "native_lr":
                inputs.native_lr,

            "carrier_state_sha256":
                state_hash,

            "carrier_state_file_sha256":
                sha256_file(
                    state_path
                ),

            "native_order_hashes":
                order_hashes,

            "q_views_shape":
                list(
                    payload[
                        "q_views"
                    ].shape
                ),

            "U_cycle_shape":
                list(
                    payload[
                        "U_cycle"
                    ].shape
                ),

            "T_rel_shape":
                list(
                    payload[
                        "T_rel"
                    ].shape
                ),

            "input_provenance":
                inputs.provenance,

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
            "CUDA_REQUIRED_FOR_P5"
        )

    torch.cuda.set_device(0)

    device = torch.device(
        "cuda:0"
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
            "P5_OUTPUT_ALREADY_EXISTS_FAIL_CLOSED"
        )

    root.mkdir(
        parents=True,
        exist_ok=False,
    )

    inputs = load_inputs(
        dataset
    )

    write_json(
        root
        / "RUN_CONFIG.json",
        {
            "stage":
                "P5-A2",

            "dataset":
                dataset,

            "training_seed":
                seed,

            "semantic_lr":
                config.semantic_lr,

            "semantic_exposure":
                config.semantic_exposure,

            "refinement_horizon":
                config.refinement_horizon,

            "carrier_epochs":
                100,

            "hyperparameter_search":
                False,

            "GT_loaded":
                False,
        },
    )

    (
        state_path,
        carrier_state_hash,
        payload,
    ) = materialize_carrier(
        config,
        inputs,
        device,
        root,
    )

    base_model = (
        new_model_from_state(
            inputs,
            seed,
            state_path,
            device,
        )
    )

    ours_model = (
        new_model_from_state(
            inputs,
            seed,
            state_path,
            device,
        )
    )

    initial_base = (
        c0._state_hash(
            base_model
        )
    )

    initial_ours = (
        c0._state_hash(
            ours_model
        )
    )

    if not (
        initial_base
        == initial_ours
        == carrier_state_hash
    ):
        raise RuntimeError(
            "BASE_OURS_CARRIER_PARITY_FAILURE"
        )

    base_optimizer = optim.Adam(
        base_model.parameters(),
        lr=inputs.native_lr,
    )

    ours_native_optimizer = optim.Adam(
        ours_model.parameters(),
        lr=inputs.native_lr,
    )

    semantic_optimizer = optim.Adam(
        ours_model.parameters(),
        lr=config.semantic_lr,
    )

    base_loader = c0._native_loader(
        inputs.views,
        inputs.adapter,
        seed,
        inputs.spec.batch_size,
    )

    ours_loader = c0._native_loader(
        inputs.views,
        inputs.adapter,
        seed,
        inputs.spec.batch_size,
    )

    # full_views[v]: [N,D_v]
    full_views = [
        torch.from_numpy(
            view
        ).to(
            device=device,
            dtype=torch.float32,
        )
        for view in inputs.views
    ]

    # [Q]
    query_rows = torch.as_tensor(
        payload[
            "query_rows"
        ],
        dtype=torch.long,
        device=device,
    )

    # [L]
    anchor_rows = torch.as_tensor(
        payload[
            "anchor_rows"
        ],
        dtype=torch.long,
        device=device,
    )

    # [Q,L,S]
    t_rel = torch.as_tensor(
        np.asarray(
            payload[
                "T_rel"
            ]
        ).copy(),
        dtype=torch.float32,
        device=device,
    ).detach()

    # [Q,S]
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
            "P5_DETACH_CONTRACT_FAILURE"
        )

    semantic_orders = (
        p4b2
        .confirmatory_semantic_orders(
            query_rows.numel(),
            seed,
            config.semantic_exposure,
        )
    )

    if (
        len(
            semantic_orders
        )
        != config.semantic_exposure
    ):
        raise RuntimeError(
            "SEMANTIC_SCHEDULE_COUNT_FAILURE"
        )

    base_orders = []
    ours_orders = []
    semantic_losses = []
    base_losses = []
    ours_losses = []

    for epoch in range(
        1,
        config.refinement_horizon
        + 1,
    ):

        if (
            epoch
            <= config.semantic_exposure
        ):

            semantic_loss = (
                c0._semantic_epoch(
                    ours_model,
                    semantic_optimizer,
                    full_views,
                    query_rows,
                    anchor_rows,
                    t_rel,
                    query_u,
                    semantic_orders[
                        epoch - 1
                    ],
                    inputs.k,
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

        # Historical exploratory order:
        # OURS semantic -> OURS native -> BASE native.
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

        if not (
            np.isfinite(
                ours_loss
            )
            and np.isfinite(
                base_loss
            )
        ):
            raise FloatingPointError(
                "NONFINITE_REFINEMENT_LOSS"
            )

        if ours_order != base_order:
            raise RuntimeError(
                "PAIRED_NATIVE_ORDER_FAILURE"
            )

        ours_orders.append(
            ours_order
        )

        base_orders.append(
            base_order
        )

        ours_losses.append(
            float(
                ours_loss
            )
        )

        base_losses.append(
            float(
                base_loss
            )
        )

        if (
            epoch % 10 == 0
            or epoch
            == config.refinement_horizon
        ):

            print(
                f"[{dataset} seed={seed}] "
                f"refine={epoch}/"
                f"{config.refinement_horizon} "
                f"BASE={base_loss:.8f} "
                f"OURS={ours_loss:.8f}",
                flush=True,
            )

    if len(
        semantic_losses
    ) != config.semantic_exposure:
        raise RuntimeError(
            "SEMANTIC_EXPOSURE_COUNT_FAILURE"
        )

    if (
        base_orders
        != ours_orders
    ):
        raise RuntimeError(
            "GLOBAL_NATIVE_ORDER_PARITY_FAILURE"
        )

    if len(
        base_orders
    ) != config.refinement_horizon:
        raise RuntimeError(
            "REFINEMENT_HORIZON_FAILURE"
        )

    artifact_paths = []

    for arm, model in (
        (
            "BASE",
            base_model,
        ),
        (
            "OURS",
            ours_model,
        ),
    ):

        arm_dir = (
            root
            / arm
        )

        arm_dir.mkdir(
            exist_ok=False
        )

        state_out = (
            arm_dir
            / "final_model_state.pt"
        )

        c0._save_state(
            model,
            state_out,
        )

        pred_out = (
            arm_dir
            / "pre_gt_predictions.npz"
        )

        np.savez_compressed(
            pred_out,
            sample_ids=
                inputs.adapter.sample_ids.copy(),
            predictions=
                c0._predict(
                    model,
                    inputs.views,
                    device,
                ),
        )

        artifact_paths.extend(
            [
                state_out,
                pred_out,
            ]
        )

    audit_path = (
        root
        / "PRE_GT_AUDIT.json"
    )

    write_json(
        audit_path,
        {
            "stage":
                "P5-A2",

            "role":
                "FIXED_CONFIG_HELD_OUT_CONFIRMATORY",

            "dataset":
                dataset,

            "training_seed":
                seed,

            "seed_role":
                "HELD_OUT_CONFIRMATORY",

            "carrier_epochs":
                100,

            "semantic_lr":
                config.semantic_lr,

            "semantic_exposure":
                config.semantic_exposure,

            "refinement_horizon":
                config.refinement_horizon,

            "native_lr":
                inputs.native_lr,

            "carrier_state_sha256":
                carrier_state_hash,

            "initial_BASE_state_sha256":
                initial_base,

            "initial_OURS_state_sha256":
                initial_ours,

            "BASE_OURS_initial_state_parity":
                (
                    initial_base
                    ==
                    initial_ours
                    ==
                    carrier_state_hash
                ),

            "BASE_OURS_native_order_parity":
                (
                    base_orders
                    ==
                    ours_orders
                ),

            "semantic_step_count":
                len(
                    semantic_losses
                ),

            "base_native_epoch_count":
                len(
                    base_orders
                ),

            "ours_native_epoch_count":
                len(
                    ours_orders
                ),

            "T_rel_shape":
                list(
                    payload[
                        "T_rel"
                    ].shape
                ),

            "U_cycle_shape":
                list(
                    payload[
                        "U_cycle"
                    ].shape
                ),

            "query_U_cycle_shape":
                list(
                    payload[
                        "query_U_cycle"
                    ].shape
                ),

            "detach_contract": {
                "T_rel":
                    True,

                "query_U_cycle":
                    True,

                "anchor_posterior":
                    True,
            },

            "hyperparameter_search":
                False,

            "GT_loaded_before_seal":
                False,

            "full_gt_used_for_training":
                False,

            "full_gt_used_for_selection":
                False,

            "full_gt_used_for_hyperparameter_selection":
                False,

            "ACC_NMI_ARI_computed":
                False,

            "runner_source_sha256":
                sha256_file(
                    Path(__file__)
                ),
        },
    )

    artifact_paths.append(
        audit_path
    )

    config_path = (
        root
        / "RUN_CONFIG.json"
    )

    artifact_paths.append(
        config_path
    )

    carrier_paths = [
        root
        / "carrier/carrier_model_state.pt",

        root
        / "carrier/carrier_pre_gt.npz",

        root
        / "carrier/CARRIER_AUDIT.json",
    ]

    artifact_paths.extend(
        carrier_paths
    )

    seal_path = (
        root
        / "PRE_GT_SEAL.json"
    )

    write_json(
        seal_path,
        {
            "stage":
                "P5-A2",

            "dataset":
                dataset,

            "training_seed":
                seed,

            "BASE_prediction_frozen":
                True,

            "OURS_prediction_frozen":
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
                for path in artifact_paths
            },
        },
    )

    write_json(
        root
        / "CELL_COMPLETE.json",
        {
            "stage":
                "P5-A2",

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
        "P5_CELL_PRE_GT=PASS"
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
        "SEMANTIC_LR =",
        config.semantic_lr
    )

    print(
        "SEMANTIC_EXPOSURE =",
        config.semantic_exposure
    )

    print(
        "REFINEMENT_HORIZON =",
        config.refinement_horizon
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
        choices=SEEDS,
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
            "refusing to launch P5 held-out GPU training without --execute"
        )

    run_cell(
        args.dataset,
        args.training_seed,
        args.output_root,
    )


if __name__ == "__main__":
    main()
