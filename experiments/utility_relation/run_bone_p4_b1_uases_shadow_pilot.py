"""
P4-B1 UASES multi-dataset validity pilot.

PRE-GT runner only.

The runner:
  1. reuses the preregistered frozen carrier,
  2. runs paired BASE / OURS refinement through physical H=100,
  3. freezes the first UASES virtual stop decision,
  4. continues a shadow trajectory to H=100 for later post-seal diagnosis,
  5. writes predictions and audit artifacts without reading full GT.

Tensor contracts
----------------
q_views:
    [N, V, K]

U0:
    [N, S]

query_U_cycle:
    [Q, S]

T_rel:
    [Q, L, S]

checkpoint prediction:
    [N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
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
from experiments.utility_relation.p3_c1_b1_optuna import (
    msrc_adapter,
)


REPO_ROOT = Path("/root/autodl-tmp/BONE")

PREREG_ROOT = (
    REPO_ROOT
    / "experiment_freeze/"
      "bone_p4_b1_uases_multidataset_validity_pilot_preregistered_20261006"
)

PROTOCOL_PATH = (
    PREREG_ROOT
    / "PROTOCOL.json"
)

INPUT_HASH_PATH = (
    PREREG_ROOT
    / "PILOT_INPUT_SHA256.txt"
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs/bone_p4/"
      "p4_b1_uases_shadow_pilot"
)

ARMS = (
    "BASE",
    "OURS",
)

H_GRID = tuple(
    range(
        0,
        es.H_MAX + 1,
        es.CHECKPOINT_INTERVAL,
    )
)

SCORED_H_GRID = H_GRID[1:]


def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

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


@dataclass(frozen=True)
class PilotCell:
    dataset: str
    training_seed: int
    fixed_h: int
    semantic_lr: float
    semantic_exposure: int
    carrier_root: Path


@dataclass(frozen=True)
class ResolvedInputs:
    dataset: str
    views: tuple[np.ndarray, ...]
    adapter: b0.SampleIdAdapter
    spec: Any
    k: int
    native_lr: float


@dataclass
class VirtualStopController:
    """
    Freeze the first UASES virtual stop.

    Once frozen, future shadow checkpoints cannot modify selected_h.
    """

    state: es.EarlyStoppingState = field(
        default_factory=es.EarlyStoppingState
    )

    frozen: bool = False
    selected_h: int | None = None
    stop_h: int | None = None

    def observe(
        self,
        h: int,
        score: float,
    ) -> dict[str, Any]:

        if self.frozen:
            return {
                "state_updated":
                    False,
                "virtual_stop_frozen":
                    True,
                "selected_h":
                    self.selected_h,
                "stop_h":
                    self.stop_h,
            }

        decision = self.state.update(
            h,
            score,
        )

        if decision["should_stop"]:
            self.frozen = True
            self.selected_h = int(
                decision["return_h"]
            )
            self.stop_h = int(h)

        return {
            "state_updated":
                True,
            "virtual_stop_frozen":
                self.frozen,
            "selected_h":
                self.selected_h,
            "stop_h":
                self.stop_h,
            "decision":
                decision,
        }


def load_protocol() -> dict[str, Any]:
    d = json.loads(
        PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    if d["stage"] != "P4-B1":
        raise RuntimeError(
            "INVALID_P4_B1_PROTOCOL_STAGE"
        )

    if (
        d["trajectory_policy"]["mode"]
        != "FULL_SHADOW_TRAJECTORY"
    ):
        raise RuntimeError(
            "INVALID_TRAJECTORY_MODE"
        )

    if (
        int(
            d["uases"]["H_max"]
        )
        != es.H_MAX
    ):
        raise RuntimeError(
            "H_MAX_PROTOCOL_MISMATCH"
        )

    if (
        int(
            d["uases"]["checkpoint_interval"]
        )
        != es.CHECKPOINT_INTERVAL
    ):
        raise RuntimeError(
            "CHECKPOINT_INTERVAL_PROTOCOL_MISMATCH"
        )

    if (
        int(
            d["uases"]["patience_checkpoints"]
        )
        != es.PATIENCE_CHECKPOINTS
    ):
        raise RuntimeError(
            "PATIENCE_PROTOCOL_MISMATCH"
        )

    return d


def pilot_cells() -> tuple[PilotCell, ...]:
    protocol = load_protocol()

    cells = []

    for item in protocol["cells"]:
        cells.append(
            PilotCell(
                dataset=str(
                    item["dataset"]
                ),
                training_seed=int(
                    item["training_seed"]
                ),
                fixed_h=int(
                    item["fixed_h"]
                ),
                semantic_lr=float(
                    item["semantic_lr"]
                ),
                semantic_exposure=int(
                    item["semantic_exposure"]
                ),
                carrier_root=(
                    REPO_ROOT
                    / str(
                        item["carrier_root"]
                    )
                ),
            )
        )

    return tuple(cells)


def get_cell(
    dataset: str,
    training_seed: int,
) -> PilotCell:

    matches = [
        cell
        for cell in pilot_cells()
        if (
            cell.dataset == dataset
            and cell.training_seed
            == training_seed
        )
    ]

    if len(matches) != 1:
        raise ValueError(
            "UNREGISTERED_P4_B1_CELL"
        )

    return matches[0]


def dataset_slug(
    dataset: str,
) -> str:
    return (
        dataset.lower()
        .replace("-", "_")
    )


def verify_preregistered_input_hashes() -> None:
    if not INPUT_HASH_PATH.is_file():
        raise FileNotFoundError(
            INPUT_HASH_PATH
        )

    lines = [
        line.strip()
        for line in INPUT_HASH_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not lines:
        raise RuntimeError(
            "EMPTY_PILOT_INPUT_HASH_FILE"
        )

    for line in lines:
        wanted, raw_path = (
            line.split(
                None,
                1,
            )
        )

        path = Path(
            raw_path.strip()
        )

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        got = sha256_file(
            path
        )

        if got != wanted:
            raise RuntimeError(
                "PILOT_INPUT_HASH_MISMATCH:"
                f"{path}"
            )


def resolve_inputs(
    dataset: str,
) -> ResolvedInputs:
    """
    Recreate exactly the historical formal noisy training input.

    Generic:
        raw -> frozen native permutation -> StandardScaler -> frozen corruption.

    MSRC:
        frozen dedicated P1 corrupted views.

    No full GT is loaded.
    """

    if dataset == "MSRC":
        (
            views,
            _labeled_ids,
            _labeled_targets,
            _split_path,
            _mask_path,
        ) = (
            msrc_adapter
            ._load_frozen_msrc_inputs()
        )

        spec = (
            msrc_adapter.MSRC_SPEC
        )

        adapter = (
            b0.SampleIdAdapter(
                np.arange(
                    spec.n,
                    dtype=np.int64,
                )
            )
        )

        k = int(
            msrc_adapter
            .p1_protocol.K
        )

        native_lr = float(
            msrc_adapter
            .p2_carrier.LR
        )

        resolved_views = tuple(
            np.ascontiguousarray(
                view,
                dtype=np.float32,
            )
            for view in views
        )

    else:
        (
            labeled_ids,
            labeled_targets,
            corruption_mask,
            _parent_hashes,
        ) = c0._load_frozen_inputs(
            dataset
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

        k = int(
            np.asarray(
                labeled_targets
            ).max()
        ) + 1

        if (
            np.unique(
                adapter.row_positions(
                    labeled_ids
                )
            ).size
            != len(labeled_ids)
        ):
            raise RuntimeError(
                "SPARSE_LABEL_ROW_MAPPING_FAILURE"
            )

        native_lr = float(
            c0_protocol.NATIVE_LR
        )

        resolved_views = tuple(
            np.ascontiguousarray(
                view,
                dtype=np.float32,
            )
            for view in views
        )

    if (
        len(resolved_views)
        != int(spec.v)
    ):
        raise RuntimeError(
            "VIEW_COUNT_MISMATCH"
        )

    if not np.array_equal(
        np.sort(
            adapter.sample_ids
        ),
        np.arange(
            int(spec.n),
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            "SAMPLE_ID_SET_MISMATCH"
        )

    return ResolvedInputs(
        dataset=dataset,
        views=resolved_views,
        adapter=adapter,
        spec=spec,
        k=k,
        native_lr=native_lr,
    )


def load_carrier_arrays(
    cell: PilotCell,
    resolved: ResolvedInputs,
) -> dict[str, np.ndarray]:

    path = (
        cell.carrier_root
        / "carrier_pre_gt.npz"
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as z:
        arrays = {
            name:
                np.asarray(
                    z[name]
                )
            for name in z.files
        }

    required = {
        "sample_ids",
        "q_views",
        "U_cycle",
        "query_U_cycle",
        "T_rel",
        "query_rows",
        "anchor_rows",
    }

    if not required.issubset(
        arrays
    ):
        raise RuntimeError(
            "CARRIER_ARRAY_SCHEMA_FAILURE"
        )

    n = int(
        resolved.spec.n
    )

    v = int(
        resolved.spec.v
    )

    k = int(
        resolved.k
    )

    actions, _, _ = (
        b0.action_arrays(v)
    )

    s = len(actions)

    if arrays[
        "q_views"
    ].shape != (
        n,
        v,
        k,
    ):
        raise RuntimeError(
            "CARRIER_Q_VIEW_SHAPE_FAILURE"
        )

    if arrays[
        "U_cycle"
    ].shape != (
        n,
        s,
    ):
        raise RuntimeError(
            "FULL_U_CYCLE_SHAPE_FAILURE"
        )

    if not np.array_equal(
        arrays["sample_ids"],
        resolved.adapter.sample_ids,
    ):
        raise RuntimeError(
            "CARRIER_SAMPLE_ORDER_FAILURE"
        )

    if (
        not np.isfinite(
            arrays["q_views"]
        ).all()
        or not np.isfinite(
            arrays["U_cycle"]
        ).all()
    ):
        raise RuntimeError(
            "NONFINITE_CARRIER_ARRAYS"
        )

    return arrays


def prediction_archive_payload(
    sample_ids: np.ndarray,
    predictions: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Build one exact pre-GT trajectory archive.

    sample_ids:
        [N]

    prediction at each H:
        [N]
    """

    ids = np.asarray(
        sample_ids,
        dtype=np.int64,
    )

    if ids.ndim != 1:
        raise ValueError(
            "sample_ids must have shape [N]"
        )

    if set(
        predictions
    ) != set(H_GRID):
        raise ValueError(
            "prediction checkpoints must equal H_GRID"
        )

    payload = {
        "sample_ids":
            np.ascontiguousarray(
                ids
            )
    }

    for h in H_GRID:
        pred = np.asarray(
            predictions[h],
            dtype=np.int64,
        )

        if pred.shape != ids.shape:
            raise ValueError(
                "prediction must have shape [N]"
            )

        payload[
            f"pred_H{h}"
        ] = (
            np.ascontiguousarray(
                pred
            )
        )

    return payload


def _load_branch(
    resolved: ResolvedInputs,
    cell: PilotCell,
    device: torch.device,
):
    model = c0._new_model(
        resolved.views,
        resolved.k,
        resolved.spec,
        cell.training_seed,
        device,
    )

    model.load_state_dict(
        c0._load_state(
            cell.carrier_root
            / "carrier_model_state.pt",
            device,
        ),
        strict=True,
    )

    return model


def run_cell(
    dataset: str,
    training_seed: int,
    output_root: Path = (
        DEFAULT_OUTPUT_ROOT
    ),
) -> Path:

    cell = get_cell(
        dataset,
        training_seed,
    )

    verify_preregistered_input_hashes()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA_REQUIRED_FOR_P4_B1"
        )

    torch.cuda.set_device(0)

    device = torch.device(
        "cuda:0"
    )

    resolved = resolve_inputs(
        dataset
    )

    arrays = load_carrier_arrays(
        cell,
        resolved,
    )

    root = (
        Path(output_root)
        / dataset_slug(
            dataset
        )
        / f"seed{training_seed}"
    )

    if root.exists():
        raise RuntimeError(
            "OUTPUT_ALREADY_EXISTS_FAIL_CLOSED"
        )

    root.mkdir(
        parents=True,
        exist_ok=False,
    )

    for arm in ARMS:
        (
            root
            / arm
        ).mkdir()

    # --------------------------------------------------------
    # Real frozen full-sample utility U0: [N,S]
    # --------------------------------------------------------
    u0 = es.freeze_utility(
        arrays["U_cycle"]
    )

    if u0.flags.writeable:
        raise RuntimeError(
            "U0_MUST_BE_IMMUTABLE"
        )

    # --------------------------------------------------------
    # BASE / OURS load the exact same frozen carrier.
    # --------------------------------------------------------
    models = {
        arm:
            _load_branch(
                resolved,
                cell,
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

    if len(
        set(
            initial_hashes.values()
        )
    ) != 1:
        raise RuntimeError(
            "BASE_OURS_CARRIER_STATE_PARITY_FAILURE"
        )

    # --------------------------------------------------------
    # Verify exact H0 q_views:
    # live [N,V,K] must reproduce frozen carrier q_views.
    # --------------------------------------------------------
    q0 = {}

    for arm in ARMS:
        q0[arm] = (
            b0.extract_q_views(
                models[arm],
                resolved.views,
                device=device,
            )
        )

        if not np.allclose(
            q0[arm],
            arrays["q_views"],
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"{arm}_H0_Q_POSTERIOR_PARITY_FAILURE"
            )

    if not np.allclose(
        q0["BASE"],
        q0["OURS"],
        rtol=0.0,
        atol=1e-7,
    ):
        raise RuntimeError(
            "BASE_OURS_H0_Q_PARITY_FAILURE"
        )

    # --------------------------------------------------------
    # Paired native schedules.
    # --------------------------------------------------------
    native_optimizers = {
        arm:
            optim.Adam(
                models[arm].parameters(),
                lr=resolved.native_lr,
            )
        for arm in ARMS
    }

    loaders = {
        arm:
            c0._native_loader(
                resolved.views,
                resolved.adapter,
                cell.training_seed,
                int(
                    resolved.spec.batch_size
                ),
            )
        for arm in ARMS
    }

    # --------------------------------------------------------
    # OURS semantic phase tensors.
    #
    # full_views[v]:
    #     [N,D_v]
    #
    # query_rows:
    #     [Q]
    #
    # anchor_rows:
    #     [L]
    #
    # T_rel:
    #     [Q,L,S]
    #
    # query_u:
    #     [Q,S]
    # --------------------------------------------------------
    full_views = [
        torch.from_numpy(
            view
        ).to(
            device=device,
            dtype=torch.float32,
        )
        for view in resolved.views
    ]

    query_rows = torch.as_tensor(
        arrays["query_rows"],
        dtype=torch.long,
        device=device,
    )

    anchor_rows = torch.as_tensor(
        arrays["anchor_rows"],
        dtype=torch.long,
        device=device,
    )

    t_rel = torch.as_tensor(
        arrays["T_rel"].copy(),
        dtype=torch.float32,
        device=device,
    ).detach()

    query_u = torch.as_tensor(
        arrays[
            "query_U_cycle"
        ].copy(),
        dtype=torch.float32,
        device=device,
    ).detach()

    if (
        t_rel.requires_grad
        or query_u.requires_grad
    ):
        raise RuntimeError(
            "SEMANTIC_DETACH_CONTRACT_FAILURE"
        )

    semantic_optimizer = (
        optim.Adam(
            models["OURS"].parameters(),
            lr=cell.semantic_lr,
        )
    )

    semantic_orders = (
        c0_protocol.semantic_orders(
            query_rows.numel(),
            cell.training_seed,
        )
    )

    if (
        len(semantic_orders)
        < cell.semantic_exposure
    ):
        raise RuntimeError(
            "INSUFFICIENT_SEMANTIC_ORDERS"
        )

    # --------------------------------------------------------
    # H0 predictions [N].
    # --------------------------------------------------------
    predictions = {
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
            predictions[arm][0].copy()
        for arm in ARMS
    }

    controllers = {
        arm:
            VirtualStopController()
        for arm in ARMS
    }

    trajectory = {
        arm: []
        for arm in ARMS
    }

    native_orders = {
        arm: []
        for arm in ARMS
    }

    native_losses = {
        arm: []
        for arm in ARMS
    }

    semantic_losses = []

    # --------------------------------------------------------
    # FULL SHADOW TRAJECTORY:
    # physical training always continues to H=100.
    # --------------------------------------------------------
    for epoch in range(
        1,
        es.H_MAX + 1,
    ):

        if (
            epoch
            <= cell.semantic_exposure
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

        # Paired native Phase B.
        for arm in ARMS:
            loss, order_hash = (
                c0._native_epoch(
                    models[arm],
                    native_optimizers[arm],
                    loaders[arm],
                    device,
                )
            )

            if not np.isfinite(loss):
                raise FloatingPointError(
                    f"NONFINITE_NATIVE_LOSS:{arm}"
                )

            native_losses[
                arm
            ].append(
                float(loss)
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
                "BASE_OURS_NATIVE_ORDER_PARITY_FAILURE"
            )

        if (
            epoch
            % es.CHECKPOINT_INTERVAL
            != 0
        ):
            continue

        print(
            f"\n[{dataset} seed={training_seed}] H={epoch}",
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

            result = (
                es.evaluate_checkpoint(
                    q_views,
                    u0,
                    previous_prediction[
                        arm
                    ],
                )
            )

            # prediction: [N]
            predictions[
                arm
            ][epoch] = (
                result[
                    "prediction"
                ].copy()
            )

            control = (
                controllers[
                    arm
                ].observe(
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
                    int(epoch),

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
                    bool(
                        control[
                            "state_updated"
                        ]
                    ),

                "virtual_stop_frozen":
                    bool(
                        control[
                            "virtual_stop_frozen"
                        ]
                    ),

                "selected_h_snapshot":
                    control[
                        "selected_h"
                    ],

                "stop_h_snapshot":
                    control[
                        "stop_h"
                    ],
            }

            if (
                control[
                    "state_updated"
                ]
            ):
                decision = (
                    control[
                        "decision"
                    ]
                )

                record.update(
                    {
                        "improved":
                            bool(
                                decision[
                                    "improved"
                                ]
                            ),

                        "bad_count":
                            int(
                                decision[
                                    "bad_count"
                                ]
                            ),

                        "best_h":
                            int(
                                decision[
                                    "best_h"
                                ]
                            ),

                        "should_stop":
                            bool(
                                decision[
                                    "should_stop"
                                ]
                            ),
                    }
                )

            else:
                record.update(
                    {
                        "improved":
                            None,
                        "bad_count":
                            None,
                        "best_h":
                            controllers[
                                arm
                            ].selected_h,
                        "should_stop":
                            False,
                    }
                )

            trajectory[
                arm
            ].append(
                record
            )

            print(
                f"{arm:4s} "
                f"S_cv={record['S_cv']:.6f} "
                f"S_bal={record['S_bal']:.6f} "
                f"S_stab={record['S_stab']:.6f} "
                f"Q={record['Q_stop']:.6f} "
                f"state_update={record['state_updated']} "
                f"selected={record['selected_h_snapshot']} "
                f"stop={record['stop_h_snapshot']}",
                flush=True,
            )

    # --------------------------------------------------------
    # Final integrity.
    # --------------------------------------------------------
    if (
        len(
            semantic_losses
        )
        != cell.semantic_exposure
    ):
        raise RuntimeError(
            "SEMANTIC_EXPOSURE_COUNT_FAILURE"
        )

    if (
        len(
            native_orders["BASE"]
        )
        != es.H_MAX
        or len(
            native_orders["OURS"]
        )
        != es.H_MAX
    ):
        raise RuntimeError(
            "NATIVE_EPOCH_COUNT_FAILURE"
        )

    if (
        native_orders["BASE"]
        != native_orders["OURS"]
    ):
        raise RuntimeError(
            "GLOBAL_NATIVE_ORDER_PARITY_FAILURE"
        )

    for arm in ARMS:
        controller = (
            controllers[arm]
        )

        if not controller.frozen:
            raise RuntimeError(
                f"UASES_DID_NOT_FREEZE:{arm}"
            )

        if (
            controller.selected_h
            not in SCORED_H_GRID
        ):
            raise RuntimeError(
                f"INVALID_SELECTED_H:{arm}"
            )

        if (
            controller.stop_h
            not in SCORED_H_GRID
        ):
            raise RuntimeError(
                f"INVALID_STOP_H:{arm}"
            )

        if (
            set(
                predictions[arm]
            )
            != set(H_GRID)
        ):
            raise RuntimeError(
                f"CHECKPOINT_PREDICTION_GRID_FAILURE:{arm}"
            )

    # --------------------------------------------------------
    # Save all pre-GT predictions.
    # --------------------------------------------------------
    artifact_paths = []

    for arm in ARMS:
        arm_dir = (
            root
            / arm
        )

        archive_path = (
            arm_dir
            / "checkpoint_predictions.npz"
        )

        np.savez_compressed(
            archive_path,
            **prediction_archive_payload(
                resolved.adapter.sample_ids,
                predictions[arm],
            ),
        )

        artifact_paths.append(
            archive_path
        )

        selected_h = (
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
                    cell.fixed_h
                ].copy(),
            fixed_h=
                np.asarray(
                    [cell.fixed_h],
                    dtype=np.int64,
                ),
        )

        artifact_paths.append(
            fixed_path
        )

    # --------------------------------------------------------
    # Trajectory record contains NO GT metrics.
    # --------------------------------------------------------
    trajectory_path = (
        root
        / "UASES_TRAJECTORY.json"
    )

    trajectory_payload = {
        "stage":
            "P4-B1",

        "dataset":
            dataset,

        "training_seed":
            training_seed,

        "fixed_h":
            cell.fixed_h,

        "semantic_lr":
            cell.semantic_lr,

        "semantic_exposure":
            cell.semantic_exposure,

        "physical_h_max":
            es.H_MAX,

        "checkpoint_interval":
            es.CHECKPOINT_INTERVAL,

        "shadow_mode":
            True,

        "GT_loaded":
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
                trajectory[
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
                trajectory[
                    "OURS"
                ],
        },
    }

    write_json(
        trajectory_path,
        trajectory_payload,
    )

    artifact_paths.append(
        trajectory_path
    )

    # --------------------------------------------------------
    # PRE-GT audit.
    # --------------------------------------------------------
    audit_path = (
        root
        / "PRE_GT_AUDIT.json"
    )

    audit = {
        "stage":
            "P4-B1",

        "dataset":
            dataset,

        "training_seed":
            training_seed,

        "role":
            "UASES_MULTIDATASET_VALIDITY_PILOT_PRE_GT",

        "paper_formal_result":
            False,

        "full_shadow_trajectory":
            True,

        "physical_h_max":
            es.H_MAX,

        "checkpoint_interval":
            es.CHECKPOINT_INTERVAL,

        "fixed_h":
            cell.fixed_h,

        "semantic_lr":
            cell.semantic_lr,

        "semantic_exposure":
            cell.semantic_exposure,

        "carrier_retrained":
            False,

        "carrier_state_sha256":
            initial_hashes["BASE"],

        "BASE_OURS_initial_state_parity":
            (
                initial_hashes[
                    "BASE"
                ]
                ==
                initial_hashes[
                    "OURS"
                ]
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

        "native_epoch_count":
            es.H_MAX,

        "semantic_step_count":
            len(
                semantic_losses
            ),

        "real_frozen_full_U_cycle":
            True,

        "U0_shape":
            list(
                u0.shape
            ),

        "U0_writeable":
            bool(
                u0.flags.writeable
            ),

        "T_rel_shape":
            list(
                arrays[
                    "T_rel"
                ].shape
            ),

        "query_U_cycle_shape":
            list(
                arrays[
                    "query_U_cycle"
                ].shape
            ),

        "U0_recomputed_during_refinement":
            False,

        "same_U0_for_BASE_and_OURS":
            True,

        "same_stopping_rule_for_BASE_and_OURS":
            True,

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

        "post_stop_shadow_can_change_selected_h":
            False,

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

        "native_order_hashes":
            native_orders["BASE"],

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
    }

    write_json(
        audit_path,
        audit,
    )

    artifact_paths.append(
        audit_path
    )

    # --------------------------------------------------------
    # Cell-level PRE-GT seal.
    # --------------------------------------------------------
    seal_path = (
        root
        / "PRE_GT_SEAL.json"
    )

    relative_hashes = {
        str(
            path.relative_to(
                root
            )
        ):
            sha256_file(
                path
            )
        for path in artifact_paths
    }

    seal = {
        "stage":
            "P4-B1",

        "dataset":
            dataset,

        "training_seed":
            training_seed,

        "GT_loaded_before_seal":
            False,

        "selected_h_frozen":
            True,

        "artifacts":
            relative_hashes,
    }

    write_json(
        seal_path,
        seal,
    )

    complete_path = (
        root
        / "CELL_COMPLETE.json"
    )

    write_json(
        complete_path,
        {
            "stage":
                "P4-B1",

            "dataset":
                dataset,

            "training_seed":
                training_seed,

            "PRE_GT_COMPLETE":
                True,

            "PRE_GT_SEAL_SHA256":
                sha256_file(
                    seal_path
                ),

            "GT_READ":
                False,
        },
    )

    print()
    print(
        "CELL_PRE_GT_SEAL=PASS"
    )

    print(
        "BASE_SELECTED_H =",
        controllers[
            "BASE"
        ].selected_h,
    )

    print(
        "BASE_STOP_H =",
        controllers[
            "BASE"
        ].stop_h,
    )

    print(
        "OURS_SELECTED_H =",
        controllers[
            "OURS"
        ].selected_h,
    )

    print(
        "OURS_STOP_H =",
        controllers[
            "OURS"
        ].stop_h,
    )

    print(
        "GT_READ=false"
    )

    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one preregistered P4-B1 "
            "UASES full shadow trajectory."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=(
            "MSRC",
            "HandWrite",
            "MNIST-USPS",
        ),
    )

    parser.add_argument(
        "--training-seed",
        required=True,
        type=int,
        choices=(
            20,
            50,
        ),
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


def main() -> None:
    args = (
        build_parser()
        .parse_args()
    )

    # Exact dataset/seed pair validation.
    get_cell(
        args.dataset,
        args.training_seed,
    )

    if not args.execute:
        raise SystemExit(
            "refusing to launch P4-B1 GPU trajectory "
            "without --execute"
        )

    run_cell(
        args.dataset,
        args.training_seed,
        args.output_root,
    )


if __name__ == "__main__":
    main()
