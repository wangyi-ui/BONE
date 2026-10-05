"""
P3-D0-A1 Matched Clean Control.

Purpose:
    Isolate the effect of weak-quality corruption.

Contract:
    - same raw data
    - same native shared permutation
    - same StandardScaler preprocessing
    - same BONE architecture / anchors / training seed
    - same 100-epoch carrier
    - same branch reload
    - same fresh Adam for +20 native epochs
    - NO 2.5 dB corruption
    - NO semantic action
    - NO full GT before seal
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import optim

from method.amv import amv

from experiments.utility_relation import (
    bone_p3_b0_generic_carrier as b0,
)
from experiments.utility_relation import (
    bone_p3_c0_cross_dataset_formal_protocol as protocol,
)
from experiments.utility_relation import (
    run_bone_p3_c0_cross_dataset_formal as c0,
)
from utils.seed_util import set_seed


OUTPUT_ROOT = Path(
    "outputs/bone_p3/p3_d0_a1_matched_clean"
)


def sha256_array(x: np.ndarray) -> str:
    arr = np.ascontiguousarray(x)
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def build_matched_clean_views(
    raw_views: list[np.ndarray],
    frozen_mask: np.ndarray,
) -> tuple[list[np.ndarray], b0.SampleIdAdapter]:
    """
    Reproduce P3-C0 preprocessing exactly up to the corruption boundary.

    raw view:
        [N, D_v]

    clean standardized view:
        [N, D_v]

    We intentionally call the frozen preprocessing helper once to recover
    its exact SampleIdAdapter / native permutation.  The corrupted arrays
    are discarded.  Clean arrays are then reconstructed using the exact
    StandardScaler expression frozen in B0.
    """

    # corrupted_views are computed only to obtain/verify the frozen adapter.
    # They are NEVER used for clean-control training.
    corrupted_views, adapter = b0.preprocess_with_raw_ids(
        raw_views,
        frozen_mask,
    )

    clean_views = [
        np.ascontiguousarray(
            StandardScaler().fit_transform(
                np.asarray(view, dtype=np.float32)[
                    adapter.sample_ids
                ]
            )
        )
        for view in raw_views
    ]

    if len(clean_views) != len(corrupted_views):
        raise RuntimeError("view-count mismatch")

    changed_mask_cells = 0

    for v, (clean, corrupted) in enumerate(
        zip(clean_views, corrupted_views)
    ):
        if clean.shape != corrupted.shape:
            raise RuntimeError(
                f"view{v}: clean/corrupted shape mismatch"
            )

        if not np.isfinite(clean).all():
            raise FloatingPointError(
                f"view{v}: non-finite clean standardized data"
            )

        omega = frozen_mask[
            adapter.sample_ids, v
        ].astype(bool)

        # P3-A4 contract:
        # corruption must not modify unmasked rows.
        if not np.array_equal(
            clean[~omega],
            corrupted[~omega],
        ):
            raise RuntimeError(
                f"view{v}: corruption changed unmasked rows"
            )

        changed_mask_cells += int(
            np.any(
                clean[omega] != corrupted[omega],
                axis=1,
            ).sum()
        )

    if changed_mask_cells <= 0:
        raise RuntimeError(
            "clean/corrupted diagnostic found no changed masked rows"
        )

    return clean_views, adapter


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=("ALOI", "CCV"),
        required=True,
    )

    parser.add_argument(
        "--training-seed",
        type=int,
        choices=(20, 30, 50),
        required=True,
    )

    parser.add_argument(
        "--execute",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.execute:
        raise SystemExit(
            "Add --execute to run matched clean control."
        )

    dataset_name = args.dataset
    training_seed = args.training_seed

    # ---------------------------------------------------------
    # Frozen provenance gates
    # ---------------------------------------------------------

    protocol.verify_native_hashes()
    protocol.verify_parent_freeze_hashes()

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    device = torch.device("cuda:0")

    spec = protocol.DATASET_SPECS[
        dataset_name
    ]

    labels, targets, mask, parent_hashes = (
        c0._load_frozen_inputs(dataset_name)
    )

    raw_views = c0.load_raw_views(
        dataset_name
    )

    # ---------------------------------------------------------
    # Exact P3-C0 preprocessing, stopping BEFORE corruption.
    # ---------------------------------------------------------

    clean_views, adapter = (
        build_matched_clean_views(
            raw_views,
            mask,
        )
    )

    if not np.array_equal(
        np.sort(adapter.sample_ids),
        np.arange(spec.n),
    ):
        raise RuntimeError(
            "sample-ID permutation mismatch"
        )

    if (
        np.unique(
            adapter.row_positions(labels)
        ).size
        != labels.size
    ):
        raise RuntimeError(
            "sparse-anchor raw-ID mapping mismatch"
        )

    k = int(targets.max()) + 1

    output_dir = (
        OUTPUT_ROOT
        / protocol.dataset_slug(dataset_name)
        / f"seed{training_seed}"
    )

    carrier_dir = output_dir / "carrier"
    base_dir = output_dir / "base"
    seal_dir = output_dir / "seal"

    for path in (
        carrier_dir,
        base_dir,
        seal_dir,
    ):
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ---------------------------------------------------------
    # Carrier: exact same schedule as P3-C0, but CLEAN views.
    # ---------------------------------------------------------

    set_seed(training_seed)

    anchors = amv(
        c0._container(
            clean_views,
            k,
        ),
        m=spec.n_anchor,
        ret_anchor=True,
        max_iter=100,
        seed=training_seed,
        info=True,
        info_iter=False,
    )

    carrier_model = c0._new_model(
        clean_views,
        k,
        spec,
        training_seed,
        device,
        anchors,
    )

    carrier_optimizer = optim.Adam(
        carrier_model.parameters(),
        lr=protocol.NATIVE_LR,
    )

    carrier_loader = c0._native_loader(
        clean_views,
        adapter,
        training_seed,
        spec.batch_size,
    )

    carrier_losses = []

    for _ in range(
        protocol.CARRIER_EPOCHS
    ):
        loss, _ = c0._native_epoch(
            carrier_model,
            carrier_optimizer,
            carrier_loader,
            device,
        )
        carrier_losses.append(loss)

    if not np.isfinite(
        carrier_losses
    ).all():
        raise FloatingPointError(
            "non-finite carrier loss"
        )

    carrier_state_path = (
        carrier_dir
        / "carrier_model_state.pt"
    )

    carrier_file_sha = c0._save_state(
        carrier_model,
        carrier_state_path,
    )

    carrier_state_sha = (
        c0._state_hash(
            carrier_model
        )
    )

    # ---------------------------------------------------------
    # BASE branch:
    # same carrier reload + NEW Adam + exactly 20 native epochs.
    # ---------------------------------------------------------

    base_model = c0._new_model(
        clean_views,
        k,
        spec,
        training_seed,
        device,
    )

    base_model.load_state_dict(
        c0._load_state(
            carrier_state_path,
            device,
        )
    )

    base_initial_sha = (
        c0._state_hash(
            base_model
        )
    )

    if (
        base_initial_sha
        != carrier_state_sha
    ):
        raise RuntimeError(
            "clean BASE carrier reload mismatch"
        )

    # Fresh Adam: matches P3-C0 branch semantics.
    base_optimizer = optim.Adam(
        base_model.parameters(),
        lr=protocol.NATIVE_LR,
    )

    base_loader = c0._native_loader(
        clean_views,
        adapter,
        training_seed,
        spec.batch_size,
    )

    base_losses = []
    native_order_hashes = []

    for _ in range(
        protocol.REFINEMENT_EPOCHS
    ):
        loss, order_hash = (
            c0._native_epoch(
                base_model,
                base_optimizer,
                base_loader,
                device,
            )
        )

        base_losses.append(loss)
        native_order_hashes.append(
            order_hash
        )

    if not np.isfinite(
        base_losses
    ).all():
        raise FloatingPointError(
            "non-finite clean BASE loss"
        )

    final_state_path = (
        base_dir
        / "final_model_state.pt"
    )

    c0._save_state(
        base_model,
        final_state_path,
    )

    predictions = c0._predict(
        base_model,
        clean_views,
        device,
    )

    prediction_path = (
        base_dir
        / "pre_gt_predictions.npz"
    )

    np.savez_compressed(
        prediction_path,
        predictions=predictions,
    )

    # ---------------------------------------------------------
    # PRE-GT audit / seal
    # ---------------------------------------------------------

    preprocessing_hashes = {
        f"view{v}": sha256_array(x)
        for v, x in enumerate(
            clean_views
        )
    }

    audit = {
        "experiment":
            "P3-D0-A1_matched_clean_control",

        "dataset":
            dataset_name,

        "training_seed":
            training_seed,

        "FORMAL_RESULT":
            False,

        "DIAGNOSTIC_ONLY":
            True,

        "clean_control":
            True,

        "weak_quality_corruption_applied":
            False,

        "semantic_action_applied":
            False,

        "GT_loaded_before_seal":
            False,

        "full_gt_used_for_training":
            False,

        "full_gt_used_for_model_selection":
            False,

        "full_gt_used_for_hyperparameter_selection":
            False,

        "carrier_native_epochs":
            protocol.CARRIER_EPOCHS,

        "base_native_epochs":
            protocol.REFINEMENT_EPOCHS,

        "native_lr":
            protocol.NATIVE_LR,

        "carrier_state_sha256":
            carrier_state_sha,

        "initial_BASE_state_sha256":
            base_initial_sha,

        "carrier_state_file_sha256":
            carrier_file_sha,

        "native_order_hashes":
            native_order_hashes,

        "clean_preprocessing_sha256":
            preprocessing_hashes,

        "parent_hashes":
            parent_hashes,
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

    targets_for_seal = [
        carrier_state_path,
        final_state_path,
        prediction_path,
        audit_path,
    ]

    seal = {
        str(path.relative_to(output_dir)):
            protocol.sha256_file(path)
        for path in targets_for_seal
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

    print("=" * 90)
    print(
        "P3-D0-A1 MATCHED CLEAN CONTROL COMPLETE"
    )
    print("=" * 90)

    print(
        json.dumps(
            {
                "dataset":
                    dataset_name,

                "training_seed":
                    training_seed,

                "carrier_state_sha256":
                    carrier_state_sha,

                "base_initial_sha256":
                    base_initial_sha,

                "paired_state_equal":
                    (
                        carrier_state_sha
                        == base_initial_sha
                    ),

                "carrier_epochs":
                    protocol.CARRIER_EPOCHS,

                "base_continuation_epochs":
                    protocol.REFINEMENT_EPOCHS,

                "corruption_applied":
                    False,

                "semantic_action_applied":
                    False,

                "GT_loaded_before_seal":
                    False,

                "root":
                    str(output_dir),
            },
            indent=2,
        )
    )

    print()
    print(
        "P3_D0_A1_MATCHED_CLEAN_RUN_PASS"
    )


if __name__ == "__main__":
    main()
