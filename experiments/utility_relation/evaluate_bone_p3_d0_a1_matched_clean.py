"""
P3-D0-A1 Matched Clean Control
Post-seal GT-only diagnostic evaluator.

This evaluator:
    1. verifies the pre-GT seal first;
    2. only then loads full GT;
    3. evaluates matched CLEAN BASE;
    4. compares it with frozen P3-C0 corrupted BASE / OURS.

DIAGNOSTIC ONLY.
No model selection.
No hyperparameter selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from experiments.utility_relation import (
    bone_p3_b0_generic_carrier as b0,
)
from experiments.utility_relation import (
    bone_p3_c0_cross_dataset_formal_protocol as protocol,
)
from experiments.utility_relation import (
    run_bone_p3_c0_cross_dataset_formal as c0,
)
from utils.data_util import (
    get_dataset,
    D_ALOI,
    D_CCV,
)
from utils.method_util import get_cluster_metrics


CLEAN_ROOT = Path(
    "outputs/bone_p3/p3_d0_a1_matched_clean"
)

FORMAL_ROOT = Path(
    "outputs/bone_p3/p3_c0_formal"
)


DATASET_CONST = {
    "ALOI": D_ALOI,
    "CCV": D_CCV,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def verify_clean_seal(
    root: Path,
) -> None:

    seal_path = (
        root
        / "seal/pre_gt_seal.json"
    )

    if not seal_path.exists():
        raise FileNotFoundError(
            f"missing seal: {seal_path}"
        )

    seal = json.loads(
        seal_path.read_text()
    )

    for rel, expected in seal.items():

        path = root / rel

        if not path.exists():
            raise FileNotFoundError(
                f"sealed artifact missing: {path}"
            )

        observed = sha256_file(path)

        if observed != expected:
            raise RuntimeError(
                f"seal hash mismatch: {path}\n"
                f"expected={expected}\n"
                f"observed={observed}"
            )

    print(
        "matched clean pre-GT seal = PASS"
    )


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
            "Add --execute to evaluate."
        )

    dataset_name = args.dataset
    seed = args.training_seed

    slug = protocol.dataset_slug(
        dataset_name
    )

    clean_root = (
        CLEAN_ROOT
        / slug
        / f"seed{seed}"
    )

    formal_root = (
        FORMAL_ROOT
        / slug
        / f"seed{seed}"
    )

    print("=" * 92)
    print(
        "P3-D0-A1 MATCHED CLEAN POST-SEAL GT DIAGNOSTIC"
    )
    print("=" * 92)

    print(
        "dataset       =",
        dataset_name,
    )

    print(
        "training seed =",
        seed,
    )

    # ========================================================
    # STEP 1:
    # Seal verification MUST happen before full GT load.
    # ========================================================

    verify_clean_seal(
        clean_root
    )

    audit_path = (
        clean_root
        / "seal/pre_gt_audit.json"
    )

    audit = json.loads(
        audit_path.read_text()
    )

    assert (
        audit["GT_loaded_before_seal"]
        is False
    )

    assert (
        audit["full_gt_used_for_training"]
        is False
    )

    assert (
        audit[
            "full_gt_used_for_model_selection"
        ]
        is False
    )

    assert (
        audit[
            "full_gt_used_for_hyperparameter_selection"
        ]
        is False
    )

    print(
        "pre-GT isolation audit = PASS"
    )

    # ========================================================
    # STEP 2:
    # Reconstruct exact P3-C0 training-order adapter.
    # Still no GT needed here.
    # ========================================================

    _, _, mask, _ = (
        c0._load_frozen_inputs(
            dataset_name
        )
    )

    raw_views = c0.load_raw_views(
        dataset_name
    )

    _, adapter = (
        b0.preprocess_with_raw_ids(
            raw_views,
            mask,
        )
    )

    # ========================================================
    # STEP 3:
    # Only NOW load FULL GT.
    # ========================================================

    dataset = get_dataset(
        data_name=DATASET_CONST[
            dataset_name
        ]
    )

    y_raw = np.asarray(
        dataset.label
    ).reshape(-1)

    if y_raw.shape[0] != adapter.sample_ids.shape[0]:
        raise RuntimeError(
            "GT/sample-ID length mismatch"
        )

    # predictions are saved in P3 native training order:
    #
    # [N]
    #
    # therefore align full GT to the exact same order.
    y_true = y_raw[
        adapter.sample_ids
    ]

    print(
        "full GT loaded after seal = PASS"
    )

    # ========================================================
    # STEP 4:
    # Matched CLEAN BASE prediction
    # ========================================================

    pred_path = (
        clean_root
        / "base/pre_gt_predictions.npz"
    )

    pred_npz = np.load(
        pred_path,
        allow_pickle=False,
    )

    if "predictions" not in pred_npz:
        raise KeyError(
            "missing predictions key"
        )

    y_pred = np.asarray(
        pred_npz["predictions"]
    ).reshape(-1)

    if y_pred.shape != y_true.shape:
        raise RuntimeError(
            "prediction / GT shape mismatch"
        )

    clean_acc, clean_nmi, clean_purity = (
        get_cluster_metrics(
            y_true,
            y_pred,
            nmi=False,
        )
    )

    clean_ari = adjusted_rand_score(
        y_true,
        y_pred,
    )

    clean_metrics = {
        "ACC": float(clean_acc),
        "ARI": float(clean_ari),
        "NMI": float(clean_nmi),
        "Purity": float(clean_purity),
    }

    # ========================================================
    # STEP 5:
    # Read already-frozen P3-C0 metrics.
    # No retraining.
    # ========================================================

    formal_metrics_path = (
        formal_root
        / "formal_metrics.json"
    )

    if not formal_metrics_path.exists():
        raise FileNotFoundError(
            formal_metrics_path
        )

    formal = json.loads(
        formal_metrics_path.read_text()
    )

    corrupted_base = {
        metric: float(
            formal["BASE"][metric]
        )
        for metric in (
            "ACC",
            "ARI",
            "NMI",
            "Purity",
        )
    }

    ours = {
        metric: float(
            formal["OURS"][metric]
        )
        for metric in (
            "ACC",
            "ARI",
            "NMI",
            "Purity",
        )
    }

    # ========================================================
    # STEP 6:
    # Causal diagnostic decomposition
    # ========================================================

    noise_effect = {
        metric:
            corrupted_base[metric]
            - clean_metrics[metric]
        for metric in clean_metrics
    }

    method_effect = {
        metric:
            ours[metric]
            - corrupted_base[metric]
        for metric in clean_metrics
    }

    total_effect = {
        metric:
            ours[metric]
            - clean_metrics[metric]
        for metric in clean_metrics
    }

    result = {
        "experiment":
            "P3-D0-A1_matched_clean_GT_diagnostic",

        "dataset":
            dataset_name,

        "training_seed":
            seed,

        "DIAGNOSTIC_ONLY":
            True,

        "GT_loaded_after_seal":
            True,

        "matched_clean_BASE":
            clean_metrics,

        "frozen_corrupted_BASE":
            corrupted_base,

        "frozen_corrupted_OURS":
            ours,

        "delta_noise_corruptedBASE_minus_cleanBASE":
            noise_effect,

        "delta_method_OURS_minus_corruptedBASE":
            method_effect,

        "delta_total_OURS_minus_cleanBASE":
            total_effect,
    }

    output_path = (
        clean_root
        / "matched_clean_diagnostic_metrics.json"
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("=" * 92)
    print(
        "MATCHED CLEAN CAUSAL DECOMPOSITION"
    )
    print("=" * 92)

    print()
    print(
        "CLEAN BASE:"
    )
    print(
        json.dumps(
            clean_metrics,
            indent=2,
        )
    )

    print()
    print(
        "CORRUPTED BASE:"
    )
    print(
        json.dumps(
            corrupted_base,
            indent=2,
        )
    )

    print()
    print(
        "OURS:"
    )
    print(
        json.dumps(
            ours,
            indent=2,
        )
    )

    print()
    print(
        "NOISE EFFECT "
        "(Corrupted BASE - Clean BASE):"
    )
    print(
        json.dumps(
            noise_effect,
            indent=2,
        )
    )

    print()
    print(
        "METHOD EFFECT "
        "(OURS - Corrupted BASE):"
    )
    print(
        json.dumps(
            method_effect,
            indent=2,
        )
    )

    print()
    print(
        "TOTAL EFFECT "
        "(OURS - Clean BASE):"
    )
    print(
        json.dumps(
            total_effect,
            indent=2,
        )
    )

    print()
    print(
        "P3_D0_A1_MATCHED_CLEAN_GT_DIAGNOSTIC_PASS"
    )


if __name__ == "__main__":
    main()
