from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

from experiments.utility_relation import (
    run_bone_p3_c0_cross_dataset_formal as c0,
)
from experiments.utility_relation import (
    bone_p3_b0_generic_carrier as b0,
)

from experiments.utility_relation.p3_c1_b1_optuna import (
    msrc_adapter,
)

from experiments.utility_relation.p3_c4_a0_utility_action_alignment.protocol import (
    DATASETS,
    SEEDS,
    verify_frozen_sources,
)

from experiments.utility_relation.p3_c4_a0_utility_action_alignment.summarize_diagnostic import (
    summarize,
)


ROOT = Path("/root/autodl-tmp/BONE")

FORMAL_ROOT = (
    ROOT
    / "outputs/bone_p3/p3_c1_b1_final_formal"
)

FULL_ROOT = (
    ROOT
    / "outputs/bone_p3/"
      "p3_c4_a0_utility_action_alignment_full24_v1"
)

LOG_ROOT = (
    ROOT
    / "logs/bone_p3/"
      "p3_c4_a0_utility_action_alignment/full24_v1"
)

WORK_ROOT = FULL_ROOT / "_working_inputs"

MODULE = (
    "experiments.utility_relation."
    "p3_c4_a0_utility_action_alignment."
    "run_diagnostic"
)

EXPECTED_DATASETS = (
    "HandWrite",
    "BDGP",
    "MSRC",
    "COIL20",
    "Caltech-5V",
    "MNIST-USPS",
    "ALOI",
    "CCV",
)

EXPECTED_SEEDS = (
    20,
    30,
    50,
)

SLUG = {
    "HandWrite": "handwrite",
    "BDGP": "bdgp",
    "MSRC": "msrc",
    "COIL20": "coil20",
    "Caltech-5V": "caltech_5v",
    "MNIST-USPS": "mnist_usps",
    "ALOI": "aloi",
    "CCV": "ccv",
}

REQUIRED_NUMERIC = (
    "A_TRUE",
    "A_UNIFORM",
    "A_SHUFFLE",
    "Delta_TRUE_UNIFORM",
    "Delta_TRUE_SHUFFLE",
    "DeltaAlignment_Q5_Q1",
    "Spearman_quintile_alignment",
    "Spearman_U_semantic_margin",
    "HIGH_U_LOW_MARGIN_CONFLICT_RATE",
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def fail(message: str) -> None:
    raise RuntimeError(
        "FAIL-CLOSED: " + message
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def logical_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)

    h = hashlib.sha256()

    h.update(
        str(values.dtype).encode("utf-8")
    )

    h.update(
        repr(tuple(values.shape)).encode("utf-8")
    )

    h.update(
        values.tobytes(order="C")
    )

    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(
        path.read_text()
    )


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )

    tmp.replace(path)


def check_protocol_manifest() -> None:
    if tuple(DATASETS) != EXPECTED_DATASETS:
        fail(
            f"dataset manifest changed: {DATASETS}"
        )

    if tuple(SEEDS) != EXPECTED_SEEDS:
        fail(
            f"seed manifest changed: {SEEDS}"
        )

    if len(DATASETS) * len(SEEDS) != 24:
        fail("manifest is not 24 cells")

    print(
        "FULL24_PROTOCOL_MANIFEST=PASS",
        flush=True,
    )


def carrier_paths(
    dataset: str,
    seed: int,
) -> dict[str, Path]:

    root = (
        FORMAL_ROOT
        / SLUG[dataset]
        / f"seed{seed}"
    )

    return {
        "root": root,
        "carrier": (
            root
            / "carrier/carrier_pre_gt.npz"
        ),
        "state": (
            root
            / "carrier/carrier_model_state.pt"
        ),
        "carrier_complete": (
            root
            / "carrier/CARRIER_COMPLETE.json"
        ),
        "audit": (
            root
            / "PRE_GT_AUDIT.json"
        ),
        "seal": (
            root
            / "PRE_GT_SEAL.json"
        ),
    }


def preflight_all_24() -> None:
    missing = []

    for dataset in DATASETS:
        for seed in SEEDS:
            paths = carrier_paths(
                dataset,
                seed,
            )

            for name in (
                "carrier",
                "state",
                "carrier_complete",
                "audit",
                "seal",
            ):
                if not paths[name].is_file():
                    missing.append(
                        str(paths[name])
                    )

    if missing:
        print(
            "MISSING_FROZEN_INPUTS:",
            flush=True,
        )

        for path in missing:
            print(
                " ",
                path,
                flush=True,
            )

        fail(
            f"{len(missing)} frozen inputs missing"
        )

    print(
        "ALL_24_FROZEN_CARRIER_INPUTS_EXIST=PASS",
        flush=True,
    )


def audit_pre_gt_flags(
    audit_path: Path,
) -> None:

    audit = load_json(
        audit_path
    )

    false_keys = (
        "GT_loaded_before_seal",
        "full_gt_used_for_training",
        "full_gt_used_for_model_selection",
        "full_gt_used_for_hyperparameter_selection",
        "ACC_NMI_ARI_computed",
    )

    for key in false_keys:
        if key in audit:
            if audit[key] not in (
                False,
                0,
                None,
            ):
                fail(
                    f"{audit_path}: "
                    f"{key} is not false"
                )


def validate_carrier_against_dataset(
    dataset: str,
    seed: int,
    adapter,
    labeled_ids: np.ndarray,
    labeled_targets: np.ndarray,
) -> dict:

    paths = carrier_paths(
        dataset,
        seed,
    )

    audit_pre_gt_flags(
        paths["audit"]
    )

    with np.load(
        paths["carrier"],
        allow_pickle=False,
    ) as z:

        required = {
            "query_rows",
            "anchor_rows",
            "T_rel",
            "U_cycle",
            "query_U_cycle",
        }

        if not required.issubset(
            set(z.files)
        ):
            fail(
                f"{dataset}/seed{seed}: "
                "carrier schema mismatch"
            )

        query_rows = np.asarray(
            z["query_rows"],
            dtype=np.int64,
        )

        anchor_rows = np.asarray(
            z["anchor_rows"],
            dtype=np.int64,
        )

        t_rel = np.asarray(
            z["T_rel"]
        )

        u_full = np.asarray(
            z["U_cycle"]
        )

        query_u = np.asarray(
            z["query_U_cycle"]
        )

        anchor_expected = np.asarray(
            adapter.row_positions(
                labeled_ids
            ),
            dtype=np.int64,
        )

        query_expected = np.setdiff1d(
            np.arange(
                adapter.sample_ids.size,
                dtype=np.int64,
            ),
            anchor_expected,
        )

        if not np.array_equal(
            anchor_rows,
            anchor_expected,
        ):
            fail(
                f"{dataset}/seed{seed}: "
                "anchor row mismatch"
            )

        if not np.array_equal(
            query_rows,
            query_expected,
        ):
            fail(
                f"{dataset}/seed{seed}: "
                "query row mismatch"
            )

        if "sample_ids" in z.files:
            if not np.array_equal(
                np.asarray(
                    z["sample_ids"],
                    dtype=np.int64,
                ),
                np.asarray(
                    adapter.sample_ids,
                    dtype=np.int64,
                ),
            ):
                fail(
                    f"{dataset}/seed{seed}: "
                    "sample ID mismatch"
                )

        if "labeled_targets" in z.files:
            if not np.array_equal(
                np.asarray(
                    z["labeled_targets"],
                    dtype=np.int64,
                ),
                labeled_targets,
            ):
                fail(
                    f"{dataset}/seed{seed}: "
                    "sparse targets mismatch"
                )

        if not np.array_equal(
            query_u,
            u_full[query_rows],
        ):
            fail(
                f"{dataset}/seed{seed}: "
                "query_U projection mismatch"
            )

        if t_rel.shape != (
            query_rows.size,
            labeled_targets.size,
            query_u.shape[1],
        ):
            fail(
                f"{dataset}/seed{seed}: "
                f"T_rel shape {t_rel.shape}"
            )

        if not np.isfinite(
            t_rel
        ).all():
            fail(
                f"{dataset}/seed{seed}: "
                "nonfinite T_rel"
            )

        if not np.isfinite(
            query_u
        ).all():
            fail(
                f"{dataset}/seed{seed}: "
                "nonfinite query_U"
            )

        if np.any(
            query_u < 0
        ) or np.any(
            query_u > 1
        ):
            fail(
                f"{dataset}/seed{seed}: "
                "U outside [0,1]"
            )

        return {
            "query_count": int(
                query_rows.size
            ),
            "anchor_count": int(
                anchor_rows.size
            ),
            "action_count": int(
                query_u.shape[1]
            ),
            "carrier_sha256":
                sha256_file(
                    paths["carrier"]
                ),
            "state_sha256":
                sha256_file(
                    paths["state"]
                ),
            "audit_sha256":
                sha256_file(
                    paths["audit"]
                ),
            "seal_sha256":
                sha256_file(
                    paths["seal"]
                ),
        }


def materialize_dataset_inputs(
    dataset: str,
) -> tuple[Path, Path, dict]:

    work = (
        WORK_ROOT
        / SLUG[dataset]
    )

    if work.exists():
        shutil.rmtree(
            work
        )

    work.mkdir(
        parents=True,
        exist_ok=False,
    )

    views_path = (
        work
        / "views_pre_gt.npz"
    )

    labels_path = (
        work
        / "sparse_anchor_labels.npy"
    )

    if dataset == "MSRC":

        (
            views,
            labeled_ids,
            labeled_targets,
            split_path,
            mask_path,
        ) = msrc_adapter._load_frozen_msrc_inputs()

        adapter = b0.SampleIdAdapter(
            np.arange(
                msrc_adapter.MSRC_SPEC.n,
                dtype=np.int64,
            )
        )

        parent_hashes = {
            "p3_a3_split":
                sha256_file(
                    Path(split_path)
                ),
            "p3_a4_mask":
                sha256_file(
                    Path(mask_path)
                ),
        }

    else:

        (
            labeled_ids,
            labeled_targets,
            corruption_mask,
            parent_hashes,
        ) = c0._load_frozen_inputs(
            dataset
        )

        raw_views = c0.load_raw_views(
            dataset
        )

        views, adapter = (
            b0.preprocess_with_raw_ids(
                raw_views,
                corruption_mask,
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

    if not np.array_equal(
        np.sort(
            adapter.sample_ids
        ),
        np.arange(
            adapter.sample_ids.size
        ),
    ):
        fail(
            f"{dataset}: sample ID set mismatch"
        )

    if np.unique(
        adapter.row_positions(
            labeled_ids
        )
    ).size != labeled_ids.size:
        fail(
            f"{dataset}: sparse row mapping mismatch"
        )

    if labeled_targets.ndim != 1:
        fail(
            f"{dataset}: sparse target ndim"
        )

    if labeled_ids.shape != (
        labeled_targets.shape
    ):
        fail(
            f"{dataset}: sparse ID/target count mismatch"
        )

    if labeled_targets.size >= (
        adapter.sample_ids.size
    ):
        fail(
            f"{dataset}: sparse set not sparse"
        )

    prepared = {}
    view_logical_hashes = {}

    n = int(
        adapter.sample_ids.size
    )

    for index, view in enumerate(
        views
    ):
        arr = np.ascontiguousarray(
            view,
            dtype=np.float32,
        )

        if arr.ndim != 2:
            fail(
                f"{dataset}: "
                f"view{index} ndim"
            )

        if arr.shape[0] != n:
            fail(
                f"{dataset}: "
                f"view{index} N mismatch"
            )

        if not np.isfinite(
            arr
        ).all():
            fail(
                f"{dataset}: "
                f"view{index} nonfinite"
            )

        key = f"view_{index}"

        prepared[key] = arr

        view_logical_hashes[key] = {
            "shape": list(
                arr.shape
            ),
            "dtype": str(
                arr.dtype
            ),
            "logical_sha256":
                logical_sha256(
                    arr
                ),
        }

    np.savez_compressed(
        views_path,
        **prepared,
    )

    np.save(
        labels_path,
        labeled_targets,
        allow_pickle=False,
    )

    seed_provenance = {}

    for seed in SEEDS:
        seed_provenance[
            str(seed)
        ] = validate_carrier_against_dataset(
            dataset,
            seed,
            adapter,
            labeled_ids,
            labeled_targets,
        )

    manifest = {
        "dataset": dataset,
        "sample_count": n,
        "view_count": len(
            prepared
        ),
        "sparse_anchor_count": int(
            labeled_targets.size
        ),
        "class_count": int(
            labeled_targets.max()
        ) + 1,
        "views": view_logical_hashes,
        "sparse_anchor_labels": {
            "shape": list(
                labeled_targets.shape
            ),
            "dtype": str(
                labeled_targets.dtype
            ),
            "logical_sha256":
                logical_sha256(
                    labeled_targets
                ),
        },
        "parent_hashes": {
            str(k): str(v)
            for k, v
            in parent_hashes.items()
        },
        "seed_carriers":
            seed_provenance,
        "full_GT_loaded": False,
    }

    manifest_path = (
        FULL_ROOT
        / "input_manifests"
        / f"{SLUG[dataset]}.json"
    )

    write_json(
        manifest_path,
        manifest,
    )

    print(
        f"{dataset}: "
        "PRE_GT_INPUT_MATERIALIZATION=PASS "
        f"N={n} "
        f"V={len(prepared)} "
        f"anchors={labeled_targets.size}",
        flush=True,
    )

    return (
        views_path,
        labels_path,
        manifest,
    )


def validate_result(
    dataset: str,
    seed: int,
    path: Path,
) -> dict:

    d = load_json(
        path
    )

    if d.get(
        "stage"
    ) != "P3-C4-A0":
        fail(
            f"{dataset}/seed{seed}: "
            "stage mismatch"
        )

    if d.get(
        "dataset"
    ) != dataset:
        fail(
            f"{dataset}/seed{seed}: "
            "dataset mismatch"
        )

    if int(
        d.get(
            "training_seed",
            -1,
        )
    ) != seed:
        fail(
            f"{dataset}/seed{seed}: "
            "seed mismatch"
        )

    if d.get(
        "status"
    ) != "VALID":
        fail(
            f"{dataset}/seed{seed}: "
            "status != VALID"
        )

    if d.get(
        "DIAGNOSTIC_ONLY"
    ) is not True:
        fail(
            f"{dataset}/seed{seed}: "
            "not diagnostic only"
        )

    if d.get(
        "full_GT_loaded"
    ) is not False:
        fail(
            f"{dataset}/seed{seed}: "
            "GT loaded"
        )

    if d.get(
        "formal_clustering_metrics_computed"
    ) is not False:
        fail(
            f"{dataset}/seed{seed}: "
            "formal metrics computed"
        )

    state_integrity = d.get(
        "model_state_sha256_before_after"
    )

    if not (
        isinstance(
            state_integrity,
            str,
        )
        and HEX64.fullmatch(
            state_integrity
        )
    ):
        fail(
            f"{dataset}/seed{seed}: "
            "model-state integrity invalid"
        )

    if d.get(
        "detach_contract"
    ) != {
        "U_cycle": True,
        "T_rel": True,
        "anchor_posterior": True,
    }:
        fail(
            f"{dataset}/seed{seed}: "
            "detach contract failed"
        )

    for key in REQUIRED_NUMERIC:

        if key not in d:
            fail(
                f"{dataset}/seed{seed}: "
                f"missing {key}"
            )

        if d[key] is None:
            fail(
                f"{dataset}/seed{seed}: "
                f"{key} null"
            )

        if not math.isfinite(
            float(
                d[key]
            )
        ):
            fail(
                f"{dataset}/seed{seed}: "
                f"{key} nonfinite"
            )

    invalid = d.get(
        "invalid_zero_norm_count"
    )

    if not isinstance(
        invalid,
        dict,
    ):
        fail(
            f"{dataset}/seed{seed}: "
            "missing zero-norm audit"
        )

    if any(
        int(v) != 0
        for v in invalid.values()
    ):
        fail(
            f"{dataset}/seed{seed}: "
            "zero-gradient invalid batch"
        )

    valid = d.get(
        "valid_batch_count",
        {},
    )

    for arm in (
        "TRUE",
        "UNIFORM",
        "SHUFFLE",
    ):
        if int(
            valid.get(
                arm,
                0,
            )
        ) <= 0:
            fail(
                f"{dataset}/seed{seed}: "
                f"no valid {arm} batches"
            )

    return d


def run_cell(
    dataset: str,
    seed: int,
    views_path: Path,
    labels_path: Path,
    dataset_manifest: dict,
) -> dict:

    slug = SLUG[
        dataset
    ]

    output = (
        FULL_ROOT
        / slug
        / f"seed{seed}"
        / "diagnostic.json"
    )

    log = (
        LOG_ROOT
        / f"{slug}_seed{seed}.log"
    )

    if output.is_file():

        result = validate_result(
            dataset,
            seed,
            output,
        )

        print(
            f"{dataset}/seed{seed}: "
            "VALID_EXISTING_RESULT -> SKIP",
            flush=True,
        )

        return result

    if log.exists():
        fail(
            f"{dataset}/seed{seed}: "
            "log exists without valid result; "
            "possible interrupted cell"
        )

    paths = carrier_paths(
        dataset,
        seed,
    )

    command = [
        sys.executable,
        "-m",
        MODULE,
        "--dataset",
        dataset,
        "--training-seed",
        str(seed),
        "--carrier-npz",
        str(
            paths["carrier"]
        ),
        "--model-state",
        str(
            paths["state"]
        ),
        "--views-npz",
        str(
            views_path
        ),
        "--sparse-anchor-labels",
        str(
            labels_path
        ),
        "--output-root",
        str(
            FULL_ROOT
        ),
        "--device",
        "cuda:0",
        "--batch-size",
        "128",
    ]

    log.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "",
        flush=True,
    )

    print(
        "=" * 76,
        flush=True,
    )

    print(
        f"RUN {dataset} / seed{seed}",
        flush=True,
    )

    print(
        "=" * 76,
        flush=True,
    )

    with log.open(
        "w"
    ) as stream:

        completed = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
            env=os.environ.copy(),
        )

    if completed.returncode != 0:

        print(
            f"FAILED LOG = {log}",
            flush=True,
        )

        fail(
            f"{dataset}/seed{seed}: "
            f"runner exit={completed.returncode}"
        )

    if not output.is_file():
        fail(
            f"{dataset}/seed{seed}: "
            "runner returned 0 but result absent"
        )

    result = validate_result(
        dataset,
        seed,
        output,
    )

    cell_root = (
        FULL_ROOT
        / slug
        / f"seed{seed}"
    )

    carrier_info = (
        dataset_manifest[
            "seed_carriers"
        ][str(seed)]
    )

    provenance = {
        "dataset": dataset,
        "training_seed": seed,
        "diagnostic_sha256":
            sha256_file(
                output
            ),
        "log_sha256":
            sha256_file(
                log
            ),
        "carrier":
            carrier_info,
        "input_manifest":
            str(
                FULL_ROOT
                / "input_manifests"
                / f"{slug}.json"
            ),
        "DIAGNOSTIC_ONLY":
            True,
        "full_GT_loaded":
            False,
        "training_performed":
            False,
    }

    write_json(
        cell_root
        / "PROVENANCE.json",
        provenance,
    )

    print(
        f"{dataset}/seed{seed}: PASS "
        f"DU={result['Delta_TRUE_UNIFORM']:+.8f} "
        f"DS={result['Delta_TRUE_SHUFFLE']:+.8f} "
        f"Q5Q1={result['DeltaAlignment_Q5_Q1']:+.8f} "
        f"rhoUM={result['Spearman_U_semantic_margin']:+.8f}",
        flush=True,
    )

    return result


def all_dataset_results_exist(
    dataset: str,
) -> bool:

    slug = SLUG[
        dataset
    ]

    for seed in SEEDS:
        path = (
            FULL_ROOT
            / slug
            / f"seed{seed}"
            / "diagnostic.json"
        )

        if not path.is_file():
            return False

        validate_result(
            dataset,
            seed,
            path,
        )

    return True


def cleanup_working_dataset(
    dataset: str,
) -> None:

    work = (
        WORK_ROOT
        / SLUG[dataset]
    )

    if work.exists():
        shutil.rmtree(
            work
        )

    print(
        f"{dataset}: "
        "EPHEMERAL_WORKING_INPUTS_REMOVED=true",
        flush=True,
    )


def collect_all_results() -> list[dict]:

    cells = []

    for dataset in DATASETS:
        for seed in SEEDS:

            path = (
                FULL_ROOT
                / SLUG[dataset]
                / f"seed{seed}"
                / "diagnostic.json"
            )

            if not path.is_file():
                fail(
                    f"missing final cell "
                    f"{dataset}/seed{seed}"
                )

            cells.append(
                validate_result(
                    dataset,
                    seed,
                    path,
                )
            )

    if len(
        cells
    ) != 24:
        fail(
            f"final cell count {len(cells)}"
        )

    return cells


def print_table(
    cells: list[dict],
) -> None:

    print()
    print(
        "=" * 118
    )

    print(
        "FULL 24-CELL DIAGNOSTIC TABLE"
    )

    print(
        "=" * 118
    )

    print(
        f"{'dataset':<14}"
        f"{'seed':>6}"
        f"{'TRUE-UNI':>14}"
        f"{'TRUE-SHUFF':>14}"
        f"{'Q5-Q1':>14}"
        f"{'rhoQ':>12}"
        f"{'rho(U,M)':>14}"
        f"{'conflict':>12}"
    )

    for d in cells:
        print(
            f"{d['dataset']:<14}"
            f"{int(d['training_seed']):>6}"
            f"{float(d['Delta_TRUE_UNIFORM']):>+14.7f}"
            f"{float(d['Delta_TRUE_SHUFFLE']):>+14.7f}"
            f"{float(d['DeltaAlignment_Q5_Q1']):>+14.7f}"
            f"{float(d['Spearman_quintile_alignment']):>+12.5f}"
            f"{float(d['Spearman_U_semantic_margin']):>+14.5f}"
            f"{float(d['HIGH_U_LOW_MARGIN_CONFLICT_RATE']):>12.5f}"
        )


def main() -> None:

    print(
        "VERIFY_FROZEN_SOURCES...",
        flush=True,
    )

    verify_frozen_sources()

    check_protocol_manifest()

    preflight_all_24()

    FULL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    WORK_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for dataset in DATASETS:

        if all_dataset_results_exist(
            dataset
        ):
            print(
                f"{dataset}: "
                "ALL_3_EXISTING_VALID -> SKIP DATASET",
                flush=True,
            )

            continue

        (
            views_path,
            labels_path,
            dataset_manifest,
        ) = materialize_dataset_inputs(
            dataset
        )

        for seed in SEEDS:
            run_cell(
                dataset,
                seed,
                views_path,
                labels_path,
                dataset_manifest,
            )

        if not all_dataset_results_exist(
            dataset
        ):
            fail(
                f"{dataset}: "
                "three-seed completion failed"
            )

        cleanup_working_dataset(
            dataset
        )

    cells = collect_all_results()

    write_json(
        FULL_ROOT
        / "FULL24_CELLS.json",
        cells,
    )

    summary = summarize(
        cells
    )

    write_json(
        FULL_ROOT
        / "FULL24_SUMMARY.json",
        summary,
    )

    print_table(
        cells
    )

    print()
    print(
        "=" * 118
    )

    print(
        "PRE-REGISTERED ROUTE SUMMARY"
    )

    print(
        "=" * 118
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )

    completion = {
        "STATUS":
            "P3_C4_A0_FULL24_DIAGNOSTIC_COMPLETE",
        "completed_cell_count":
            len(cells),
        "dataset_count":
            len(DATASETS),
        "seed_count":
            len(SEEDS),
        "DIAGNOSTIC_ONLY":
            True,
        "full_GT_loaded":
            False,
        "formal_clustering_metrics_computed":
            False,
        "training_performed":
            False,
        "parameter_update_performed":
            False,
        "ENGINEERING_SMOKE_AFFECTS_ROUTE_DECISION":
            False,
        "summary":
            summary,
    }

    write_json(
        FULL_ROOT
        / "FULL24_COMPLETE.json",
        completion,
    )

    verify_frozen_sources()

    print()
    print(
        "=" * 76
    )

    print(
        "FINAL FULL24 STATUS"
    )

    print(
        "=" * 76
    )

    print(
        "P3_C4_A0_FULL24_DIAGNOSTIC=PASS"
    )

    print(
        "COMPLETED_CELL_COUNT=24"
    )

    print(
        "DATASET_COUNT=8"
    )

    print(
        "SEED_COUNT=3"
    )

    print(
        "GPU_USED=true"
    )

    print(
        "DIAGNOSTIC_ONLY=true"
    )

    print(
        "TRAINING_PERFORMED=false"
    )

    print(
        "PARAMETER_UPDATE_PERFORMED=false"
    )

    print(
        "GT_READ=false"
    )

    print(
        "ACC_NMI_ARI_COMPUTED=false"
    )

    print(
        "POST_RUN_FROZEN_SOURCE_INTEGRITY=PASS"
    )

    print(
        "SUMMARY_PATH="
        + str(
            FULL_ROOT
            / "FULL24_SUMMARY.json"
        )
    )

    print(
        "CELLS_PATH="
        + str(
            FULL_ROOT
            / "FULL24_CELLS.json"
        )
    )

    print(
        "NEXT_STAGE="
        "P3_C4_A0_FULL24_SCIENTIFIC_ROUTE_REVIEW"
    )


if __name__ == "__main__":
    main()
