"""Reporting-only three-seed summary and immutable B3 result freeze."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from . import protocol
from .run_confirmatory import verify_complete_run


METRICS = ("ACC", "NMI", "ARI")


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_rows(output_root: str | Path) -> list[dict]:
    rows = []
    for seed in protocol.CONFIRMATORY_SEEDS:
        root = protocol.run_root(output_root, seed)
        verify_complete_run(root)
        path = root / "CONFIRMATORY_METRICS.json"
        if not path.is_file():
            raise FileNotFoundError(f"B3_CONFIRMATORY_METRICS_MISSING:seed{seed}")
        item = json.loads(path.read_text(encoding="utf-8"))
        if (
            item.get("status") != "REPORTING_ONLY"
            or item.get("dataset") != protocol.DATASET
            or item.get("training_seed") != seed
            or item.get("parameter_retuning_allowed") is not False
            or item.get("metrics_used_for_parameter_selection") is not False
            or item.get("pre_gt_seal_sha256")
            != protocol.sha256_file(root / "PRE_GT_SEAL.json")
        ):
            raise RuntimeError(f"B3_CONFIRMATORY_METRICS_INVALID:seed{seed}")
        rows.append(item)
    return rows


def summarize_confirmatory(
    output_root: str | Path = protocol.OUTPUT_ROOT,
    write_outputs: bool = True,
) -> dict:
    rows = _load_rows(output_root)
    per_seed = []
    for item in rows:
        row = {"SEED": item["training_seed"]}
        for metric in METRICS:
            row[f"BASE_{metric}"] = item["BASE"][metric]
            row[f"B1_{metric}"] = item["OURS_B1"][metric]
            row[f"B2_{metric}"] = item["OURS_B2"][metric]
            row[f"B1_MINUS_BASE_{metric}"] = item["B1_minus_BASE"][metric]
            row[f"B2_MINUS_BASE_{metric}"] = item["B2_minus_BASE"][metric]
            row[f"B2_MINUS_B1_{metric}"] = item["B2_minus_B1"][metric]
        per_seed.append(row)

    summary_rows = []
    summary_json = {}
    for metric in METRICS:
        base = np.asarray([item["BASE"][metric] for item in rows], dtype=float)
        b1 = np.asarray([item["OURS_B1"][metric] for item in rows], dtype=float)
        b2 = np.asarray([item["OURS_B2"][metric] for item in rows], dtype=float)
        b1_base, b2_base, b2_b1 = b1 - base, b2 - base, b2 - b1
        row = {"METRIC": metric}
        for name, values in (
            ("BASE", base),
            ("B1", b1),
            ("B2", b2),
            ("B1_MINUS_BASE", b1_base),
            ("B2_MINUS_BASE", b2_base),
            ("B2_MINUS_B1", b2_b1),
        ):
            row[f"{name}_MEAN"] = float(values.mean())
            row[f"{name}_STD_POPULATION"] = float(values.std(ddof=0))
        row["B2_GT_B1_POSITIVE_SEED_COUNT"] = int(np.count_nonzero(b2_b1 > 0))
        row["B2_GT_BASE_POSITIVE_SEED_COUNT"] = int(np.count_nonzero(b2_base > 0))
        summary_rows.append(row)
        summary_json[metric] = {
            "BASE": {"mean": row["BASE_MEAN"], "std_population": row["BASE_STD_POPULATION"]},
            "B1": {"mean": row["B1_MEAN"], "std_population": row["B1_STD_POPULATION"]},
            "B2": {"mean": row["B2_MEAN"], "std_population": row["B2_STD_POPULATION"]},
            "B1_minus_BASE": {"mean": row["B1_MINUS_BASE_MEAN"], "std_population": row["B1_MINUS_BASE_STD_POPULATION"]},
            "B2_minus_BASE": {"mean": row["B2_MINUS_BASE_MEAN"], "std_population": row["B2_MINUS_BASE_STD_POPULATION"]},
            "B2_minus_B1": {"mean": row["B2_MINUS_B1_MEAN"], "std_population": row["B2_MINUS_B1_STD_POPULATION"]},
            "B2_gt_B1_positive_seed_count": row["B2_GT_B1_POSITIVE_SEED_COUNT"],
            "B2_gt_BASE_positive_seed_count": row["B2_GT_BASE_POSITIVE_SEED_COUNT"],
        }
    result = {
        "status": "REPORTING_ONLY",
        "dataset": protocol.DATASET,
        "seeds": list(protocol.CONFIRMATORY_SEEDS),
        "arms": list(protocol.ARMS),
        "run_count": len(protocol.CONFIRMATORY_SEEDS) * len(protocol.ARMS),
        "all_runs_complete": True,
        "population_std_ddof": 0,
        "parameter_retuning_allowed": False,
        "metrics": summary_json,
    }
    if write_outputs:
        output = Path(output_root)
        _write_csv(output / "PER_SEED_RESULTS.csv", per_seed, list(per_seed[0]))
        _write_csv(
            output / "THREE_SEED_SUMMARY.csv", summary_rows, list(summary_rows[0])
        )
        protocol.write_json(output / "THREE_SEED_SUMMARY.json", result)
    return result


def freeze_results(
    output_root: str | Path = protocol.OUTPUT_ROOT,
    freeze_dir: str | Path = protocol.RESULT_FREEZE,
    preregistration_freeze: str | Path = protocol.PREREGISTERED_FREEZE,
) -> Path:
    target = Path(freeze_dir)
    if target.exists():
        raise FileExistsError("B3_RESULTS_FREEZE_ALREADY_EXISTS")
    protocol.verify_preregistration(preregistration_freeze)
    summary = summarize_confirmatory(output_root, write_outputs=True)
    if summary["run_count"] != 9 or summary["all_runs_complete"] is not True:
        raise RuntimeError("B3_NINE_RUN_COMPLETION_FAILURE")
    target.mkdir(parents=True)
    status = """STATUS=P3_C1_B3_MSRC_B2_CONFIRMATORY_RESULTS_FROZEN
DATASET=MSRC
SEEDS=60,70,90
RUN_COUNT=9
ALL_RUNS_COMPLETE=true
B1_PARAMETER_FROZEN=true
B2_PARAMETER_FROZEN=true
METHOD_CHANGED=false
OPTIMIZER_CHANGED=false
FULL_GT_USED_FOR_TRAINING=false
GT_USED_FOR_REPORTING_ONLY=true
PARAMETER_RETUNING_ALLOWED=false
NEXT_STAGE=SCIENTIFIC_INTERPRETATION
"""
    (target / "STATUS.txt").write_text(status, encoding="utf-8")
    output = Path(output_root)
    for name in (
        "PER_SEED_RESULTS.csv",
        "THREE_SEED_SUMMARY.csv",
        "THREE_SEED_SUMMARY.json",
    ):
        shutil.copy2(output / name, target / name)
    source = protocol.source_hashes()
    (target / "source_sha256.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in source.items()),
        encoding="utf-8",
    )
    prereg_root = Path(preregistration_freeze)
    prereg_files = sorted(path for path in prereg_root.iterdir() if path.is_file())
    (target / "parent_freeze_sha256.txt").write_text(
        "".join(
            f"{protocol.sha256_file(path)}  "
            f"{path.relative_to(protocol.REPO_ROOT) if path.is_relative_to(protocol.REPO_ROOT) else Path('preregistration') / path.relative_to(prereg_root)}\n"
            for path in prereg_files
        ),
        encoding="utf-8",
    )
    artifacts = sorted(path for path in output.rglob("*") if path.is_file())
    (target / "artifact_sha256.txt").write_text(
        "".join(
            f"{protocol.sha256_file(path)}  "
            f"{path.relative_to(protocol.REPO_ROOT) if path.is_relative_to(protocol.REPO_ROOT) else Path('output') / path.relative_to(output)}\n"
            for path in artifacts
        ),
        encoding="utf-8",
    )
    protocol.write_json(
        target / "PARENT_INTEGRITY.txt",
        {
            "verified": True,
            "parent_source_modification": False,
            "preregistration_verified": True,
            "b2_parent_verified": True,
            "final_formal_parent_verified": True,
            "full_gt_used_for_training": False,
            "gt_used_for_reporting_only": True,
            "parameter_retuning_allowed": False,
        },
    )
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=protocol.OUTPUT_ROOT)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--freeze-results", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.freeze_results:
        print(freeze_results(args.output_root))
        return
    result = summarize_confirmatory(args.output_root, write_outputs=args.write)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
