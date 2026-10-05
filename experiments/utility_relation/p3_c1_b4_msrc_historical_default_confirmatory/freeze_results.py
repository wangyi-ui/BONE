"""Fail-closed immutable B4 result freezer; reporting values are never selection inputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from . import audit, protocol
from .run_confirmatory import verify_complete_run
from .summarize_confirmatory import summarize_confirmatory


RESULT_FREEZE = protocol.REPO_ROOT / "experiment_freeze/bone_p3_c1_b4_msrc_historical_default_confirmatory_results_20261004"


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _manifest(path: Path, files: list[Path]) -> None:
    path.write_text("".join(f"{protocol.sha256_file(item)}  {item.relative_to(protocol.REPO_ROOT)}\n" for item in files) + "\n", encoding="utf-8")


def _verify_summary(summary: dict) -> list[dict]:
    if summary.get("HISTORICAL_DEFAULT_CONFIRMATION") != "NOT_CONFIRMED":
        raise RuntimeError("B4_FROZEN_PRIMARY_GATE_NOT_NOT_CONFIRMED")
    expected = {
        "ACC": -0.004761904761904782,
        "NMI": 0.0007722994311792907,
        "ARI": 0.0007172712231704463,
    }
    rows = []
    for item in summary["per_seed"]:
        if item.get("status") != "REPORTING_ONLY" or item.get("metrics_used_for_parameter_selection") is not False or item.get("full_gt_access_phase") != "POST_TRAINING_AFTER_PRE_GT_SEAL":
            raise RuntimeError("B4_REPORTING_METRICS_INVALID")
        rows.append({"SEED": item["training_seed"], **{f"BASE_{metric}": item["BASE"][metric] for metric in ("ACC", "NMI", "ARI")}, **{f"HISTORICAL_DEFAULT_{metric}": item["OURS_HISTORICAL_DEFAULT"][metric] for metric in ("ACC", "NMI", "ARI")}, **{f"DELTA_{metric}": item["delta_historical_default_minus_base"][metric] for metric in ("ACC", "NMI", "ARI")}})
    for metric, value in expected.items():
        # The parent request publishes decimal expectations; JSON recomputation
        # is authoritative and can differ by one IEEE-754 rounding unit.
        if abs(summary["mean_deltas"][metric]["mean"] - value) > 1e-15:
            raise RuntimeError(f"B4_RECOMPUTED_MEAN_MISMATCH:{metric}")
    return rows


def freeze_results(output_root: str | Path = protocol.OUTPUT_ROOT, result_freeze: str | Path = RESULT_FREEZE) -> Path:
    target = Path(result_freeze)
    if target.exists():
        raise FileExistsError("B4_RESULT_FREEZE_ALREADY_EXISTS")
    preregistration = audit.verify_preregistration()
    parent = audit.verify_parent_integrity()
    for seed in protocol.CONFIRMATORY_SEEDS:
        root = protocol.run_root(output_root, seed)
        verify_complete_run(root)
        if not (root / "CONFIRMATORY_METRICS.json").is_file():
            raise FileNotFoundError(f"B4_METRICS_MISSING:seed{seed}")
    summary = summarize_confirmatory(output_root)
    rows = _verify_summary(summary)
    target.mkdir(parents=True)
    status = """STATUS=P3_C1_B4_MSRC_HISTORICAL_DEFAULT_CONFIRMATORY_RESULTS_FROZEN
DATASET=MSRC
SEEDS=100,110,130
ARMS=BASE,OURS_HISTORICAL_DEFAULT
RUN_COUNT=6
ALL_RUNS_COMPLETE=true
HISTORICAL_DEFAULT_SEMANTIC_LR=0.0001
HISTORICAL_DEFAULT_SEMANTIC_EXPOSURE=20
MEAN_DELTA_ACC=-0.004761904761904782
MEAN_DELTA_NMI=0.0007722994311792907
MEAN_DELTA_ARI=0.0007172712231704463
HISTORICAL_DEFAULT_CONFIRMATION=NOT_CONFIRMED
GT_USED_FOR_REPORTING_ONLY=true
METRICS_USED_FOR_PARAMETER_SELECTION=false
PARAMETER_RETUNING_ALLOWED=false
METHOD_CHANGED=false
OPTIMIZER_CHANGED=false
U_CYCLE_CHANGED=false
T_REL_CHANGED=false
B4_CLOSED=true
NEXT_STAGE=P3-C1-B5
"""
    (target / "STATUS.txt").write_text(status, encoding="utf-8")
    protocol.write_json(target / "THREE_SEED_SUMMARY.json", summary)
    _write_csv(target / "PER_SEED_RESULTS.csv", rows)
    protocol.write_json(target / "PARENT_INTEGRITY.txt", {"verified": True, "preregistration_verified": preregistration["verified"], "parent_integrity_verified": parent["verified"], "source_hash_status": preregistration["source_sha256"] == protocol.source_hashes(), "historical_default_confirmation": summary["HISTORICAL_DEFAULT_CONFIRMATION"]})
    source_paths = [protocol.REPO_ROOT / name for name in (*protocol.SOURCE_RELATIVE_PATHS, "experiments/utility_relation/p3_c1_b4_msrc_historical_default_confirmatory/freeze_results.py", "tests/test_p3_c1_b4_result_freeze.py")]
    _manifest(target / "source_sha256.txt", source_paths)
    output_files = sorted(path for path in Path(output_root).rglob("*") if path.is_file())
    _manifest(target / "artifact_sha256.txt", output_files)
    freeze_files = sorted(path for path in target.iterdir() if path.is_file())
    _manifest(target / "FREEZE_MANIFEST.sha256", freeze_files)
    return target


if __name__ == "__main__":
    print(freeze_results())
