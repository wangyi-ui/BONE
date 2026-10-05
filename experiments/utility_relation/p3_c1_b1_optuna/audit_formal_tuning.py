"""Read-only audit for one dataset's formal tuning artifacts and study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna

try:
    from . import protocol
    from .formal_base_reference import load_verified_base_reference
    from .formal_candidate_evaluator import candidate_directory, load_verified_candidate
    from .formal_tuning_context import build_formal_tuning_context
    from .run_formal_optuna_tuning import audit_study_coverage, dataset_slug, protocol_fingerprint, storage_url, study_name_for_dataset
    from .optuna_driver import expected_main_candidates
    from .study_schema import candidate_key
except ImportError:  # pragma: no cover
    from experiments.utility_relation.p3_c1_b1_optuna import protocol
    from experiments.utility_relation.p3_c1_b1_optuna.formal_base_reference import load_verified_base_reference
    from experiments.utility_relation.p3_c1_b1_optuna.formal_candidate_evaluator import candidate_directory, load_verified_candidate
    from experiments.utility_relation.p3_c1_b1_optuna.formal_tuning_context import build_formal_tuning_context
    from experiments.utility_relation.p3_c1_b1_optuna.run_formal_optuna_tuning import audit_study_coverage, dataset_slug, protocol_fingerprint, storage_url, study_name_for_dataset
    from experiments.utility_relation.p3_c1_b1_optuna.optuna_driver import expected_main_candidates
    from experiments.utility_relation.p3_c1_b1_optuna.study_schema import candidate_key


def audit_formal_tuning(dataset: str, formal_tuning_root: str | Path) -> dict:
    root = Path(formal_tuning_root)
    context = build_formal_tuning_context(dataset, root)
    carriers = {str(seed): True for seed in protocol.DEV_SEEDS}
    bases = {}
    for seed in protocol.DEV_SEEDS:
        try:
            load_verified_base_reference(context, seed)
            bases[str(seed)] = True
        except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
            bases[str(seed)] = False
    candidate_records = []
    for candidate in expected_main_candidates():
        path = candidate_directory(root, dataset, candidate)
        verified = False
        try:
            load_verified_candidate(context, candidate); verified = True
        except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
            pass
        candidate_records.append({"candidate_key": candidate_key(candidate), "path": str(path), "verified": verified})

    db_path = root / dataset_slug(dataset) / "study.db"
    study_exists = db_path.is_file()
    study_audit = None
    if study_exists:
        study = optuna.load_study(study_name=study_name_for_dataset(dataset), storage=storage_url(db_path))
        if study.user_attrs.get("protocol_fingerprint") != protocol_fingerprint() or study.user_attrs.get("dataset") != dataset:
            raise RuntimeError("STUDY_PROTOCOL_OR_DATASET_MISMATCH")
        study_audit = audit_study_coverage(study)
    complete = sum(record["verified"] for record in candidate_records)
    study_complete = 0 if study_audit is None else study_audit["complete_candidate_count"]
    candidate_keys = {record["candidate_key"] for record in candidate_records if record["verified"]}
    study_keys = set()
    if study_exists:
        study_keys = {
            trial.user_attrs.get("candidate_key", "")
            for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
        }
    return {
        "dataset": dataset, "formal_tuning_root": str(root),
        "carrier11_verified": carriers.get("11", False), "carrier37_verified": carriers.get("37", False),
        "base_reference11_verified": bases.get("11", False), "base_reference37_verified": bases.get("37", False),
        "expected_candidate_count": 25, "complete_candidate_count": complete,
        "missing_candidate_count": 25 - complete,
        "invalid_candidate_count": sum(path["path"] != "" and not path["verified"] and Path(path["path"]).exists() for path in candidate_records),
        "sqlite_study_exists": study_exists,
        "study_complete_count": study_complete,
        "candidate_study_parity": study_audit is not None and candidate_keys == study_keys,
        "best_params_exists": (root / dataset_slug(dataset) / "best_params.json").is_file() if complete == 25 and study_complete == 25 else False,
        "full_gt_used": False,
        "protocol_fingerprint": protocol_fingerprint(),
        "MAIN_GRID_COMPLETE": complete == 25 and study_complete == 25 and candidate_keys == study_keys,
        "candidates": candidate_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(protocol.DATASETS), required=True)
    parser.add_argument("--formal-tuning-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"REFUSE_TO_OVERWRITE:{args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit_formal_tuning(args.dataset, args.formal_tuning_root), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()


audit = audit_formal_tuning
