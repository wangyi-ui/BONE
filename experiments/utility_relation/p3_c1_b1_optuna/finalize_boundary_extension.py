"""Combine exactly 25 main plus 5 extension candidates using frozen selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from . import objective
from .boundary_extension_context import (
    BoundaryExtensionContext,
    extension_candidate_directory,
    load_verified_extension_candidate,
)
from .boundary_extension_protocol import extension_candidates
from .formal_candidate_evaluator_r2 import (
    candidate_directory,
    load_verified_candidate,
)
from .optuna_driver import expected_main_candidates


def _record(config, evaluation, source: str, path: Path) -> dict:
    return {
        "config": config,
        "evaluation": evaluation,
        "source": source,
        "candidate_path": str(path),
    }


def _score(record: Mapping) -> dict:
    evaluation = record["evaluation"]
    return {
        "mean_delta_J": evaluation.mean_delta_j,
        "worst_delta_J": evaluation.worst_delta_j,
        "positive_count": evaluation.positive_count,
    }


def _select(records: Iterable[Mapping]) -> Mapping:
    records = tuple(records)
    scores = {record["config"]: _score(record) for record in records}
    selected = objective.select_best_candidate(scores)
    return next(record for record in records if record["config"] == selected)


def _serialize(record: Mapping) -> dict:
    config = record["config"]
    evaluation = record["evaluation"]
    return {
        "semantic_lr": config.semantic_lr,
        "semantic_exposure": config.semantic_exposure,
        "mean_delta_j": evaluation.mean_delta_j,
        "worst_delta_j": evaluation.worst_delta_j,
        "positive_count": evaluation.positive_count,
        "source": record["source"],
        "candidate_path": record["candidate_path"],
    }


def select_final_from_records(
    dataset: str,
    main_records: Iterable[Mapping],
    extension_records: Iterable[Mapping],
) -> dict:
    main_records = tuple(main_records)
    extension_records = tuple(extension_records)
    if len(main_records) != 25:
        raise RuntimeError("FINALIZER_REQUIRES_EXACTLY_25_MAIN_CANDIDATES")
    if len(extension_records) != 5:
        raise RuntimeError("FINALIZER_REQUIRES_EXACTLY_5_EXTENSION_CANDIDATES")
    if {record["config"] for record in main_records} != set(expected_main_candidates()):
        raise RuntimeError("FINALIZER_MAIN_CANDIDATE_IDENTITY_MISMATCH")
    if {record["config"] for record in extension_records} != set(extension_candidates()):
        raise RuntimeError("FINALIZER_EXTENSION_CANDIDATE_IDENTITY_MISMATCH")
    if any(record["source"] != "MAIN_GRID" for record in main_records):
        raise ValueError("INVALID_MAIN_CANDIDATE_SOURCE")
    if any(record["source"] != "UPPER_BOUNDARY_EXTENSION" for record in extension_records):
        raise ValueError("INVALID_EXTENSION_CANDIDATE_SOURCE")
    main_best = _select(main_records)
    extension_best = _select(extension_records)
    final_best = _select((*main_records, *extension_records))
    return {
        "dataset": dataset,
        **_serialize(final_best),
        "main_grid_best": _serialize(main_best),
        "extension_best": _serialize(extension_best),
        "final_best": _serialize(final_best),
        "main_grid_candidate_count": 25,
        "extension_candidate_count": 5,
        "final_selection_pool_count": 30,
        "selection_rule": "frozen_rounded_lexicographic",
        "second_boundary_extension_allowed": False,
        "full_gt_used": False,
    }


def finalize_boundary_extension(
    context: BoundaryExtensionContext,
    output_path: str | Path | None = None,
) -> dict:
    main_records = []
    for candidate in expected_main_candidates():
        evaluation = load_verified_candidate(context.base_context, candidate)
        main_records.append(_record(
            candidate,
            evaluation,
            "MAIN_GRID",
            candidate_directory(context.formal_tuning_root, context.dataset, candidate),
        ))
    extension_records = []
    for candidate in extension_candidates():
        evaluation = load_verified_extension_candidate(context, candidate)
        extension_records.append(_record(
            candidate,
            evaluation,
            "UPPER_BOUNDARY_EXTENSION",
            extension_candidate_directory(context, candidate),
        ))
    result = select_final_from_records(
        context.dataset, main_records, extension_records
    )
    target = Path(output_path) if output_path is not None else context.extension_root / "final_best_params.json"
    if target.exists():
        raise FileExistsError(f"REFUSE_TO_OVERWRITE:{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
