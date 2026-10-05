from pathlib import Path

import pytest

from .boundary_extension_protocol import (
    ExtensionCandidateEvaluation,
    extension_candidates,
)
from .finalize_boundary_extension import select_final_from_records
from .optuna_driver import expected_main_candidates
from .study_schema import CandidateEvaluation


def _evaluation(dataset, config, deltas):
    cls = (
        ExtensionCandidateEvaluation
        if config.semantic_lr == 1e-3 else CandidateEvaluation
    )
    return cls(dataset, config.semantic_lr, config.semantic_exposure, *deltas)


def _records(dataset="BDGP", value=-10.0):
    main = [
        {
            "config": config,
            "evaluation": _evaluation(dataset, config, (value,) * 4),
            "source": "MAIN_GRID",
            "candidate_path": f"main/{index}",
        }
        for index, config in enumerate(expected_main_candidates())
    ]
    extension = [
        {
            "config": config,
            "evaluation": _evaluation(dataset, config, (value,) * 4),
            "source": "UPPER_BOUNDARY_EXTENSION",
            "candidate_path": f"extension/{index}",
        }
        for index, config in enumerate(extension_candidates())
    ]
    return main, extension


def _replace_score(records, config, deltas):
    for record in records:
        if record["config"] == config:
            record["evaluation"] = _evaluation("BDGP", config, deltas)
            return
    raise AssertionError("config missing")


def test_finalizer_requires_and_reports_exact_25_plus_5_pool():
    main, extension = _records()
    _replace_score(extension, extension_candidates()[2], (1.0,) * 4)
    result = select_final_from_records("BDGP", main, extension)
    assert result["main_grid_candidate_count"] == 25
    assert result["extension_candidate_count"] == 5
    assert result["final_selection_pool_count"] == 30
    assert result["source"] == "UPPER_BOUNDARY_EXTENSION"
    assert result["candidate_path"] == "extension/2"
    assert result["second_boundary_extension_allowed"] is False


@pytest.mark.parametrize("main_drop,extension_drop,error", [
    (1, 0, "25_MAIN"), (0, 1, "5_EXTENSION"),
])
def test_finalizer_rejects_incomplete_pool(main_drop, extension_drop, error):
    main, extension = _records()
    with pytest.raises(RuntimeError, match=error):
        select_final_from_records(
            "BDGP",
            main[:-main_drop] if main_drop else main,
            extension[:-extension_drop] if extension_drop else extension,
        )


def test_finalizer_rejects_duplicate_candidate_identity():
    main, extension = _records()
    main[-1]["config"] = main[0]["config"]
    with pytest.raises(RuntimeError, match="MAIN_CANDIDATE_IDENTITY_MISMATCH"):
        select_final_from_records("BDGP", main, extension)


def test_frozen_tie_break_mean_then_worst_then_positive_then_exposure_then_lr():
    main, extension = _records()
    a, b, c, d, e = main[:5]
    _replace_score(main, a["config"], (1.0, 1.0, 1.0, 1.0))
    _replace_score(main, b["config"], (2.0, 2.0, 2.0, 2.0))
    assert select_final_from_records("BDGP", main, extension)["semantic_lr"] == b["config"].semantic_lr

    main, extension = _records()
    a, b = main[:2]
    _replace_score(main, a["config"], (0.0, 2.0, 1.0, 1.0))
    _replace_score(main, b["config"], (0.5, 0.5, 1.5, 1.5))
    assert select_final_from_records("BDGP", main, extension)["semantic_exposure"] == b["config"].semantic_exposure

    main, extension = _records()
    a, b = main[:2]
    _replace_score(main, a["config"], (-1.0, 0.0, 2.0, 3.0))
    _replace_score(main, b["config"], (-1.0, 1.0, 1.0, 3.0))
    assert select_final_from_records("BDGP", main, extension)["semantic_exposure"] == b["config"].semantic_exposure

    main, extension = _records(value=0.0)
    result = select_final_from_records("BDGP", main, extension)
    assert result["semantic_exposure"] == 1
    assert result["semantic_lr"] == 3e-6


def test_rounding_to_12_decimals_is_used_before_other_ties():
    main, extension = _records()
    first, second = main[:2]
    _replace_score(main, first["config"], (1.0000000000001,) * 4)
    _replace_score(main, second["config"], (1.0000000000002,) * 4)
    result = select_final_from_records("BDGP", main, extension)
    assert result["semantic_exposure"] == min(
        first["config"].semantic_exposure, second["config"].semantic_exposure
    )
