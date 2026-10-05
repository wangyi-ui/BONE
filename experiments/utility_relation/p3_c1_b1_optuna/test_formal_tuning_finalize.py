import json

import pytest

from . import finalize_formal_tuning as module
from .run_formal_optuna_tuning import run_formal_optuna_tuning
from .study_schema import CandidateEvaluation


def _evaluator(candidate):
    return CandidateEvaluation("BDGP", candidate.semantic_lr, candidate.semantic_exposure, 0.0, 0.0, 0.0, 0.0)


def _fake_context(*args, **kwargs):
    return object()


def _fake_verified(context, candidate):
    return CandidateEvaluation("BDGP", candidate.semantic_lr, candidate.semantic_exposure, 0.0, 0.0, 0.0, 0.0)


def test_24_of_25_cannot_finalize(tmp_path, monkeypatch):
    run_formal_optuna_tuning("BDGP", "cpu", tmp_path, max_new_trials=24, evaluator=_evaluator, context=object())
    monkeypatch.setattr(module, "build_formal_tuning_context", _fake_context)
    monkeypatch.setattr(module, "load_verified_candidate", _fake_verified)
    with pytest.raises(RuntimeError, match="CANNOT_FINALIZE_BEFORE_25_OF_25"):
        module.finalize_formal_tuning("BDGP", tmp_path)


def test_25_of_25_finalizes_with_preregistered_tie_break(tmp_path, monkeypatch):
    run_formal_optuna_tuning("BDGP", "cpu", tmp_path, evaluator=_evaluator, context=object())
    monkeypatch.setattr(module, "build_formal_tuning_context", _fake_context)
    monkeypatch.setattr(module, "load_verified_candidate", _fake_verified)
    result = module.finalize_formal_tuning("BDGP", tmp_path)
    assert result["semantic_lr"] == 3e-6
    assert result["semantic_exposure"] == 1
    assert result["selection_rule"] == "preregistered_lexicographic"
    assert result["full_gt_used_for_selection"] is False
    assert (tmp_path / "bdgp" / "best_params.json").is_file()
    assert (tmp_path / "bdgp" / "trials.csv").is_file()
    assert (tmp_path / "bdgp" / "selection_audit.json").is_file()

