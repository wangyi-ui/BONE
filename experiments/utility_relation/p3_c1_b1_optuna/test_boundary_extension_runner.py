from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest

from . import protocol
from .boundary_extension_context import (
    BoundaryExtensionContext,
    build_boundary_extension_context,
)
from .boundary_extension_protocol import (
    A2_FREEZE_RELATIVE,
    REQUIRED_A2_STATUS,
    REQUIRED_TRIGGER_STATUS,
    TRIGGER_FREEZE_RELATIVE,
    ExtensionCandidateEvaluation,
)
from .formal_base_reference import BaseReference
from .formal_tuning_context_a2 import FormalTuningContextA2
from .run_boundary_extension import audit_extension_study, run_boundary_extension


def _context(tmp_path, dataset="BDGP"):
    base = SimpleNamespace(dataset=dataset)
    return BoundaryExtensionContext(
        base, tmp_path / dataset.lower() / "boundary_extension_upper_1e3", {}
    )


def _references(dataset="BDGP"):
    return {
        seed: BaseReference(
            dataset, seed, Path(f"base{seed}.json"), f"s{seed}", f"p{seed}",
            20, 0.0, 0.0, {}, tuple("h" for _ in range(20)), "fp",
        )
        for seed in protocol.DEV_SEEDS
    }


def _write_status(root, relative, values):
    path = root / relative / "STATUS.txt"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )


def _evaluator(dataset, calls=None):
    def evaluate(candidate):
        if calls is not None:
            calls.append((candidate.semantic_lr, candidate.semantic_exposure))
        value = candidate.semantic_exposure * 0.01
        return ExtensionCandidateEvaluation(
            dataset, candidate.semantic_lr, candidate.semantic_exposure,
            value, value, value, value,
        )
    return evaluate


def test_runner_orders_gate_base_reuse_study_and_candidate(tmp_path):
    context = _context(tmp_path)
    order = []

    def gate(root):
        order.append("gate")
        return {}

    def bootstrap(base_context, **kwargs):
        order.append("base")
        assert not context.extension_root.exists()
        return _references()

    def evaluator(candidate):
        order.append("candidate")
        return _evaluator("BDGP")(candidate)

    study = run_boundary_extension(
        "BDGP", "cpu", tmp_path, tmp_path,
        max_new_trials=1, evaluator=evaluator, context=context,
        bootstrap=bootstrap, precondition=gate,
    )
    assert order == ["gate", "base", "candidate"]
    assert isinstance(study.sampler, optuna.samplers.GridSampler)
    assert study.user_attrs["a2_feasibility_active"] is True
    assert study.user_attrs["second_boundary_extension_allowed"] is False


def test_extension_study_is_exactly_five_and_has_no_hidden_candidate(tmp_path):
    study = run_boundary_extension(
        "BDGP", "cpu", tmp_path, tmp_path,
        evaluator=_evaluator("BDGP"), context=_context(tmp_path),
        bootstrap=lambda context, **kwargs: _references(),
        precondition=lambda root: {},
    )
    audit = audit_extension_study(study)
    assert audit == {
        "expected_candidate_count": 5,
        "complete_candidate_count": 5,
        "unique_complete_candidate_count": 5,
        "duplicate_candidate_count": 0,
        "missing_candidates": [],
        "unregistered_candidates": [],
        "failed_trial_count": 0,
        "pruned_trial_count": 0,
        "extension_grid_complete": True,
    }
    assert {trial.params["semantic_lr"] for trial in study.trials} == {1e-3}


def test_extension_sqlite_resume_two_to_five_without_duplicates(tmp_path):
    calls = []
    context = _context(tmp_path, "CCV")
    kwargs = dict(
        dataset="CCV", device="cpu", formal_tuning_root=tmp_path,
        repository_root=tmp_path, evaluator=_evaluator("CCV", calls),
        context=context, bootstrap=lambda context, **kwargs: _references("CCV"),
        precondition=lambda root: {},
    )
    first = run_boundary_extension(max_new_trials=2, **kwargs)
    assert audit_extension_study(first)["complete_candidate_count"] == 2
    resumed = run_boundary_extension(**kwargs)
    assert audit_extension_study(resumed)["extension_grid_complete"] is True
    assert len(calls) == len(set(calls)) == 5


def test_missing_base_stops_before_extension_study_creation(tmp_path):
    context = _context(tmp_path)
    with pytest.raises(RuntimeError, match="NO_VERIFIED_BASES"):
        run_boundary_extension(
            "BDGP", "cpu", tmp_path, tmp_path, context=context,
            bootstrap=lambda context, **kwargs: {11: _references()[11]},
            precondition=lambda root: {},
        )
    assert not context.extension_root.exists()


def test_runner_never_touches_main_candidate_tree(tmp_path):
    main = tmp_path / "bdgp" / "candidates" / "main-sentinel.bin"
    main.parent.mkdir(parents=True)
    main.write_bytes(b"original-25-byte-identical")
    before = main.read_bytes()
    run_boundary_extension(
        "BDGP", "cpu", tmp_path, tmp_path,
        evaluator=_evaluator("BDGP"), context=_context(tmp_path),
        bootstrap=lambda context, **kwargs: _references(),
        precondition=lambda root: {},
    )
    assert main.read_bytes() == before


@pytest.mark.parametrize("dataset", ["ALOI", "CCV"])
def test_aloi_and_ccv_extension_context_explicitly_routes_through_a2(
    tmp_path, dataset
):
    _write_status(
        tmp_path, TRIGGER_FREEZE_RELATIVE, REQUIRED_TRIGGER_STATUS
    )
    _write_status(tmp_path, A2_FREEZE_RELATIVE, REQUIRED_A2_STATUS)
    built = FormalTuningContextA2(
        dataset=dataset,
        formal_tuning_root=tmp_path / "formal_tuning",
        carriers={},
        feasibility={
            11: {"A2_FEASIBILITY": "PASS"},
            37: {"A2_FEASIBILITY": "PASS"},
        },
        protocol_fingerprint="a2-route-sentinel",
    )
    context = build_boundary_extension_context(
        dataset,
        tmp_path / "formal_tuning",
        tmp_path,
        builder=lambda selected_dataset, selected_root: built,
    )
    assert context.base_context is built
    assert context.base_context.active_protocol == "P3-C1-B1-A2"
    assert context.base_context.candidate_protocol == "P3-C1-B1-A1"
    assert context.precondition_audit["a2_feasibility_active"] is True
