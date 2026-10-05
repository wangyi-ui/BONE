import json
from types import SimpleNamespace

import pytest

from . import formal_candidate_evaluator as frozen_evaluator
from . import formal_candidate_evaluator_r2 as module
from . import protocol
from .formal_base_reference import BaseReference
from .formal_tuning_context import CarrierContext, FormalTuningContext
from .parent_integrity import sha256_file
from .study_schema import CandidateConfig, CandidateEvaluation, candidate_key


def _context(tmp_path):
    carriers = {
        seed: CarrierContext(
            dataset="BDGP",
            dev_seed=seed,
            directory=tmp_path,
            state_path=tmp_path / f"state{seed}",
            pre_gt_path=tmp_path / f"pre{seed}",
            audit_path=tmp_path / f"audit{seed}",
            arrays={"seed": seed},
            views=tuple(),
            adapter=SimpleNamespace(seed=seed),
            spec=SimpleNamespace(batch_size=1),
            k=2,
            folds=("folds", seed),
            state_file_sha256=f"s{seed}",
            pre_gt_sha256=f"p{seed}",
        )
        for seed in protocol.DEV_SEEDS
    }
    return FormalTuningContext(
        "BDGP", tmp_path, carriers, {11: {}, 37: {}}, "fp"
    )


def _bases(tmp_path):
    return {
        seed: BaseReference(
            "BDGP",
            seed,
            tmp_path / f"base{seed}.json",
            f"s{seed}",
            f"p{seed}",
            protocol.REFINEMENT_HORIZON,
            1.0,
            1.0,
            {},
            tuple(f"h{i}" for i in range(protocol.REFINEMENT_HORIZON)),
            "fp",
        )
        for seed in protocol.DEV_SEEDS
    }


def test_r2_explicitly_binds_trial_runner_a1():
    module._bind_trial_runner_a1()
    assert frozen_evaluator.trial_runner_a1 is module.trial_runner_a1
    source = open(module.__file__, encoding="utf-8").read()
    assert "from . import trial_runner_a1" in source
    assert "frozen_evaluator.trial_runner_a1 = trial_runner_a1" in source


def test_fold_builder_is_called_and_all_four_branches_execute(monkeypatch, tmp_path):
    context = _context(tmp_path)
    candidate = CandidateConfig(1e-5, 3)
    builder_calls = []
    branch_calls = []

    def build_fold_semantics_a1(arrays, adapter, folds):
        builder_calls.append((arrays["seed"], adapter.seed, folds))
        return {"A": {"fold": "A"}, "B": {"fold": "B"}}

    monkeypatch.setattr(
        module,
        "trial_runner_a1",
        SimpleNamespace(build_fold_semantics_a1=build_fold_semantics_a1),
    )

    def runner(carrier, fold, received, device, payload):
        branch_calls.append((carrier.dev_seed, fold, payload["fold"]))
        return {
            "J_OURS": 1.0,
            "semantic_steps": received.semantic_exposure,
            "native_epochs": protocol.REFINEMENT_HORIZON,
            "initial_state_sha256": f"state-{carrier.dev_seed}",
            "audit": {},
        }

    results = module._evaluate_with_runner(context, candidate, "cpu", runner)
    assert [call[0] for call in builder_calls] == [11, 37]
    assert set(branch_calls) == {
        (11, "A", "A"),
        (11, "B", "B"),
        (37, "A", "A"),
        (37, "B", "B"),
    }
    assert set(results) == {
        (11, "A"),
        (11, "B"),
        (37, "A"),
        (37, "B"),
    }


def test_r2_evaluation_delegates_frozen_candidate_semantics(monkeypatch, tmp_path):
    context = _context(tmp_path)
    candidate = CandidateConfig(3e-5, 5)
    bases = _bases(tmp_path)
    builder_calls = []
    branch_calls = []

    def build_fold_semantics_a1(arrays, adapter, folds):
        builder_calls.append(arrays["seed"])
        return {"A": {"fold": "A"}, "B": {"fold": "B"}}

    monkeypatch.setattr(
        module,
        "trial_runner_a1",
        SimpleNamespace(build_fold_semantics_a1=build_fold_semantics_a1),
    )
    monkeypatch.setattr(
        frozen_evaluator,
        "load_verified_base_reference",
        lambda context, seed: bases[seed],
    )
    ours = {(11, "A"): 1.1, (11, "B"): 1.2, (37, "A"): 0.9, (37, "B"): 1.4}

    def runner(carrier, fold, received, device, payload):
        branch_calls.append((carrier.dev_seed, fold))
        return {
            "J_OURS": ours[(carrier.dev_seed, fold)],
            "semantic_steps": received.semantic_exposure,
            "native_epochs": protocol.REFINEMENT_HORIZON,
            "initial_state_sha256": f"state-{carrier.dev_seed}",
            "audit": {
                "supported_anchor_count": 1,
                "abstained_anchor_count": 0,
                "coverage": 1.0,
            },
        }

    actual = module.evaluate_candidate(context, candidate, device="cpu", runner=runner)
    expected = CandidateEvaluation(
        "BDGP", candidate.semantic_lr, candidate.semantic_exposure,
        0.1, 0.2, -0.1, 0.4,
    )
    assert actual.dataset == expected.dataset
    assert actual.candidate_config() == expected.candidate_config()
    assert actual.delta_scores() == pytest.approx(expected.delta_scores())
    assert actual.mean_delta_j == pytest.approx(expected.mean_delta_j)
    assert actual.worst_delta_j == pytest.approx(expected.worst_delta_j)
    assert actual.positive_count == expected.positive_count
    assert builder_calls == [11, 37]
    assert len(branch_calls) == 4


def test_r2_candidate_cache_reuse_never_calls_branch_runner(tmp_path):
    context = _context(tmp_path)
    candidate = CandidateConfig(3e-6, 1)
    path = module.candidate_directory(tmp_path, "BDGP", candidate)
    path.mkdir(parents=True)
    payload = {
        "dataset": "BDGP",
        "semantic_lr": candidate.semantic_lr,
        "semantic_exposure": candidate.semantic_exposure,
        "delta_j_seed11_foldA": 0.1,
        "delta_j_seed11_foldB": 0.2,
        "delta_j_seed37_foldA": 0.3,
        "delta_j_seed37_foldB": 0.4,
        "protocol_fingerprint": "fp",
        "full_gt_used": False,
        "candidate_key": candidate_key(candidate),
    }
    evaluation_path = path / "candidate_evaluation.json"
    audit_path = path / "audit.json"
    evaluation_path.write_text(json.dumps(payload), encoding="utf-8")
    audit_path.write_text("{}", encoding="utf-8")
    (path / "COMPLETE.json").write_text(
        json.dumps({
            "status": "COMPLETE",
            "evaluation_sha256": sha256_file(evaluation_path),
            "audit_sha256": sha256_file(audit_path),
        }),
        encoding="utf-8",
    )
    calls = []
    result = module.evaluate_candidate(
        context, candidate, device="cpu", runner=lambda *args: calls.append(args)
    )
    assert result.mean_delta_j == 0.25
    assert calls == []
