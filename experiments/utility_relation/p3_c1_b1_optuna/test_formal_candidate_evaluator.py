from types import SimpleNamespace

import numpy as np

from . import formal_candidate_evaluator as module
from .formal_base_reference import BaseReference
from .formal_tuning_context import CarrierContext, FormalTuningContext
from .study_schema import CandidateConfig, candidate_key
from . import protocol


def _context(tmp_path):
    carriers = {}
    for seed in protocol.DEV_SEEDS:
        carriers[seed] = CarrierContext(
            dataset="BDGP", dev_seed=seed, directory=tmp_path,
            state_path=tmp_path / f"state{seed}", pre_gt_path=tmp_path / f"pre{seed}", audit_path=tmp_path / f"audit{seed}",
            arrays={}, views=tuple(), adapter=object(), spec=SimpleNamespace(batch_size=1), k=2,
            folds=tuple(), state_file_sha256=f"s{seed}", pre_gt_sha256=f"p{seed}",
        )
    return FormalTuningContext("BDGP", tmp_path, carriers, {11: {}, 37: {}}, "fp")


def test_four_branch_candidate_evaluator_and_exact_schedule(monkeypatch, tmp_path):
    context = _context(tmp_path)
    candidate = CandidateConfig(1e-5, 5)
    bases = {
        seed: BaseReference("BDGP", seed, tmp_path / "base.json", f"s{seed}", f"p{seed}", 20,
                            1.0, 1.0, {}, tuple(f"n{i}" for i in range(20)), "fp")
        for seed in protocol.DEV_SEEDS
    }
    monkeypatch.setattr(module, "load_verified_base_reference", lambda ctx, seed: bases[seed])
    calls = []

    def runner(carrier, fold, candidate, device, payload):
        calls.append((carrier.dev_seed, fold, candidate.semantic_exposure))
        return {
            "J_OURS": 1.5, "semantic_steps": candidate.semantic_exposure, "native_epochs": 20,
            "initial_state_sha256": f"carrier-{carrier.dev_seed}",
            "audit": {"supported_anchor_count": 1, "abstained_anchor_count": 0, "coverage": 1.0},
        }

    # The real fold builder is not involved in this CPU contract test.
    monkeypatch.setattr(module, "_evaluate_with_runner", lambda ctx, cand, device, runner: {
        (seed, fold): runner(ctx.carrier(seed), fold, cand, device, {})
        for seed in protocol.DEV_SEEDS for fold in protocol.FOLDS
    })
    result = module.evaluate_candidate(context, candidate, device="cpu", runner=runner)
    assert result.mean_delta_j == 0.5
    assert result.positive_count == 4
    assert len(calls) == 4
    assert {call[2] for call in calls} == {5}


def test_candidate_cache_reuse_does_not_call_runner(monkeypatch, tmp_path):
    context = _context(tmp_path)
    candidate = CandidateConfig(3e-6, 1)
    payload = {
        "dataset": "BDGP", "semantic_lr": 3e-6, "semantic_exposure": 1,
        "delta_j_seed11_foldA": 1.0, "delta_j_seed11_foldB": 1.0,
        "delta_j_seed37_foldA": 1.0, "delta_j_seed37_foldB": 1.0,
        "protocol_fingerprint": "fp", "full_gt_used": False,
        "candidate_key": candidate_key(candidate),
    }
    path = module.candidate_directory(tmp_path, "BDGP", candidate)
    path.mkdir(parents=True)
    (path / "candidate_evaluation.json").write_text(__import__("json").dumps(payload), encoding="utf-8")
    (path / "audit.json").write_text("{}", encoding="utf-8")
    from .parent_integrity import sha256_file
    (path / "COMPLETE.json").write_text(__import__("json").dumps({
        "status": "COMPLETE", "evaluation_sha256": sha256_file(path / "candidate_evaluation.json"),
        "audit_sha256": sha256_file(path / "audit.json"),
    }), encoding="utf-8")
    called = []
    result = module.evaluate_candidate(context, candidate, device="cpu", runner=lambda *args: called.append(args))
    assert result.mean_delta_j == 1.0
    assert called == []
