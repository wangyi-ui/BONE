import hashlib
from types import SimpleNamespace

import pytest

from . import boundary_extension_context as module
from . import protocol
from .boundary_extension_protocol import ExtensionCandidateConfig
from .formal_base_reference import BaseReference
from .formal_tuning_context import FormalTuningContext
from .formal_tuning_context_a2 import FormalTuningContextA2


def _context(tmp_path):
    carriers = {
        seed: SimpleNamespace(dev_seed=seed) for seed in protocol.DEV_SEEDS
    }
    base = FormalTuningContextA2(
        "BDGP", tmp_path, carriers, {11: {}, 37: {}}, "fp"
    )
    return module.BoundaryExtensionContext(
        base, tmp_path / "bdgp" / "boundary_extension_upper_1e3", {}
    )


def _base(tmp_path, seed):
    return BaseReference(
        "BDGP", seed, tmp_path / f"base{seed}.json", f"s{seed}", f"p{seed}",
        20, 1.0, 1.0, {}, tuple("h" for _ in range(20)), "fp",
    )


def _results(candidate, calls):
    result = {}
    for seed in protocol.DEV_SEEDS:
        for fold in protocol.FOLDS:
            calls.append((seed, fold))
            result[(seed, fold)] = {
                "J_OURS": 1.5,
                "semantic_steps": candidate.semantic_exposure,
                "native_epochs": 20,
                "initial_state_sha256": f"s{seed}",
                "audit": {
                    "supported_anchor_count": 1,
                    "abstained_anchor_count": 0,
                    "coverage": 1.0,
                },
            }
    return result


def _tree_hash(path):
    return {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.iterdir()) if item.is_file()
    }


def test_extension_evaluator_uses_four_frozen_branches_and_isolated_path(monkeypatch, tmp_path):
    context = _context(tmp_path)
    candidate = ExtensionCandidateConfig(1e-3, 5)
    calls = []
    monkeypatch.setattr(
        module, "load_verified_base_reference",
        lambda context, seed: _base(tmp_path, seed),
    )
    monkeypatch.setattr(
        module.frozen_evaluator, "_evaluate_with_runner",
        lambda context, received, device, runner: _results(received, calls),
    )
    result = module.evaluate_extension_candidate(context, candidate, device="cpu")
    assert result.mean_delta_j == 0.5
    assert set(calls) == {(11, "A"), (11, "B"), (37, "A"), (37, "B")}
    path = module.extension_candidate_directory(context, candidate)
    assert "boundary_extension_upper_1e3/candidates/lr_0.001__es_5" in path.as_posix()
    assert not (tmp_path / "bdgp" / "candidates").exists()


def test_verified_extension_candidate_cache_is_reused_byte_identically(monkeypatch, tmp_path):
    context = _context(tmp_path)
    candidate = ExtensionCandidateConfig(1e-3, 3)
    calls = []
    monkeypatch.setattr(
        module, "load_verified_base_reference",
        lambda context, seed: _base(tmp_path, seed),
    )
    monkeypatch.setattr(
        module.frozen_evaluator, "_evaluate_with_runner",
        lambda context, received, device, runner: _results(received, calls),
    )
    module.evaluate_extension_candidate(context, candidate, device="cpu")
    path = module.extension_candidate_directory(context, candidate)
    before = _tree_hash(path)
    calls.clear()
    module.evaluate_extension_candidate(
        context, candidate, device="cpu",
        runner=lambda *args: pytest.fail("cache reuse must not train"),
    )
    assert calls == []
    assert _tree_hash(path) == before


def test_main_schema_still_rejects_extension_lr():
    from .study_schema import CandidateConfig

    with pytest.raises(ValueError, match="UNREGISTERED_SEMANTIC_LR"):
        CandidateConfig(1e-3, 1)

