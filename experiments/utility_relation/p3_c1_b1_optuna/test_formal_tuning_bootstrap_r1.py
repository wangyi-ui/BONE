from types import SimpleNamespace

import pytest

from . import formal_tuning_bootstrap_r1 as module
from .formal_base_reference import BaseReference
from . import protocol


def _references(dataset="BDGP"):
    return {
        seed: BaseReference(
            dataset, seed, __import__("pathlib").Path("base.json"), f"s{seed}", f"p{seed}",
            protocol.REFINEMENT_HORIZON, 0.0, 0.0, {}, tuple("h" for _ in range(20)), "fp",
        )
        for seed in protocol.DEV_SEEDS
    }


def test_bootstrap_materializes_and_reverifies_both_seeds_once():
    context = SimpleNamespace(dataset="BDGP")
    calls = []
    references = _references()

    def builder(ctx, seed, **kwargs):
        calls.append(("build", seed, kwargs["device"]))
        return references[seed]

    def verifier(ctx, seed):
        calls.append(("verify", seed))
        return references[seed]

    result = module.bootstrap_base_references(context, "cpu", builder=builder, verifier=verifier)
    assert set(result) == set(protocol.DEV_SEEDS)
    assert calls == [("build", 11, "cpu"), ("build", 37, "cpu"), ("verify", 11), ("verify", 37)]
    assert module.verify_bootstrap_complete(context, result) is True


def test_bootstrap_failure_is_fail_closed():
    context = SimpleNamespace(dataset="BDGP")

    def failing_builder(ctx, seed, **kwargs):
        raise FileNotFoundError("BASE_REFERENCE_MISSING")

    with pytest.raises(FileNotFoundError, match="BASE_REFERENCE_MISSING"):
        module.bootstrap_base_references(context, "cpu", builder=failing_builder)


def test_bootstrap_uses_frozen_20_epoch_contract():
    assert protocol.REFINEMENT_HORIZON == 20
    assert "build_base_reference" in open(module.__file__, encoding="utf-8").read()
    assert "load_verified_base_reference" in open(module.__file__, encoding="utf-8").read()


def test_r1_orchestration_is_gt_free_and_protocol_neutral():
    from pathlib import Path
    root = Path(module.__file__).parent
    text = "\n".join((root / name).read_text(encoding="utf-8") for name in (
        "formal_tuning_bootstrap_r1.py", "run_formal_optuna_tuning_r1.py",
    ))
    assert "full_gt_used" in text
    assert not any(token in text for token in ("ACC", "NMI", "Purity", "ARI", "FORMAL_SEEDS"))
    assert protocol.DEV_SEEDS == (11, 37)
    assert protocol.SEMANTIC_LR_GRID == (3e-6, 1e-5, 3e-5, 1e-4, 3e-4)
    assert protocol.SEMANTIC_EXPOSURE_GRID == (1, 3, 5, 10, 20)
