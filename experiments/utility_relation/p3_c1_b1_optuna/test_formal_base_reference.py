from types import SimpleNamespace

import numpy as np
import pytest

from . import formal_base_reference as module
from .formal_tuning_context import CarrierContext, FormalTuningContext
from . import protocol


def _context(tmp_path):
    carriers = {}
    for seed in protocol.DEV_SEEDS:
        carriers[seed] = CarrierContext(
            dataset="BDGP", dev_seed=seed, directory=tmp_path, state_path=tmp_path / f"state{seed}",
            pre_gt_path=tmp_path / f"pre{seed}", audit_path=tmp_path / f"audit{seed}",
            arrays={}, views=tuple(), adapter=object(), spec=SimpleNamespace(batch_size=1), k=2,
            folds=tuple(), state_file_sha256=f"state-{seed}", pre_gt_sha256=f"pre-{seed}",
        )
    return FormalTuningContext("BDGP", tmp_path, carriers, {11: {}, 37: {}}, "fp")


def test_base_reference_runs_once_per_seed_and_resumes(tmp_path, monkeypatch):
    context = _context(tmp_path)
    calls = {11: 0}

    def runner(carrier, device):
        calls[carrier.dev_seed] = calls.get(carrier.dev_seed, 0) + 1
        return {"model": object(), "native_order_hashes": [f"h{i}" for i in range(20)]}

    monkeypatch.setattr(module, "_evaluate_base", lambda carrier, model, device: (
        {"A": 1.0, "B": 2.0},
        {"A": {"supported_anchor_count": 1, "abstained_anchor_count": 0, "coverage": 1.0},
         "B": {"supported_anchor_count": 1, "abstained_anchor_count": 0, "coverage": 1.0}},
    ))
    first = module.build_base_reference(context, 11, runner=runner)
    second = module.build_base_reference(context, 11, runner=runner)
    assert first.J_BASE_A == second.J_BASE_A == 1.0
    assert calls[11] == 1
    assert first.native_epochs == 20


def test_base_reference_incomplete_cache_fails_closed(tmp_path, monkeypatch):
    context = _context(tmp_path)
    path = module.base_reference_directory(tmp_path, "BDGP", 11)
    path.mkdir(parents=True)
    (path / "partial.json").write_text("partial", encoding="utf-8")
    with pytest.raises(RuntimeError, match="INCOMPLETE_BASE_REFERENCE"):
        module.build_base_reference(context, 11, runner=lambda *_: {})


def test_base_schema_records_single_20_native_schedule():
    source = open(module.__file__, encoding="utf-8").read()
    assert "native_epochs" in source
    assert "J_BASE_A" in source and "J_BASE_B" in source
    assert "candidate" not in source.lower().split("def build_base_reference", 1)[0]

