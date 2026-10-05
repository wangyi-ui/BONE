from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.utility_relation.p3_c1_b4_msrc_historical_default_confirmatory import freeze_results


def test_primary_gate_is_not_confirmed_from_existing_metrics():
    summary = freeze_results.summarize_confirmatory()
    assert summary["HISTORICAL_DEFAULT_CONFIRMATION"] == "NOT_CONFIRMED"
    assert summary["mean_deltas"]["ACC"]["mean"] == -0.004761904761904782


def test_freeze_fails_closed_when_metric_identity_is_invalid(monkeypatch, tmp_path):
    bad = {"HISTORICAL_DEFAULT_CONFIRMATION": "CONFIRMED", "per_seed": [], "mean_deltas": {}}
    monkeypatch.setattr(freeze_results, "summarize_confirmatory", lambda _root: bad)
    with pytest.raises(RuntimeError, match="PRIMARY_GATE"):
        freeze_results._verify_summary(bad)
