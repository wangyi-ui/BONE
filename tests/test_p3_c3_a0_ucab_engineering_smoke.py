from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from experiments.utility_relation.p3_c3_a0_ucab import protocol
from experiments.utility_relation.p3_c3_a0_ucab import run_engineering_smoke as smoke


SOURCE = Path(smoke.__file__)


def test_parent_freeze_paths_and_stage_are_fixed():
    assert protocol.STAGE == "P3-C3-A0"
    assert smoke.SMOKE_DATASET == "CCV" and smoke.SMOKE_SEED == 20
    assert all((path / "FREEZE_MANIFEST.sha256").is_file() for path in (
        smoke.DEFINITION_FREEZE, smoke.CONFIRMATORY_FREEZE, smoke.CPU_IMPLEMENTATION_FREEZE,
    ))


def test_fixed_config_rejects_all_non_smoke_inputs():
    smoke.validate_fixed_config("CCV", 20)
    with pytest.raises(ValueError): smoke.validate_fixed_config("ALOI", 20)
    for seed in (140, 150, 170):
        with pytest.raises(ValueError): smoke.validate_fixed_config("CCV", seed)


def test_counting_adam_counts_one_step():
    value = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = smoke.CountingAdam([value], lr=1e-5)
    (value.square().sum()).backward()
    optimizer.step()
    assert optimizer.step_count == 1


def test_output_dir_is_new_and_source_paths_fail_closed(tmp_path: Path):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError): smoke.prepare_output_dir(existing)
    created = smoke.prepare_output_dir(tmp_path / "new")
    assert created.is_dir()
    sources = smoke.SmokeSources(tmp_path / "state", tmp_path / "bundle", tmp_path / "audit", tmp_path / "seal")
    with pytest.raises(FileNotFoundError): smoke.validate_source_paths(sources)


def test_runner_source_has_smoke_arms_parity_step_order_and_firewalls():
    text = SOURCE.read_text()
    assert smoke.SCIENCE_ARMS == ("CURRENT", "TRUE_BUDGET", "MEAN_BUDGET", "SHUFFLE_BUDGET")
    assert "DIRECT_LEGACY_REFERENCE" in text and "legacy._semantic_epoch" in text
    assert "step_count != 1" in text and "NATIVE_ORDER_PARITY_FAILED" in text
    for field in ("FULL_GT_LOADED", "FULL_GT_USED", "ACC_COMPUTED", "NMI_COMPUTED", "ARI_COMPUTED", "METRICS_IMPORTED", "PREDICTION_PATH_USED"):
        assert field in text
    forbidden = ("build_development_carrier", "build_formal_", "amv(", "accuracy_score", "normalized_mutual_info_score", "adjusted_rand_score", "compute_metrics", "_predict")
    assert not any(item in text for item in forbidden)
    assert not any(value in text for value in ("140", "150", "170"))
    tree = ast.parse(text)
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    assert not any("metric" in name.lower() or "evaluate" in name.lower() for name in imports)
