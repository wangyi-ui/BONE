from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/utility_relation/p3_c2_b0_utility_mass_action_budget_audit.py"
SPEC = importlib.util.spec_from_file_location("p3c2b0", SOURCE)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def test_exact_loss_reconstruction_and_chunk_parity():
    torch.manual_seed(7)
    logits = torch.randn(5, 3, dtype=torch.float64)
    target = torch.rand(5, 3, 2, dtype=torch.float64)
    utility = torch.tensor([[.2, .7], [.1, .4], [.6, .3], [.8, .5], [.2, .9]], dtype=torch.float64)
    relation = torch.sigmoid(logits)
    direct, _ = audit.semantic_relation_loss(relation, target, utility, 4)
    chunked = audit.reconstruct_semantic_loss([relation], target, utility, 4, 2)
    assert torch.allclose(direct, chunked, rtol=1e-13, atol=1e-13)
    data = audit.synthetic_scale_audit()
    assert data["UTILITY_MASS_SCALE_INVARIANT"] is True
    assert data["GRADIENT_MASS_SCALE_INVARIANT"] is True


def test_utility_validation_and_zero_safe_behavior():
    with pytest.raises(ValueError): audit._validate_utility(np.array([[np.nan]]))
    with pytest.raises(ValueError): audit._validate_utility(np.array([[-.1]]))
    with pytest.raises(ValueError): audit.semantic_relation_loss(torch.full((1, 1), .5), torch.full((1, 1, 1), .5), torch.zeros((1, 1)), 2)


def test_chunked_relation_mass_parity():
    rng = np.random.default_rng(3)
    t = rng.random((7, 4, 2), dtype=np.float64)
    u = rng.random((7, 2), dtype=np.float64)
    observed = audit.chunked_relation_mass(t, u, 5, chunk_size=2)
    positive = np.sum(u[:, None, :] * t)
    negative = np.sum(u[:, None, :] * (1 - t))
    assert observed["positive_relation_weight_mass"] == pytest.approx(positive)
    assert observed["negative_relation_weight_mass"] == pytest.approx(negative)


def test_source_is_diagnostic_only_and_parent_hashes_are_readable():
    tree = ast.parse(SOURCE.read_text())
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    assert not any("evaluate" in name or "train" in name for name in imported)
    text = SOURCE.read_text().lower()
    assert "loadmat" not in text
    assert text.count("formal_metrics") == 1  # required status declaration only; no reader is imported.
    assert audit.verify_parent_integrity() == audit.EXPECTED_PARENT_HASHES
