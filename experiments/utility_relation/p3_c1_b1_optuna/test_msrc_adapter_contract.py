from pathlib import Path

import numpy as np
import pytest

from experiments.utility_relation.p3_c1_b1_optuna import protocol
from experiments.utility_relation.p3_c1_b1_optuna.msrc_adapter import (
    MSRC_SPEC, _load_frozen_msrc_inputs, expected_msrc_carrier_dir, validate_msrc_request,
)
from experiments.utility_relation.p3_c1_b1_optuna.parent_integrity import sha256_file


ROOT = Path(__file__).resolve().parents[3]


def test_msrc_uses_frozen_common_search_protocol_and_candidate():
    assert "MSRC" in protocol.DATASETS and len(protocol.DATASETS) == 8
    assert 1e-5 in protocol.SEMANTIC_LR_GRID
    assert 5 in protocol.SEMANTIC_EXPOSURE_GRID
    candidate = validate_msrc_request("MSRC", 11, 1e-5, 5, "cuda:0")
    assert candidate.semantic_lr == 1e-5 and candidate.semantic_exposure == 5
    with pytest.raises(ValueError):
        validate_msrc_request("MSRC", 37, 1e-5, 5, "cuda:0")
    with pytest.raises(ValueError):
        validate_msrc_request("MSRC", 11, 3e-5, 5, "cuda:0")


def test_frozen_sparse_corruption_and_action_inputs_are_reused_without_gt():
    views, ids, targets, split_path, mask_path = _load_frozen_msrc_inputs()
    assert len(views) == MSRC_SPEC.v == 4
    assert all(view.shape[0] == MSRC_SPEC.n == 210 for view in views)
    assert ids.shape == targets.shape == (14,)
    assert np.array_equal(np.bincount(targets, minlength=7), np.full(7, 2))
    assert split_path.name == "MSRC_2perclass.npz"
    assert mask_path.name == "MSRC_mask.npz"


def test_output_namespace_and_parent_sources_are_unchanged():
    output = expected_msrc_carrier_dir(
        "outputs/bone_p3/p3_c1_b1_optuna/engineering_msrc_adapter_smoke"
    )
    assert output == Path(
        "outputs/bone_p3/p3_c1_b1_optuna/engineering_msrc_adapter_smoke/msrc/dev_seed11/carrier"
    )
    assert "p3_c0_formal" not in str(output) and "formal_tuning" not in str(output)
    parent = ROOT / "method/bone.py"
    before = sha256_file(parent)
    validate_msrc_request("MSRC", 11, 1e-5, 5, "cuda:0")
    assert sha256_file(parent) == before


def test_adapter_defines_no_independent_grid_or_full_gt_loader():
    source = Path(__file__).with_name("msrc_adapter.py").read_text(encoding="utf-8")
    assert "SEMANTIC_LR_GRID =" not in source
    assert "SEMANTIC_EXPOSURE_GRID =" not in source
    assert "loadmat" not in source and '["labels"]' not in source
    assert "get_cluster_metrics" not in source
