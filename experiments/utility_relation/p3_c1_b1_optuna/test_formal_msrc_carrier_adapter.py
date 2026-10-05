from pathlib import Path

import numpy as np
import pytest

from experiments.utility_relation import bone_p3_b0_generic_carrier as b0
from experiments.utility_relation.p3_c1_b1_optuna import (
    formal_msrc_carrier_adapter as formal_adapter,
    msrc_adapter,
    protocol,
)
from experiments.utility_relation.p3_c1_b1_optuna.carrier_adapter import _carrier_arrays


def test_formal_seed11_and_seed37_supported_from_frozen_protocol():
    assert protocol.DEV_SEEDS == (11, 37)
    for seed in protocol.DEV_SEEDS:
        formal_adapter.validate_formal_msrc_request(seed, "cuda:0")
    for seed in (20, 30, 50, 123):
        with pytest.raises(ValueError, match="UNREGISTERED_DEV_SEED"):
            formal_adapter.validate_formal_msrc_request(seed, "cuda:0")
    source = Path(formal_adapter.__file__).read_text(encoding="utf-8")
    assert "protocol.DEV_SEEDS" in source
    assert "ONLY_DEV_SEED11_IS_ALLOWED" not in source


def test_seed_independent_scientific_configuration_matches_frozen_adapter():
    config = formal_adapter.scientific_configuration()
    assert formal_adapter.MSRC_SPEC is msrc_adapter.MSRC_SPEC
    assert config["dataset"] == "MSRC"
    assert config["carrier_epochs"] == protocol.CARRIER_EPOCHS == 100
    assert config["native_lr"] == msrc_adapter.p2_carrier.LR
    assert config["batch_size"] == msrc_adapter.MSRC_SPEC.batch_size
    assert config["feature_dims"] == tuple(msrc_adapter.MSRC_SPEC.dims)
    assert config["n_anchor"] == msrc_adapter.MSRC_SPEC.n_anchor
    assert config["layer"] == msrc_adapter.MSRC_SPEC.layer
    assert config["snr_db"] == msrc_adapter.p1_protocol.TARGET_SNR_DB == 2.5
    assert config["corruption_seed"] == msrc_adapter.p1_protocol.CORRUPTION_SEED == 20
    assert config["action_count"] == 6
    assert config["sparse_label_sha256"] == msrc_adapter.p3_b0_materializer.sha256_file(
        config["sparse_label_path"]
    )
    assert config["corruption_sha256"] == msrc_adapter.p3_b0_materializer.sha256_file(
        config["corruption_path"]
    )


def test_pre_gt_schema_action_space_and_u_formula_match_frozen_primitives():
    q = np.full((msrc_adapter.MSRC_SPEC.n, msrc_adapter.MSRC_SPEC.v, 7), 1 / 7, dtype=np.float32)
    arrays = _carrier_arrays(q)
    _, labeled_ids, labeled_targets, _, _ = msrc_adapter._load_frozen_msrc_inputs()
    arrays.update({
        "sample_ids": np.arange(msrc_adapter.MSRC_SPEC.n, dtype=np.int64),
        "labeled_ids": labeled_ids,
        "labeled_targets": labeled_targets,
    })
    formal_adapter.validate_generated_arrays(arrays)
    actions, generator, verifier = b0.action_arrays(msrc_adapter.MSRC_SPEC.v)
    assert np.array_equal(arrays["generator_actions"], generator)
    assert np.array_equal(arrays["verifier_actions"], verifier)
    assert arrays["U_cycle"].shape == (210, len(actions))
    broken = dict(arrays); broken["U_cycle"] = arrays["U_cycle"].copy(); broken["U_cycle"][0, 0] += .1
    with pytest.raises(ValueError, match="U_FORMULA_PARITY_FAILURE"):
        formal_adapter.validate_generated_arrays(broken)


def test_formal_change_is_execution_restriction_removal_only():
    assert formal_adapter.ACTIVE_PROTOCOL == "P3-C1-B1-A1"
    source = Path(formal_adapter.__file__).read_text(encoding="utf-8")
    assert "_load_frozen_msrc_inputs()" in source
    assert "_carrier_arrays(q_views)" in source
    assert "protocol.CARRIER_EPOCHS" in source
    assert "objective" not in source.lower()
