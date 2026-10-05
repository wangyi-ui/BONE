"""P3-A4 deterministic balanced half-view corruption-mask tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from experiments.utility_relation import bone_p3_a4_corruption_protocol as protocol
from experiments.utility_relation import materialize_bone_p3_a4_corruptions as materializer


REPO_ROOT = Path(__file__).resolve().parents[1]
P1_ARTIFACT = REPO_ROOT / "outputs/bone_p1/msrc_snr2p5_exact2of4_seed20/bone_p1_msrc_corruption.npz"
P2_CARRIER = REPO_ROOT / "outputs/bone_p2/p2_a0_msrc_seed20/p2_a0_carrier_pre_gt.npz"
P1_SHA = "2b05b7b5adfe872c60475fe8d3d9f4959229a57ffb1eb50ff29adb015f38db87"
P2_SHA = "83ea7681491e7ab3e51285446075a3362a43d6122a1397d04a0e059bdc7841ea"
BONE_SHA = "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092"
AMV_SHA = "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(("view_num", "n"), ((2, 100), (3, 60), (4, 60), (5, 200), (6, 200), (7, 210)))
def test_generic_mask_invariants_and_determinism(view_num: int, n: int) -> None:
    first = protocol.build_balanced_half_view_mask("Toy", n, view_num)
    second = protocol.build_balanced_half_view_mask("Toy", n, view_num)
    assert first.dtype == np.bool_ and first.shape == (n, view_num)
    assert np.array_equal(first, second)
    assert protocol.mask_sha256(first) == protocol.mask_sha256(second)
    protocol.validate_balanced_half_view_mask(first, "Toy")
    assert int(first.sum()) == n * view_num // 2
    assert np.all(first.sum(axis=0) == n // 2)


def test_odd_cardinality_groups_and_balanced_extra_patterns() -> None:
    assert protocol.corruption_cardinality_groups(5, 3) == {1: 3, 2: 2}
    mask = protocol.build_balanced_half_view_mask("Odd", 5, 3)
    assert sorted(mask.sum(axis=1).tolist()) == [1, 1, 1, 2, 2]
    assert max(mask.sum(axis=0)) - min(mask.sum(axis=0)) <= 1


def test_current_expected_generic_statistics() -> None:
    expected = {
        "MNIST-USPS": (5000, 2, {"1": 5000}, 2500),
        "BDGP": (2500, 2, {"1": 2500}, 1250),
        "Caltech-5V": (1400, 5, {"2": 700, "3": 700}, 700),
        "HandWrite": (2000, 6, {"3": 2000}, 1000),
        "COIL20": (1440, 3, {"1": 720, "2": 720}, 720),
        "ALOI": (10800, 4, {"2": 10800}, 5400),
    }
    for dataset_name, (n, v, histogram, per_view) in expected.items():
        mask = protocol.build_balanced_half_view_mask(dataset_name, n, v)
        observed = {str(int(value)): int(count) for value, count in zip(*np.unique(mask.sum(axis=1), return_counts=True))}
        assert observed == histogram
        assert np.array_equal(mask.sum(axis=0), np.full(v, per_view))
        for cardinality in protocol.corruption_cardinality_groups(n, v):
            usages = protocol._group_pattern_counts(mask, cardinality)
            assert max(usages) - min(usages) <= 1


def test_msrc_legacy_artifact_is_authoritative_and_balanced() -> None:
    mask, raw_row_ids = materializer._legacy_msrc_mask()
    assert _sha256(P1_ARTIFACT) == P1_SHA
    assert mask.shape == (210, 4)
    assert np.array_equal(raw_row_ids, np.arange(210, dtype=np.int64))
    assert np.array_equal(mask.sum(axis=1), np.full(210, 2))
    assert protocol.pattern_counts(mask) == {"0,1": 35, "0,2": 35, "0,3": 35, "1,2": 35, "1,3": 35, "2,3": 35}
    assert np.array_equal(mask.sum(axis=0), np.full(4, 105))


def test_training_seed_is_not_an_input_and_wrong_corruption_seed_fails_closed() -> None:
    assert "training_seed" not in protocol.build_balanced_half_view_mask.__annotations__
    with pytest.raises(ValueError, match="corruption_seed"):
        protocol.build_balanced_half_view_mask("Toy", 10, 2, corruption_seed=30)


def test_minimal_artifact_repeatability_and_missing_dataset_block(tmp_path: Path) -> None:
    first = materializer.materialize_dataset("BDGP", tmp_path)
    second = materializer.materialize_dataset("BDGP", tmp_path)
    assert first["npz_sha256"] == second["npz_sha256"]
    with np.load(tmp_path / "BDGP_mask.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"corruption_mask", "raw_row_ids"}
    with pytest.raises(ValueError, match="not an approved"):
        materializer.materialize_dataset("CCV", tmp_path)


def test_no_label_feature_rng_or_model_dependency_and_protected_hashes() -> None:
    source = Path(protocol.__file__).read_text(encoding="utf-8") + Path(materializer.__file__).read_text(encoding="utf-8")
    for forbidden in ("np.random", "random.", "import torch", "from torch", "DataLoader(", "optim.", "BONE(", "amv(", "get_cluster_metrics("):
        assert forbidden not in source
    assert _sha256(REPO_ROOT / "method/bone.py") == BONE_SHA
    assert _sha256(REPO_ROOT / "method/amv.py") == AMV_SHA
    assert _sha256(P1_ARTIFACT) == P1_SHA
    assert _sha256(P2_CARRIER) == P2_SHA
