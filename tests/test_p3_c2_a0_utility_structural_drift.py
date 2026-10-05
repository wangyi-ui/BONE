from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.utility_relation import p3_c2_a0_utility_structural_drift as drift


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_pair(
    root: Path,
    *,
    carrier_ids=(10, 20, 30, 40),
    query_ids=(10, 20, 30, 40),
    base_ids=(10, 20, 30, 40),
    base=(0, 0, 1, 1),
    ours_ids=(10, 20, 30, 40),
    ours=(1, 1, 0, 0),
    utility=None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    carrier_ids = np.asarray(carrier_ids, dtype=np.int64)
    query_ids = np.asarray(query_ids, dtype=np.int64)
    positions = np.asarray([np.where(carrier_ids == value)[0][0] for value in query_ids], dtype=np.int64)
    if utility is None:
        utility = np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    utility = np.asarray(utility)
    full_utility = np.zeros((carrier_ids.size, utility.shape[1]), dtype=utility.dtype)
    full_utility[positions] = utility
    carrier = root / "carrier" / "carrier_pre_gt.npz"
    carrier.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        carrier, sample_ids=carrier_ids, query_sample_ids=query_ids, query_rows=positions,
        U_cycle=full_utility, query_U_cycle=utility,
    )
    files = []
    for arm, ids, labels in (("BASE", base_ids, base), ("OURS", ours_ids, ours)):
        path = root / arm / "pre_gt_predictions.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, sample_ids=np.asarray(ids, dtype=np.int64), predictions=np.asarray(labels, dtype=np.int64))
        files.append(path)
    config = {"dataset": "HandWrite", "training_seed": 20, "semantic_lr": 1e-5, "semantic_exposure": 1}
    audit = {"dataset": "HandWrite", "training_seed": 20}
    for field in (
        "GT_loaded_before_seal", "full_gt_used_for_training", "full_gt_used_for_model_selection",
        "full_gt_used_for_hyperparameter_selection", "ACC_NMI_ARI_computed",
    ):
        audit[field] = False
    _write_json(root / "RUN_CONFIG.json", config)
    _write_json(root / "PRE_GT_AUDIT.json", audit)
    hashed = {str(path.relative_to(root)): _sha(path) for path in [root / "RUN_CONFIG.json", root / "PRE_GT_AUDIT.json", carrier, *files]}
    _write_json(root / "PRE_GT_SEAL.json", {"status": "PRE_GT_SEALED", "GT_loaded_before_seal": False, "artifact_sha256": hashed})
    _write_json(root / "COMPLETE.json", {"status": "COMPLETE", "dataset": "HandWrite", "training_seed": 20})
    return root


def _loaded(root: Path) -> dict:
    source = root.parents[1]
    return drift.load_validated_pre_gt_pair("HandWrite", 20, source)


def test_sample_id_permutation_is_explicitly_aligned(tmp_path):
    root = _make_pair(
        tmp_path / "handwrite" / "seed20",
        base_ids=(40, 10, 30, 20), base=(1, 0, 1, 0),
        ours_ids=(20, 30, 10, 40), ours=(1, 0, 1, 0),
    )
    loaded = _loaded(root)
    assert loaded["base_query"].tolist() == [0, 0, 1, 1]
    assert loaded["ours_query"].tolist() == [1, 1, 0, 0]


def test_duplicate_and_missing_ids_fail_closed(tmp_path):
    duplicate = _make_pair(tmp_path / "duplicate" / "handwrite" / "seed20", base_ids=(10, 10, 30, 40))
    with pytest.raises(ValueError, match="DUPLICATE"):
        _loaded(duplicate)
    missing = _make_pair(tmp_path / "missing" / "handwrite" / "seed20", ours_ids=(10, 20, 30, 99))
    with pytest.raises(ValueError, match="FULL_SAMPLE_ID_SET_MISMATCH"):
        _loaded(missing)


def test_pure_cluster_permutation_has_zero_flips_and_full_agreement(tmp_path):
    result = drift.compute_diagnostic(_loaded(_make_pair(tmp_path / "handwrite" / "seed20")))
    assert result["query_flip_count"] == 0
    assert result["query_flip_rate"] == 0.0
    assert result["agreement_nmi_query"] == pytest.approx(1.0)
    assert result["agreement_ari_query"] == pytest.approx(1.0)
    assert result["flip_supported_fraction"] is None
    assert result["mean_utility_flip"] is None


def test_true_change_rectangular_mapping_utility_rates_and_occupancy(tmp_path):
    root = _make_pair(
        tmp_path / "handwrite" / "seed20",
        carrier_ids=(10, 20, 30, 40, 50, 60, 70, 80),
        query_ids=(10, 20, 30, 40, 50, 60, 70, 80),
        base_ids=(10, 20, 30, 40, 50, 60, 70, 80), base=(0, 0, 0, 0, 1, 1, 1, 1),
        ours_ids=(10, 20, 30, 40, 50, 60, 70, 80), ours=(0, 0, 0, 0, 1, 1, 1, 2),
        utility=[[1.0], [1.0], [2.0], [3.0], [0.0], [0.0], [0.0], [0.0]],
    )
    result = drift.compute_diagnostic(_loaded(root))
    assert result["query_flip_count"] == 1
    assert result["supported_flip_count"] == 0
    assert result["abstained_flip_count"] == 1
    assert result["supported_flip_rate"] == 0.0
    assert result["abstained_flip_rate"] == 0.25
    assert result["flip_abstained_fraction"] == 1.0
    assert result["hungarian_mapping_audit"]["contingency_shape"] == [2, 3]
    assert result["hungarian_mapping_audit"]["unmatched_ours_clusters"] == [2]
    assert result["occupancy_tv_query"] == pytest.approx(0.125)
    assert result["base_query_entropy_norm"] == pytest.approx(1.0)
    assert result["ours_query_entropy_norm"] == pytest.approx(
        -(0.5 * np.log(0.5) + 0.375 * np.log(0.375) + 0.125 * np.log(0.125)) / np.log(3)
    )


def test_negative_nonfinite_and_shape_mismatch_utility_fail(tmp_path):
    for name, utility, expected in (
        ("negative", [[-1.0], [1.0], [1.0], [1.0]], "INVALID_QUERY_UTILITY"),
        ("nan", [[np.nan], [1.0], [1.0], [1.0]], "INVALID_QUERY_UTILITY"),
    ):
        root = _make_pair(tmp_path / name / "handwrite" / "seed20", utility=utility)
        with pytest.raises(ValueError, match=expected):
            _loaded(root)
    root = _make_pair(tmp_path / "shape" / "handwrite" / "seed20")
    carrier = root / "carrier" / "carrier_pre_gt.npz"
    np.savez_compressed(carrier, sample_ids=np.arange(4), query_sample_ids=np.arange(4), query_rows=np.arange(4), U_cycle=np.ones((4, 1)), query_U_cycle=np.ones((3, 1)))
    with pytest.raises(RuntimeError, match="SEAL_HASH_MISMATCH"):
        _loaded(root)


def test_all_flip_primitives_and_tied_quartiles_are_safe_and_deterministic(tmp_path):
    root = _make_pair(
        tmp_path / "handwrite" / "seed20", carrier_ids=(40, 10, 30, 20), query_ids=(40, 10, 30, 20),
        base_ids=(40, 10, 30, 20), base=(0, 0, 0, 0), ours_ids=(40, 10, 30, 20), ours=(1, 1, 1, 1),
        utility=[[1.0], [1.0], [1.0], [1.0]],
    )
    result = drift.compute_diagnostic(_loaded(root))
    quartiles = result["supported_utility_rank_quartiles"]
    assert [item["sample_ids_ascending_utility_tiebreak"] for item in quartiles] == [[10], [20], [30], [40]]
    assert result["spearman_all"] == {"rho": None, "p_value": None, "reason": "CONSTANT_INPUT"}
    assert drift._safe_mean(np.asarray([], dtype=float)) is None
    assert drift._spearman(np.ones(4), np.ones(4, dtype=np.int64))["reason"] == "CONSTANT_INPUT"


def test_no_gt_metrics_or_training_imports_and_exact_pairs():
    source = Path(drift.__file__).read_text(encoding="utf-8")
    assert "evaluate_final_formal" not in source
    assert "FORMAL_METRICS.json" not in source
    assert "loadmat" not in source
    assert "torch" not in source
    assert drift.DATASETS == ("HandWrite", "BDGP", "MSRC", "COIL20", "Caltech-5V", "MNIST-USPS", "ALOI", "CCV")
    assert drift.SEEDS == (20, 30, 50)
    assert len([(dataset, seed) for dataset in drift.DATASETS for seed in drift.SEEDS]) == 24


def test_schema_preflight_exact_24_with_mock_loader(monkeypatch):
    def fake_loader(dataset, seed, _root):
        return {"carrier_ids": np.arange(8), "query_ids": np.arange(4), "query_utility": np.ones((4, 2)), "root": Path(f"/{dataset}/{seed}")}
    monkeypatch.setattr(drift, "load_validated_pre_gt_pair", fake_loader)
    result = drift.schema_preflight("irrelevant")
    assert result["ALL_24_SCHEMA_PREFLIGHT_PASS"] is True
    assert result["RESOLVABLE_PAIR_COUNT"] == 24
    assert result["FULL_GT_READ"] is False
