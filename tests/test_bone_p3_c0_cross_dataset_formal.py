"""Offline protocol tests; no P3-C0 GPU training or GT evaluation is started."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.utility_relation import bone_p3_b0_generic_carrier as b0
from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as protocol
from experiments.utility_relation import evaluate_bone_p3_c0_cross_dataset_formal as evaluator
from experiments.utility_relation import run_bone_p3_c0_cross_dataset_formal as runner
from experiments.utility_relation import summarize_bone_p3_c0_cross_dataset_formal as summarizer


REPO_ROOT = Path(__file__).resolve().parents[1]
B1_SHA = "d686d59a2a7000eecb2626413b0640edb95a4d25de0db2c8834fa402cddef771"
PARENT_HASHES = {
    "a1": "5564354b41f78917b1ad4f98ae8737fe997d3726902e5e2a97e1ec024e20d61a",
    "a3": "07571e80de1917e7358e8d614a3fb8d159b8ee635331f2f4c83e709acc387e7a",
    "a4": "1865741c1ac3f2f65ffb9c325dc30a7d34978f96adc1be0ad000b45ed899b3c9",
    "b0": "5a9f3c817d0e0ea07e311f0899fdb926cef993d0f3b7edbd65faf7f83dfabc27",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_constants_schedule_seed_set_and_dataset_registry() -> None:
    assert protocol.FORMAL_SEEDS == (20, 30, 50)
    assert (protocol.CARRIER_EPOCHS, protocol.REFINEMENT_EPOCHS) == (100, 20)
    assert (protocol.CORRUPTION_SEED, protocol.SNR_DB, protocol.QUERY_CHUNK_SIZE) == (20, 2.5, 256)
    assert set(protocol.FORMAL_DATASETS) == {"Caltech-5V", "BDGP", "HandWrite", "COIL20", "MNIST-USPS", "ALOI", "CCV"}
    assert "MSRC" not in protocol.FORMAL_DATASETS
    assert protocol.DATASET_SPECS["Caltech-5V"].dims == (40, 254, 1984, 512, 928)


def test_formal_semantic_schedule_is_seeded_complete_and_not_selection_driven() -> None:
    first, same, other = protocol.semantic_orders(31, 20), protocol.semantic_orders(31, 20), protocol.semantic_orders(31, 30)
    assert first.shape == (20, 31) and np.array_equal(first, same) and not np.array_equal(first, other)
    assert all(np.array_equal(np.sort(order), np.arange(31)) for order in first)


def test_frozen_a1_a3_a4_b0_reuse_and_caltech_shape_contract() -> None:
    labels, targets, mask, _hashes = runner._load_frozen_inputs("Caltech-5V")
    assert labels.shape == targets.shape == (14,) and mask.shape == (1400, 5)
    assert np.array_equal(np.bincount(targets, minlength=7), np.full(7, 2))
    shapes = protocol.expected_shapes("Caltech-5V", 7)
    assert shapes == {"q_views": (1400, 5, 7), "p_gen": (1400, 20, 7), "p_ver": (1400, 20, 7),
                      "U_cycle": (1400, 20), "Pi": (1386, 20, 7), "T_rel": (1386, 14, 20)}
    actions, _generator, _verifier = b0.action_arrays(5)
    assert len(actions) == 20


def test_query_chunking_is_carrier_equivalent() -> None:
    n, v, k = 23, 5, 3
    values = np.arange(1, n * v * k + 1, dtype=np.float32).reshape(n, v, k)
    q = values / values.sum(axis=2, keepdims=True)
    ids = b0.native_shared_permutation(n)
    labeled_ids, targets = ids[[1, 2, 5, 6, 9, 10]], np.repeat(np.arange(k), 2)
    full, chunked = b0.build_generic_carrier(q, ids, labeled_ids, targets), b0.build_generic_carrier(q, ids, labeled_ids, targets, query_chunk_size=4)
    for name in ("p_gen", "p_ver", "U_cycle", "Pi", "T_rel"):
        assert np.array_equal(full[name], chunked[name])


def test_state_hash_shared_carrier_parity_contract() -> None:
    state = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    assert protocol.state_dict_sha256(state) == protocol.state_dict_sha256({"weight": state["weight"].clone()})
    changed = {"weight": state["weight"].clone()}; changed["weight"][0, 0] += 1
    assert protocol.state_dict_sha256(state) != protocol.state_dict_sha256(changed)


def test_runner_is_opt_in_pre_gt_and_uses_separate_phase_optimizers() -> None:
    args = runner.build_parser().parse_args(["--dataset", "Caltech-5V", "--training-seed", "20"])
    assert args.execute is False
    source = Path(runner.__file__).read_text(encoding="utf-8")
    for forbidden in ('mat["Y"]', "get_cluster_metrics", "adjusted_rand_score", "native_loss +", "best_epoch", "best_seed"):
        assert forbidden not in source
    assert "optimizer.step()  # Exactly one semantic optimizer step" in source and "ours_native_optimizer" in source
    assert "Exactly one semantic optimizer step" in source
    assert '"base_native_epochs": 20' in source and '"ours_native_phase_b_epochs": 20' in source
    assert "final_model_state.pt" in source and "best_checkpoint" not in source


def test_evaluator_metrics_are_deterministic_and_include_ari() -> None:
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    predictions = np.asarray([7, 7, 3, 3, 9, 9])
    result = evaluator.compute_metrics(labels, predictions)
    assert set(result) == {"ACC", "NMI", "Purity", "ARI"}
    assert result == pytest.approx({"ACC": 1.0, "NMI": 1.0, "Purity": 1.0, "ARI": 1.0})
    assert evaluator.cluster_acc_hungarian(labels, predictions) == evaluator.cluster_acc_hungarian(labels, predictions)


def test_evaluator_refuses_bad_seal_before_gt_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(evaluator.scipy.io, "loadmat", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("GT must remain sealed")))
    with pytest.raises(ValueError, match="missing P3-C0"):
        evaluator.evaluate_seed("Caltech-5V", 20, tmp_path)


def _write_metrics(root: Path, dataset: str, seed: int, base: float, ours: float) -> None:
    path = root / protocol.dataset_slug(dataset) / f"seed{seed}"; path.mkdir(parents=True)
    metrics = {"dataset": dataset, "training_seed": seed,
               "BASE": {name: base for name in summarizer.METRICS}, "OURS": {name: ours for name in summarizer.METRICS},
               "delta_OURS_minus_BASE": {name: ours - base for name in summarizer.METRICS}}
    (path / "formal_metrics.json").write_text(json.dumps(metrics))


def test_one_dataset_summarizer_reports_mean_std_and_no_performance_gate(tmp_path: Path) -> None:
    for seed, base, ours in ((20, .4, .5), (30, .5, .4), (50, .6, .8)): _write_metrics(tmp_path, "Caltech-5V", seed, base, ours)
    summary = summarizer.summarize_dataset("Caltech-5V", tmp_path)
    item = summary["metrics"]["ACC"]
    assert item["BASE"]["mean"] == pytest.approx(.5) and item["delta_OURS_minus_BASE"]["positive_delta_seed_count"] == 2
    assert "verdict" not in summary and "selection" in summary["aggregation"]


def test_native_and_all_parent_freezes_including_b1_are_unchanged() -> None:
    assert protocol.verify_native_hashes() == protocol.NATIVE_HASHES
    assert protocol.verify_parent_freeze_hashes() == protocol.PARENT_FREEZE_HASHES
    assert _sha(REPO_ROOT / "experiments/utility_relation/bone_p3_a1_action_space_protocol.py") == PARENT_HASHES["a1"]
    assert _sha(REPO_ROOT / "experiments/utility_relation/bone_p3_a3_sparse_label_protocol.py") == PARENT_HASHES["a3"]
    assert _sha(REPO_ROOT / "experiments/utility_relation/bone_p3_a4_corruption_protocol.py") == PARENT_HASHES["a4"]
    assert _sha(REPO_ROOT / "experiments/utility_relation/bone_p3_b0_generic_carrier.py") == PARENT_HASHES["b0"]
    assert _sha(REPO_ROOT / "experiments/utility_relation/run_bone_p3_b1_caltech_engineering_smoke.py") == B1_SHA
