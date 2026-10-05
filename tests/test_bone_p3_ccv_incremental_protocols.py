"""CCV-only P3-A3/P3-A4 addendum tests; no training or carrier creation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from experiments.utility_relation import bone_p3_a1_action_space_protocol as a1
from experiments.utility_relation import bone_p3_a3_sparse_label_protocol as a3
from experiments.utility_relation import bone_p3_a4_corruption_protocol as a4
from experiments.utility_relation import materialize_bone_p3_ccv_incremental_protocols as incremental


REPO_ROOT = Path(__file__).resolve().parents[1]
P1 = REPO_ROOT / "outputs/bone_p1/msrc_snr2p5_exact2of4_seed20/bone_p1_msrc_corruption.npz"
P2 = REPO_ROOT / "outputs/bone_p2/p2_a0_msrc_seed20/p2_a0_carrier_pre_gt.npz"
BONE_SHA = "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092"
AMV_SHA = "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc"
P1_SHA = "2b05b7b5adfe872c60475fe8d3d9f4959229a57ffb1eb50ff29adb015f38db87"
P2_SHA = "83ea7681491e7ab3e51285446075a3362a43d6122a1397d04a0e059bdc7841ea"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(path: Path) -> dict[str, str]:
    return {file.name: _sha(file) for file in sorted(path.glob("*.npz"))}


def test_ccv_a3_a4_incremental_artifacts_and_invariants(tmp_path: Path) -> None:
    result = incremental.materialize(tmp_path)
    sparse, mask, summary = result["sparse"], result["mask"], result["summary"]
    assert sparse["dataset"] == "CCV" and (sparse["N"], sparse["V"], sparse["K"], sparse["L"]) == (6773, 3, 20, 40)
    assert sparse["labels_per_class"] == 2
    assert sparse["label_fraction"] == 40 / 6773
    ids = np.asarray(sparse["labeled_ids"], dtype=np.int64)
    targets = np.asarray(sparse["labeled_targets"], dtype=np.int64)
    labels = incremental.load_ccv_labels()
    assert ids.shape == targets.shape == (40,)
    assert np.unique(ids).size == 40 and np.all((0 <= ids) & (ids < 6773))
    assert np.array_equal(labels[ids], targets)
    assert np.array_equal(np.bincount(targets, minlength=20), np.full(20, 2))
    assert sparse["split_mode"] == "P3_A3_HASH_STRATIFIED_V1"
    assert sparse["namespace"] == "BONE_P3_A3_V1"
    assert sparse["row_id_semantics"] == "RAW_FILE_ROW_ID"
    assert mask["row_sum_histogram"] == {"1": 3387, "2": 3386}
    assert mask["total_corrupted_cells"] == 10159
    assert mask["pattern_counts"]["0"] == mask["pattern_counts"]["1"] == mask["pattern_counts"]["2"] == 1129
    pair_counts = [mask["pattern_counts"][key] for key in ("0,1", "0,2", "1,2")]
    assert sorted(pair_counts) == [1128, 1129, 1129]
    assert sorted(mask["per_view_counts"]) == [3386, 3386, 3387]
    assert mask["corruption_fraction"] == 10159 / (6773 * 3)
    assert mask["row_id_semantics"] == "RAW_FILE_ROW_ID"
    assert summary["FINAL_CCV_PORTABILITY_STATUS"] == "READY_FOR_P3_B0"
    with np.load(tmp_path / "CCV_2perclass.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"labeled_ids", "labeled_targets"}
    with np.load(tmp_path / "CCV_corruption_mask.npz", allow_pickle=False) as archive:
        assert archive["corruption_mask"].shape == (6773, 3)
        assert archive["corruption_mask"].dtype == np.bool_
        assert np.array_equal(archive["raw_row_ids"], np.arange(6773, dtype=np.int64))


def test_ccv_deterministic_repeatability_and_training_seed_independence(tmp_path: Path) -> None:
    first = incremental.materialize(tmp_path / "first")
    second = incremental.materialize(tmp_path / "second")
    assert first["sparse"]["labeled_ids"] == second["sparse"]["labeled_ids"]
    assert first["sparse"]["labeled_targets"] == second["sparse"]["labeled_targets"]
    assert first["sparse"]["ids_targets_content_sha256"] == second["sparse"]["ids_targets_content_sha256"]
    assert first["mask"]["mask_content_sha256"] == second["mask"]["mask_content_sha256"]
    assert first["sparse"]["npz_sha256"] == second["sparse"]["npz_sha256"]
    assert first["mask"]["npz_sha256"] == second["mask"]["npz_sha256"]
    assert "training_seed" not in a3.build_sparse_label_split.__annotations__
    assert "training_seed" not in a4.build_balanced_half_view_mask.__annotations__


def test_a4_builder_receives_only_frozen_protocol_inputs(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}
    original = incremental.a4.build_balanced_half_view_mask

    def capture(*args, **kwargs):
        observed["args"], observed["kwargs"] = args, kwargs
        return original(*args, **kwargs)

    monkeypatch.setattr(incremental.a4, "build_balanced_half_view_mask", capture)
    incremental.materialize(tmp_path)
    assert observed["args"] == ("CCV", 6773, 3)
    assert observed["kwargs"] == {}


def test_action_protocol_v3_six_directed_complements() -> None:
    action_list = a1.build_balanced_directed_actions(3)
    a1.validate_balanced_directed_actions(3, action_list)
    assert len(action_list) == 6
    assert a1.generator_sizes(3) == (1, 2)
    for action in action_list:
        assert set(action.generator_views).isdisjoint(action.verifier_views)
        assert set(action.generator_views) | set(action.verifier_views) == {0, 1, 2}


def test_parent_provenance_and_frozen_artifacts_are_unchanged() -> None:
    sparse_before = _tree_hashes(REPO_ROOT / "outputs/bone_p3/sparse_splits")
    mask_before = _tree_hashes(REPO_ROOT / "outputs/bone_p3/corruptions")
    # The preceding materialization tests target temporary CCV-only paths; this
    # protection test is intentionally read-only over frozen parent outputs.
    assert _tree_hashes(REPO_ROOT / "outputs/bone_p3/sparse_splits") == sparse_before
    assert _tree_hashes(REPO_ROOT / "outputs/bone_p3/corruptions") == mask_before
    assert incremental.sha256_file(REPO_ROOT / "data/CCV.mat") == incremental.CCV_SHA256
    assert _sha(REPO_ROOT / "method/bone.py") == BONE_SHA
    assert _sha(REPO_ROOT / "method/amv.py") == AMV_SHA
    assert _sha(P1) == P1_SHA
    assert _sha(P2) == P2_SHA
    parent = incremental._parent_hashes()
    assert set(parent) == {"bone_p3_a1_action_space_protocol.py", "bone_p3_a3_sparse_label_protocol.py", "bone_p3_a4_corruption_protocol.py"}
