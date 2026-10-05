"""P3-B0 generic carrier mechanics, with no new BONE training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from experiments.utility_relation import bone_p3_a1_action_space_protocol as actions
from experiments.utility_relation import bone_p3_b0_generic_carrier as generic
from experiments.utility_relation import materialize_bone_p3_b0_carrier as materializer
from experiments.weak_quality.materialize_bone_p1_msrc_corruption import corrupt_standardized_views


REPO_ROOT = Path(__file__).resolve().parents[1]
P1_ARTIFACT = REPO_ROOT / "outputs/bone_p1/msrc_snr2p5_exact2of4_seed20/bone_p1_msrc_corruption.npz"
P2_CARRIER = REPO_ROOT / "outputs/bone_p2/p2_a0_msrc_seed20/p2_a0_carrier_pre_gt.npz"
P2_STATE = REPO_ROOT / "outputs/bone_p2/p2_a0_msrc_seed20/carrier_model_state.pt"
P1_SHA = "2b05b7b5adfe872c60475fe8d3d9f4959229a57ffb1eb50ff29adb015f38db87"
P2_SHA = "83ea7681491e7ab3e51285446075a3362a43d6122a1397d04a0e059bdc7841ea"
BONE_SHA = "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092"
AMV_SHA = "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _q(n: int = 11, v: int = 3, k: int = 2) -> np.ndarray:
    values = np.arange(1, n * v * k + 1, dtype=np.float32).reshape(n, v, k)
    return values / values.sum(axis=2, keepdims=True)


def _sparse_ids_for_permuted_rows(sample_ids: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    # Use raw IDs in deliberately non-tensor order to prove adapter mapping.
    return np.asarray((sample_ids[4], sample_ids[1], sample_ids[8], sample_ids[2]), dtype=np.int64), np.asarray((0, 0, 1, 1), dtype=np.int64)


def test_sample_id_adapter_shuffle_and_dataloader_alignment() -> None:
    adapter = generic.SampleIdAdapter.from_native_shuffle(9)
    assert np.array_equal(adapter.sample_ids, np.random.RandomState(42).permutation(9))
    native_views = [np.arange(18, dtype=np.float32).reshape(2, 9), np.arange(27, dtype=np.float32).reshape(3, 9)]
    dataset = generic.RawIdDataset(native_views, adapter)
    batches = list(DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0))
    observed_ids = torch.cat([batch_ids for _views, batch_ids in batches]).numpy()
    assert np.array_equal(observed_ids, adapter.sample_ids)
    first_views, first_ids = batches[0]
    assert int(first_ids[0]) == int(adapter.sample_ids[0])
    assert np.array_equal(first_views[0][0].numpy(), native_views[0][:, 0])


def test_preprocessing_maps_mask_by_raw_id_and_noise_formula_matches_p1() -> None:
    raw_views = [np.arange(18, dtype=np.float32).reshape(6, 3), np.arange(12, dtype=np.float32).reshape(6, 2)]
    raw_mask = np.zeros((6, 2), dtype=bool)
    raw_mask[[0, 2, 5], 0] = True
    raw_mask[[1, 3, 4], 1] = True
    corrupted, adapter = generic.preprocess_with_raw_ids(raw_views, raw_mask)
    assert np.array_equal(adapter.sample_ids, generic.native_shared_permutation(6))
    assert all(view.shape[0] == 6 for view in corrupted)
    with np.load(P1_ARTIFACT, allow_pickle=False) as archive:
        clean = [archive[f"x_clean_v{i}"] for i in range(4)]
        mask = archive["corruption_mask"]
    historical, _noise, _alpha, _snr = corrupt_standardized_views(clean, mask)
    generic_result = generic.apply_historical_noise_formula(clean, mask)
    for actual, expected in zip(generic_result, historical):
        assert np.array_equal(actual, expected)


def test_sparse_raw_anchor_mapping_fails_closed_on_missing_or_duplicate() -> None:
    adapter = generic.SampleIdAdapter(np.asarray((3, 0, 2, 1), dtype=np.int64))
    assert np.array_equal(adapter.row_positions((0, 3, 1)), np.asarray((1, 0, 3)))
    with pytest.raises(ValueError, match="unique"):
        adapter.row_positions((0, 0))
    with pytest.raises(ValueError, match="absent"):
        adapter.row_positions((4,))


def test_q_hook_shape_and_action_protocol_is_imported() -> None:
    class HookModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.label_fc = torch.nn.Identity()

        def forward(self, views, test: bool = False):
            assert test
            posteriors = [self.label_fc(view) for view in views]
            return {"label": torch.argmax(torch.stack(posteriors).mean(dim=0), dim=1)}

    q = _q(n=7, v=3, k=2)
    observed = generic.extract_q_views(HookModel(), [q[:, view] for view in range(3)])
    assert observed.shape == (7, 3, 2)
    assert np.allclose(observed.sum(axis=2), 1.0)
    action_list, generator, verifier = generic.action_arrays(3)
    assert action_list == actions.build_balanced_directed_actions(3)
    assert np.array_equal(generator, np.asarray(((0, -1), (1, -1), (2, -1), (0, 1), (0, 2), (1, 2))))
    assert np.array_equal(verifier, np.asarray(((1, 2), (0, 2), (0, 1), (2, -1), (1, -1), (0, -1))))


def test_generic_u_t_shapes_formula_and_detached_artifact() -> None:
    q = _q()
    sample_ids = generic.native_shared_permutation(q.shape[0])
    labeled_ids, labeled_targets = _sparse_ids_for_permuted_rows(sample_ids)
    payload = generic.build_generic_carrier(q, sample_ids, labeled_ids, labeled_targets)
    assert payload["p_gen"].shape == (11, 6, 2)
    assert payload["p_ver"].shape == (11, 6, 2)
    assert payload["U_cycle"].shape == (11, 6)
    assert payload["T_rel"].shape == (7, 4, 6)
    assert payload["Pi"].shape == (7, 6, 2)
    y = np.argmax(payload["p_gen"], axis=2)
    expected = (np.argmax(payload["p_ver"], axis=2) == y) * np.sqrt(
        np.take_along_axis(payload["p_gen"], y[..., None], axis=2)[..., 0]
        * np.take_along_axis(payload["p_ver"], y[..., None], axis=2)[..., 0]
    )
    assert np.array_equal(payload["U_cycle"], expected)
    assert not isinstance(payload["U_cycle"], torch.Tensor)
    assert np.allclose(payload["Pi"].sum(axis=2), 1.0)
    assert np.array_equal(payload["anchor_rows"], generic.SampleIdAdapter(sample_ids).row_positions(labeled_ids))


@pytest.mark.parametrize("chunk_size", (1, 4, 11))
def test_full_and_chunked_construction_are_identical(chunk_size: int) -> None:
    q = _q()
    sample_ids = generic.native_shared_permutation(q.shape[0])
    labeled_ids, labeled_targets = _sparse_ids_for_permuted_rows(sample_ids)
    full = generic.build_generic_carrier(q, sample_ids, labeled_ids, labeled_targets)
    chunked = generic.build_generic_carrier(q, sample_ids, labeled_ids, labeled_targets, query_chunk_size=chunk_size)
    for name in ("p_gen", "p_ver", "U_cycle", "Pi", "T_rel", "query_sample_ids", "generator_actions", "verifier_actions"):
        assert np.array_equal(chunked[name], full[name]), name


def test_msrc_backward_parity_for_actions_sparse_split_u_and_t() -> None:
    audit = materializer.msrc_backward_parity()
    assert audit["status"] == "PASS"
    for name, comparison in audit["comparison"].items():
        assert comparison["shape_equal"], name
        if "allclose_1e-7" in comparison:
            assert comparison["allclose_1e-7"], name
        else:
            assert comparison["exact"], name
    assert audit["GT_loaded_before_seal"] is False


def test_generic_artifact_schema_audit_and_gt_isolation(tmp_path: Path) -> None:
    with np.load(P2_CARRIER, allow_pickle=False) as archive:
        q_views = archive["q_views"]
        sample_ids = archive["sample_ids"]
    paths = materializer.materialize_from_q_views("MSRC", q_views, sample_ids, 20, P2_STATE, tmp_path, query_chunk_size=17)
    with np.load(paths["carrier"], allow_pickle=False) as archive:
        required = {"sample_ids", "query_sample_ids", "anchor_sample_ids", "labeled_ids", "labeled_targets", "generator_actions", "verifier_actions", "U_cycle", "T_rel"}
        assert required <= set(archive.files)
    audit = json.loads(paths["audit"].read_text())
    assert audit["GT_loaded_before_seal"] is False
    assert audit["sample_id_semantics"] == "RAW_FILE_ROW_ID"
    assert audit["query_chunk_size"] == 17


def test_pre_gt_sources_do_not_load_full_gt_or_metrics() -> None:
    source = Path(generic.__file__).read_text(encoding="utf-8") + Path(materializer.__file__).read_text(encoding="utf-8")
    lower = source.lower()
    for forbidden in ("scipy.io", "get_dataset", "get_cluster_metrics", "accuracy", "purity", "nmi", 'archive["labels"]'):
        assert forbidden not in lower
    assert "bone_p3_a1_action_space_protocol" in source


def test_native_and_frozen_artifacts_are_unchanged() -> None:
    assert _sha(REPO_ROOT / "method/bone.py") == BONE_SHA
    assert _sha(REPO_ROOT / "method/amv.py") == AMV_SHA
    assert _sha(P1_ARTIFACT) == P1_SHA
    assert _sha(P2_CARRIER) == P2_SHA
