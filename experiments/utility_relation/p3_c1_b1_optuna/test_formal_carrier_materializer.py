from pathlib import Path

import numpy as np
import pytest

from experiments.utility_relation.p3_c1_b1_optuna import formal_carrier_materializer as materializer
from experiments.utility_relation.p3_c1_b1_optuna import protocol


def _write_fake_adapter_artifacts(staging: Path) -> None:
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "carrier_model_state.pt").write_bytes(b"mock-state")
    q = np.full((4, 2, 2), .5, dtype=np.float32)
    p_gen = np.array([[[.8, .2], [.2, .8]]] * 4, dtype=np.float32)
    p_ver = p_gen.copy()
    confidence = np.max(p_gen, axis=2)
    support = np.take_along_axis(p_ver, np.argmax(p_gen, axis=2)[..., None], axis=2)[..., 0]
    utility = np.sqrt(confidence * support)
    np.savez_compressed(
        staging / "carrier_pre_gt.npz",
        q_views=q, p_gen=p_gen, p_ver=p_ver, U_cycle=utility,
        generator_actions=np.array([[0], [1]], dtype=np.int64),
        verifier_actions=np.array([[1], [0]], dtype=np.int64),
        sample_ids=np.arange(4, dtype=np.int64),
        labeled_ids=np.arange(4, dtype=np.int64),
        labeled_targets=np.array([0, 0, 1, 1], dtype=np.int64),
    )


def install_fake_adapters(monkeypatch, tmp_path, calls):
    split, corruption = tmp_path / "split.npz", tmp_path / "mask.npz"
    split.write_bytes(b"split"); corruption.write_bytes(b"mask")
    monkeypatch.setattr(materializer, "_expected_dimensions", lambda _dataset: (4, 2))
    monkeypatch.setattr(materializer, "_artifact_paths", lambda _dataset: (split, corruption))
    monkeypatch.setattr(materializer, "_source_provenance", lambda _dataset: {"mock": "hash"})
    def generic(dataset, seed, device, staging):
        calls.append(("generic", dataset, seed, device)); _write_fake_adapter_artifacts(staging); return {}
    def msrc(seed, device, staging):
        calls.append(("msrc", "MSRC", seed, device)); _write_fake_adapter_artifacts(staging); return {}
    monkeypatch.setattr(materializer.carrier_adapter, "build_development_carrier", generic)
    monkeypatch.setattr(materializer.msrc_adapter, "build_msrc_development_carrier", msrc)


def test_exact_protocol_identities_epochs_and_adapter_routes():
    assert materializer.amendment_a1.AMENDMENT_ID == "P3-C1-B1-A1"
    assert materializer.amendment_a1.AMENDED_DATASETS is protocol.DATASETS
    assert len(protocol.DATASETS) == 8
    assert protocol.DEV_SEEDS == (11, 37)
    assert len(materializer.expected_identities()) == 16
    assert protocol.CARRIER_EPOCHS == 100
    assert [dataset for dataset in protocol.DATASETS if materializer.adapter_route(dataset) == "generic"] == [
        "HandWrite", "BDGP", "COIL20", "Caltech-5V", "MNIST-USPS", "ALOI", "CCV"
    ]
    assert materializer.adapter_route("MSRC") == "msrc"


@pytest.mark.parametrize("seed", [20, 30, 50, 123])
def test_formal_and_unknown_seeds_are_rejected(seed):
    with pytest.raises(ValueError, match="UNREGISTERED_DEV_SEED"):
        materializer.validate_request("MSRC", seed, "cuda:0")
    with pytest.raises(ValueError, match="UNREGISTERED_DATASET"):
        materializer.validate_request("unknown", 11, "cuda:0")


def test_correct_path_and_generic_msrc_routing(monkeypatch, tmp_path):
    calls = []; install_fake_adapters(monkeypatch, tmp_path, calls)
    root = tmp_path / "formal"
    one = materializer.materialize_one("HandWrite", 11, "cuda:0", root)
    two = materializer.materialize_one("MSRC", 37, "cuda:0", root)
    assert one["status"] == two["status"] == "COMPLETE"
    assert Path(one["path"]) == root / "handwrite/dev_seed11/carrier"
    assert Path(two["path"]) == root / "msrc/dev_seed37/carrier"
    assert [call[0] for call in calls] == ["generic", "msrc"]


def test_complete_written_last_then_verified_resume_skips(monkeypatch, tmp_path):
    calls = []; install_fake_adapters(monkeypatch, tmp_path, calls)
    root = tmp_path / "formal"
    result = materializer.materialize_one("BDGP", 11, "cuda:0", root)
    target = Path(result["path"])
    assert all((target / name).is_file() for name in (
        "carrier_model_state.pt", "carrier_pre_gt.npz", "carrier_audit.json", "COMPLETE.json"
    ))
    resumed = materializer.materialize_one("BDGP", 11, "cuda:0", root)
    assert resumed["status"] == "ALREADY_COMPLETE_VERIFIED"
    assert len(calls) == 1


def test_complete_hash_mismatch_fails_and_never_overwrites(monkeypatch, tmp_path):
    calls = []; install_fake_adapters(monkeypatch, tmp_path, calls)
    root = tmp_path / "formal"
    result = materializer.materialize_one("COIL20", 11, "cuda:0", root)
    state = Path(result["path"]) / "carrier_model_state.pt"
    state.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="COMPLETE_HASH_MISMATCH"):
        materializer.materialize_one("COIL20", 11, "cuda:0", root, replace_incomplete=True)
    assert state.read_bytes() == b"tampered"
    assert len(calls) == 1


def test_partial_fails_by_default_and_explicit_replace_is_target_scoped(monkeypatch, tmp_path):
    calls = []; install_fake_adapters(monkeypatch, tmp_path, calls)
    root = tmp_path / "formal"
    target = materializer.carrier_directory(root, "ALOI", 37)
    target.mkdir(parents=True); (target / "partial.bin").write_bytes(b"partial")
    neighbor = tmp_path / "do_not_touch.txt"; neighbor.write_text("safe")
    with pytest.raises(RuntimeError, match="INCOMPLETE_EXISTING_OUTPUT"):
        materializer.materialize_one("ALOI", 37, "cuda:0", root)
    result = materializer.materialize_one("ALOI", 37, "cuda:0", root, replace_incomplete=True)
    assert result["status"] == "COMPLETE"
    assert not (target / "partial.bin").exists()
    assert neighbor.read_text() == "safe"


def test_legacy_zero_utility_stop_is_accepted_only_after_artifacts(monkeypatch, tmp_path):
    calls = []; install_fake_adapters(monkeypatch, tmp_path, calls)
    original = materializer.msrc_adapter.build_msrc_development_carrier
    def stopped(seed, device, staging):
        original(seed, device, staging)
        raise RuntimeError("ZERO_UTILITY_MASS")
    monkeypatch.setattr(materializer.msrc_adapter, "build_msrc_development_carrier", stopped)
    result = materializer.materialize_one("MSRC", 11, "cuda:0", tmp_path / "formal")
    assert result["status"] == "COMPLETE"
    assert result["audit"]["legacy_zero_utility_gate_observed"] is True


def test_zero_utility_stop_before_artifacts_fails(monkeypatch, tmp_path):
    calls = []; install_fake_adapters(monkeypatch, tmp_path, calls)
    def stopped(_seed, _device, _staging):
        raise RuntimeError("ZERO_UTILITY_MASS")
    monkeypatch.setattr(materializer.msrc_adapter, "build_msrc_development_carrier", stopped)
    with pytest.raises(RuntimeError, match="BEFORE_ARTIFACT_COMPLETION"):
        materializer.materialize_one("MSRC", 11, "cuda:0", tmp_path / "formal")
