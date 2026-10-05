import json
from pathlib import Path

import pytest

from experiments.utility_relation.p3_c1_b1_optuna import formal_carrier_materializer as materializer
from experiments.utility_relation.p3_c1_b1_optuna.audit_formal_dev_carriers import audit_inventory
from experiments.utility_relation.p3_c1_b1_optuna.test_formal_carrier_materializer import install_fake_adapters


def _populate(monkeypatch, tmp_path, count):
    calls = []; install_fake_adapters(monkeypatch, tmp_path, calls)
    root = tmp_path / "formal"
    for dataset, seed in materializer.expected_identities()[:count]:
        materializer.materialize_one(dataset, seed, "cuda:0", root)
    return root


def test_zero_of_sixteen_not_ready(tmp_path):
    inventory = audit_inventory(tmp_path / "formal")
    assert inventory["expected_carrier_count"] == 16
    assert inventory["complete_verified_count"] == 0
    assert inventory["missing_count"] == 16
    assert inventory["ALL_16_FORMAL_DEV_CARRIERS_READY"] is False


def test_fifteen_of_sixteen_not_ready(monkeypatch, tmp_path):
    inventory = audit_inventory(_populate(monkeypatch, tmp_path, 15))
    assert inventory["complete_verified_count"] == 15
    assert inventory["missing_count"] == 1
    assert inventory["ALL_16_FORMAL_DEV_CARRIERS_READY"] is False


def test_sixteen_valid_ready(monkeypatch, tmp_path):
    inventory = audit_inventory(_populate(monkeypatch, tmp_path, 16))
    assert inventory["complete_verified_count"] == 16
    assert inventory["invalid_count"] == inventory["missing_count"] == 0
    assert inventory["ALL_16_FORMAL_DEV_CARRIERS_READY"] is True


def test_one_hash_mismatch_not_ready(monkeypatch, tmp_path):
    root = _populate(monkeypatch, tmp_path, 16)
    target = materializer.carrier_directory(root, *materializer.expected_identities()[0])
    (target / "carrier_model_state.pt").write_bytes(b"mismatch")
    inventory = audit_inventory(root)
    assert inventory["complete_verified_count"] == 15 and inventory["invalid_count"] == 1
    assert inventory["ALL_16_FORMAL_DEV_CARRIERS_READY"] is False


def test_one_full_gt_flag_not_ready(monkeypatch, tmp_path):
    root = _populate(monkeypatch, tmp_path, 16)
    target = materializer.carrier_directory(root, *materializer.expected_identities()[0])
    audit_path, marker_path = target / "carrier_audit.json", target / "COMPLETE.json"
    audit = json.loads(audit_path.read_text()); audit["full_gt_loaded"] = True
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    marker = json.loads(marker_path.read_text()); marker["audit_sha256"] = materializer.sha256_file(audit_path)
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    inventory = audit_inventory(root)
    assert inventory["invalid_count"] == 1 and inventory["ALL_16_FORMAL_DEV_CARRIERS_READY"] is False


def test_one_missing_complete_not_ready(monkeypatch, tmp_path):
    root = _populate(monkeypatch, tmp_path, 16)
    target = materializer.carrier_directory(root, *materializer.expected_identities()[0])
    (target / "COMPLETE.json").unlink()
    inventory = audit_inventory(root)
    assert inventory["invalid_count"] == 1 and inventory["ALL_16_FORMAL_DEV_CARRIERS_READY"] is False


def test_unexpected_identity_does_not_replace_missing_expected(monkeypatch, tmp_path):
    root = _populate(monkeypatch, tmp_path, 15)
    unexpected = root / "unexpected/dev_seed11/carrier"
    unexpected.mkdir(parents=True); (unexpected / "COMPLETE.json").write_text("{}")
    inventory = audit_inventory(root)
    assert inventory["complete_verified_count"] == 15 and inventory["missing_count"] == 1
    assert inventory["ALL_16_FORMAL_DEV_CARRIERS_READY"] is False
