from pathlib import Path

import pytest

from experiments.utility_relation.p3_c1_b1_optuna.parent_integrity import (
    NATIVE_EXACT_HASHES, REGISTERED_PARENT_FILES, read_manifest, sha256_file,
    verify_manifest, verify_native_exact_hashes, verify_registered_parents,
)


ROOT = Path(__file__).resolve().parents[3]
B0_FREEZE = ROOT / "experiment_freeze/bone_p3_c1_b0_transfer_validation_final_20261002"


def test_current_b0_freeze_manifests_verify():
    assert verify_manifest(B0_FREEZE / "final_source_sha256.txt", ROOT)
    assert verify_manifest(B0_FREEZE / "freeze_core_sha256.txt", ROOT)


def test_native_hashes_and_registered_parent_existence_verify():
    assert verify_native_exact_hashes(ROOT)
    assert all((ROOT / path).is_file() for path in REGISTERED_PARENT_FILES)
    expected = {path: sha256_file(ROOT / path) for path in REGISTERED_PARENT_FILES}
    assert verify_registered_parents(expected, ROOT)
    assert {path: sha256_file(ROOT / path) for path in NATIVE_EXACT_HASHES} == NATIVE_EXACT_HASHES


def test_synthetic_alteration_fails_without_source_mutation(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("original", encoding="utf-8")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"{sha256_file(source)}  source.txt\n", encoding="utf-8")
    assert verify_manifest(manifest, tmp_path)
    source.write_text("altered", encoding="utf-8")
    with pytest.raises(ValueError, match="PARENT_INTEGRITY_MISMATCH"):
        verify_manifest(manifest, tmp_path)


def test_manifest_reader_is_read_only(tmp_path):
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("a" * 64 + "  source.py\n", encoding="utf-8")
    before = manifest.read_bytes()
    assert read_manifest(manifest) == {"source.py": "a" * 64}
    assert manifest.read_bytes() == before
