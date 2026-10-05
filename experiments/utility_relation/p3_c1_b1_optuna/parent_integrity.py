"""Read-only SHA256 checks for preregistered parent scientific sources."""

import hashlib
from pathlib import Path
from typing import Mapping


REGISTERED_PARENT_FILES = (
    "method/bone.py",
    "method/amv.py",
    "test_cmvsc.py",
    "experiments/utility_relation/bone_p3_b0_generic_carrier.py",
    "experiments/utility_relation/bone_p3_c0_cross_dataset_formal_protocol.py",
    "experiments/utility_relation/run_bone_p3_c0_cross_dataset_formal.py",
    "experiments/utility_relation/evaluate_bone_p3_c0_cross_dataset_formal.py",
    "experiments/utility_relation/bone_p3_c1_b0_candidate_protocol.py",
    "experiments/utility_relation/run_bone_p3_c1_b0_candidate_validation.py",
    "experiments/utility_relation/evaluate_bone_p3_c1_b0_candidate_validation.py",
    "experiments/utility_relation/summarize_bone_p3_c1_b0_candidate_validation.py",
)
NATIVE_EXACT_HASHES = {
    "method/bone.py": "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092",
    "method/amv.py": "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc",
    "test_cmvsc.py": "fc836c5a8e053624eee7bcf200a592b3a064a9eec71e719012eb976dd1641d8d",
}


def sha256_file(path: str | Path) -> str:
    """Hash a file without opening it for writing."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(manifest_path: str | Path) -> dict[str, str]:
    """Read standard ``sha256sum`` manifest lines into a path/hash mapping."""
    manifest = {}
    for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative_path = line.split(maxsplit=1)
        manifest[relative_path.strip()] = digest
    return manifest


def verify_manifest(manifest_path: str | Path, root: str | Path = ".") -> bool:
    """Fail closed when a listed file is absent or its SHA256 differs."""
    root = Path(root)
    manifest = read_manifest(manifest_path)
    if not manifest:
        raise ValueError("EMPTY_PARENT_MANIFEST")
    for relative_path, expected in manifest.items():
        source = root / relative_path
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"PARENT_INTEGRITY_MISMATCH:{relative_path}")
    return True


def verify_registered_parents(expected_hashes: Mapping[str, str], root: str | Path = ".") -> bool:
    """Require hashes for every registered source and compare them read-only."""
    if set(expected_hashes) != set(REGISTERED_PARENT_FILES):
        raise ValueError("PARENT_MANIFEST_COVERAGE_MISMATCH")
    root = Path(root)
    for relative_path in REGISTERED_PARENT_FILES:
        source = root / relative_path
        if not source.is_file() or sha256_file(source) != expected_hashes[relative_path]:
            raise ValueError(f"PARENT_INTEGRITY_MISMATCH:{relative_path}")
    return True


def verify_native_exact_hashes(root: str | Path = ".") -> bool:
    """Verify the three fixed native source hashes supplied by the protocol."""
    root = Path(root)
    for relative_path, expected in NATIVE_EXACT_HASHES.items():
        if sha256_file(root / relative_path) != expected:
            raise ValueError(f"NATIVE_HASH_MISMATCH:{relative_path}")
    return True
