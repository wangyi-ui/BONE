"""Read-only provenance audit for the frozen MSRC P2 sparse-label split.

This audit intentionally does not generate a split, invoke an RNG, or search
seeds.  It only records the available source and artifact evidence for the
historical IDs already frozen by P2-A0/P2-A1B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "bone_p3"
P1_ARTIFACT = REPO_ROOT / "outputs" / "bone_p1" / "msrc_snr2p5_exact2of4_seed20" / "bone_p1_msrc_corruption.npz"
P1_SEAL = P1_ARTIFACT.with_name("corruption_seal.json")
P2_CARRIER = REPO_ROOT / "outputs" / "bone_p2" / "p2_a0_msrc_seed20" / "p2_a0_carrier_pre_gt.npz"
P2_SEAL = P2_CARRIER.with_name("p2_a0_carrier_seal.json")
MSRC_MAT = REPO_ROOT / "data" / "MSRC.mat"
P1_MATERIALIZER = REPO_ROOT / "experiments" / "weak_quality" / "materialize_bone_p1_msrc_corruption.py"
P2_PROTOCOL = REPO_ROOT / "experiments" / "utility_relation" / "bone_p2_a0_protocol.py"
P2_MATERIALIZER = REPO_ROOT / "experiments" / "utility_relation" / "materialize_bone_p2_a0_carrier.py"

FROZEN_IDS = np.asarray((17, 20, 30, 42, 60, 81, 96, 107, 139, 142, 153, 175, 182, 201), dtype=np.int64)
FROZEN_TARGETS = np.asarray((0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6), dtype=np.int64)
ORIGINAL_BONE_SHA256 = "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092"
ORIGINAL_AMV_SHA256 = "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_record(path: Path) -> dict[str, str]:
    return {"relative_path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}


def _textual_search() -> dict[str, Any]:
    """Search textual repository evidence without interpreting it as a generator."""
    suffixes = {".py", ".json", ".txt", ".log", ".md"}
    keywords = ("LABELED_IDS", "EXPECTED_TARGETS", "labeled_ids", "labeled_targets", "sparse", "split_seed", "anchor_seed", "2 labels")
    matches: dict[str, list[str]] = {keyword: [] for keyword in keywords}
    excluded_parts = {".git", "__pycache__", ".pytest_cache"}
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes or excluded_parts.intersection(path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        relative = str(path.relative_to(REPO_ROOT))
        for keyword in keywords:
            if keyword.lower() in text.lower():
                matches[keyword].append(relative)
    return {
        "scope": "all repository text files with suffix .py/.json/.txt/.log/.md, excluding .git, __pycache__, and .pytest_cache; NPZ metadata is inspected separately",
        "keywords": list(keywords),
        "matches": matches,
        "selection_generator_found": False,
        "selection_generator_search_conclusion": "The only exact-ID definition found is the frozen LABELED_IDS literal in P2-A0 protocol source and its frozen snapshots; no function constructs this list.",
    }


def _artifact_evidence() -> dict[str, Any]:
    for path in (P1_ARTIFACT, P1_SEAL, P2_CARRIER, P2_SEAL, MSRC_MAT):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(P1_ARTIFACT, allow_pickle=False) as parent:
        row_ids = np.asarray(parent["row_ids"], dtype=np.int64)
        parent_labels = np.asarray(parent["labels"], dtype=np.int64)
    with np.load(P2_CARRIER, allow_pickle=False) as carrier:
        sample_ids = np.asarray(carrier["sample_ids"], dtype=np.int64)
        carrier_ids = np.asarray(carrier["labeled_ids"], dtype=np.int64)
        carrier_targets = np.asarray(carrier["labeled_targets"], dtype=np.int64)
    raw_labels = np.asarray(scipy.io.loadmat(MSRC_MAT, variable_names=["Y"])["Y"], dtype=np.int64).reshape(-1)
    parent_seal = json.loads(P1_SEAL.read_text(encoding="utf-8"))
    carrier_seal = json.loads(P2_SEAL.read_text(encoding="utf-8"))

    if not np.array_equal(row_ids, np.arange(210, dtype=np.int64)):
        raise ValueError("P1 parent row_ids are not canonical arange(210)")
    if not np.array_equal(sample_ids, np.arange(210, dtype=np.int64)):
        raise ValueError("P2 carrier sample_ids are not canonical arange(210)")
    if not np.array_equal(carrier_ids, FROZEN_IDS) or not np.array_equal(carrier_targets, FROZEN_TARGETS):
        raise ValueError("P2 carrier does not contain the historical frozen split")
    if not np.array_equal(parent_labels[FROZEN_IDS], FROZEN_TARGETS):
        raise ValueError("P1 parent labels do not validate frozen targets")
    if not np.array_equal(raw_labels[FROZEN_IDS], FROZEN_TARGETS):
        raise ValueError("raw MSRC labels do not validate frozen targets")

    parent_hash = sha256_file(P1_ARTIFACT)
    carrier_hash = sha256_file(P2_CARRIER)
    if parent_seal["sha256"]["bone_p1_msrc_corruption.npz"] != parent_hash:
        raise ValueError("P1 artifact hash does not match its seal")
    if carrier_seal["sha256"]["carrier_npz"] != carrier_hash:
        raise ValueError("P2 carrier hash does not match its seal")
    if carrier_seal["sha256"]["parent_corruption_artifact"] != parent_hash:
        raise ValueError("P2 carrier seal parent hash does not match P1 artifact")
    return {
        "raw_msrc_file": {"path": "data/MSRC.mat", "sha256": sha256_file(MSRC_MAT), "label_key": "Y", "raw_labels_at_frozen_ids": raw_labels[FROZEN_IDS].tolist()},
        "p1_parent_artifact": {
            "path": str(P1_ARTIFACT.relative_to(REPO_ROOT)), "sha256": parent_hash,
            "sealed_sha256_matches": True, "row_ids": "int64 arange(210)",
            "labels_at_frozen_ids": parent_labels[FROZEN_IDS].tolist(),
        },
        "p2_seed20_carrier": {
            "path": str(P2_CARRIER.relative_to(REPO_ROOT)), "sha256": carrier_hash,
            "sealed_sha256_matches": True, "sample_ids": "int64 arange(210)",
            "labeled_ids": carrier_ids.tolist(), "labeled_targets": carrier_targets.tolist(),
        },
        "chain_validation": "raw MSRC Y[IDs] == P1 parent labels[IDs] == P2 carrier labeled_targets == frozen targets",
    }


def build_audit() -> dict[str, Any]:
    native = {
        "method/bone.py": sha256_file(REPO_ROOT / "method" / "bone.py"),
        "method/amv.py": sha256_file(REPO_ROOT / "method" / "amv.py"),
    }
    if native["method/bone.py"] != ORIGINAL_BONE_SHA256 or native["method/amv.py"] != ORIGINAL_AMV_SHA256:
        raise RuntimeError("native hash protection failed")
    evidence = _artifact_evidence()
    p1_source = P1_MATERIALIZER.read_text(encoding="utf-8")
    p2_source = P2_PROTOCOL.read_text(encoding="utf-8")
    p2_materializer_source = P2_MATERIALIZER.read_text(encoding="utf-8")
    required_p1 = ("shuffle is intentionally never called", "dataset.standard_data()", '"row_ids": np.arange(N, dtype=np.int64)')
    required_p2 = ("LABELED_IDS = np.asarray", "EXPECTED_TARGETS = np.asarray", "canonical-row complement")
    required_p2_materializer = ('archive["labels"][LABELED_IDS]', '"sample_ids": np.arange(N, dtype=np.int64)')
    if not all(snippet in p1_source for snippet in required_p1):
        raise ValueError("P1 canonical-row source evidence is absent")
    if not all(snippet in p2_source for snippet in required_p2):
        raise ValueError("P2 frozen-ID source evidence is absent")
    if not all(snippet in p2_materializer_source for snippet in required_p2_materializer):
        raise ValueError("P2 parent-label validation evidence is absent")
    return {
        "MSRC_SPLIT_PROVENANCE_STATUS": "PARTIAL_PROVENANCE",
        "status_rationale": "Canonical row semantics, exact frozen values, target validation, and sealed artifacts are proven. The source contains the IDs as a frozen literal, but contains no selection/generator function, RNG type, split seed, or construction sorting rule.",
        "frozen_ids": FROZEN_IDS.tolist(),
        "frozen_targets": FROZEN_TARGETS.tolist(),
        "row_id_semantics": {
            "value": "RAW_FILE_ROW_ID / P1_CANONICAL_PARENT_ARTIFACT_ROW_ID",
            "proven": True,
            "proof": [
                "P1 materializer calls get_dataset(D_MSRC), calls standard_data(), and explicitly states shuffle is intentionally never called.",
                "P1 artifact stores row_ids=int64 arange(210) and labels in that unshuffled canonical order.",
                "P2 sparse_targets_from_parent reads archive['labels'][LABELED_IDS]; P2 carrier stores sample_ids=int64 arange(210).",
                "Raw MSRC Y[IDs], P1 artifact labels[IDs], and P2 carrier targets all exactly equal the frozen target vector.",
            ],
            "not_post_native_shuffle_seed_42": True,
            "MVDataset_shuffle_42_relation": "The native entry scripts call shuffle(42), but P1 materialization explicitly does not; therefore those shuffled positions are not the frozen-ID coordinate system.",
        },
        "generator_source": "experiments/utility_relation/bone_p2_a0_protocol.py: LABELED_IDS and EXPECTED_TARGETS frozen literals",
        "generator_function": "NOT_FOUND",
        "ids_explicitly_generated_by_code": False,
        "rng_type": "NOT_FOUND",
        "split_seed": "NOT_FOUND",
        "sorting_rule": "NOT_FOUND; IDs are strictly ascending and targets are class-grouped in the literal, but no selection/construction rule is present.",
        "non_split_seeds_observed": {
            "P1_corruption_seed": 20,
            "P1_corruption_seed_scope": "label-independent corruption mask/noise only; not a sparse-split seed",
            "P2_training_seed": 20,
            "P2_training_seed_scope": "native replay/model randomness only; not a sparse-split seed",
            "native_entry_shuffle_seed": 42,
            "native_entry_shuffle_seed_scope": "not used by P1 materializer, which deliberately preserves canonical raw rows",
        },
        "artifact_source": evidence,
        "artifact_sha256": {
            "p1_parent_corruption_npz": evidence["p1_parent_artifact"]["sha256"],
            "p2_seed20_carrier_npz": evidence["p2_seed20_carrier"]["sha256"],
        },
        "scope_and_safety": {
            "brute_force_seed_search_performed": False,
            "new_rng_algorithm_inferred": False,
            "new_sparse_split_generated": False,
            "other_dataset_labeled_ids_generated": False,
            "historical_split_modified": False,
            "interpretation": "The MSRC split remains a legacy frozen split. Its unavailable selection provenance must not be reconstructed or generalized.",
        },
        "textual_search": _textual_search(),
        "source_evidence": [source_record(path) for path in (P1_MATERIALIZER, P2_PROTOCOL, P2_MATERIALIZER, MSRC_MAT, P1_ARTIFACT, P1_SEAL, P2_CARRIER, P2_SEAL)],
        "native_hashes": native,
    }


def text_report(audit: dict[str, Any]) -> str:
    evidence = audit["artifact_source"]
    return "\n".join((
        "P3-A3-A0 MSRC Sparse-Label Split Provenance Audit",
        "====================================================",
        f"MSRC_SPLIT_PROVENANCE_STATUS: {audit['MSRC_SPLIT_PROVENANCE_STATUS']}",
        "",
        f"frozen_ids: {audit['frozen_ids']}",
        f"frozen_targets: {audit['frozen_targets']}",
        "",
        "Row-ID semantics: RAW_FILE_ROW_ID / P1_CANONICAL_PARENT_ARTIFACT_ROW_ID.",
        "P1 materialization loads MSRC, standardizes it, and explicitly never calls shuffle; it writes row_ids=arange(210).",
        "P2 validates archive labels at the frozen IDs and writes carrier sample_ids=arange(210).",
        "raw Y[IDs] == P1 labels[IDs] == P2 labeled_targets == frozen targets.",
        "",
        f"generator_source: {audit['generator_source']}",
        "generator_function: NOT_FOUND",
        "rng_type: NOT_FOUND", "split_seed: NOT_FOUND",
        "sorting_rule: NOT_FOUND (ascending IDs/class-grouped targets are observed literal properties, not a documented selection rule).",
        "",
        f"P1 artifact: {evidence['p1_parent_artifact']['path']}",
        f"P1 artifact SHA256: {evidence['p1_parent_artifact']['sha256']}",
        f"P2 carrier: {evidence['p2_seed20_carrier']['path']}",
        f"P2 carrier SHA256: {evidence['p2_seed20_carrier']['sha256']}",
        "",
        "Conclusion: PARTIAL_PROVENANCE. The historical MSRC split remains a legacy frozen split and cannot reasonably be generalized to other datasets.",
        "No brute-force seed search, new RNG inference, new sparse split, training, GPU use, or historical artifact modification was performed.",
        "",
        "Native hashes:",
        f"method/bone.py {audit['native_hashes']['method/bone.py']}",
        f"method/amv.py  {audit['native_hashes']['method/amv.py']}",
    )) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    output_dir = build_parser().parse_args().output_dir
    audit = build_audit()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "p3_a3_a0_sparse_split_provenance_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "p3_a3_a0_sparse_split_provenance_audit.txt").write_text(text_report(audit), encoding="utf-8")


if __name__ == "__main__":
    main()
