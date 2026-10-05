"""Frozen B4 contract; no tunable scientific parameters are exposed here."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET = "MSRC"
CONFIRMATORY_SEEDS = (100, 110, 130)
FORBIDDEN_SELECTION_SEEDS = (20, 30, 50, 60, 70, 90)
ARMS = ("BASE", "OURS_HISTORICAL_DEFAULT")
HISTORICAL_DEFAULT_SEMANTIC_LR = 1e-4
HISTORICAL_DEFAULT_SEMANTIC_EXPOSURE = 20
CARRIER_EPOCHS = 100
REFINEMENT_EPOCHS = 20
SEMANTIC_BATCH_SIZE = 128
NATIVE_BATCH_SIZE = 128
NATIVE_LR = 1e-2
SEMANTIC_OPTIMIZER = "Adam"
QUERY_CHUNK_SIZE = 256
OUTPUT_ROOT = REPO_ROOT / "outputs/bone_p3/p3_c1_b4_msrc_historical_default_confirmatory"
PREREGISTERED_FREEZE = REPO_ROOT / "experiment_freeze/bone_p3_c1_b4_msrc_historical_default_confirmatory_preregistered_20261004"

P2_PREFORMAL = REPO_ROOT / "experiment_freeze/bone_p2_a1b_multiseed_preformal_20261001"
P2_PASS = REPO_ROOT / "experiment_freeze/bone_p2_a1b_multiseed_pass_20261001"
P3_A3 = REPO_ROOT / "experiment_freeze/bone_p3_a3_sparse_label_protocol_20261001"
P3_A4 = REPO_ROOT / "experiment_freeze/bone_p3_a4_corruption_protocol_20261001"
P3_B0 = REPO_ROOT / "experiment_freeze/bone_p3_b0_generic_carrier_20261001"
P3_B2 = REPO_ROOT / "experiment_freeze/bone_p3_c1_b2_targeted_local_refinement_results_20261004"
P3_B3 = REPO_ROOT / "experiment_freeze/bone_p3_c1_b3_msrc_b2_confirmatory_results_20261004"

PARENT_FREEZE_SHA256 = {
    "p2_preformal_protocol": (P2_PREFORMAL / "PREREGISTERED_PROTOCOL.txt", "0e02059dbc1aa8657833dd4ed51896a1f4ec7d5103163926b76a1263a6b75038"),
    "p2_preformal_manifest": (P2_PREFORMAL / "FREEZE_MANIFEST.sha256", "5b3efe449003f18fcf76f8e685721b8c64bf914cf70ad995ed592b10a6014e9a"),
    "p2_pass_status": (P2_PASS / "STATUS.txt", "4ca64a905217de52800b0fd6e0b26f61538ff78d31248e61872ede9bf3ce07c2"),
    "p2_pass_manifest": (P2_PASS / "FREEZE_MANIFEST.sha256", "5557ae5d351b7e2017fff9afa7a7b12d1d144265cbdb417f6bb4016579047c85"),
    "p3_a3_manifest": (P3_A3 / "FREEZE_MANIFEST.sha256", "0b67a404a11e75f4fb19c1facfce11b842e25f1078c43bca3f5e4a355a91b7c7"),
    "p3_a4_manifest": (P3_A4 / "FREEZE_MANIFEST.sha256", "f8cac93a3ffeb1d6ff860e66acf2ff468e40a347100ba1a5b9d1badf8281dda1"),
    "p3_b0_manifest": (P3_B0 / "FREEZE_MANIFEST.sha256", "fa8cab49c9fdcc4f921268b5001e586215b736a9935de90c74107ea5a46d55cd"),
    "p3_b2_status": (P3_B2 / "STATUS.txt", "a16db7bf30182097eb97169e7a63df972ad9d5b60dda0009a2867487fdb42a0b"),
    "p3_b3_status": (P3_B3 / "STATUS.txt", "928b128fac682a9943ecfe058353efa8ac148359b8366d3f1c31949a8d6c60d7"),
}
FROZEN_METHOD_SHA256 = {
    "method/bone.py": "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092",
    "method/amv.py": "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc",
}
SOURCE_RELATIVE_PATHS = (
    "experiments/utility_relation/p3_c1_b4_msrc_historical_default_confirmatory/__init__.py",
    "experiments/utility_relation/p3_c1_b4_msrc_historical_default_confirmatory/protocol.py",
    "experiments/utility_relation/p3_c1_b4_msrc_historical_default_confirmatory/context.py",
    "experiments/utility_relation/p3_c1_b4_msrc_historical_default_confirmatory/run_confirmatory.py",
    "experiments/utility_relation/p3_c1_b4_msrc_historical_default_confirmatory/evaluate_confirmatory.py",
    "experiments/utility_relation/p3_c1_b4_msrc_historical_default_confirmatory/summarize_confirmatory.py",
    "experiments/utility_relation/p3_c1_b4_msrc_historical_default_confirmatory/audit.py",
    "tests/test_p3_c1_b4_msrc_historical_default_confirmatory.py",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temp.replace(target)


def source_hashes() -> dict[str, str]:
    return {relative: sha256_file(REPO_ROOT / relative) for relative in SOURCE_RELATIVE_PATHS}


def run_root(output_root: str | Path, training_seed: int) -> Path:
    if training_seed not in CONFIRMATORY_SEEDS:
        raise ValueError("B4_UNREGISTERED_CONFIRMATORY_SEED")
    return Path(output_root) / f"seed{training_seed}"


def primary_confirmation_gate(deltas: dict[str, float]) -> str:
    """The preregistered, non-tunable primary confirmation decision."""
    if set(deltas) != {"ACC", "NMI", "ARI"}:
        raise ValueError("B4_PRIMARY_GATE_METRIC_SET_INVALID")
    return "CONFIRMED" if all(float(deltas[name]) > 0.0 for name in deltas) else "NOT_CONFIRMED"
