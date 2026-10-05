"""Fail-closed B5 search space: optimizer family and semantic LR only."""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASETS = ("CCV", "ALOI")
DEV_SEEDS = (11, 37)
FOLDS = ("A", "B")
OPTIMIZERS = ("Adam", "AdamW", "SGD_MOMENTUM")
EXPOSURES = {"CCV": 3, "ALOI": 1}
LR_GRID = {
    "CCV": {"Adam": (3e-6, 1e-5, 3e-5), "AdamW": (3e-6, 1e-5, 3e-5), "SGD_MOMENTUM": (1e-3, 3e-3, 1e-2)},
    "ALOI": {"Adam": (3e-5, 1e-4, 3e-4), "AdamW": (3e-5, 1e-4, 3e-4), "SGD_MOMENTUM": (1e-3, 3e-3, 1e-2)},
}
REFINEMENT_EPOCHS = 20
OUTPUT_ROOT = REPO_ROOT / "outputs/bone_p3/p3_c1_b5_optimizer_pilot"
PREREGISTERED_FREEZE = REPO_ROOT / "experiment_freeze/bone_p3_c1_b5_optimizer_pilot_preregistered_20261004"
B4_RESULT_FREEZE = REPO_ROOT / "experiment_freeze/bone_p3_c1_b4_msrc_historical_default_confirmatory_results_20261004"
B2_RESULT_FREEZE = REPO_ROOT / "experiment_freeze/bone_p3_c1_b2_targeted_local_refinement_results_20261004"
B1_FREEZE = REPO_ROOT / "experiment_freeze/bone_p3_c1_b1_a2_final_hyperparameters_20261004"

SOURCE_RELATIVE_PATHS = (
 "experiments/utility_relation/p3_c1_b5_optimizer_pilot/__init__.py", "experiments/utility_relation/p3_c1_b5_optimizer_pilot/protocol.py", "experiments/utility_relation/p3_c1_b5_optimizer_pilot/optimizer_factory.py", "experiments/utility_relation/p3_c1_b5_optimizer_pilot/candidate_reuse.py", "experiments/utility_relation/p3_c1_b5_optimizer_pilot/context.py", "experiments/utility_relation/p3_c1_b5_optimizer_pilot/run_candidate.py", "experiments/utility_relation/p3_c1_b5_optimizer_pilot/audit_optimizer_pilot.py", "experiments/utility_relation/p3_c1_b5_optimizer_pilot/finalize_optimizer_pilot.py", "tests/test_p3_c1_b5_optimizer_pilot.py")

@dataclass(frozen=True)
class Candidate:
    dataset: str; optimizer_name: str; semantic_lr: float; semantic_exposure: int
    def key(self) -> str: return f"{self.dataset}|{self.optimizer_name}|lr={self.semantic_lr:.17g}|es={self.semantic_exposure}"

def candidates(dataset: str | None = None) -> tuple[Candidate, ...]:
    datasets = DATASETS if dataset is None else (dataset,)
    if any(name not in DATASETS for name in datasets): raise ValueError("B5_UNREGISTERED_DATASET")
    return tuple(Candidate(name, optimizer, lr, EXPOSURES[name]) for name in datasets for optimizer in OPTIMIZERS for lr in LR_GRID[name][optimizer])

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as h:
        for block in iter(lambda: h.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

def write_json(path: str | Path, payload: object) -> None:
    target=Path(path); target.parent.mkdir(parents=True, exist_ok=True); tmp=target.with_name(target.name+".tmp"); tmp.write_text(json.dumps(payload, sort_keys=True, indent=2)+"\n", encoding="utf-8"); tmp.replace(target)

def source_hashes() -> dict[str, str]: return {name: sha256_file(REPO_ROOT/name) for name in SOURCE_RELATIVE_PATHS}

def selection_key(record: dict) -> tuple:
    c=record["candidate"]
    priority={"Adam": 2, "AdamW": 1, "SGD_MOMENTUM": 0}[c.optimizer_name]
    return (round(float(record["mean_delta_J"]),12), round(float(record["worst_delta_J"]),12), int(record["positive_count"]), priority, -float(c.semantic_lr))
