"""B5 binds frozen B1 sparse-CV contexts; it contains no GT loader."""
from __future__ import annotations
from experiments.utility_relation.p3_c1_b1_optuna.formal_tuning_context_a2 import build_formal_tuning_context_a2
from . import protocol

def build_context(dataset: str):
    if dataset not in protocol.DATASETS: raise ValueError("B5_UNREGISTERED_DATASET")
    return build_formal_tuning_context_a2(dataset, protocol.REPO_ROOT/"outputs/bone_p3/p3_c1_b1_optuna/formal_tuning")
