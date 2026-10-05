"""Frozen protocol constants for the BONE-P2-A1 relation-action pilot."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from experiments.utility_relation.bone_p2_a0_protocol import K, L, N, NU, S, V
from experiments.utility_relation.evaluate_bone_p2_a0_carrier_v2 import global_derangement, permutation_sha256


TRAINING_SEED = 20
CORRUPTION_SEED = 20
TARGET_SNR_DB = 2.5
ARMS = ("BASE", "TRUE_UNIFORM", "TRUE_U", "SHUFFLE_REL")
ACTIVE_SEMANTIC_ARMS = ("TRUE_UNIFORM", "TRUE_U", "SHUFFLE_REL")
REFINEMENT_EPOCHS = 20
SEMANTIC_BATCH_SIZE = 128
NATIVE_BATCH_SIZE = 128
SEMANTIC_LR = 1e-4
NATIVE_LR = 0.01
N_ANCHOR = 3
LAYER = 5
LAMBDA = 0.1
CARRIER_DIR = Path("outputs/bone_p2/p2_a0_msrc_seed20")
CARRIER_NPZ = CARRIER_DIR / "p2_a0_carrier_pre_gt.npz"
INITIAL_STATE = CARRIER_DIR / "carrier_model_state.pt"
DEFAULT_OUTPUT_DIR = Path("outputs/bone_p2/p2_a1_msrc_seed20")
FROZEN_PERMUTATION_SHA256 = "e97d91e550d9c3f2d9c7c85a922b72d47cb3b0223cf2e04dee3e90bc2951a155"


def frozen_derangement() -> np.ndarray:
    """Return the single registered P2-A0 V2 derangement, fail-closed on drift."""
    permutation = global_derangement(NU, seed=TRAINING_SEED)
    if permutation_sha256(permutation) != FROZEN_PERMUTATION_SHA256:
        raise ValueError("frozen P2-A0 V2 derangement hash mismatch")
    return permutation


def semantic_orders() -> np.ndarray:
    """Pre-generate the shared query-position order for all active semantic arms."""
    rng = np.random.default_rng(TRAINING_SEED)
    return np.stack([rng.permutation(NU).astype(np.int64) for _ in range(REFINEMENT_EPOCHS)])


def array_sha256(values: np.ndarray) -> str:
    """Stable digest for audit-only fixed integer schedules."""
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()
