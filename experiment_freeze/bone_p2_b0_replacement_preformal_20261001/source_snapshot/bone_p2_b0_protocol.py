"""Fixed seed20 protocol for the P2-B0 native-consistency replacement diagnostic."""

from __future__ import annotations

from pathlib import Path

from experiments.utility_relation.bone_p2_a1_protocol import (
    LAMBDA, LAYER, N_ANCHOR, NATIVE_BATCH_SIZE, NATIVE_LR, REFINEMENT_EPOCHS,
    SEMANTIC_LR, SEMANTIC_BATCH_SIZE,
)


TRAINING_SEED = 20
CORRUPTION_SEED = 20
CARRIER_DIR = Path("outputs/bone_p2/p2_a0_msrc_seed20")
PARENT_P2_A1_DIR = Path("outputs/bone_p2/p2_a1_seed20_formal")
DEFAULT_OUTPUT_DIR = Path("outputs/bone_p2/p2_b0_seed20")
ARMS = ("NATIVE", "ADDON_TRUE_U", "REG_ONLY", "REPLACE_TRUE_U")
NEW_ARMS = ("REG_ONLY", "REPLACE_TRUE_U")
PARENT_ARM_SOURCES = {"NATIVE": "BASE", "ADDON_TRUE_U": "TRUE_U"}
FROZEN_INITIAL_MODEL_SHA256 = "4cd943984c4d31d5655767706241695b6ad3cd3825c62d986a3e05e21eafd42f"
