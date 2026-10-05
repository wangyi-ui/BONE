"""
P3-C1-B0 Cross-dataset Candidate Validation Protocol.

Hyperparameters were selected ONLY from ALOI diagnostics.

Frozen candidate:
    semantic_lr = 1e-5
    refinement_budget = 5

Held-out validation datasets MUST NOT modify this configuration.
"""

from pathlib import Path

SEMANTIC_LR = 1e-5
REFINEMENT_BUDGET = 5

TUNING_DATASET = "ALOI"

VALIDATION_DATASETS = (
    "Caltech-5V",
    "BDGP",
    "HandWrite",
    "COIL20",
    "MNIST-USPS",
    "CCV",
)

SEEDS = (
    20,
    30,
    50,
)

OUTPUT_ROOT = Path(
    "outputs/bone_p3/p3_c1_b0_candidate_validation"
)

CONFIG_FROZEN = True
NO_DATASET_SPECIFIC_RETUNING = True
