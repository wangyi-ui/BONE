from pathlib import Path
import re

import numpy as np
import pytest

from . import amendment_a1, amendment_a2, objective_a1, protocol


@pytest.mark.parametrize(
    "dataset,K",
    [
        ("HandWrite", 10), ("BDGP", 5), ("MSRC", 7),
        ("COIL20", 20), ("Caltech-5V", 7), ("MNIST-USPS", 10),
    ],
)
def test_existing_a1_feasible_datasets_also_pass_a2_without_semantic_change(dataset, K):
    del dataset
    targets = np.repeat(np.arange(K), 2)
    raw_ids = np.arange(2 * K)
    masses = np.ones(2 * K)
    a1 = amendment_a1.validate_class_feasibility(targets, masses, range(K))
    a2 = amendment_a2.validate_coverage_aware_feasibility(
        targets, raw_ids, masses, range(K)
    )
    assert a1["all_classes_feasible"] is True
    assert a2["A2_FEASIBILITY"] == "PASS"
    assert a2["supported_anchor_count"] == 2 * K
    assert a2["abstained_class_count"] == 0


def test_a2_uses_exact_a1_objective_with_zero_contribution_and_denominator_K():
    utility = np.asarray([[0.0], [2.0]])
    probabilities = np.asarray([[[0.5, 0.5]], [[0.25, 0.75]]])
    targets = np.asarray([0, 1], dtype=np.int64)
    score, audit = objective_a1.validation_j_with_audit(
        utility, probabilities, targets
    )
    assert audit["anchor_scores"][0] == 0.0
    assert audit["anchor_scores"][1] == pytest.approx(np.log(2 * 0.75))
    assert audit["fold_denominator"] == 2
    assert score == pytest.approx(np.log(1.5) / 2)


def test_entire_unsupported_class_contributes_zero_without_renormalization():
    utility = np.asarray([[0.0], [1.0], [0.0]])
    probabilities = np.asarray([
        [[0.9, 0.05, 0.05]],
        [[0.1, 0.8, 0.1]],
        [[0.1, 0.1, 0.8]],
    ])
    targets = np.asarray([0, 1, 2], dtype=np.int64)
    score, audit = objective_a1.validation_j_with_audit(
        utility, probabilities, targets
    )
    assert audit["anchor_scores"] == pytest.approx([0.0, np.log(2.4), 0.0])
    assert audit["fold_denominator"] == 3
    assert score == pytest.approx(np.log(2.4) / 3)


def test_a2_sources_have_no_threshold_gt_or_metric_selection_path():
    root = Path(__file__).parent
    text = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in (
            "amendment_a2.py", "formal_tuning_context_a2.py",
            "run_formal_optuna_tuning_a2.py",
        )
    )
    assert re.search(r"\b(?:ACC|NMI|ARI)\b", text) is None
    forbidden = (
        "full_ground_truth", "FORMAL_SEEDS", "seed20", "seed30", "seed50",
        "epsilon", "objective_a2",
    )
    assert not any(token in text for token in forbidden)
    assert "from .formal_candidate_evaluator_r2 import evaluate_candidate" in text
    assert amendment_a2.COVERAGE_THRESHOLD_ADDED is False
    assert amendment_a2.UTILITY_THRESHOLD_ADDED is False
    assert protocol.SEMANTIC_LR_GRID == (3e-6, 1e-5, 3e-5, 1e-4, 3e-4)
    assert protocol.SEMANTIC_EXPOSURE_GRID == (1, 3, 5, 10, 20)
