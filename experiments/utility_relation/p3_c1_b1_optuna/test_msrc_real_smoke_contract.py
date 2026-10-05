from pathlib import Path
from types import SimpleNamespace
import re

import numpy as np

from experiments.utility_relation.p3_c1_b1_optuna import objective, trial_runner
from experiments.utility_relation.p3_c1_b1_optuna.run_msrc_adapter_smoke import build_msrc_summary


ROOT = Path(__file__).resolve().parent


def _mock_trial():
    schedule = SimpleNamespace(
        native_counts={"base": 20, "A": 20, "B": 20},
        semantic_counts={"A": 5, "B": 5},
        initial_hashes={"base": "same", "A": "same", "B": "same"},
    )
    scores = {
        "A": {"J_BASE": .1, "J_OURS": .2, "DeltaJ": .1},
        "B": {"J_BASE": .3, "J_OURS": .25, "DeltaJ": -.05},
    }
    folds = {
        "A": {"validation_utility_mass": np.ones(7)},
        "B": {"validation_utility_mass": np.ones(7)},
    }
    return {"schedule": schedule, "scores": scores, "fold_payloads": folds}


def test_mock_real_smoke_counts_parity_utility_and_partial_objective():
    carrier = {"arrays": {"q_views": np.ones((210, 4, 7)), "U_cycle": np.ones((210, 6))}}
    summary = build_msrc_summary(carrier, _mock_trial(), 1e-5, 5)
    assert summary["carrier_epochs"] == 100
    assert summary["base_native_epochs"] == 20
    assert (summary["ours_foldA_semantic_steps"], summary["ours_foldA_native_epochs"]) == (5, 20)
    assert (summary["ours_foldB_semantic_steps"], summary["ours_foldB_native_epochs"]) == (5, 20)
    assert summary["initial_state_parity"] is True
    assert summary["fold_A_zero_utility_mass"] is False
    assert summary["fold_B_zero_utility_mass"] is False
    assert summary["SMOKE_PARTIAL_OBJECTIVE_ONLY"] is True
    assert (summary["view_count"], summary["cluster_count"], summary["action_count"]) == (4, 7, 6)


def test_frozen_trial_runner_and_objective_are_reused():
    import experiments.utility_relation.p3_c1_b1_optuna.run_msrc_adapter_smoke as runner
    assert runner.trial_runner.run_single_candidate is trial_runner.run_single_candidate
    assert runner.objective.validation_j is objective.validation_j
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "CandidateEvaluation" not in source
    assert "aggregate_candidate_scores" not in source
    assert "create_study" not in source and "study.db" not in source


def test_no_fake_seed37_full_gt_metric_or_other_dataset_path():
    source = (ROOT / "run_msrc_adapter_smoke.py").read_text(encoding="utf-8")
    assert "protocol.DEV_SEEDS[1]" not in source
    assert not re.search(r"\b(?:ACC|NMI|Purity|ARI)\b", source)
    assert "evaluate_bone_p3_c0_cross_dataset_formal" not in source
    assert "select_best_candidate" not in source
    assert "for dataset in" not in source


def test_u_is_built_once_and_validation_exclusion_delegates_to_frozen_runner():
    adapter_source = (ROOT / "msrc_adapter.py").read_text(encoding="utf-8")
    runner_source = (ROOT / "run_msrc_adapter_smoke.py").read_text(encoding="utf-8")
    assert adapter_source.count("_carrier_arrays(q_views)") == 1
    assert runner_source.count("trial_runner.run_single_candidate(") == 1
    frozen_source = Path(trial_runner.__file__).read_text(encoding="utf-8")
    assert '"validation_rows_excluded_from_semantic_training": True' in frozen_source
