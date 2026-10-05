from pathlib import Path
import re

from experiments.utility_relation.p3_c1_b1_optuna import amendment_a1, protocol
from experiments.utility_relation.p3_c1_b1_optuna.run_msrc_a1_smoke import (
    EXPECTED_CARRIER_PRE_GT, EXPECTED_CARRIER_STATE, EXPECTED_OUTPUT_ROOT,
)


ROOT = Path(__file__).resolve().parent


def test_a1_applies_to_msrc_without_changing_base_search_protocol():
    assert amendment_a1.APPLIES_TO_ALL_DATASETS is True
    assert amendment_a1.AMENDED_DATASETS is protocol.DATASETS
    assert "MSRC" in amendment_a1.AMENDED_DATASETS
    assert protocol.SEMANTIC_LR_GRID == (3e-6, 1e-5, 3e-5, 1e-4, 3e-4)
    assert protocol.SEMANTIC_EXPOSURE_GRID == (1, 3, 5, 10, 20)
    assert protocol.REFINEMENT_HORIZON == 20


def test_wrapper_is_carrier_reuse_only_and_output_isolated():
    source = (ROOT / "run_msrc_a1_smoke.py").read_text(encoding="utf-8")
    assert "build_msrc_development_carrier" not in source
    assert "build_development_carrier" not in source
    assert str(EXPECTED_CARRIER_STATE).endswith("dev_seed11/carrier/carrier_model_state.pt")
    assert str(EXPECTED_CARRIER_PRE_GT).endswith("dev_seed11/carrier/carrier_pre_gt.npz")
    assert str(EXPECTED_OUTPUT_ROOT).endswith("engineering_msrc_a1_smoke")
    assert "engineering_msrc_adapter_smoke" not in str(EXPECTED_OUTPUT_ROOT)


def test_a1_objective_and_frozen_training_primitives_are_explicitly_used():
    runner = (ROOT / "trial_runner_a1.py").read_text(encoding="utf-8")
    assert "objective_a1.validation_j_with_audit" in runner
    assert "trial_runner.execute_isolated_schedule" in runner
    assert "c0._semantic_epoch" in runner and "c0._native_epoch" in runner
    assert "objective.validation_j(" not in runner


def test_no_seed37_optuna_formal_gt_metrics_or_other_dataset_loop():
    source = (ROOT / "run_msrc_a1_smoke.py").read_text(encoding="utf-8")
    source += (ROOT / "trial_runner_a1.py").read_text(encoding="utf-8")
    assert "protocol.DEV_SEEDS[1]" not in source
    assert "create_study" not in source and "study.db" not in source
    assert "select_best_candidate" not in source and "CandidateEvaluation" not in source
    assert not re.search(r"\b(?:ACC|NMI|Purity|ARI)\b", source)
    assert "for dataset in" not in source


def test_summary_contract_declares_no_training_or_full_gt():
    source = (ROOT / "run_msrc_a1_smoke.py").read_text(encoding="utf-8")
    assert '"carrier_reused": True' in source
    assert '"carrier_training_run": False' in source
    assert '"full_gt_loaded": False' in source
    assert '"SMOKE_PARTIAL_OBJECTIVE_ONLY": True' in source
