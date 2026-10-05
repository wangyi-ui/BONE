from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_NAMES = (
    "formal_tuning_context.py", "formal_base_reference.py", "formal_candidate_evaluator.py",
    "run_formal_optuna_tuning.py", "finalize_formal_tuning.py", "audit_formal_tuning.py",
)


def test_new_runner_sources_are_gt_free_and_metric_free():
    text = "\n".join((ROOT / name).read_text(encoding="utf-8") for name in SOURCE_NAMES)
    forbidden = ("ACC", "NMI", "Purity", "ARI", "FORMAL_SEEDS", "seed20", "seed30", "seed50")
    assert not any(token in text for token in forbidden)
    assert "full_gt_loaded" in text
    assert "full_gt_used" in text


def test_no_p3_c0_evaluator_or_automatic_extension_path():
    text = (ROOT / "formal_candidate_evaluator.py").read_text(encoding="utf-8")
    assert "evaluate_bone_p3_c0" not in text
    assert "required_lr_extension" not in text
    assert "optuna" not in text.lower()


def test_new_runner_declares_no_real_execution_by_itself():
    text = (ROOT / "run_formal_optuna_tuning.py").read_text(encoding="utf-8")
    assert "max_new_trials" in text
    assert "n_jobs=1" in text
    assert "GridSampler" in text

