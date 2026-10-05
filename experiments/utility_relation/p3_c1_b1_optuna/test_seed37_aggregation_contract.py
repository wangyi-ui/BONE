import ast
from pathlib import Path

from experiments.utility_relation.p3_c1_b1_optuna.parent_integrity import sha256_file
from experiments.utility_relation.p3_c1_b1_optuna.run_seed37_aggregation_smoke import (
    BRIDGE_FREEZE, load_frozen_seed11_result, seed37_carrier_dir,
)


ROOT = Path(__file__).resolve().parent


def test_seed11_source_is_verified_frozen_artifact_and_not_rerun():
    frozen = BRIDGE_FREEZE / "smoke_summary.json"
    before = sha256_file(frozen)
    result = load_frozen_seed11_result()
    assert result["dev_seed"] == 11
    assert result["delta_j_foldA"] == -0.21656791713400114
    assert result["delta_j_foldB"] == 0.08451769549816235
    assert sha256_file(frozen) == before


def test_seed37_namespace_is_isolated_and_exact():
    root = Path("outputs/bone_p3/p3_c1_b1_optuna/engineering_evaluator_smoke")
    path = seed37_carrier_dir(root)
    assert path == root / "caltech_5v/dev_seed37/carrier"
    assert "engineering_smoke/caltech_5v/dev_seed11" not in str(path)


def test_runner_has_no_optuna_study_formal_evaluator_or_other_dataset_loop():
    source = (ROOT / "run_seed37_aggregation_smoke.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert "optuna" not in imported
    assert "create_study" not in source and "run_remaining_trials" not in source
    assert "study.db" not in source and "trials.csv" not in source
    assert "evaluate_bone_p3_c0_cross_dataset_formal" not in source
    assert "MSRC" not in source
    assert "select_best_candidate" not in source


def test_new_gpu_call_is_exact_seed37_candidate_and_parent_gate_precedes_it():
    source = (ROOT / "run_seed37_aggregation_smoke.py").read_text(encoding="utf-8")
    run_body = source[source.index("def run(output_root"):source.index("def build_parser")]
    assert run_body.index("verify_frozen_inputs()") < run_body.index("build_development_carrier(")
    assert 'candidate = CandidateConfig(1e-5, 5)' in run_body
    assert 'build_development_carrier("Caltech-5V", protocol.DEV_SEEDS[1]' in run_body
    assert "protocol.DEV_SEEDS[0]" not in run_body
