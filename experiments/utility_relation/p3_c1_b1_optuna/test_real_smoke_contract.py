import ast
from pathlib import Path

import pytest

from experiments.utility_relation.p3_c1_b1_optuna.run_real_smoke import (
    build_parser, validate_smoke_args,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path("outputs/bone_p3/p3_c1_b1_optuna/engineering_smoke")


def test_cli_exact_candidate_is_accepted_and_unregistered_is_rejected():
    args = build_parser().parse_args([
        "--dataset", "Caltech-5V", "--dev-seed", "11", "--semantic-lr", "1e-5",
        "--semantic-exposure", "5", "--device", "cuda:0", "--output-root", str(OUTPUT_ROOT),
    ])
    validate_smoke_args(args.dataset, args.dev_seed, args.semantic_lr, args.semantic_exposure, args.device, args.output_root)
    with pytest.raises(ValueError):
        validate_smoke_args("Caltech-5V", 11, 2e-5, 5, "cuda:0", OUTPUT_ROOT)
    with pytest.raises(ValueError):
        validate_smoke_args("Caltech-5V", 37, 1e-5, 5, "cuda:0", OUTPUT_ROOT)


def test_no_optuna_study_or_formal_output_path_exists_in_bridge_sources():
    for filename in ("carrier_adapter.py", "trial_runner.py", "run_real_smoke.py"):
        source = (ROOT / filename).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        assert "optuna" not in imports
        assert "study.db" not in source
        assert "p3_c0_formal" not in source
    assert "engineering_smoke" in (ROOT / "run_real_smoke.py").read_text(encoding="utf-8")


def test_parent_integrity_gate_precedes_any_real_carrier_call():
    source = (ROOT / "run_real_smoke.py").read_text(encoding="utf-8")
    run_body = source[source.index("def run_smoke"):source.index("def build_parser")]
    assert run_body.index("verify_parent_integrity_gate()") < run_body.index("build_development_carrier(")
    assert "loadmat" not in source
