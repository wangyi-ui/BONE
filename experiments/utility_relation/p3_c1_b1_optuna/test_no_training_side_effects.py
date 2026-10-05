import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
IMPLEMENTATION = ("study_schema.py", "optuna_driver.py", "audit.py", "summarize.py")


def test_implementation_source_has_no_training_or_data_imports():
    forbidden_roots = {"method", "torch", "tensorflow", "cuda", "datasets"}
    for filename in IMPLEMENTATION:
        tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not {name.name.split(".")[0] for name in node.names} & forbidden_roots
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden_roots


def test_only_explicit_implementation_files_are_present():
    present = {path.name for path in ROOT.glob("*.py")}
    assert set(IMPLEMENTATION) <= present
    assert "carrier_adapter.py" not in present
    assert "trial_runner.py" not in present
