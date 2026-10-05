import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
SOURCES = (
    "formal_carrier_materializer.py",
    "materialize_formal_dev_carrier.py",
    "audit_formal_dev_carriers.py",
)


def test_new_sources_have_no_gt_metric_optuna_or_refinement_dependency():
    combined = "\n".join((ROOT / name).read_text(encoding="utf-8") for name in SOURCES)
    assert not re.search(r"\b(?:ACC|NMI|Purity|ARI)\b", combined)
    assert "optuna.create_study" not in combined and "create_study(" not in combined
    assert "evaluate_bone_p3_c0_cross_dataset_formal" not in combined
    assert "trial_runner" not in combined and "run_single_candidate" not in combined
    assert "FORMAL_SEEDS" not in combined


def test_materializer_routes_to_frozen_adapters_without_training_reimplementation():
    source = (ROOT / "formal_carrier_materializer.py").read_text(encoding="utf-8")
    assert "carrier_adapter.build_development_carrier" in source
    assert "msrc_adapter.build_msrc_development_carrier" in source
    tree = ast.parse(source)
    called = {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "backward" not in called and "step" not in called
    assert "SEMANTIC_LR_GRID =" not in source
    assert "DEV_SEEDS =" not in source and "CARRIER_EPOCHS =" not in source


def test_real_formal_output_namespace_is_not_created_by_tests():
    real = ROOT.parents[2] / "outputs/bone_p3/p3_c1_b1_optuna/formal_tuning"
    assert not real.exists()
