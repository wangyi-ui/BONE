from pathlib import Path
import re


ROOT = Path(__file__).parent
SOURCES = (
    "boundary_extension_protocol.py",
    "boundary_extension_context.py",
    "run_boundary_extension.py",
    "finalize_boundary_extension.py",
    "audit_boundary_extension.py",
)


def test_boundary_extension_sources_have_no_gt_or_metric_selection_path():
    text = "\n".join((ROOT / name).read_text(encoding="utf-8") for name in SOURCES)
    assert re.search(r"\b(?:ACC|NMI|ARI)\b", text) is None
    assert not any(token in text for token in (
        "full_ground_truth", "FORMAL_SEEDS", "seed20", "seed30", "seed50",
        "evaluate_bone_p3_c0", "required_lr_extension",
    ))
    assert "full_gt_used" in text


def test_extension_sources_declare_no_second_stage_or_posthoc_lr():
    text = "\n".join((ROOT / name).read_text(encoding="utf-8") for name in SOURCES)
    assert "SECOND_BOUNDARY_EXTENSION_ALLOWED = False" in text
    assert "3e-3" not in text
    assert "1e-2" not in text
    assert "1e-6" not in text

