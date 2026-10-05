from pathlib import Path
import re

from experiments.utility_relation.p3_c1_b1_optuna import formal_carrier_materializer as frozen
from experiments.utility_relation.p3_c1_b1_optuna import formal_carrier_materializer_r1 as r1
from experiments.utility_relation.p3_c1_b1_optuna import formal_msrc_carrier_adapter
from experiments.utility_relation.p3_c1_b1_optuna.materialize_formal_dev_carrier_r1 import (
    AUDIT_CLI_OUTPUT_OPTION,
)
from experiments.utility_relation.p3_c1_b1_optuna.test_formal_carrier_materializer import (
    _write_fake_adapter_artifacts,
)


def install_r1_fakes(monkeypatch, tmp_path, calls):
    split, corruption = tmp_path / "split.npz", tmp_path / "mask.npz"
    split.write_bytes(b"split"); corruption.write_bytes(b"mask")
    monkeypatch.setattr(frozen, "_expected_dimensions", lambda _dataset: (4, 2))
    monkeypatch.setattr(frozen, "_artifact_paths", lambda _dataset: (split, corruption))
    def generic(dataset, seed, device, staging):
        calls.append(("generic", dataset, seed, device)); _write_fake_adapter_artifacts(staging); return {}
    def msrc(seed, device, staging):
        calls.append(("formal_msrc", "MSRC", seed, device)); _write_fake_adapter_artifacts(staging); return {}
    monkeypatch.setattr(frozen.carrier_adapter, "build_development_carrier", generic)
    monkeypatch.setattr(formal_msrc_carrier_adapter, "build_formal_msrc_development_carrier", msrc)


def test_msrc_seed37_reaches_formal_adapter_and_completes(monkeypatch, tmp_path):
    calls = []; install_r1_fakes(monkeypatch, tmp_path, calls)
    result = r1.materialize_one_r1("MSRC", 37, "cuda:0", tmp_path / "formal")
    assert result["status"] == "COMPLETE"
    assert calls == [("formal_msrc", "MSRC", 37, "cuda:0")]
    assert "formal_msrc_adapter_source" in result["audit"]["frozen_source_sha256"]


def test_generic_route_remains_frozen_generic_adapter(monkeypatch, tmp_path):
    calls = []; install_r1_fakes(monkeypatch, tmp_path, calls)
    result = r1.materialize_one_r1("CCV", 37, "cuda:0", tmp_path / "formal")
    assert result["status"] == "COMPLETE"
    assert calls == [("generic", "CCV", 37, "cuda:0")]


def test_r1_is_thin_composition_and_restores_frozen_extension_points(monkeypatch, tmp_path):
    calls = []; install_r1_fakes(monkeypatch, tmp_path, calls)
    invoke_before, provenance_before = frozen._invoke_frozen_adapter, frozen._source_provenance
    r1.materialize_one_r1("MSRC", 37, "cuda:0", tmp_path / "formal")
    assert frozen._invoke_frozen_adapter is invoke_before
    assert frozen._source_provenance is provenance_before
    source = Path(r1.__file__).read_text(encoding="utf-8")
    assert "frozen_materializer.materialize_one(" in source
    assert "os.replace" not in source and "COMPLETE.json" not in source


def test_scientific_protocol_flags_are_all_unchanged():
    assert r1.CHANGE_TYPE == "IMPLEMENTATION_ROUTING_ONLY"
    assert r1.SCIENTIFIC_PROTOCOL_CHANGED is False
    assert r1.U_CYCLE_CHANGED is False
    assert r1.T_REL_CHANGED is False
    assert r1.A1_CHANGED is False
    assert r1.SPARSE_LABELS_CHANGED is False
    assert r1.SEARCH_SPACE_CHANGED is False
    assert r1.DEV_SEEDS_CHANGED is False
    assert r1.CARRIER_EPOCHS_CHANGED is False


def test_audit_cli_documentation_uses_output_not_output_json():
    assert AUDIT_CLI_OUTPUT_OPTION == "--output"
    source = Path(__file__).with_name("materialize_formal_dev_carrier_r1.py").read_text(encoding="utf-8")
    assert "--output-json" not in source
    frozen_audit = Path(__file__).with_name("audit_formal_dev_carriers.py").read_text(encoding="utf-8")
    assert 'add_argument("--output"' in frozen_audit


def test_r1_sources_have_no_gt_metric_optuna_or_refinement_path():
    names = (
        "formal_msrc_carrier_adapter.py",
        "formal_carrier_materializer_r1.py",
        "materialize_formal_dev_carrier_r1.py",
    )
    source = "\n".join(Path(__file__).with_name(name).read_text(encoding="utf-8") for name in names)
    assert not re.search(r"\b(?:ACC|NMI|Purity|ARI)\b", source)
    assert "create_study" not in source
    assert "evaluate_bone_p3_c0_cross_dataset_formal" not in source
    assert "trial_runner" not in source and "run_single_candidate" not in source
