from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from experiments.utility_relation.p3_c1_b4_msrc_historical_default_confirmatory import audit, context, protocol
from experiments.utility_relation.p3_c1_b4_msrc_historical_default_confirmatory import evaluate_confirmatory as evaluator
from experiments.utility_relation.p3_c1_b4_msrc_historical_default_confirmatory import run_confirmatory as runner
from experiments.utility_relation.p3_c1_b4_msrc_historical_default_confirmatory import summarize_confirmatory as summarizer


@pytest.fixture
def preregistration(tmp_path):
    path = tmp_path / "preregistered"
    audit.preregister(path)
    return path


def _mock_backend(config: dict, root: Path) -> dict:
    shared = "c" * 64
    artifacts = []
    for arm, pred in (("BASE", [0, 0, 1, 1]), ("OURS_HISTORICAL_DEFAULT", [0, 0, 1, 0])):
        directory = root / arm; directory.mkdir(parents=True, exist_ok=True)
        (directory / "final_model_state.pt").write_bytes(arm.encode())
        np.savez_compressed(directory / "pre_gt_predictions.npz", sample_ids=np.arange(4), predictions=np.asarray(pred))
        artifacts += [f"{arm}/final_model_state.pt", f"{arm}/pre_gt_predictions.npz"]
    return {"artifacts": artifacts, "audit": {
        "dataset": "MSRC", "training_seed": config["training_seed"], "arms": list(protocol.ARMS),
        "carrier_state_sha256": shared, "base_initial_state_sha256": shared, "historical_default_initial_state_sha256": shared,
        "corruption_artifact_sha256": config["corruption_artifact_sha256"], "sparse_split_sha256": config["sparse_split_sha256"], "carrier_pre_gt_sha256": "d" * 64,
        "semantic_optimizer": "Adam", "semantic_optimizer_steps_per_epoch": [1] * 20,
        "base_semantic_steps": 0, "historical_default_semantic_steps": 20,
        "GT_loaded_before_seal": False, "full_gt_used_for_training": False, "full_gt_used_for_selection": False,
        "ACC_NMI_ARI_computed": False, "b1_b2_result_used_for_parameter_selection": False,
        "optimizer_search": False, "parameter_search": False,
    }}


def test_exact_dataset_seeds_arms_and_historical_contract():
    assert protocol.DATASET == "MSRC" and protocol.CONFIRMATORY_SEEDS == (100, 110, 130)
    assert protocol.ARMS == ("BASE", "OURS_HISTORICAL_DEFAULT")
    assert (protocol.HISTORICAL_DEFAULT_SEMANTIC_LR, protocol.HISTORICAL_DEFAULT_SEMANTIC_EXPOSURE) == (1e-4, 20)
    assert (protocol.REFINEMENT_EPOCHS, protocol.SEMANTIC_BATCH_SIZE, protocol.NATIVE_BATCH_SIZE, protocol.NATIVE_LR) == (20, 128, 128, 1e-2)
    assert protocol.FORBIDDEN_SELECTION_SEEDS == (20, 30, 50, 60, 70, 90)
    with pytest.raises(ValueError, match="MSRC_ONLY"): context.resolve_training_inputs("ALOI")
    with pytest.raises(ValueError, match="UNREGISTERED"): protocol.run_root(protocol.OUTPUT_ROOT, 90)


def test_no_cli_parameter_optimizer_or_dataset_override():
    destinations = {item.dest for item in runner.build_parser()._actions}
    assert not destinations.intersection({"semantic_lr", "semantic_exposure", "optimizer", "dataset", "batch_size", "native_lr"})
    with pytest.raises(SystemExit): runner.build_parser().parse_args(["--training-seed", "100", "--semantic-lr", "0.1", "--execute"])


def test_parent_provenance_method_integrity_and_precedence(monkeypatch):
    result = audit.verify_parent_integrity()
    assert result["verified"] and result["historical_precedes_b1_b2_b3"]
    assert result["historical_default_source"] == "P2_A1B_FROZEN_PRE_B1_B2_B3"
    altered = dict(protocol.FROZEN_METHOD_SHA256); altered["method/bone.py"] = "0" * 64
    monkeypatch.setattr(protocol, "FROZEN_METHOD_SHA256", altered)
    with pytest.raises(RuntimeError, match="FROZEN_METHOD_HASH_MISMATCH"): audit.verify_parent_integrity()


def test_preregistration_has_frozen_gate_and_no_real_run(preregistration):
    result = audit.verify_preregistration(preregistration)
    assert result["verified"] is True
    status = (preregistration / "STATUS.txt").read_text()
    assert "REAL_CONFIRMATORY_RUN=false" in status and "PARAMETER_SEARCH=false" in status
    protocol_text = (preregistration / "PROTOCOL.txt").read_text()
    assert "mean Delta_ACC > 0 and mean Delta_NMI > 0 and mean Delta_ARI > 0" in protocol_text


def test_mock_pre_gt_pairing_state_and_output_separation(tmp_path, preregistration):
    output = tmp_path / "outputs"
    result = runner.run_confirmatory(100, output, preregistration_freeze=preregistration, backend=_mock_backend)
    assert result["status"] == "COMPLETE"
    root = output / "seed100"; audit_row = json.loads((root / "PRE_GT_AUDIT.json").read_text())
    assert audit_row["carrier_state_sha256"] == audit_row["base_initial_state_sha256"] == audit_row["historical_default_initial_state_sha256"]
    assert audit_row["semantic_optimizer_steps_per_epoch"] == [1] * 20
    assert audit_row["base_semantic_steps"] == 0 and audit_row["historical_default_semantic_steps"] == 20
    assert all((root / arm / "pre_gt_predictions.npz").is_file() for arm in protocol.ARMS)


def test_evaluator_reads_gt_only_after_seal(monkeypatch, tmp_path, preregistration):
    output = tmp_path / "outputs"; runner.run_confirmatory(110, output, preregistration_freeze=preregistration, backend=_mock_backend)
    calls = []
    def fake_loadmat(path, variable_names): calls.append((Path(path).name, tuple(variable_names))); return {"Y": np.asarray([[0], [0], [1], [1]])}
    monkeypatch.setattr(evaluator.scipy.io, "loadmat", fake_loadmat)
    result = evaluator.evaluate_confirmatory(110, output)
    assert calls == [("MSRC.mat", ("Y",))] and result["metrics_used_for_parameter_selection"] is False


def test_corrupt_seal_blocks_gt(monkeypatch, tmp_path, preregistration):
    output = tmp_path / "outputs"; runner.run_confirmatory(130, output, preregistration_freeze=preregistration, backend=_mock_backend)
    (output / "seed130/BASE/pre_gt_predictions.npz").write_bytes(b"broken")
    monkeypatch.setattr(evaluator.scipy.io, "loadmat", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("GT read")))
    with pytest.raises(RuntimeError, match="SEAL_INVALID"): evaluator.evaluate_confirmatory(130, output)


def test_frozen_semantic_schedule_no_search_and_exact_gate():
    source = Path(runner.__file__).read_text()
    assert "for epoch in range(protocol.REFINEMENT_EPOCHS)" in source
    assert "c0._semantic_epoch" in source and "semantic_steps.append(1)" in source
    assert "optim.Adam" in source and "loadmat" not in source
    assert '"optimizer_search": False' in source
    assert protocol.primary_confirmation_gate({"ACC": 0.1, "NMI": 0.1, "ARI": 0.1}) == "CONFIRMED"
    assert protocol.primary_confirmation_gate({"ACC": 0.1, "NMI": 0.0, "ARI": 0.1}) == "NOT_CONFIRMED"


def test_summary_uses_population_std_and_fixed_gate(tmp_path):
    output = tmp_path / "outputs"
    values = ((0.2, 0.3, 0.4), (-0.1, 0.2, 0.4), (0.1, 0.2, 0.4))
    for seed, delta in zip(protocol.CONFIRMATORY_SEEDS, values):
        root = output / f"seed{seed}"; root.mkdir(parents=True)
        protocol.write_json(root / "CONFIRMATORY_METRICS.json", {"metrics_used_for_parameter_selection": False, "delta_historical_default_minus_base": dict(zip(("ACC", "NMI", "ARI"), delta))})
    result = summarizer.summarize_confirmatory(output)
    assert result["positive_seed_counts"]["ACC"] == 2
    assert result["mean_deltas"]["ACC"]["std_population"] == pytest.approx(np.std([.2, -.1, .1], ddof=0))
    assert result["HISTORICAL_DEFAULT_CONFIRMATION"] == "CONFIRMED"
