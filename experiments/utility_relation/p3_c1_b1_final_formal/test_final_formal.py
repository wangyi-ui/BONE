from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from . import evaluate_final_formal as evaluator
from . import final_formal_context as context
from . import final_formal_protocol as protocol
from . import run_final_formal as runner


EXPECTED = {
    "HandWrite": (1e-5, 1),
    "BDGP": (3e-4, 5),
    "MSRC": (1e-3, 1),
    "COIL20": (1e-5, 5),
    "Caltech-5V": (1e-3, 5),
    "MNIST-USPS": (1e-4, 5),
    "ALOI": (1e-4, 1),
    "CCV": (1e-5, 3),
}


def _audit(config: dict) -> dict:
    return {
        "dataset": config["dataset"],
        "training_seed": config["training_seed"],
        "dataset_adapter": config["dataset_adapter"],
        "semantic_lr": config["semantic_lr"],
        "semantic_exposure": config["semantic_exposure"],
        "final_parameter_freeze_sha256": config["final_parameter_freeze_sha256"],
        "BASE_OURS_same_dataset_seed_corruption": True,
        "A2_FEASIBILITY_PRESERVED": True,
        "R3_CANONICALIZATION_PRESERVED": True,
        "GT_loaded_before_seal": False,
        "full_gt_used_for_training": False,
        "full_gt_used_for_model_selection": False,
        "full_gt_used_for_hyperparameter_selection": False,
        "ACC_NMI_ARI_computed": False,
    }


def _mock_backend(config: dict, root: Path) -> dict:
    artifacts = []
    for arm, predictions in (
        ("BASE", np.asarray([0, 0, 1, 1], dtype=np.int64)),
        ("OURS", np.asarray([0, 1, 1, 0], dtype=np.int64)),
    ):
        directory = root / arm
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "final_model_state.pt").write_bytes(arm.encode("ascii"))
        np.savez_compressed(
            directory / "pre_gt_predictions.npz",
            sample_ids=np.arange(4, dtype=np.int64),
            predictions=predictions,
        )
        artifacts.extend(
            [f"{arm}/final_model_state.pt", f"{arm}/pre_gt_predictions.npz"]
        )
    return {"artifacts": artifacts, "audit": _audit(config)}


def test_exact_dataset_seed_and_frozen_parameter_bindings():
    assert protocol.DATASETS == tuple(EXPECTED)
    assert protocol.FORMAL_SEEDS == (20, 30, 50)
    bindings = protocol.load_final_parameter_bindings()
    assert {
        name: (item.semantic_lr, item.semantic_exposure)
        for name, item in bindings.items()
    } == EXPECTED
    assert {item.final_parameter_freeze_sha256 for item in bindings.values()} == {
        protocol.FINAL_HYPERPARAMETERS_SHA256
    }


def test_cli_has_no_hyperparameter_override():
    parser = runner.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "semantic_lr" not in destinations
    assert "semantic_exposure" not in destinations
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--dataset", "MSRC", "--training-seed", "20",
                "--semantic-lr", "0.9",
            ]
        )


def test_missing_and_malformed_freeze_fail_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="FREEZE_MISSING"):
        protocol.load_final_parameter_bindings(tmp_path / "missing")
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "FINAL_HYPERPARAMETERS.json").write_text("{}", encoding="utf-8")
    (bad / "STATUS.txt").write_text("not frozen", encoding="utf-8")
    with pytest.raises(RuntimeError, match="HASH_MISMATCH"):
        protocol.load_final_parameter_bindings(bad)


def test_read_only_preflight_resolves_all_24_and_msrc_actual_route():
    result = context.preflight_all_24()
    assert result["INTENDED_RUN_COUNT"] == 24
    assert result["RESOLVABLE_RUN_COUNT"] == 24
    assert result["UNRESOLVED_RUN_COUNT"] == 0
    assert result["ALL_24_FORMAL_RUNS_READY"] is True
    assert result["REAL_FORMAL_RUN"] is False
    assert result["FULL_GT_READ"] is False
    msrc = [row for row in result["runs"] if row["dataset"] == "MSRC"]
    assert len(msrc) == 3
    assert all(row["dataset_adapter"] == protocol.MSRC_ROUTE for row in msrc)
    assert "msrc_adapter._load_frozen_msrc_inputs" in protocol.MSRC_ROUTE
    assert callable(context.msrc_adapter._load_frozen_msrc_inputs)


def test_mock_run_output_isolated_audited_resumable_and_idempotent(tmp_path):
    output = tmp_path / "new_final_only"
    result = runner.run_final_formal(
        "HandWrite", 20, output_root=output, backend=_mock_backend
    )
    assert result["status"] == "COMPLETE"
    root = output / "handwrite" / "seed20"
    assert root != protocol.REPO_ROOT / "outputs/bone_p3/p3_c0_formal/handwrite/seed20"
    for name in ("RUN_CONFIG.json", "PRE_GT_AUDIT.json", "PRE_GT_SEAL.json", "COMPLETE.json"):
        assert (root / name).is_file()
    config = json.loads((root / "RUN_CONFIG.json").read_text())
    audit = json.loads((root / "PRE_GT_AUDIT.json").read_text())
    assert (config["semantic_lr"], config["semantic_exposure"]) == EXPECTED["HandWrite"]
    assert audit["BASE_OURS_same_dataset_seed_corruption"] is True
    assert audit["GT_loaded_before_seal"] is False
    assert audit["A2_FEASIBILITY_PRESERVED"] is True
    assert audit["R3_CANONICALIZATION_PRESERVED"] is True

    def forbidden_backend(_config, _root):
        raise AssertionError("sealed COMPLETE must never retrain")

    again = runner.run_final_formal(
        "HandWrite", 20, output_root=output, backend=forbidden_backend
    )
    assert again["status"] == "ALREADY_COMPLETE_VERIFIED"


def test_partial_config_resumes_without_overwrite(tmp_path):
    output = tmp_path / "resume"
    calls = []

    def interrupted(config, root):
        calls.append((config["dataset"], config["training_seed"]))
        raise RuntimeError("simulated interruption before training artifacts")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.run_final_formal("BDGP", 30, output_root=output, backend=interrupted)
    config_path = output / "bdgp" / "seed30" / "RUN_CONFIG.json"
    before = config_path.read_bytes()
    result = runner.run_final_formal("BDGP", 30, output_root=output, backend=_mock_backend)
    assert result["status"] == "COMPLETE"
    assert config_path.read_bytes() == before
    assert calls == [("BDGP", 30)]


def test_evaluator_reads_gt_only_after_verified_seal_and_is_reporting_only(monkeypatch, tmp_path):
    output = tmp_path / "evaluation"
    runner.run_final_formal("MSRC", 50, output_root=output, backend=_mock_backend)
    calls = []

    def fake_loadmat(path, variable_names):
        calls.append((Path(path).name, tuple(variable_names)))
        return {"Y": np.asarray([[0], [0], [1], [1]], dtype=np.int64)}

    monkeypatch.setattr(evaluator.scipy.io, "loadmat", fake_loadmat)
    before = protocol.sha256_file(
        protocol.DEFAULT_FREEZE / "FINAL_HYPERPARAMETERS.json"
    )
    result = evaluator.evaluate_final_formal("MSRC", 50, output)
    after = protocol.sha256_file(
        protocol.DEFAULT_FREEZE / "FINAL_HYPERPARAMETERS.json"
    )
    assert calls == [("MSRC.mat", ("Y",))]
    assert set(result["BASE"]) == {"ACC", "NMI", "ARI"}
    assert set(result["OURS"]) == {"ACC", "NMI", "ARI"}
    assert set(result["delta_OURS_minus_BASE"]) == {"ACC", "NMI", "ARI"}
    assert result["status"] == "REPORTING_ONLY"
    assert result["metrics_used_for_parameter_selection"] is False
    assert result["parameter_retuning_allowed"] is False
    assert before == after == protocol.FINAL_HYPERPARAMETERS_SHA256


def test_evaluator_never_reads_gt_when_seal_is_corrupt(monkeypatch, tmp_path):
    output = tmp_path / "corrupt"
    runner.run_final_formal("COIL20", 20, output_root=output, backend=_mock_backend)
    prediction = output / "coil20" / "seed20" / "BASE" / "pre_gt_predictions.npz"
    prediction.write_bytes(b"corrupt")
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("GT must not be read")

    monkeypatch.setattr(evaluator.scipy.io, "loadmat", forbidden)
    with pytest.raises(RuntimeError, match="SEAL_INVALID"):
        evaluator.evaluate_final_formal("COIL20", 20, output)
    assert called is False


def test_parent_sha_integrity_fails_closed(monkeypatch):
    altered = dict(protocol.PARENT_SOURCE_SHA256)
    target = next(iter(altered))
    altered[target] = "0" * 64
    monkeypatch.setattr(protocol, "PARENT_SOURCE_SHA256", altered)
    with pytest.raises(RuntimeError, match="FROZEN_PARENT_HASH_MISMATCH"):
        protocol.verify_parent_sources()


def test_default_backend_contract_encodes_frozen_100_plus_20_schedule():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "range(protocol.CARRIER_EPOCHS)" in source
    assert "range(protocol.REFINEMENT_EPOCHS)" in source
    assert "epoch < semantic_exposure" in source
    assert "c0._semantic_epoch" in source
    assert "c0._native_epoch" in source
    assert protocol.CARRIER_EPOCHS == 100
    assert protocol.REFINEMENT_EPOCHS == 20

