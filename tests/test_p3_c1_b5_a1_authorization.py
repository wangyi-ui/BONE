from __future__ import annotations

import json
from pathlib import Path
import pytest

from experiments.utility_relation.p3_c1_b5_optimizer_pilot import protocol
from experiments.utility_relation.p3_c1_b5_optimizer_pilot import run_candidate


def _candidate(name="AdamW", dataset="CCV", lr=3e-6):
    return next(item for item in protocol.candidates(dataset) if item.optimizer_name == name and item.semantic_lr == lr)


def test_without_authorization_remains_fail_closed():
    with pytest.raises(RuntimeError, match="REQUIRES_SEPARATE_AUTHORIZATION"):
        run_candidate.run_candidate(_candidate(), execute=True, authorize_real_candidate=False)
    with pytest.raises(RuntimeError, match="REQUIRES_EXECUTE"):
        run_candidate.run_candidate(_candidate(), execute=False, authorize_real_candidate=True)


def test_authorized_ccv_adamw_is_preflight_only():
    result = run_candidate.authorization_preflight(_candidate())
    assert result["authorization_preflight"] == "PASS"
    assert result["status"] == "PASS"


def test_authorized_sgd_preflight_passes():
    result = run_candidate.authorization_preflight(_candidate("SGD_MOMENTUM", "CCV", 1e-3))
    assert result["authorization_preflight"] == "PASS"


def test_adam_is_reuse_only_and_invalid_candidates_rejected():
    with pytest.raises(RuntimeError, match="REUSE_ONLY"):
        run_candidate.run_candidate(_candidate("Adam", "CCV", 3e-6), execute=True, authorize_real_candidate=True)
    with pytest.raises(ValueError, match="AUTHORIZED_NEW_SET"):
        run_candidate.authorization_preflight(protocol.Candidate("CCV", "AdamW", 9e-6, 3))
    with pytest.raises(ValueError, match="AUTHORIZED_NEW_SET"):
        run_candidate.authorization_preflight(protocol.Candidate("ALOI", "AdamW", 9e-6, 1))
    with pytest.raises(ValueError, match="AUTHORIZED_NEW_SET"):
        run_candidate.authorization_preflight(protocol.Candidate("MSRC", "AdamW", 3e-6, 3))
    with pytest.raises(ValueError, match="AUTHORIZED_NEW_SET"):
        run_candidate.authorization_preflight(protocol.Candidate("CCV", "RMSPROP", 3e-6, 3))
    with pytest.raises(ValueError, match="UNREGISTERED_DATASET"):
        protocol.candidates("MSRC")


def test_old_preregistration_and_scientific_contract_unchanged():
    old_root = protocol.PREREGISTERED_FREEZE
    old = old_root / "STATUS.txt"
    assert "REAL_OPTIMIZER_CANDIDATES_RUN=false" in old.read_text()
    assert len(protocol.candidates()) == 18
    assert protocol.EXPOSURES == {"CCV": 3, "ALOI": 1}
    assert "--authorize-real-candidate" in " ".join(run_candidate.build_parser()._option_string_actions)
    assert protocol.sha256_file(old) == "efb153610c24e3d82d4ba1f9f5824b1698964946c2d887ec2df804b4d851955d"
    assert protocol.sha256_file(old_root / "SEARCH_SPACE.json") == "1e6b8d5257861b9f6b23aeab3feab78beb355d8b73ef826597b0a19b1cbead3b"
    assert protocol.sha256_file(old_root / "SELECTION_RULE.json") == "3d3bcc5fd820bb5decce4c649f34a3821e0c591b1a87c63a9f4264effe6d533b"


def test_frozen_search_space_and_selection_rule_are_exact():
    search = json.loads((protocol.PREREGISTERED_FREEZE / "SEARCH_SPACE.json").read_text())
    assert search["datasets"] == ["CCV", "ALOI"]
    assert search["dev_seeds"] == [11, 37]
    assert search["folds"] == ["A", "B"]
    assert len(search["candidates"]) == 18
    assert search["absolute_utility_mass_cancellation"] == "frozen_and_unresolved"
    selection = json.loads((protocol.PREREGISTERED_FREEZE / "SELECTION_RULE.json").read_text())
    assert selection == {
        "full_gt_used": False,
        "rule": [
            "maximize round(mean_delta_J,12)",
            "maximize round(worst_delta_J,12)",
            "maximize positive_count",
            "tie: Adam then AdamW then SGD_MOMENTUM",
            "same-family tie: smaller semantic_lr",
        ],
    }


def test_methods_and_failed_attempt_artifact_absent():
    assert protocol.sha256_file(protocol.REPO_ROOT / "method/bone.py") == "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092"
    assert protocol.sha256_file(protocol.REPO_ROOT / "method/amv.py") == "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc"
    assert not (protocol.OUTPUT_ROOT / "ccv" / "adamw" / "lr_3e-6").exists()
