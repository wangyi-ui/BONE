"""P3-A1 deterministic balanced directed complementary action-space tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from experiments.utility_relation import audit_bone_p3_a1_action_space as audit
from experiments.utility_relation import bone_p3_a1_action_space_protocol as protocol


REPO_ROOT = Path(__file__).resolve().parents[1]
P2_CARRIER = REPO_ROOT / protocol.MSRC_CARRIER_PATH
ORIGINAL_P2_CARRIER_SHA256 = "83ea7681491e7ab3e51285446075a3362a43d6122a1397d04a0e059bdc7841ea"
ORIGINAL_BONE_SHA256 = "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092"
ORIGINAL_AMV_SHA256 = "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("view_num", [0, 1, -1, True, 2.0])
def test_view_num_below_or_outside_integer_domain_fails(view_num: object) -> None:
    with pytest.raises(ValueError):
        protocol.build_balanced_directed_actions(view_num)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("view_num", "count"), ((2, 2), (3, 6), (4, 6), (5, 20), (6, 20), (7, 70), (8, 70))
)
def test_action_count_contract(view_num: int, count: int) -> None:
    assert protocol.expected_action_count(view_num) == count
    assert len(protocol.build_balanced_directed_actions(view_num)) == count


def test_v2_and_v3_exact_canonical_order() -> None:
    v2 = protocol.build_balanced_directed_actions(2)
    assert [action.generator_views for action in v2] == [(0,), (1,)]
    assert [action.verifier_views for action in v2] == [(1,), (0,)]
    v3 = protocol.build_balanced_directed_actions(3)
    assert [action.generator_views for action in v3] == [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]
    assert [action.generator_size for action in v3] == [1, 1, 1, 2, 2, 2]


def test_v4_exact_msrc_order_and_existing_carrier_parity() -> None:
    actions = protocol.build_balanced_directed_actions(4)
    assert tuple(action.generator_views for action in actions) == protocol.MSRC_GENERATOR_SUBSETS
    assert tuple(action.verifier_views for action in actions) == protocol.MSRC_VERIFIER_SUBSETS
    assert protocol.verify_msrc_action_order(P2_CARRIER) is True


def test_v5_odd_directed_reverse_closure() -> None:
    actions = protocol.build_balanced_directed_actions(5)
    assert [action.generator_size for action in actions] == [2] * 10 + [3] * 10
    first = next(action for action in actions if action.generator_views == (0, 1))
    reverse = actions[first.reverse_action_id]
    assert first.verifier_views == (2, 3, 4)
    assert reverse.generator_views == (2, 3, 4)
    assert reverse.verifier_views == (0, 1)
    assert reverse.action_id >= 10


def test_v6_even_complements_are_naturally_enumerated_once() -> None:
    actions = protocol.build_balanced_directed_actions(6)
    assert len(actions) == 20
    assert {action.generator_size for action in actions} == {3}
    assert {action.verifier_size for action in actions} == {3}


@pytest.mark.parametrize("view_num", range(2, 9))
def test_partition_reverse_uniqueness_and_involution(view_num: int) -> None:
    actions = protocol.build_balanced_directed_actions(view_num)
    all_views = set(range(view_num))
    assert len({(item.generator_views, item.verifier_views) for item in actions}) == len(actions)
    for action in actions:
        generator, verifier = set(action.generator_views), set(action.verifier_views)
        reverse = actions[action.reverse_action_id]
        assert generator and verifier
        assert not (generator & verifier)
        assert generator | verifier == all_views
        assert action.generator_views == reverse.verifier_views
        assert action.verifier_views == reverse.generator_views
        assert reverse.reverse_action_id == action.action_id


@pytest.mark.parametrize("view_num", range(2, 9))
def test_serialization_and_sha_are_deterministic(view_num: int) -> None:
    assert protocol.canonical_action_payload(view_num) == protocol.canonical_action_payload(view_num)
    assert protocol.canonical_action_json(view_num) == protocol.canonical_action_json(view_num)
    assert protocol.canonical_action_sha256(view_num) == protocol.canonical_action_sha256(view_num)
    assert len(protocol.canonical_action_sha256(view_num)) == 64


def test_formal_dataset_expectations_have_no_dataset_specific_action_exception() -> None:
    for expectation in protocol.FORMAL_DATASET_ACTION_EXPECTATIONS.values():
        view_num = expectation["view_num"]
        actions = protocol.build_balanced_directed_actions(view_num)
        assert tuple(protocol.generator_sizes(view_num)) == expectation["generator_sizes"]
        assert len(actions) == expectation["action_count"]


def test_protocol_has_no_randomness_gt_or_utility_threshold() -> None:
    source = Path(protocol.__file__).read_text(encoding="utf-8").lower()
    # Warning/contract prose may name historical U/T and memory, but the action
    # builder must not access labels/GT, a loader, randomness, or a threshold.
    for forbidden in ("random", "get_dataset", "threshold", "pseudo-label", "labels"):
        assert forbidden not in source


def test_p2_carrier_is_untouched_by_action_construction() -> None:
    before = _sha256(P2_CARRIER)
    protocol.build_balanced_directed_actions(2)
    protocol.build_balanced_directed_actions(8)
    assert before == ORIGINAL_P2_CARRIER_SHA256
    assert _sha256(P2_CARRIER) == before


def test_native_bone_hashes_are_unchanged() -> None:
    assert _sha256(REPO_ROOT / "method/bone.py") == ORIGINAL_BONE_SHA256
    assert _sha256(REPO_ROOT / "method/amv.py") == ORIGINAL_AMV_SHA256


def test_warnings_epoch_metadata_and_static_audit_payload() -> None:
    assert protocol.YOUTUBE_MEMORY_WARNING is True
    assert "Caltech-5V" in protocol.ACTION_SPACE_PORTABILITY_WARNINGS[0]
    assert protocol.NATIVE_EPOCHS == 100
    assert "20" in protocol.PROPOSED_EPOCH_PROTOCOL["matched_base"]
    payload = audit.build_audit_payload()
    assert payload["MSRC_ACTION_ORDER_PARITY"] is True
    assert payload["YOUTUBE_MEMORY_WARNING"] is True
    assert payload["formal_dataset_expectations"]["HandWrite"]["action_count"] == 20
