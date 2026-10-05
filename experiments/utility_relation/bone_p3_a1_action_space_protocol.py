"""Deterministic, dataset-agnostic balanced directed action-space protocol.

This module only defines partitions of view indices.  It has no training,
dataset loading, ground-truth access, utility computation, or relation loss.
The mathematics is independent of execution strategy: a future implementation
may chunk a query dimension, but must preserve this action order, every action,
and the historical U_cycle and T_rel definitions unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable


MSRC_CARRIER_PATH = Path("outputs/bone_p2/p2_a0_msrc_seed20/p2_a0_carrier_pre_gt.npz")
MSRC_GENERATOR_SUBSETS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
MSRC_VERIFIER_SUBSETS = ((2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1))

YOUTUBE_MEMORY_WARNING = True
ACTION_SPACE_PORTABILITY_WARNINGS = (
    "Caltech-5V: dataset_info view dimensions disagree with the current X1..X5 loader order; "
    "complete a loader-order audit before a Caltech engineering smoke run.",
    "YouTubeFace50: U/p_gen/p_ver/T_rel estimate is about 607.84 decimal MB in float32; "
    "a future carrier must use query chunking or streaming without action sampling.",
)

NATIVE_EPOCHS = 100
PROPOSED_EPOCH_PROTOCOL = {
    "official_bone": "100 epochs",
    "matched_base": "100 native epochs + 20 native continuation epochs",
    "ours": "100 native carrier epochs + 20 alternating refinement epochs",
}

# Metadata only: the builder never branches on a dataset name.
FORMAL_DATASET_ACTION_EXPECTATIONS = {
    "MNIST-USPS": {"view_num": 2, "generator_sizes": (1,), "action_count": 2},
    "BDGP": {"view_num": 2, "generator_sizes": (1,), "action_count": 2},
    "CCV": {"view_num": 3, "generator_sizes": (1, 2), "action_count": 6},
    "COIL20": {"view_num": 3, "generator_sizes": (1, 2), "action_count": 6},
    "MSRC": {"view_num": 4, "generator_sizes": (2,), "action_count": 6},
    "ALOI": {"view_num": 4, "generator_sizes": (2,), "action_count": 6},
    "YouTubeFace50": {"view_num": 4, "generator_sizes": (2,), "action_count": 6},
    "Caltech-5V": {"view_num": 5, "generator_sizes": (2, 3), "action_count": 20},
    "HandWrite": {"view_num": 6, "generator_sizes": (3,), "action_count": 20},
}


@dataclass(frozen=True)
class BalancedDirectedAction:
    """One ordered generator-to-verifier complementary view partition."""

    action_id: int
    generator_views: tuple[int, ...]
    verifier_views: tuple[int, ...]
    generator_size: int
    verifier_size: int
    reverse_action_id: int


def generator_sizes(view_num: int) -> tuple[int, ...]:
    """Return the balanced generator cardinalities for ``view_num >= 2``."""
    if isinstance(view_num, bool) or not isinstance(view_num, int) or view_num < 2:
        raise ValueError("view_num must be an integer greater than or equal to 2")
    lower = view_num // 2
    return (lower,) if view_num % 2 == 0 else (lower, lower + 1)


def expected_action_count(view_num: int) -> int:
    """Return the specified count of directed balanced complementary actions."""
    sizes = generator_sizes(view_num)
    return sum(len(tuple(itertools.combinations(range(view_num), size))) for size in sizes)


def build_balanced_directed_actions(view_num: int) -> tuple[BalancedDirectedAction, ...]:
    """Build canonical actions sorted by size and lexicographic generator tuple."""
    sizes = generator_sizes(view_num)
    all_views = tuple(range(view_num))
    pairs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for size in sizes:
        for generator in itertools.combinations(all_views, size):
            generator_set = set(generator)
            verifier = tuple(view for view in all_views if view not in generator_set)
            pairs.append((generator, verifier))

    reverse_index = {pair: action_id for action_id, pair in enumerate(pairs)}
    if len(reverse_index) != len(pairs):
        raise RuntimeError("balanced directed action enumeration is not unique")
    actions = tuple(
        BalancedDirectedAction(
            action_id=action_id,
            generator_views=generator,
            verifier_views=verifier,
            generator_size=len(generator),
            verifier_size=len(verifier),
            reverse_action_id=reverse_index[(verifier, generator)],
        )
        for action_id, (generator, verifier) in enumerate(pairs)
    )
    validate_balanced_directed_actions(view_num, actions)
    return actions


def validate_balanced_directed_actions(
    view_num: int, actions: Iterable[BalancedDirectedAction]
) -> None:
    """Fail closed unless every ordering, partition, and reverse invariant holds."""
    expected_sizes = generator_sizes(view_num)
    actions = tuple(actions)
    if len(actions) != expected_action_count(view_num):
        raise ValueError("unexpected balanced directed action count")
    all_views = tuple(range(view_num))
    action_pairs = [(action.generator_views, action.verifier_views) for action in actions]
    if len(set(action_pairs)) != len(actions):
        raise ValueError("balanced directed actions must be unique")
    for expected_id, action in enumerate(actions):
        if action.action_id != expected_id:
            raise ValueError("action IDs must be contiguous and canonical")
        if action.generator_size not in expected_sizes or action.generator_size != len(action.generator_views):
            raise ValueError("invalid generator size")
        if action.verifier_size != len(action.verifier_views):
            raise ValueError("invalid verifier size")
        generator, verifier = set(action.generator_views), set(action.verifier_views)
        if not generator or not verifier or generator & verifier or generator | verifier != set(all_views):
            raise ValueError("generator and verifier must be nonempty strict complements")
        if tuple(sorted(generator)) != action.generator_views or tuple(sorted(verifier)) != action.verifier_views:
            raise ValueError("view tuples must be sorted")
        if not 0 <= action.reverse_action_id < len(actions):
            raise ValueError("invalid reverse action ID")
        reverse = actions[action.reverse_action_id]
        if reverse.generator_views != action.verifier_views or reverse.verifier_views != action.generator_views:
            raise ValueError("reverse action is not the directed complement")
        if reverse.reverse_action_id != action.action_id:
            raise ValueError("reverse action must be involutive")

    expected_pairs = []
    for size in expected_sizes:
        for generator in itertools.combinations(all_views, size):
            generator_set = set(generator)
            expected_pairs.append((generator, tuple(view for view in all_views if view not in generator_set)))
    if action_pairs != expected_pairs:
        raise ValueError("actions do not satisfy canonical size/lexicographic ordering")


def canonical_action_payload(view_num: int) -> dict[str, object]:
    """Return a deterministic JSON-serializable representation of the action space."""
    actions = build_balanced_directed_actions(view_num)
    return {
        "view_num": view_num,
        "generator_sizes": list(generator_sizes(view_num)),
        "action_count": len(actions),
        "actions": [
            {
                **asdict(action),
                "generator_views": list(action.generator_views),
                "verifier_views": list(action.verifier_views),
            }
            for action in actions
        ],
    }


def canonical_action_json(view_num: int) -> str:
    """Serialize the canonical payload without whitespace-dependent variation."""
    return json.dumps(canonical_action_payload(view_num), sort_keys=True, separators=(",", ":"))


def canonical_action_sha256(view_num: int) -> str:
    """Hash the exact deterministic JSON representation for one view count."""
    return hashlib.sha256(canonical_action_json(view_num).encode("utf-8")).hexdigest()


def verify_msrc_action_order(carrier_path: Path | str = MSRC_CARRIER_PATH) -> bool:
    """Require exact V=4 parity with frozen P2-A0 tuples and carrier metadata."""
    actions = build_balanced_directed_actions(4)
    generator = tuple(action.generator_views for action in actions)
    verifier = tuple(action.verifier_views for action in actions)
    if generator != MSRC_GENERATOR_SUBSETS or verifier != MSRC_VERIFIER_SUBSETS:
        raise ValueError("generic V=4 action order is not the frozen MSRC order")

    import numpy as np

    path = Path(carrier_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        try:
            carrier_generator = tuple(tuple(map(int, row)) for row in archive["generator_subsets"])
            carrier_verifier = tuple(tuple(map(int, row)) for row in archive["verifier_subsets"])
        except KeyError as error:
            raise ValueError("P2-A0 carrier does not contain action ordering metadata") from error
    if carrier_generator != generator or carrier_verifier != verifier:
        raise ValueError("P2-A0 carrier action order does not exactly match generic V=4 order")
    return True
