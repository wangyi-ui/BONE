"""Write a deterministic, no-data P3-A1 action-space audit payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.utility_relation import bone_p3_a1_action_space_protocol as protocol


DEFAULT_OUTPUT = Path("outputs/bone_p3/p3_a1_action_space_audit.json")


def build_audit_payload() -> dict[str, object]:
    """Build metadata from deterministic combinatorics and the existing carrier order only."""
    formal = {}
    for name, expectation in protocol.FORMAL_DATASET_ACTION_EXPECTATIONS.items():
        view_num = expectation["view_num"]
        formal[name] = {
            "view_num": view_num,
            "generator_sizes": list(protocol.generator_sizes(view_num)),
            "action_count": protocol.expected_action_count(view_num),
            "canonical_sha256": protocol.canonical_action_sha256(view_num),
        }
    return {
        "scope": "static action-space audit only; no training, GPU, GT, or dataset loading",
        "action_space_mathematics_independent_of_execution_strategy": True,
        "chunking_contract": "Future query chunking may not change action order, U_cycle, T_rel, or action coverage, and may not sample actions.",
        "canonical_by_view_count": {
            str(view_num): {
                "generator_sizes": list(protocol.generator_sizes(view_num)),
                "action_count": protocol.expected_action_count(view_num),
                "canonical_sha256": protocol.canonical_action_sha256(view_num),
            }
            for view_num in range(2, 9)
        },
        "formal_dataset_expectations": formal,
        "MSRC_ACTION_ORDER_PARITY": protocol.verify_msrc_action_order(),
        "ACTION_SPACE_PORTABILITY_WARNINGS": list(protocol.ACTION_SPACE_PORTABILITY_WARNINGS),
        "YOUTUBE_MEMORY_WARNING": protocol.YOUTUBE_MEMORY_WARNING,
        "native_epochs": protocol.NATIVE_EPOCHS,
        "proposed_epoch_protocol_metadata_only": protocol.PROPOSED_EPOCH_PROTOCOL,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    output = build_parser().parse_args().output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_audit_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
