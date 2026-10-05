"""CLI to materialize one B1-A1 formal development carrier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.utility_relation.p3_c1_b1_optuna import protocol
from experiments.utility_relation.p3_c1_b1_optuna.formal_carrier_materializer import (
    FORMAL_OUTPUT_SUBDIR, materialize_one,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def validate_output_root(output_root: Path) -> Path:
    expected = (REPO_ROOT / FORMAL_OUTPUT_SUBDIR).resolve()
    if output_root.resolve() != expected:
        raise ValueError("FORMAL_CARRIER_OUTPUT_ROOT_MISMATCH")
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=protocol.DATASETS)
    parser.add_argument("--dev-seed", required=True, type=int, choices=protocol.DEV_SEEDS)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--replace-incomplete", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_output_root(args.output_root)
    result = materialize_one(
        args.dataset, args.dev_seed, args.device, args.output_root, args.replace_incomplete
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
