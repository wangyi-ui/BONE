"""R1 CLI for one formal dev carrier; audit inventory uses ``--output``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.utility_relation.p3_c1_b1_optuna import protocol
from experiments.utility_relation.p3_c1_b1_optuna.formal_carrier_materializer_r1 import materialize_one_r1
from experiments.utility_relation.p3_c1_b1_optuna.materialize_formal_dev_carrier import validate_output_root


AUDIT_CLI_OUTPUT_OPTION = "--output"


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
    result = materialize_one_r1(
        args.dataset, args.dev_seed, args.device, args.output_root, args.replace_incomplete
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
