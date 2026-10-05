import pytest

from pathlib import Path

from experiments.utility_relation.p3_c1_b1_optuna.materialize_formal_dev_carrier import (
    REPO_ROOT, build_parser, validate_output_root,
)
from experiments.utility_relation.p3_c1_b1_optuna.formal_carrier_materializer import FORMAL_OUTPUT_SUBDIR


@pytest.mark.parametrize("dataset,seed", [
    ("HandWrite", "11"), ("MSRC", "37"), ("Caltech-5V", "11"),
])
def test_valid_single_identity_cli(dataset, seed):
    args = build_parser().parse_args([
        "--dataset", dataset, "--dev-seed", seed, "--device", "cuda:0",
        "--output-root", "/tmp/formal-test",
    ])
    assert args.dataset == dataset and args.dev_seed == int(seed)
    assert not isinstance(args.dataset, list) and not isinstance(args.dev_seed, list)


@pytest.mark.parametrize("arguments", [
    ["--dataset", "unknown", "--dev-seed", "11"],
    ["--dataset", "MSRC", "--dev-seed", "20"],
    ["--dataset", "MSRC", "--dev-seed", "30"],
    ["--dataset", "MSRC", "--dev-seed", "50"],
    ["--dataset", "MSRC", "--dev-seed", "123"],
])
def test_invalid_dataset_or_seed_cli(arguments):
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments + ["--device", "cuda:0", "--output-root", "/tmp/formal-test"])


def test_cli_has_no_batch_or_loop_argument():
    destinations = {action.dest for action in build_parser()._actions}
    assert "datasets" not in destinations and "dev_seeds" not in destinations
    assert {"dataset", "dev_seed", "device", "output_root", "replace_incomplete"} <= destinations


def test_cli_output_root_is_locked_to_formal_namespace(tmp_path):
    expected = REPO_ROOT / FORMAL_OUTPUT_SUBDIR
    assert validate_output_root(expected) == expected
    with pytest.raises(ValueError, match="OUTPUT_ROOT_MISMATCH"):
        validate_output_root(tmp_path / "wrong")
