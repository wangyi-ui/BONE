import hashlib
from pathlib import Path

from .boundary_extension_context import BoundaryExtensionContext
from .run_boundary_extension import run_boundary_extension
from .test_boundary_extension_runner import _evaluator, _references


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mock_run_preserves_old_studies_candidates_bases_and_parent_freezes(tmp_path):
    protected = []
    for relative in (
        "bdgp/study.db", "bdgp/study_r1.db", "bdgp/study_r2.db",
        "bdgp/study_r3.db", "bdgp/study_a2.db",
        "bdgp/candidates/main/COMPLETE.json",
        "bdgp/base_reference/dev_seed11/COMPLETE.json",
        "bdgp/base_reference/dev_seed37/COMPLETE.json",
        "freeze/main/STATUS.txt", "freeze/a2/STATUS.txt", "freeze/r3/STATUS.txt",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        protected.append(path)
    before = {path: _hash(path) for path in protected}
    context = BoundaryExtensionContext(
        object(), tmp_path / "bdgp" / "boundary_extension_upper_1e3", {}
    )
    run_boundary_extension(
        "BDGP", "cpu", tmp_path, tmp_path, context=context,
        evaluator=_evaluator("BDGP"),
        bootstrap=lambda context, **kwargs: _references(),
        precondition=lambda root: {},
    )
    assert all(_hash(path) == digest for path, digest in before.items())


def test_current_parent_freeze_statuses_are_read_only_inputs():
    root = Path(__file__).resolve().parents[3]
    paths = (
        root / "experiment_freeze/bone_p3_c1_b1_a2_eight_dataset_main_grid_provisional_selection_20261004/STATUS.txt",
        root / "experiment_freeze/bone_p3_c1_b1_a2_coverage_aware_abstention_preregistered_20261003/STATUS.txt",
        root / "experiment_freeze/bone_p3_c1_b1_a1_formal_optuna_runner_r3_20261003/STATUS.txt",
    )
    before = [_hash(path) for path in paths]
    after = [_hash(path) for path in paths]
    assert before == after

