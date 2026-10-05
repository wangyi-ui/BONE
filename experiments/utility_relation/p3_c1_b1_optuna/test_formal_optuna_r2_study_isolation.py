import hashlib

from . import protocol
from .run_formal_optuna_tuning_r2 import (
    R2_STUDY_DB_NAME,
    create_or_load_study_r2,
    study_name_for_dataset_r2,
    study_r2_path,
)


def test_all_datasets_use_uniform_r2_paths_and_names(tmp_path):
    paths = set()
    names = set()
    for dataset in protocol.DATASETS:
        path = study_r2_path(tmp_path, dataset)
        name = study_name_for_dataset_r2(dataset)
        assert path.name == R2_STUDY_DB_NAME
        assert name.startswith("p3_c1_b1_a1__")
        assert name.endswith("__main_r2")
        paths.add(path)
        names.add(name)
    assert len(paths) == len(names) == len(protocol.DATASETS) == 8


def test_r2_study_starts_with_zero_historical_trials_and_old_dbs_unchanged(tmp_path):
    directory = tmp_path / "handwrite"
    directory.mkdir(parents=True)
    old = directory / "study.db"
    r1 = directory / "study_r1.db"
    old.write_bytes(b"old failed trial")
    r1.write_bytes(b"r1 failed pre-refinement trial")
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (old, r1)
    }
    r2 = directory / R2_STUDY_DB_NAME
    study = create_or_load_study_r2("HandWrite", r2)
    assert study.study_name == "p3_c1_b1_a1__handwrite__main_r2"
    assert study.trials == []
    assert study.user_attrs["namespace_version"] == "R2"
    assert study.user_attrs["full_gt_used"] is False
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for path, digest in before.items()
    )

