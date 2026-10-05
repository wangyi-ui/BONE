import hashlib
from pathlib import Path

from . import protocol
from .formal_base_reference import BaseReference
from .run_formal_optuna_tuning_r1 import R1_STUDY_DB_NAME, run_formal_optuna_tuning_r1
from .study_schema import CandidateEvaluation


def test_old_failed_study_is_byte_identical_and_r1_study_starts_clean(tmp_path):
    old = tmp_path / "handwrite" / "study.db"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"sqlite historical FAIL artifact")
    before_hash = hashlib.sha256(old.read_bytes()).hexdigest()
    references = {
        seed: BaseReference("HandWrite", seed, Path("base.json"), f"s{seed}", f"p{seed}", 20,
                            0.0, 0.0, {}, tuple("h" for _ in range(20)), "fp")
        for seed in protocol.DEV_SEEDS
    }

    def bootstrap(context, **kwargs):
        return references

    def evaluator(candidate):
        return CandidateEvaluation("HandWrite", candidate.semantic_lr, candidate.semantic_exposure, 0.0, 0.0, 0.0, 0.0)

    study = run_formal_optuna_tuning_r1("HandWrite", "cpu", tmp_path, max_new_trials=1,
                                        evaluator=evaluator, context=object(), bootstrap=bootstrap)
    assert hashlib.sha256(old.read_bytes()).hexdigest() == before_hash
    assert study.study_name.endswith("__main_r1")
    assert len(study.trials) == 1
    assert study.trials[0].state.name == "COMPLETE"
    assert (tmp_path / "handwrite" / R1_STUDY_DB_NAME).is_file()
