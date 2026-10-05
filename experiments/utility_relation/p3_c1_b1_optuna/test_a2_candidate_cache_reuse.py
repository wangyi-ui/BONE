import hashlib
import json
from pathlib import Path

from . import formal_candidate_evaluator_r2, protocol
from .formal_base_reference import BaseReference
from .formal_tuning_context import FormalTuningContext
from .optuna_driver import expected_main_candidates
from .parent_integrity import sha256_file
from .run_formal_optuna_tuning import audit_study_coverage
from .run_formal_optuna_tuning_a2 import run_formal_optuna_tuning_a2
from .study_schema import CandidateConfig, candidate_key


def _context(tmp_path):
    return FormalTuningContext("BDGP", tmp_path, {}, {11: {}, 37: {}}, "fp")


def _references():
    return {
        seed: BaseReference(
            "BDGP", seed, Path(f"base{seed}.json"), f"s{seed}", f"p{seed}",
            20, 0.0, 0.0, {}, tuple("h" for _ in range(20)), "fp",
        )
        for seed in protocol.DEV_SEEDS
    }


def _write_cache(context, candidate):
    path = formal_candidate_evaluator_r2.candidate_directory(
        context.formal_tuning_root, context.dataset, candidate
    )
    path.mkdir(parents=True)
    payload = {
        "dataset": context.dataset,
        "semantic_lr": candidate.semantic_lr,
        "semantic_exposure": candidate.semantic_exposure,
        "delta_j_seed11_foldA": 0.1,
        "delta_j_seed11_foldB": 0.2,
        "delta_j_seed37_foldA": 0.3,
        "delta_j_seed37_foldB": 0.4,
        "protocol_fingerprint": context.protocol_fingerprint,
        "full_gt_used": False,
        "candidate_key": candidate_key(candidate),
    }
    evaluation = path / "candidate_evaluation.json"
    audit = path / "audit.json"
    evaluation.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    audit.write_text(json.dumps({"full_gt_used": False}), encoding="utf-8")
    (path / "COMPLETE.json").write_text(json.dumps({
        "status": "COMPLETE",
        "evaluation_sha256": sha256_file(evaluation),
        "audit_sha256": sha256_file(audit),
    }, sort_keys=True), encoding="utf-8")


def _tree_hashes(root):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def test_existing_25_candidate_cache_populates_a2_study_without_branch_training(tmp_path):
    context = _context(tmp_path)
    for registered in expected_main_candidates():
        _write_cache(
            context,
            CandidateConfig(registered.semantic_lr, registered.semantic_exposure),
        )
    root = tmp_path / "bdgp" / "candidates"
    before = _tree_hashes(root)
    branch_calls = []

    def evaluator(candidate):
        return formal_candidate_evaluator_r2.evaluate_candidate(
            context, candidate, device="cpu",
            runner=lambda *args: branch_calls.append(args),
        )

    study = run_formal_optuna_tuning_a2(
        "BDGP", "cpu", tmp_path, evaluator=evaluator, context=context,
        bootstrap=lambda context, **kwargs: _references(),
    )
    assert audit_study_coverage(study)["main_grid_complete"] is True
    assert branch_calls == []
    assert _tree_hashes(root) == before

