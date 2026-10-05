import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from . import formal_candidate_evaluator_r2
from . import protocol
from .formal_base_reference import BaseReference
from .formal_tuning_context import CarrierContext, FormalTuningContext
from .formal_tuning_context_r3 import canonicalize_formal_tuning_context_r3
from .optuna_driver import expected_main_candidates
from .parent_integrity import sha256_file
from .run_formal_optuna_tuning import audit_study_coverage
from .run_formal_optuna_tuning_r3 import run_formal_optuna_tuning_r3
from .sparse_cv import SparseFold
from .study_schema import CandidateConfig, candidate_key


def _context(tmp_path):
    folds = (
        SparseFold("A", (0, 1), (0, 1), (2, 3), (0, 1)),
        SparseFold("B", (2, 3), (0, 1), (0, 1), (0, 1)),
    )
    carriers = {
        seed: CarrierContext(
            dataset="BDGP", dev_seed=seed, directory=tmp_path,
            state_path=tmp_path / f"state{seed}", pre_gt_path=tmp_path / f"pre{seed}",
            audit_path=tmp_path / f"audit{seed}", arrays={}, views=tuple(),
            adapter=object(), spec=SimpleNamespace(batch_size=1), k=2, folds=folds,
            state_file_sha256=f"s{seed}", pre_gt_sha256=f"p{seed}",
        )
        for seed in protocol.DEV_SEEDS
    }
    frozen = FormalTuningContext("BDGP", tmp_path, carriers, {11: {}, 37: {}}, "fp")
    return canonicalize_formal_tuning_context_r3(frozen)[0]


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
    (path / "COMPLETE.json").write_text(
        json.dumps({
            "status": "COMPLETE",
            "evaluation_sha256": sha256_file(evaluation),
            "audit_sha256": sha256_file(audit),
        }, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _tree_hashes(root):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def test_all_25_existing_candidate_caches_build_clean_r3_study_without_training(tmp_path):
    context = _context(tmp_path)
    for registered in expected_main_candidates():
        _write_cache(
            context,
            CandidateConfig(registered.semantic_lr, registered.semantic_exposure),
        )
    candidate_root = tmp_path / "bdgp" / "candidates"
    before = _tree_hashes(candidate_root)
    branch_calls = []

    def evaluator(candidate):
        return formal_candidate_evaluator_r2.evaluate_candidate(
            context, candidate, device="cpu",
            runner=lambda *args: branch_calls.append(args),
        )

    study = run_formal_optuna_tuning_r3(
        "BDGP", "cpu", tmp_path, evaluator=evaluator, context=context,
        bootstrap=lambda context, **kwargs: _references(),
    )
    assert audit_study_coverage(study)["main_grid_complete"] is True
    assert branch_calls == []
    assert _tree_hashes(candidate_root) == before


def test_single_r2_candidate_cache_is_identical_before_and_after_r3_reuse(tmp_path):
    context = _context(tmp_path)
    candidate = CandidateConfig(3e-6, 1)
    path = _write_cache(context, candidate)
    before = _tree_hashes(path)
    result = formal_candidate_evaluator_r2.evaluate_candidate(
        context, candidate, device="cpu",
        runner=lambda *args: (_ for _ in ()).throw(AssertionError("must not train")),
    )
    assert result.mean_delta_j == 0.25
    assert _tree_hashes(path) == before

