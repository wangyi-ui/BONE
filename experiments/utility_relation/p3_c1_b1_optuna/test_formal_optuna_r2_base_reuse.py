import hashlib
from types import SimpleNamespace

from . import formal_base_reference
from . import formal_tuning_bootstrap_r1
from . import protocol
from .formal_tuning_context import CarrierContext, FormalTuningContext
from .run_formal_optuna_tuning_r2 import run_formal_optuna_tuning_r2


def _context(tmp_path):
    carriers = {
        seed: CarrierContext(
            dataset="BDGP",
            dev_seed=seed,
            directory=tmp_path,
            state_path=tmp_path / f"state{seed}",
            pre_gt_path=tmp_path / f"pre{seed}",
            audit_path=tmp_path / f"audit{seed}",
            arrays={},
            views=tuple(),
            adapter=object(),
            spec=SimpleNamespace(batch_size=1),
            k=2,
            folds=tuple(),
            state_file_sha256=f"state-{seed}",
            pre_gt_sha256=f"pre-{seed}",
        )
        for seed in protocol.DEV_SEEDS
    }
    return FormalTuningContext(
        "BDGP", tmp_path, carriers, {11: {}, 37: {}}, "fp"
    )


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_existing_base_references_are_verified_reused_and_byte_identical(
    monkeypatch, tmp_path
):
    context = _context(tmp_path)
    training_calls = []

    def initial_runner(carrier, device):
        training_calls.append(carrier.dev_seed)
        return {
            "model": object(),
            "native_order_hashes": [
                f"seed{carrier.dev_seed}-epoch{epoch}"
                for epoch in range(protocol.REFINEMENT_HORIZON)
            ],
        }

    monkeypatch.setattr(
        formal_base_reference,
        "_evaluate_base",
        lambda carrier, model, device: (
            {"A": 1.0, "B": 1.0},
            {
                "A": {"supported_anchor_count": 1, "abstained_anchor_count": 0,
                      "coverage": 1.0},
                "B": {"supported_anchor_count": 1, "abstained_anchor_count": 0,
                      "coverage": 1.0},
            },
        ),
    )
    for seed in protocol.DEV_SEEDS:
        formal_base_reference.build_base_reference(
            context, seed, device="cpu", runner=initial_runner
        )
    assert training_calls == [11, 37]

    protected_paths = [
        path
        for seed in protocol.DEV_SEEDS
        for path in (
            formal_base_reference.base_reference_path(tmp_path, "BDGP", seed),
            formal_base_reference.base_reference_directory(
                tmp_path, "BDGP", seed
            ) / "COMPLETE.json",
        )
    ]
    before = {path: _digest(path) for path in protected_paths}

    def forbidden_training(*args, **kwargs):
        raise AssertionError("verified BASE cache must not retrain")

    def cached_builder(ctx, seed, **kwargs):
        return formal_base_reference.build_base_reference(
            ctx,
            seed,
            device=kwargs["device"],
            replace_incomplete=kwargs["replace_incomplete"],
            runner=forbidden_training,
        )

    def bootstrap(ctx, **kwargs):
        return formal_tuning_bootstrap_r1.bootstrap_base_references(
            ctx,
            device=kwargs["device"],
            builder=cached_builder,
            verifier=formal_base_reference.load_verified_base_reference,
        )

    study = run_formal_optuna_tuning_r2(
        "BDGP",
        "cpu",
        tmp_path,
        max_new_trials=0,
        context=context,
        bootstrap=bootstrap,
    )
    assert study.trials == []
    assert all(_digest(path) == digest for path, digest in before.items())
    assert training_calls == [11, 37]


def test_r2_sources_are_gt_free_and_keep_frozen_protocol_constants():
    root = __import__("pathlib").Path(__file__).parent
    text = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in (
            "formal_candidate_evaluator_r2.py",
            "run_formal_optuna_tuning_r2.py",
        )
    )
    assert not any(
        token in text
        for token in (
            "ACC", "NMI", "Purity", "ARI", "FORMAL_SEEDS",
            "seed20", "seed30", "seed50",
        )
    )
    assert "GridSampler" not in text
    assert protocol.SEMANTIC_LR_GRID == (3e-6, 1e-5, 3e-5, 1e-4, 3e-4)
    assert protocol.SEMANTIC_EXPOSURE_GRID == (1, 3, 5, 10, 20)
    assert protocol.DEV_SEEDS == (11, 37)
    assert protocol.REFINEMENT_HORIZON == 20

