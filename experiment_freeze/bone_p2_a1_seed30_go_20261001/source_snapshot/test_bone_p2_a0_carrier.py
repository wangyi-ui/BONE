"""Unit tests for P2-A0 carrier mechanics; no 100-epoch replay is started."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.utility_relation import bone_p2_a0_protocol as protocol
from experiments.utility_relation import evaluate_bone_p2_a0_carrier as evaluator
from experiments.utility_relation import evaluate_bone_p2_a0_carrier_v2 as evaluator_v2
from experiments.utility_relation import materialize_bone_p2_a0_carrier as materializer


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_BONE_SHA256 = "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092"
ORIGINAL_AMV_SHA256 = "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc"


def _posterior_tensor() -> np.ndarray:
    raw = np.arange(protocol.N * protocol.V * protocol.K, dtype=np.float64).reshape(
        protocol.N, protocol.V, protocol.K
    ) + 1.0
    return raw / raw.sum(axis=2, keepdims=True)


def test_action_enumeration_and_strict_complements() -> None:
    generator, verifier = protocol.action_subsets()
    assert np.array_equal(generator, np.asarray(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))))
    assert np.array_equal(verifier, np.asarray(((2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1))))
    protocol.validate_action_subsets(generator, verifier)
    for left, right in zip(generator, verifier):
        assert not (set(left) & set(right))
        assert set(left) | set(right) == {0, 1, 2, 3}


def test_q_hook_shape_row_sums_and_fused_label_parity() -> None:
    q_by_view = _posterior_tensor()

    class HookModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.label_fc = torch.nn.Identity()

        def forward(self, views, test: bool = False):
            assert test is True
            posteriors = [self.label_fc(view) for view in views]
            return {"label": torch.argmax(torch.stack(posteriors).mean(dim=0), dim=1)}

    model = HookModel()
    views = [q_by_view[:, view, :].astype(np.float32) for view in range(protocol.V)]
    q_views, fused = materializer.extract_q_views(model, views)
    assert q_views.shape == (protocol.N, protocol.V, protocol.K)
    assert np.allclose(q_views.sum(axis=2), 1.0)
    assert np.array_equal(np.argmax(q_views.mean(axis=1), axis=1), fused)


def test_u_cycle_toy_formula_exactness() -> None:
    p_gen = np.zeros((1, protocol.S, protocol.K), dtype=np.float64)
    p_ver = np.zeros_like(p_gen)
    p_gen[0, 0, :2], p_ver[0, 0, :2] = (0.8, 0.2), (0.7, 0.3)
    p_gen[0, 1, :2], p_ver[0, 1, :2] = (0.2, 0.8), (0.4, 0.6)
    p_gen[0, 2:, 0], p_ver[0, 2:, 0] = 1.0, 1.0
    labels, confidence, support, closure, utility = materializer.compute_u_cycle(p_gen, p_ver)
    assert np.array_equal(labels[0, :2], [0, 1])
    assert np.allclose(confidence[0, :2], [0.8, 0.8])
    assert np.allclose(support[0, :2], [0.7, 0.6])
    assert np.array_equal(closure[0, :2], [1, 1])
    assert np.allclose(utility[0, :2], [np.sqrt(0.8 * 0.7), np.sqrt(0.8 * 0.6)])


def test_u_shape_range_finite_and_nonzero_mass_per_action() -> None:
    q_views = _posterior_tensor()
    p_gen, p_ver = materializer.compute_action_posteriors(q_views)
    _labels, _confidence, _support, _closure, utility = materializer.compute_u_cycle(p_gen, p_ver)
    assert utility.shape == (protocol.N, protocol.S)
    assert np.isfinite(utility).all()
    assert np.all((0 <= utility) & (utility <= 1))
    assert np.all(utility.sum(axis=0) > 0)


def test_sparse_split_contract_and_unlabeled_canonical_complement() -> None:
    protocol.validate_sparse_targets(protocol.EXPECTED_TARGETS)
    with pytest.raises(ValueError, match="actual targets"):
        protocol.validate_sparse_targets(np.roll(protocol.EXPECTED_TARGETS, 1))
    unlabeled = protocol.unlabeled_complement()
    assert unlabeled.shape == (protocol.NU,)
    assert not (set(unlabeled) & set(protocol.LABELED_IDS))
    assert np.array_equal(np.sort(np.r_[unlabeled, protocol.LABELED_IDS]), np.arange(protocol.N))


def test_anchor_signature_is_not_utility_weighted_and_pi_is_normalized() -> None:
    q_views = _posterior_tensor()
    p_gen, _p_ver = materializer.compute_action_posteriors(q_views)
    unlabeled, anchor_signature, class_signature, pi, t_rel = materializer.build_semantic_relation(
        p_gen, protocol.LABELED_IDS, protocol.EXPECTED_TARGETS
    )
    assert np.allclose(anchor_signature, p_gen[protocol.LABELED_IDS].mean(axis=1))
    assert class_signature.shape == (protocol.K, protocol.K)
    assert pi.shape == (protocol.NU, protocol.S, protocol.K)
    assert np.allclose(pi.sum(axis=2), 1.0)
    assert t_rel.shape == (protocol.NU, protocol.L, protocol.S)
    assert np.isfinite(t_rel).all()
    assert np.all((0 <= pi) & (pi <= 1))
    assert np.all((0 <= t_rel) & (t_rel <= 1))
    assert np.array_equal(unlabeled, protocol.unlabeled_complement())


def test_preseal_source_has_no_forbidden_mechanisms_or_full_gt_semantics() -> None:
    source = Path(materializer.__file__).read_text(encoding="utf-8")
    lower = source.lower()
    for forbidden in ("epsilon", "temperature", "u_tilde", "gating", "pseudo-label", "memory", "mnn"):
        assert forbidden not in lower
    assert "standard_data(" not in source
    assert ".shuffle(" not in source
    assert source.count('archive["labels"]') == 1
    assert "get_cluster_metrics" not in source


def test_evaluator_refuses_missing_or_invalid_seal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing carrier seal"):
        evaluator.verify_seal(tmp_path)
    (tmp_path / "p2_a0_carrier_seal.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid carrier seal"):
        evaluator.verify_seal(tmp_path)


def test_parent_artifact_and_native_sources_are_frozen() -> None:
    assert protocol.PARENT_ARTIFACT == Path(
        "outputs/bone_p1/msrc_snr2p5_exact2of4_seed20/bone_p1_msrc_corruption.npz"
    )
    assert "--artifact" not in materializer.build_parser().format_help()
    assert materializer.sha256_file(REPO_ROOT / "method/bone.py") == ORIGINAL_BONE_SHA256
    assert materializer.sha256_file(REPO_ROOT / "method/amv.py") == ORIGINAL_AMV_SHA256



def _query_posteriors() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    p_gen = rng.random((protocol.NU, protocol.S, protocol.K))
    p_ver = rng.random((protocol.NU, protocol.S, protocol.K))
    return p_gen / p_gen.sum(axis=2, keepdims=True), p_ver / p_ver.sum(axis=2, keepdims=True)


def test_v2_derangement_is_deterministic_complete_and_fixed_point_free() -> None:
    first = evaluator_v2.global_derangement()
    second = evaluator_v2.global_derangement(seed=evaluator_v2.NULL_SEED)
    assert np.array_equal(first, second)
    assert first.shape == (protocol.NU,)
    assert first.dtype == np.int64
    assert first.min() == 0 and first.max() == protocol.NU - 1
    assert np.array_equal(np.sort(first), np.arange(protocol.NU, dtype=np.int64))
    assert np.count_nonzero(first == np.arange(protocol.NU)) == 0
    assert evaluator_v2.permutation_sha256(first) == evaluator_v2.permutation_sha256(second)


def test_v2_permutation_constructor_accepts_no_gt_or_carrier_tensors() -> None:
    assert set(evaluator_v2.global_derangement.__annotations__) == {"size", "seed", "return"}
    with pytest.raises(TypeError):
        evaluator_v2.global_derangement(labels=np.zeros(protocol.NU, dtype=np.int64))


def test_v2_null_changes_only_verifier_correspondence() -> None:
    p_gen, p_ver = _query_posteriors()
    p_gen_before, p_ver_before = p_gen.copy(), p_ver.copy()
    result = evaluator_v2.build_null_utility(p_gen, p_ver)
    permutation, p_ver_null, y_true, conf_true, y_null, conf_null, _support, _closure, _u_null = result
    assert np.array_equal(p_gen, p_gen_before)
    assert np.array_equal(p_ver, p_ver_before)
    assert np.array_equal(p_ver_null, p_ver_before[permutation])
    assert np.array_equal(y_null, y_true)
    assert np.array_equal(conf_null, conf_true)
    assert not np.array_equal(p_ver_null, p_ver_before)


def test_v2_null_leaves_true_u_and_relation_inputs_untouched() -> None:
    p_gen, p_ver = _query_posteriors()
    true_u = materializer.compute_u_cycle(p_gen, p_ver)[-1]
    t_rel = np.arange(protocol.NU * protocol.L * protocol.S, dtype=np.float64).reshape(
        protocol.NU, protocol.L, protocol.S
    )
    true_u_before, t_rel_before = true_u.copy(), t_rel.copy()
    evaluator_v2.build_null_utility(p_gen, p_ver)
    assert np.array_equal(true_u, true_u_before)
    assert np.array_equal(t_rel, t_rel_before)


def test_v2_preserves_old_evaluator_and_result_paths() -> None:
    source = Path(evaluator_v2.__file__).read_text(encoding="utf-8")
    assert "p2_a0_postseal_metrics.json" not in source
    assert evaluator_v2.DEFAULT_OUTPUT_NAME == "p2_a0_postseal_metrics_v2.json"
    assert Path(evaluator.__file__).is_file()


def test_v2_invalid_seal_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing carrier seal"):
        evaluator_v2.evaluate(tmp_path)


@pytest.mark.parametrize("training_seed", [20, 30, 50])
def test_materializer_cli_accepts_all_registered_training_seeds(tmp_path: Path, training_seed: int) -> None:
    args = materializer.build_parser().parse_args(
        ["--execute", "--output-dir", str(tmp_path), "--training-seed", str(training_seed)]
    )
    assert args.training_seed == training_seed


def test_materializer_rejects_unregistered_training_seed(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        materializer.build_parser().parse_args(
            ["--execute", "--output-dir", str(tmp_path), "--training-seed", "40"]
        )


def test_materializer_training_seed_reaches_native_stochastic_components(monkeypatch) -> None:
    training_seed = 30
    views = [np.zeros((protocol.N, dimension), dtype=np.float32) for dimension in protocol.FEATURE_DIMS]
    observed: dict[str, list[int]] = {"set_seed": [], "amv": [], "dataloader": [], "bone": []}
    original_loader = materializer.DataLoader
    monkeypatch.setattr(materializer, "set_seed", lambda value: observed["set_seed"].append(value))

    def fake_amv(*_args, seed: int, **_kwargs):
        observed["amv"].append(seed)
        return object()

    def recording_loader(*args, generator: torch.Generator, **kwargs):
        observed["dataloader"].append(generator.initial_seed())
        return original_loader(*args, generator=generator, **kwargs)

    class FakeBONE(torch.nn.Module):
        def __init__(self, *_args, seed: int, **_kwargs) -> None:
            super().__init__()
            observed["bone"].append(seed)
            self.parameter = torch.nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(materializer, "amv", fake_amv)
    monkeypatch.setattr(materializer, "DataLoader", recording_loader)
    monkeypatch.setattr(materializer, "BONE", FakeBONE)
    monkeypatch.setattr(materializer, "EPOCHS", 0)
    materializer.replay_corrupt_base(views, training_seed)
    assert observed == {name: [training_seed] for name in observed}


def test_materializer_audit_seed_and_parent_artifact_remain_fixed() -> None:
    payload = {"u_cycle": np.ones((protocol.NU, protocol.S)), "pi": np.full((protocol.NU, protocol.S, protocol.K), 1 / protocol.K)}
    audit = materializer._audit(payload, protocol.PARENT_ARTIFACT, training_seed=50)
    assert audit["training_seed"] == 50
    assert audit["corruption_seed"] == 20
    assert protocol.PARENT_ARTIFACT == Path(
        "outputs/bone_p1/msrc_snr2p5_exact2of4_seed20/bone_p1_msrc_corruption.npz"
    )
