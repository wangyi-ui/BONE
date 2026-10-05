"""P2-A1 protocol tests; formal four-arm training is never started here."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.utility_relation import bone_p2_a1_protocol as protocol
from experiments.utility_relation import bone_p2_a1_relation_action as action
from experiments.utility_relation import evaluate_bone_p2_a1_pilot as evaluator
from experiments.utility_relation import run_bone_p2_a1_pilot as runner
from experiments.utility_relation.materialize_bone_p2_a0_carrier import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_BONE_SHA256 = "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092"
ORIGINAL_AMV_SHA256 = "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc"


class ToyBONE(torch.nn.Module):
    """Four-view hook-compatible model for relation-action unit tests."""

    def __init__(self) -> None:
        super().__init__()
        linear = torch.nn.Linear(7, 7, bias=False)
        with torch.no_grad():
            linear.weight.copy_(torch.eye(7))
        self.label_fc = torch.nn.Sequential(linear, torch.nn.Softmax(dim=1))

    def forward(self, views, test: bool = False):
        labels = [self.label_fc(view) for view in views]
        return {"label": torch.argmax(torch.stack(labels).mean(dim=0), dim=1)}


def _views() -> list[torch.Tensor]:
    values = torch.arange(210 * 7, dtype=torch.float32).reshape(210, 7) + 1
    values = values / values.sum(dim=1, keepdim=True)
    return [values + view * 1e-4 for view in range(4)]


def _frozen_batch(batch_size: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.full((batch_size, 14, 6), 0.5)
    weights = torch.ones((batch_size, 6))
    return targets.detach(), weights.detach()


def test_p2a0_seal_validation_and_exact_arm_registry() -> None:
    carrier = runner.load_frozen_carrier()
    assert protocol.ARMS == ("BASE", "TRUE_UNIFORM", "TRUE_U", "SHUFFLE_REL")
    assert carrier.u_cycle.shape == (196, 6)
    assert carrier.t_rel.shape == (196, 14, 6)


def test_all_arms_load_the_exact_same_initial_model_hash() -> None:
    hashes = [runner.model_state_sha256(runner.instantiate_initial_model()) for _ in protocol.ARMS]
    assert len(set(hashes)) == 1


def test_u_t_are_detached_and_arm_transformations_are_isolated() -> None:
    carrier = runner.load_frozen_carrier()
    true_t, true_u = runner.arm_frozen_tensors("TRUE_U", carrier)
    uniform_t, uniform_u = runner.arm_frozen_tensors("TRUE_UNIFORM", carrier)
    shuffled_t, shuffled_u = runner.arm_frozen_tensors("SHUFFLE_REL", carrier)
    assert np.array_equal(true_t, carrier.t_rel) and np.array_equal(true_u, carrier.u_cycle)
    assert np.array_equal(uniform_t, carrier.t_rel) and np.array_equal(uniform_u, np.ones_like(carrier.u_cycle))
    assert np.array_equal(shuffled_t, carrier.t_rel[protocol.frozen_derangement()])
    assert np.array_equal(shuffled_u, carrier.u_cycle)
    assert protocol.array_sha256(protocol.frozen_derangement()) == protocol.FROZEN_PERMUTATION_SHA256
    assert not action.detached_frozen(true_t, torch.device("cpu")).requires_grad
    assert not action.detached_frozen(true_u, torch.device("cpu")).requires_grad


def test_query_anchor_same_forward_shapes_and_anchor_detach() -> None:
    model, views = ToyBONE(), _views()
    queries, anchors = action.same_forward_posteriors(model, [view[:3] for view in views], [view[17:31] for view in views])
    assert len(queries) == len(anchors) == 4
    assert all(query.shape == (3, 7) for query in queries)
    assert all(anchor.shape == (14, 7) for anchor in anchors)
    assert all(query.requires_grad for query in queries)
    assert all(not anchor.requires_grad for anchor in anchors)


def test_relation_is_raw_posterior_dot_and_action_loss_has_bys_shape() -> None:
    model, views = ToyBONE(), _views()
    queries, anchors = action.same_forward_posteriors(model, [view[:3] for view in views], [view[17:31] for view in views])
    targets, weights = _frozen_batch()
    relation, action_loss = action.relation_action_loss(queries[0], anchors[0], targets, weights)
    assert relation.shape == (3, 14)
    assert torch.allclose(relation, queries[0] @ anchors[0].T)
    assert action_loss.shape == (3, 6)


def test_full_epoch_denominator_is_not_a_minibatch_denominator() -> None:
    model, views = ToyBONE(), _views()
    targets, weights = _frozen_batch()
    first, _ = action.semantic_batch_loss(
        model, [view[:3] for view in views], [view[17:31] for view in views], targets, weights, torch.tensor(1176.0)
    )
    second, _ = action.semantic_batch_loss(
        model, [view[:3] for view in views], [view[17:31] for view in views], targets, weights, torch.tensor(2352.0)
    )
    assert torch.allclose(first, second * 2)


def test_exactly_one_semantic_step_per_epoch_and_gradients_are_cleared(monkeypatch) -> None:
    model, views = ToyBONE(), _views()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    targets = torch.full((196, 14, 6), 0.5).detach()
    weights = torch.ones((196, 6)).detach()
    calls = 0
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", counted_step)
    assert action.semantic_epoch(
        model, optimizer, views, torch.arange(196), torch.arange(14, 28), targets, weights, np.arange(196, dtype=np.int64)
    ) == 1
    assert calls == 1
    assert all(parameter.grad is None for parameter in model.parameters())


def test_semantic_and_native_optimizers_are_independent_objects() -> None:
    model = ToyBONE()
    semantic = torch.optim.Adam(model.parameters(), lr=1e-4)
    native = torch.optim.Adam(model.parameters(), lr=.01)
    assert semantic is not native
    assert semantic.param_groups[0]["lr"] == 1e-4
    assert native.param_groups[0]["lr"] == .01


def test_zero_action_graph_is_exact_model_identity() -> None:
    model, views = ToyBONE(), _views()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = runner.model_state_sha256(model)
    targets = torch.full((196, 14, 6), 0.5).detach()
    weights = torch.ones((196, 6)).detach()
    action.zero_action_semantic_graph(model, optimizer, views, torch.arange(196), torch.arange(14, 28), targets, weights)
    assert runner.model_state_sha256(model) == before


def test_native_sample_order_and_active_semantic_order_are_paired() -> None:
    views = [np.zeros((210, dimension), dtype=np.float32) for dimension in (24, 512, 256, 254)]
    orders = []
    for _ in range(2):
        loader = runner._native_loader(views)
        observed = np.concatenate([audit_ids.numpy().astype(np.int64) for _data, _dummy, audit_ids in loader])
        orders.append(protocol.array_sha256(observed))
    assert orders[0] == orders[1]
    schedule_hash = protocol.array_sha256(protocol.semantic_orders())
    assert {schedule_hash for _ in protocol.ACTIVE_SEMANTIC_ARMS} == {schedule_hash}


def test_runner_keeps_gt_out_of_preseal_training_and_never_adds_objectives() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    action_source = Path(action.__file__).read_text(encoding="utf-8")
    assert 'archive["labels"]' not in source
    assert "get_cluster_metrics" not in source
    assert "action_scale" not in source
    assert "native_loss +" not in action_source
    assert "semantic_batch_loss" in action_source and "optimizer.step()" in action_source


def test_p2a1_evaluator_refuses_missing_pre_gt_seal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing P2-A1 pre-GT seal"):
        evaluator.verify_pre_gt_seal(tmp_path)


def test_native_bone_and_amv_sources_are_unmodified() -> None:
    assert sha256_file(REPO_ROOT / "method/bone.py") == ORIGINAL_BONE_SHA256
    assert sha256_file(REPO_ROOT / "method/amv.py") == ORIGINAL_AMV_SHA256


@pytest.mark.parametrize("training_seed", [20, 30, 50])
def test_runner_cli_accepts_registered_training_seeds(tmp_path: Path, training_seed: int) -> None:
    args = runner.build_parser().parse_args(
        ["--carrier-dir", "outputs/bone_p2/p2_a0_msrc_seed20", "--output-dir", str(tmp_path),
         "--training-seed", str(training_seed)]
    )
    assert args.training_seed == training_seed


def test_runner_requires_carrier_dir() -> None:
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args([])


def test_carrier_training_seed_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="carrier training_seed mismatch"):
        runner.load_frozen_carrier("outputs/bone_p2/p2_a0_msrc_seed20", training_seed=30)


def test_requested_training_seed_controls_semantic_and_native_orders() -> None:
    assert np.array_equal(protocol.semantic_orders(), protocol.semantic_orders(20))
    assert not np.array_equal(protocol.semantic_orders(20), protocol.semantic_orders(30))
    views = [np.zeros((210, dimension), dtype=np.float32) for dimension in (24, 512, 256, 254)]
    def first_order(seed: int) -> str:
        loader = runner._native_loader(views, seed)
        observed = np.concatenate([ids.numpy().astype(np.int64) for _data, _dummy, ids in loader])
        return protocol.array_sha256(observed)
    assert first_order(20) == first_order(20)
    assert first_order(20) != first_order(30)


def test_seed20_default_parity_and_frozen_null_across_training_seeds() -> None:
    default_model = runner.model_state_sha256(runner.instantiate_initial_model())
    explicit_model = runner.model_state_sha256(
        runner.instantiate_initial_model("outputs/bone_p2/p2_a0_msrc_seed20", training_seed=20)
    )
    assert default_model == explicit_model
    permutations = [protocol.frozen_derangement() for _seed in (20, 30, 50)]
    assert all(np.array_equal(permutations[0], permutation) for permutation in permutations[1:])
    assert protocol.array_sha256(permutations[0]) == protocol.FROZEN_PERMUTATION_SHA256



def test_runner_training_seed_reaches_initial_model_and_native_generator(monkeypatch) -> None:
    observed: dict[str, list[int]] = {"set_seed": [], "bone": []}
    original_bone = runner.BONE
    monkeypatch.setattr(runner, "set_seed", lambda seed: observed["set_seed"].append(seed))

    class RecordingBONE(original_bone):
        def __init__(self, *args, seed: int, **kwargs) -> None:
            observed["bone"].append(seed)
            super().__init__(*args, seed=seed, **kwargs)

    monkeypatch.setattr(runner, "BONE", RecordingBONE)
    runner.instantiate_initial_model("outputs/bone_p2/p2_a0_msrc_seed20", training_seed=30)
    assert observed == {"set_seed": [30], "bone": [30]}
    views = [np.zeros((210, dimension), dtype=np.float32) for dimension in (24, 512, 256, 254)]
    assert runner._native_loader(views, training_seed=30).generator.initial_seed() == 30


def test_existing_seed20_pre_gt_seal_remains_verifiable_after_parameterization() -> None:
    evaluator.verify_pre_gt_seal("outputs/bone_p2/p2_a1_seed20_formal/BASE")



def _write_dynamic_sealed_arm(
    root: Path, training_seed: int, carrier_seed: int | None = None, extra_seal_entry: bool = False
) -> tuple[Path, Path]:
    """Build a minimal sealed arm and matching P2-A0 carrier without any GT data."""
    carrier_seed = training_seed if carrier_seed is None else carrier_seed
    carrier_dir = root / f"p2_a0_msrc_seed{training_seed}"
    carrier_dir.mkdir(parents=True)
    carrier_npz = carrier_dir / "p2_a0_carrier_pre_gt.npz"
    carrier_npz.write_bytes(f"carrier-seed-{training_seed}".encode("ascii"))
    state = carrier_dir / "carrier_model_state.pt"
    state.write_bytes(f"state-seed-{training_seed}".encode("ascii"))
    carrier_hash = sha256_file(carrier_npz)
    (carrier_dir / "p2_a0_carrier_audit.json").write_text(
        json.dumps({"training_seed": carrier_seed}), encoding="utf-8"
    )
    (carrier_dir / "p2_a0_carrier_seal.json").write_text(
        json.dumps({"sha256": {"carrier_npz": carrier_hash}}), encoding="utf-8"
    )
    arm_dir = root / "BASE"
    arm_dir.mkdir()
    predictions = arm_dir / "final_predictions.npz"
    predictions.write_bytes(b"predictions")
    audit = arm_dir / "pre_gt_audit.json"
    audit.write_text(json.dumps({"training_seed": training_seed, "carrier_dir": str(carrier_dir)}), encoding="utf-8")
    seal_hashes = {
        "predictions": sha256_file(predictions),
        "audit": sha256_file(audit),
        "runner_source": sha256_file(REPO_ROOT / "experiments/utility_relation/run_bone_p2_a1_pilot.py"),
        "action_source": sha256_file(REPO_ROOT / "experiments/utility_relation/bone_p2_a1_relation_action.py"),
        "protocol_source": sha256_file(REPO_ROOT / "experiments/utility_relation/bone_p2_a1_protocol.py"),
        "method_bone": sha256_file(REPO_ROOT / "method/bone.py"),
        "method_amv": sha256_file(REPO_ROOT / "method/amv.py"),
        "carrier_npz": carrier_hash,
        "initial_model_state": sha256_file(state),
    }
    if extra_seal_entry:
        seal_hashes["unexpected"] = "not-allowed"
    seal_path = arm_dir / "pre_gt_seal.json"
    seal_path.write_text(json.dumps({"sha256": seal_hashes}), encoding="utf-8")
    return arm_dir, carrier_dir


@pytest.mark.parametrize("training_seed", [30, 50])
def test_dynamic_seed_specific_carrier_three_way_hash_is_accepted(tmp_path: Path, training_seed: int) -> None:
    arm_dir, carrier_dir = _write_dynamic_sealed_arm(tmp_path, training_seed)
    evaluator.verify_pre_gt_seal(arm_dir)
    expected = json.loads((arm_dir / "pre_gt_seal.json").read_text())["sha256"]["carrier_npz"]
    carrier_seal = json.loads((carrier_dir / "p2_a0_carrier_seal.json").read_text())["sha256"]["carrier_npz"]
    assert expected == carrier_seal == sha256_file(carrier_dir / "p2_a0_carrier_pre_gt.npz")


def test_dynamic_carrier_arm_seal_hash_tampering_fails_closed(tmp_path: Path) -> None:
    arm_dir, _carrier_dir = _write_dynamic_sealed_arm(tmp_path, 30)
    seal_path = arm_dir / "pre_gt_seal.json"
    seal = json.loads(seal_path.read_text())
    seal["sha256"]["carrier_npz"] = "0" * 64
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(ValueError, match="carrier_npz"):
        evaluator.verify_pre_gt_seal(arm_dir)


def test_dynamic_carrier_file_tampering_fails_closed(tmp_path: Path) -> None:
    arm_dir, carrier_dir = _write_dynamic_sealed_arm(tmp_path, 30)
    (carrier_dir / "p2_a0_carrier_pre_gt.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="carrier_npz"):
        evaluator.verify_pre_gt_seal(arm_dir)


def test_p2a0_carrier_seal_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    arm_dir, carrier_dir = _write_dynamic_sealed_arm(tmp_path, 30)
    carrier_seal_path = carrier_dir / "p2_a0_carrier_seal.json"
    carrier_seal_path.write_text(json.dumps({"sha256": {"carrier_npz": "f" * 64}}), encoding="utf-8")
    with pytest.raises(ValueError, match="carrier_npz"):
        evaluator.verify_pre_gt_seal(arm_dir)


def test_arm_and_carrier_training_seed_mismatch_fails_closed(tmp_path: Path) -> None:
    arm_dir, _carrier_dir = _write_dynamic_sealed_arm(tmp_path, 30, carrier_seed=50)
    with pytest.raises(ValueError, match="training_seed mismatch"):
        evaluator.verify_pre_gt_seal(arm_dir)


def test_missing_or_unknown_pre_gt_seal_schema_entries_fail_closed(tmp_path: Path) -> None:
    arm_dir, _carrier_dir = _write_dynamic_sealed_arm(tmp_path / "missing", 30)
    seal_path = arm_dir / "pre_gt_seal.json"
    seal = json.loads(seal_path.read_text())
    del seal["sha256"]["carrier_npz"]
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        evaluator.verify_pre_gt_seal(arm_dir)
    extra_arm, _extra_carrier = _write_dynamic_sealed_arm(tmp_path / "extra", 30, extra_seal_entry=True)
    with pytest.raises(ValueError, match="schema"):
        evaluator.verify_pre_gt_seal(extra_arm)


def test_evaluator_does_not_load_gt_when_first_seal_validation_fails(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "BASE").mkdir()
    def gt_load_forbidden(*_args, **_kwargs):
        raise AssertionError("GT loader must not be called before seal validation")
    monkeypatch.setattr(evaluator.np, "load", gt_load_forbidden)
    with pytest.raises(ValueError, match="missing P2-A1 pre-GT seal"):
        evaluator.evaluate(tmp_path)
