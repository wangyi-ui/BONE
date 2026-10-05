"""Protocol tests only: no formal CLEAN_BASE/CORRUPT_BASE training is started."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.weak_quality import materialize_bone_p1_msrc_corruption as materializer
from experiments.weak_quality import run_bone_p1_msrc_pair as runner
from experiments.weak_quality.bone_p1_msrc_protocol import (
    CORRUPTION_SEED, FEATURE_DIMS, N, PAIRS, TARGET_SNR_DB, V, build_corruption_mask, pair_counts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_BONE_SHA256 = "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092"
ORIGINAL_AMV_SHA256 = "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc"


def _clean_views() -> list[np.ndarray]:
    rng = np.random.RandomState(123)
    return [rng.normal(size=(N, dimension)).astype(np.float64) for dimension in FEATURE_DIMS]


def test_mask_exact_structure_and_pair_counts() -> None:
    mask = build_corruption_mask()
    assert mask.shape == (210, 4)
    assert np.all(mask.sum(axis=1) == 2)
    assert np.array_equal(mask.sum(axis=0), np.array([105, 105, 105, 105]))
    assert int(mask.sum()) == 420
    assert pair_counts(mask) == {f"{a}-{b}": 35 for a, b in PAIRS}


def test_mask_seed_determinism_and_assignment_variation() -> None:
    assert np.array_equal(build_corruption_mask(CORRUPTION_SEED), build_corruption_mask(CORRUPTION_SEED))
    assert not np.array_equal(build_corruption_mask(CORRUPTION_SEED), build_corruption_mask(CORRUPTION_SEED + 1))


def test_noise_changes_only_masked_locations_and_hits_snr() -> None:
    clean = _clean_views()
    mask = build_corruption_mask()
    corrupt, _noise, _alphas, achieved = materializer.corrupt_standardized_views(clean, mask)
    for view_index in range(V):
        omega = mask[:, view_index]
        assert np.array_equal(clean[view_index][~omega], corrupt[view_index][~omega])
        assert np.any(clean[view_index][omega] != corrupt[view_index][omega])
        assert abs(achieved[view_index] - TARGET_SNR_DB) <= 0.05
        assert np.isfinite(corrupt[view_index]).all()


def test_materialized_artifact_schema_and_audit(tmp_path: Path) -> None:
    paths = materializer.materialize(tmp_path)
    with np.load(paths["artifact"], allow_pickle=False) as artifact:
        assert set(artifact.files) == {"x_clean_v0", "x_clean_v1", "x_clean_v2", "x_clean_v3",
                                       "x_corrupt_v0", "x_corrupt_v1", "x_corrupt_v2", "x_corrupt_v3",
                                       "corruption_mask", "labels", "row_ids"}
        assert artifact["corruption_mask"].shape == (N, V)
        assert np.array_equal(artifact["row_ids"], np.arange(N))
        assert all(artifact[key].dtype != object for key in artifact.files)
        for view_index, dimension in enumerate(FEATURE_DIMS):
            assert artifact[f"x_clean_v{view_index}"].shape == (N, dimension)
            assert artifact[f"x_corrupt_v{view_index}"].shape == (N, dimension)
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    assert audit["corrupted_count_per_view"] == [105] * V
    assert audit["pair_counts"] == {f"{a}-{b}": 35 for a, b in PAIRS}
    assert audit["clean_locations_exactly_unchanged"] is True
    assert audit["labels_used_for_corruption"] is False
    assert audit["post_corruption_standardization"] is False
    assert all(abs(snr - TARGET_SNR_DB) <= 0.05 for snr in audit["achieved_snr_db"])


def test_labels_do_not_affect_corruption(monkeypatch, tmp_path: Path) -> None:
    views = _clean_views()

    class FakeDataset:
        def __init__(self, labels: np.ndarray) -> None:
            self.data = [view.T.copy() for view in views]
            self.label = labels

        def standard_data(self) -> None:
            pass

    monkeypatch.setattr(materializer, "get_dataset", lambda _name: FakeDataset(np.arange(N)))
    first = materializer.materialize(tmp_path / "first")
    monkeypatch.setattr(materializer, "get_dataset", lambda _name: FakeDataset(np.arange(N)[::-1]))
    second = materializer.materialize(tmp_path / "second")
    with np.load(first["artifact"], allow_pickle=False) as left, np.load(second["artifact"], allow_pickle=False) as right:
        assert not np.array_equal(left["labels"], right["labels"])
        for key in ("corruption_mask", "x_corrupt_v0", "x_corrupt_v1", "x_corrupt_v2", "x_corrupt_v3"):
            assert np.array_equal(left[key], right[key])


def test_runner_does_not_reprocess_artifact_or_use_gt_in_training() -> None:
    source = (REPO_ROOT / "experiments/weak_quality/run_bone_p1_msrc_pair.py").read_text(encoding="utf-8")
    assert ".standard_data(" not in source
    assert ".shuffle(" not in source
    assert "dummy_labels" in source
    assert "labels = np.ascontiguousarray(archive[\"labels\"])" in source


def test_native_bone_files_are_unmodified() -> None:
    assert materializer.sha256_file(REPO_ROOT / "method/bone.py") == ORIGINAL_BONE_SHA256
    assert materializer.sha256_file(REPO_ROOT / "method/amv.py") == ORIGINAL_AMV_SHA256


@pytest.mark.parametrize("training_seed", [20, 30, 50])
def test_runner_cli_accepts_allowed_training_seeds(tmp_path: Path, training_seed: int) -> None:
    args = runner.build_parser().parse_args(
        ["--execute", "--output-dir", str(tmp_path), "--training-seed", str(training_seed)]
    )
    assert args.training_seed == training_seed


def test_runner_cli_rejects_invalid_training_seed(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(
            ["--execute", "--output-dir", str(tmp_path), "--training-seed", "40"]
        )


def test_training_seed_reaches_each_stochastic_native_component(monkeypatch) -> None:
    seed = 30
    views = [np.zeros((N, dimension), dtype=np.float32) for dimension in FEATURE_DIMS]
    observed: dict[str, list[int]] = {"set_seed": [], "amv": [], "dataloader": [], "bone": []}
    original_dataloader = runner.DataLoader

    monkeypatch.setattr(runner, "set_seed", lambda value: observed["set_seed"].append(value))

    def fake_amv(*_args, seed: int, **_kwargs):
        observed["amv"].append(seed)
        return object()

    def recording_dataloader(*args, generator: torch.Generator, **kwargs):
        observed["dataloader"].append(generator.initial_seed())
        return original_dataloader(*args, generator=generator, **kwargs)

    class FakeBONE(torch.nn.Module):
        def __init__(self, *_args, seed: int, **_kwargs) -> None:
            super().__init__()
            observed["bone"].append(seed)
            self.parameter = torch.nn.Parameter(torch.zeros(()))

        def forward(self, data, test: bool = False):
            assert test is True
            return {"label": torch.zeros(len(data[0]), dtype=torch.long)}

    monkeypatch.setattr(runner, "amv", fake_amv)
    monkeypatch.setattr(runner, "DataLoader", recording_dataloader)
    monkeypatch.setattr(runner, "BONE", FakeBONE)
    monkeypatch.setattr(runner, "EPOCHS", 0)

    runner._train_arm(views, seed)

    assert observed == {name: [seed] for name in observed}


def test_training_seed_does_not_change_corruption_artifact(monkeypatch, tmp_path: Path) -> None:
    artifact = tmp_path / "bone_p1_msrc_corruption.npz"
    contents = {"labels": np.zeros(N, dtype=np.int64)}
    for prefix in ("x_clean", "x_corrupt"):
        for view_index, dimension in enumerate(FEATURE_DIMS):
            contents[f"{prefix}_v{view_index}"] = np.zeros((N, dimension), dtype=np.float32)
    np.savez(artifact, **contents)
    before = materializer.sha256_file(artifact)
    seen_seeds: list[int] = []

    def fake_train(_views, training_seed: int) -> np.ndarray:
        seen_seeds.append(training_seed)
        return np.zeros(N, dtype=np.int64)

    monkeypatch.setattr(runner, "_train_arm", fake_train)
    runner.run_paired_experiment(artifact, tmp_path / "seed30", training_seed=30)

    audit = json.loads((tmp_path / "seed30" / "run_audit.json").read_text(encoding="utf-8"))
    assert seen_seeds == [30, 30]
    assert audit["training_seed_reset_per_arm"] == 30
    assert audit["dataloader"]["generator_seed"] == 30
    assert materializer.sha256_file(artifact) == before
