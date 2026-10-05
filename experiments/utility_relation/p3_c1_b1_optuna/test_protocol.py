import pytest

from experiments.utility_relation.p3_c1_b1_optuna import protocol as p


def test_exact_dataset_registry_and_msrc():
    assert p.DATASETS == (
        "HandWrite", "BDGP", "MSRC", "COIL20", "Caltech-5V", "MNIST-USPS", "ALOI", "CCV",
    )
    assert len(p.DATASETS) == 8
    assert "MSRC" in p.DATASETS


def test_main_grid_and_epoch_constants():
    assert p.SEMANTIC_LR_GRID == (3e-6, 1e-5, 3e-5, 1e-4, 3e-4)
    assert p.SEMANTIC_EXPOSURE_GRID == (1, 3, 5, 10, 20)
    assert len(p.main_search_candidates()) == 25
    assert p.CARRIER_EPOCHS == 100
    assert p.REFINEMENT_HORIZON == 20
    assert all(x.semantic_exposure <= p.REFINEMENT_HORIZON for x in p.main_search_candidates())


def test_seeds_folds_and_metadata_are_frozen():
    assert p.DEV_SEEDS == (11, 37)
    assert p.FORMAL_SEEDS == (20, 30, 50)
    assert not set(p.DEV_SEEDS) & set(p.FORMAL_SEEDS)
    assert p.FOLDS == ("A", "B")
    assert p.FUTURE_OPTUNA_SAMPLER == "GridSampler"
    assert p.SEED_IS_A_HYPERPARAMETER is False
    assert p.ZERO_UTILITY_POLICY == "FAIL_CLOSED"


@pytest.mark.parametrize("exposure", [0, -1, 21, 2])
def test_exposure_rejects_invalid_or_unregistered_values(exposure):
    with pytest.raises(ValueError):
        p.validate_semantic_exposure(exposure)


def test_refinement_schedule_keeps_horizon_fixed():
    assert p.refinement_schedule(1) == (1, 19)
    assert p.refinement_schedule(20) == (20, 0)
    assert p.exposure_upper_bound_hit(20) is True
    assert p.exposure_upper_bound_hit(10) is False


def test_lr_extension_rules_are_global_and_one_stage():
    mid = {dataset: 1e-5 for dataset in p.DATASETS}
    assert p.required_lr_extension(mid) == ()
    lower = dict(mid, **{"HandWrite": 3e-6})
    assert p.required_lr_extension(lower) == (1e-6,)
    upper = dict(mid, **{"HandWrite": 3e-4})
    assert p.required_lr_extension(upper) == (1e-3,)
    both = dict(lower, **{"BDGP": 3e-4})
    assert p.required_lr_extension(both) == (1e-6, 1e-3)
    with pytest.raises(ValueError):
        p.required_lr_extension({"HandWrite": 1e-5})
