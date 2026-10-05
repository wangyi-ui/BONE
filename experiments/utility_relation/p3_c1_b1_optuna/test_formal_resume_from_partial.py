from experiments.utility_relation.p3_c1_b1_optuna import formal_carrier_materializer as frozen
from experiments.utility_relation.p3_c1_b1_optuna import formal_carrier_materializer_r1 as r1
from experiments.utility_relation.p3_c1_b1_optuna.test_formal_carrier_materializer_r1 import install_r1_fakes


def test_first_five_complete_skip_and_sixth_msrc37_enters_r1(monkeypatch, tmp_path):
    initial_calls = []; install_r1_fakes(monkeypatch, tmp_path, initial_calls)
    root = tmp_path / "formal"
    first_six = r1.expected_identities()[:6]
    assert first_six == (
        ("HandWrite", 11), ("HandWrite", 37),
        ("BDGP", 11), ("BDGP", 37),
        ("MSRC", 11), ("MSRC", 37),
    )
    for dataset, seed in first_six[:5]:
        assert r1.materialize_one_r1(dataset, seed, "cuda:0", root)["status"] == "COMPLETE"

    resumed_calls = []
    def generic(dataset, seed, device, staging):
        resumed_calls.append(("generic", dataset, seed));
        raise AssertionError("completed generic carrier must not retrain")
    monkeypatch.setattr(frozen.carrier_adapter, "build_development_carrier", generic)
    original_formal = r1.formal_msrc_carrier_adapter.build_formal_msrc_development_carrier
    def msrc(seed, device, staging):
        resumed_calls.append(("formal_msrc", "MSRC", seed))
        return original_formal(seed, device, staging)
    monkeypatch.setattr(r1.formal_msrc_carrier_adapter, "build_formal_msrc_development_carrier", msrc)

    statuses = []
    for dataset, seed in first_six:
        statuses.append(r1.materialize_one_r1(dataset, seed, "cuda:0", root)["status"])
    assert statuses[:5] == ["ALREADY_COMPLETE_VERIFIED"] * 5
    assert statuses[5] == "COMPLETE"
    assert resumed_calls == [("formal_msrc", "MSRC", 37)]
