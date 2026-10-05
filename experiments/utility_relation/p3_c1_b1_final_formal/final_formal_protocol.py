"""Fail-closed constants and parameter binding for final formal evaluation.

No hyperparameter value is defined here.  The byte-identical frozen JSON is
the sole source of ``semantic_lr`` and ``semantic_exposure``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FREEZE = (
    REPO_ROOT
    / "experiment_freeze"
    / "bone_p3_c1_b1_a2_final_hyperparameters_20261004"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "bone_p3" / "p3_c1_b1_final_formal"
FINAL_HYPERPARAMETERS_SHA256 = (
    "1e296aa4ea8cd562857044c1dca8432f854d32dc67868002f240af2c0d53b9d0"
)
FINAL_STATUS_SHA256 = (
    "f9ac49ba966a573cf59dff542792eac25d37ab93f3f154e03a6b12cb41d72d18"
)
DATASETS = (
    "HandWrite",
    "BDGP",
    "MSRC",
    "COIL20",
    "Caltech-5V",
    "MNIST-USPS",
    "ALOI",
    "CCV",
)
FORMAL_SEEDS = (20, 30, 50)
CARRIER_EPOCHS = 100
REFINEMENT_EPOCHS = 20
NATIVE_LR = 1e-2
QUERY_CHUNK_SIZE = 256
OUTPUT_STAGE = "p3_c1_b1_final_formal"
MSRC_ROUTE = (
    "p3_c1_b1_optuna.msrc_adapter._load_frozen_msrc_inputs"
    " -> P3-B0 generic carrier -> frozen C0 lower-level training primitives"
)
GENERIC_ROUTE = (
    "P3-C0 frozen A3/A4 input adapter -> P3-B0 generic carrier"
    " -> frozen C0 lower-level training primitives"
)

# These are immutable scientific parents, not tunable parameters.
PARENT_SOURCE_SHA256 = MappingProxyType({
    "method/bone.py": "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092",
    "method/amv.py": "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc",
    "test_cmvsc.py": "fc836c5a8e053624eee7bcf200a592b3a064a9eec71e719012eb976dd1641d8d",
    "experiments/utility_relation/run_bone_p3_c0_cross_dataset_formal.py": "330ac8999db35cae64092fcfeb0dc8dc900370996001a82d214a1eca90233f6b",
    "experiments/utility_relation/evaluate_bone_p3_c0_cross_dataset_formal.py": "5ae9f85a988e42a7bdee7b6a068f4b103b94118307b7edfbf149bbcf0a93d4ba",
    "experiments/utility_relation/bone_p3_a1_action_space_protocol.py": "5564354b41f78917b1ad4f98ae8737fe997d3726902e5e2a97e1ec024e20d61a",
    "experiments/utility_relation/bone_p3_a3_sparse_label_protocol.py": "07571e80de1917e7358e8d614a3fb8d159b8ee635331f2f4c83e709acc387e7a",
    "experiments/utility_relation/bone_p3_a4_corruption_protocol.py": "1865741c1ac3f2f65ffb9c325dc30a7d34978f96adc1be0ad000b45ed899b3c9",
    "experiments/utility_relation/bone_p3_b0_generic_carrier.py": "5a9f3c817d0e0ea07e311f0899fdb926cef993d0f3b7edbd65faf7f83dfabc27",
    "experiments/utility_relation/run_bone_p3_b1_caltech_engineering_smoke.py": "d686d59a2a7000eecb2626413b0640edb95a4d25de0db2c8834fa402cddef771",
    "experiments/utility_relation/p3_c1_b1_optuna/trial_runner.py": "b7f98f180cd3a81bd9fca3ef2b75fe611e98b0d2521c43352ae68407242eecc3",
    "experiments/utility_relation/p3_c1_b1_optuna/trial_runner_a1.py": "654e1ce7554c5c01a672a7473e82a1fe732d171845c89e6a1ae09cc6a0a05b22",
    "experiments/utility_relation/p3_c1_b1_optuna/protocol.py": "d78436123de23739fe65064c0a7d9640c472cb0f7f88bf92e6f4dfa69179212f",
    "experiments/utility_relation/p3_c1_b1_optuna/amendment_a2.py": "b3237d057e11051909c721828c3efb3d123651f3f8b7e1961694ba9f03ee46c0",
    "experiments/utility_relation/p3_c1_b1_optuna/canonical_fold_order_r3.py": "98702c3c4af819a5229b9b05da576c81a67aa1bbcf9ce0fd9e67fab122e43caf",
    "experiments/utility_relation/p3_c1_b1_optuna/formal_tuning_context_a2.py": "7186c647a5612f5ca53e219c6f4766d563df5b7a73e88aa1f25d2af2f466eb2b",
    "experiments/utility_relation/p3_c1_b1_optuna/formal_tuning_context_r3.py": "4a2d45cb6c64d23c7cd443425b3b794985a6adf97029750a696f6cf1b85b15de",
    "experiments/utility_relation/p3_c1_b1_optuna/msrc_adapter.py": "71b6108269f3e225d7ea45116ac1fddf3c970d4330d117c40da086bb95d1ee06",
    "experiments/utility_relation/p3_c1_b1_optuna/formal_msrc_carrier_adapter.py": "3661a5541fee2f3f521c0c6dc3fd1b276ce3e478b40631d30027d452d0f5ef56",
    "experiments/utility_relation/materialize_bone_p2_a0_carrier.py": "70583ad56eba43da26adbf95e50ed287a69c3b13c259474e1cf81eeada1fb04f",
    "experiments/utility_relation/materialize_bone_p3_a4_corruptions.py": "01b3bf750c1abec19d3367c55927603e13197a3bbf2d8647d805a07362351cba",
    "experiments/utility_relation/materialize_bone_p3_b0_carrier.py": "159d6a7a3e5a58d49f4d7da6fc0f2a0ca43bccf200de57f59de5781c86f15a53",
    "experiments/weak_quality/bone_p1_msrc_protocol.py": "198b368a65271d034657d9ca075a9725880609e771d3a0866d4d72f27b42676a",
})


@dataclass(frozen=True)
class ParameterBinding:
    dataset: str
    semantic_lr: float
    semantic_exposure: int
    final_parameter_freeze_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_freeze_identity(freeze_dir: str | Path) -> tuple[Path, Path]:
    root = Path(freeze_dir)
    parameter_path, status_path = root / "FINAL_HYPERPARAMETERS.json", root / "STATUS.txt"
    if not parameter_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("FINAL_HYPERPARAMETER_FREEZE_MISSING")
    if sha256_file(parameter_path) != FINAL_HYPERPARAMETERS_SHA256:
        raise RuntimeError("FINAL_HYPERPARAMETER_FREEZE_HASH_MISMATCH")
    if sha256_file(status_path) != FINAL_STATUS_SHA256:
        raise RuntimeError("FINAL_HYPERPARAMETER_STATUS_HASH_MISMATCH")
    status = status_path.read_text(encoding="utf-8")
    required = (
        "STATUS=P3_C1_B1_A2_FINAL_HYPERPARAMETERS_FROZEN",
        "FINAL_HYPERPARAMETERS_FROZEN=true",
        "SECOND_BOUNDARY_EXTENSION_ALLOWED=false",
        "PARAMETER_RETUNING_AFTER_FORMAL_GT_ALLOWED=false",
    )
    if any(item not in status for item in required):
        raise RuntimeError("FINAL_HYPERPARAMETER_STATUS_INVALID")
    return parameter_path, status_path


def load_final_parameter_bindings(
    freeze_dir: str | Path = DEFAULT_FREEZE,
) -> Mapping[str, ParameterBinding]:
    parameter_path, _ = _validate_freeze_identity(freeze_dir)
    try:
        payload = json.loads(parameter_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("FINAL_HYPERPARAMETER_JSON_INVALID") from error
    if (
        payload.get("status") != "FINAL_HYPERPARAMETERS_FROZEN"
        or payload.get("stage") != "P3-C1-B1-A2"
        or payload.get("second_boundary_extension_allowed") is not False
        or payload.get("full_gt_used_for_selection") is not False
        or payload.get("acc_used_for_selection") is not False
        or payload.get("nmi_used_for_selection") is not False
        or payload.get("ari_used_for_selection") is not False
    ):
        raise RuntimeError("FINAL_HYPERPARAMETER_JSON_NOT_FROZEN")
    rows = payload.get("datasets")
    if not isinstance(rows, list) or len(rows) != len(DATASETS):
        raise RuntimeError("FINAL_HYPERPARAMETER_DATASET_SET_MISMATCH")
    bindings: dict[str, ParameterBinding] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("FINAL_HYPERPARAMETER_ROW_INVALID")
        dataset = row.get("dataset")
        lr, exposure = row.get("semantic_lr"), row.get("semantic_exposure")
        if (
            dataset not in DATASETS
            or dataset in bindings
            or isinstance(lr, bool)
            or not isinstance(lr, (int, float))
            or not (0.0 < float(lr))
            or isinstance(exposure, bool)
            or not isinstance(exposure, int)
            or not (1 <= exposure <= REFINEMENT_EPOCHS)
        ):
            raise RuntimeError("FINAL_HYPERPARAMETER_ROW_INVALID")
        bindings[dataset] = ParameterBinding(
            dataset=dataset,
            semantic_lr=float(lr),
            semantic_exposure=exposure,
            final_parameter_freeze_sha256=FINAL_HYPERPARAMETERS_SHA256,
        )
    if tuple(bindings) != DATASETS or set(bindings) != set(DATASETS):
        raise RuntimeError("FINAL_HYPERPARAMETER_DATASET_ORDER_MISMATCH")
    return MappingProxyType(bindings)


def resolve_binding(dataset: str, freeze_dir: str | Path = DEFAULT_FREEZE) -> ParameterBinding:
    if dataset not in DATASETS:
        raise ValueError("UNREGISTERED_FINAL_FORMAL_DATASET")
    return load_final_parameter_bindings(freeze_dir)[dataset]


def dataset_slug(dataset: str) -> str:
    if dataset not in DATASETS:
        raise ValueError("UNREGISTERED_FINAL_FORMAL_DATASET")
    return dataset.lower().replace("-", "_")


def run_root(output_root: str | Path, dataset: str, training_seed: int) -> Path:
    if training_seed not in FORMAL_SEEDS:
        raise ValueError("UNREGISTERED_FINAL_FORMAL_SEED")
    return Path(output_root) / dataset_slug(dataset) / f"seed{training_seed}"


def verify_parent_sources() -> dict[str, str]:
    observed = {}
    for relative, expected in PARENT_SOURCE_SHA256.items():
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"FROZEN_PARENT_MISSING:{relative}")
        observed[relative] = sha256_file(path)
        if observed[relative] != expected:
            raise RuntimeError(f"FROZEN_PARENT_HASH_MISMATCH:{relative}")
    return observed


def write_json(path: str | Path, payload: object) -> None:
    """Atomic deterministic JSON write within an already-authorized output root."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
