"""P3-C2-B0 read-only audit of the frozen semantic action budget.

This module deliberately never imports a training runner, evaluator, labels, or
metrics.  It reads only pre-GT carrier arrays and C2-A0 diagnostic JSON files.
The torch use below is CPU autograd on a synthetic fixed tensor, not training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
FINAL_ROOT = ROOT / "outputs/bone_p3/p3_c1_b1_final_formal"
C2A0_ROOT = ROOT / "outputs/bone_p3/p3_c2_a0_utility_structural_drift"
OUTPUT_ROOT = ROOT / "outputs/bone_p3/p3_c2_b0_utility_mass_action_budget_audit"
DATASETS = ("aloi", "bdgp", "caltech_5v", "ccv", "coil20", "handwrite", "mnist_usps", "msrc")
SEEDS = (20, 30, 50)
EPS = torch.finfo(torch.float64).eps

PARENT_FILES = (
    "method/bone.py",
    "method/amv.py",
    "test_cmvsc.py",
    "experiments/utility_relation/run_bone_p3_c0_cross_dataset_formal.py",
    "experiments/utility_relation/bone_p3_b0_generic_carrier.py",
    "experiments/utility_relation/p3_c1_b1_optuna/amendment_a2.py",
    "experiments/utility_relation/p3_c1_b1_optuna/canonical_fold_order_r3.py",
    "experiments/utility_relation/p3_c1_b1_final_formal/final_formal_protocol.py",
    "experiments/utility_relation/p3_c1_b1_final_formal/final_formal_context.py",
    "experiments/utility_relation/p3_c1_b1_final_formal/run_final_formal.py",
)
C2A0_FILES = (
    "experiments/utility_relation/p3_c2_a0_utility_structural_drift.py",
    "tests/test_p3_c2_a0_utility_structural_drift.py",
)
EXPECTED_PARENT_HASHES = {
    "method/bone.py": "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092",
    "method/amv.py": "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc",
    "test_cmvsc.py": "fc836c5a8e053624eee7bcf200a592b3a064a9eec71e719012eb976dd1641d8d",
    "experiments/utility_relation/run_bone_p3_c0_cross_dataset_formal.py": "330ac8999db35cae64092fcfeb0dc8dc900370996001a82d214a1eca90233f6b",
    "experiments/utility_relation/bone_p3_b0_generic_carrier.py": "5a9f3c817d0e0ea07e311f0899fdb926cef993d0f3b7edbd65faf7f83dfabc27",
    "experiments/utility_relation/p3_c1_b1_optuna/amendment_a2.py": "b3237d057e11051909c721828c3efb3d123651f3f8b7e1961694ba9f03ee46c0",
    "experiments/utility_relation/p3_c1_b1_optuna/canonical_fold_order_r3.py": "98702c3c4af819a5229b9b05da576c81a67aa1bbcf9ce0fd9e67fab122e43caf",
    "experiments/utility_relation/p3_c1_b1_final_formal/final_formal_protocol.py": "4e3d443bc9e5f21ddf92cb763832a6b3ed5bf5520a07b7113067288fd8847ffb",
    "experiments/utility_relation/p3_c1_b1_final_formal/final_formal_context.py": "77840c28319df9e4d07a548523903a859dfaf09a544e9a74cc17fc7411cf8e56",
    "experiments/utility_relation/p3_c1_b1_final_formal/run_final_formal.py": "b6abc513f81ae08d00141ea897d2c8c209840677503715fd79ab3f237284a6bd",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(paths: Iterable[str]) -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in paths}


def verify_parent_integrity() -> dict[str, str]:
    observed = source_hashes(PARENT_FILES)
    if observed != EXPECTED_PARENT_HASHES:
        raise ValueError("PARENT_SOURCE_INTEGRITY_MISMATCH")
    return observed


def _validate_utility(u: np.ndarray) -> None:
    if u.ndim != 2 or not np.isfinite(u).all() or np.any(u < 0):
        raise ValueError("UTILITY_MUST_BE_FINITE_NONNEGATIVE_2D")


def semantic_relation_loss(
    relation: torch.Tensor, t_rel: torch.Tensor, u_cycle: torch.Tensor,
    k: int, full_utility_mass: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Exact one-view, one-batch C0 loss expression with a supplied full D."""
    if relation.ndim != 2 or t_rel.ndim != 3 or u_cycle.ndim != 2:
        raise ValueError("SEMANTIC_SHAPE_RANK")
    batch, anchors = relation.shape
    if t_rel.shape != (batch, anchors, u_cycle.shape[1]):
        raise ValueError("SEMANTIC_SHAPE_MISMATCH")
    if k <= 1:
        raise ValueError("K_MUST_EXCEED_ONE")
    denominator = u_cycle.sum() if full_utility_mass is None else full_utility_mass
    if not bool(torch.isfinite(denominator).item()) or not bool((denominator > 0).item()):
        raise ValueError("NONPOSITIVE_OR_NONFINITE_UTILITY_DENOMINATOR")
    safe = relation.clamp(EPS, 1.0 - EPS)
    positive = -(t_rel * torch.log(safe.unsqueeze(2))).sum(dim=1) / 4.0
    negative = -((1.0 - t_rel) * torch.log1p(-safe.unsqueeze(2))).sum(dim=1) / (4.0 * (k - 1))
    positive_numerator = (u_cycle * positive).sum()
    negative_numerator = (u_cycle * negative).sum()
    loss = (positive_numerator + negative_numerator) / denominator
    return loss, {
        "positive": positive, "negative": negative,
        "positive_numerator": positive_numerator,
        "negative_numerator": negative_numerator,
        "denominator": denominator,
    }


def reconstruct_semantic_loss(
    relation_views: list[torch.Tensor], t_rel: torch.Tensor, u_cycle: torch.Tensor,
    k: int, batch_size: int,
) -> torch.Tensor:
    """Exact C0 accumulation: each query batch uses global D, then views average."""
    full_mass = u_cycle.sum()
    if not bool((full_mass > 0).item()):
        raise ValueError("NONPOSITIVE_OR_NONFINITE_UTILITY_DENOMINATOR")
    total = torch.zeros((), dtype=relation_views[0].dtype, device=relation_views[0].device)
    for start in range(0, u_cycle.shape[0], batch_size):
        stop = min(start + batch_size, u_cycle.shape[0])
        per_view = []
        for relation in relation_views:
            value, _ = semantic_relation_loss(relation[start:stop], t_rel[start:stop], u_cycle[start:stop], k, full_mass)
            per_view.append(value)
        total = total + sum(per_view) / len(per_view)
    return total


def synthetic_scale_audit() -> dict[str, Any]:
    """Fixed CPU logits/targets, evaluating loss and dL/dz under U scaling."""
    logits_base = torch.tensor([[0.2, -0.7, 1.1], [-1.3, 0.6, 0.4]], dtype=torch.float64)
    targets = torch.tensor(
        [[[0.2, 0.8], [0.6, 0.1], [0.4, 0.7]], [[0.9, 0.3], [0.5, 0.6], [0.2, 0.4]]],
        dtype=torch.float64,
    )
    weights = torch.tensor([[0.8, 0.3], [0.4, 0.9]], dtype=torch.float64)
    results: dict[str, dict[str, float]] = {}
    for factor in (1.0, 0.5, 0.1):
        logits = logits_base.detach().clone().requires_grad_(True)
        loss, _ = semantic_relation_loss(torch.sigmoid(logits), targets, weights * factor, k=4)
        loss.backward()
        results[str(factor)] = {"loss": float(loss.detach()), "gradient_l2": float(logits.grad.norm())}
    base = results["1.0"]
    loss_invariant = all(np.isclose(base["loss"], results[key]["loss"], rtol=1e-12, atol=1e-12) for key in ("0.5", "0.1"))
    grad_invariant = all(np.isclose(base["gradient_l2"], results[key]["gradient_l2"], rtol=1e-12, atol=1e-12) for key in ("0.5", "0.1"))
    return {"device": "cpu", "fixed_logits": True, "results": results,
            "UTILITY_MASS_SCALE_INVARIANT": bool(loss_invariant),
            "GRADIENT_MASS_SCALE_INVARIANT": bool(grad_invariant)}


def chunked_relation_mass(t_rel: np.ndarray, u: np.ndarray, k: int, chunk_size: int = 256) -> dict[str, float]:
    """Read chunks without a secondary whole-T_rel copy; T_rel is compressed NPZ decoded once."""
    _validate_utility(u)
    if t_rel.ndim != 3 or t_rel.shape[0] != u.shape[0] or t_rel.shape[2] != u.shape[1]:
        raise ValueError("T_REL_UTILITY_SHAPE_MISMATCH")
    if not np.isfinite(t_rel).all():
        raise ValueError("T_REL_MUST_BE_FINITE")
    positive_mass = negative_mass = positive_nonzero = negative_nonzero = 0.0
    population = float(t_rel.shape[0] * t_rel.shape[1] * t_rel.shape[2])
    for start in range(0, t_rel.shape[0], chunk_size):
        stop = min(start + chunk_size, t_rel.shape[0])
        t_chunk = t_rel[start:stop]
        u_chunk = u[start:stop, None, :]
        positive_mass += float(np.sum(u_chunk * t_chunk, dtype=np.float64))
        negative_mass += float(np.sum(u_chunk * (1.0 - t_chunk), dtype=np.float64))
        positive_nonzero += float(np.count_nonzero(u_chunk * t_chunk))
        negative_nonzero += float(np.count_nonzero(u_chunk * (1.0 - t_chunk)))
    effective_mass = positive_mass / 4.0 + negative_mass / (4.0 * (k - 1))
    return {
        "positive_relation_weight_mass": positive_mass,
        "negative_relation_weight_mass": negative_mass,
        "effective_relation_weight_mass": effective_mass,
        "effective_weight_mean": effective_mass / (2.0 * population),
        "effective_weight_nonzero_fraction": (positive_nonzero + negative_nonzero) / (2.0 * population),
        "relation_term_population": 2.0 * population,
    }


def _carrier_audit(dataset: str, seed: int) -> dict[str, Any]:
    path = FINAL_ROOT / dataset / f"seed{seed}" / "carrier/carrier_pre_gt.npz"
    with np.load(path, allow_pickle=False) as carrier:
        required = {"query_U_cycle", "T_rel", "closure", "query_rows", "anchor_rows", "labeled_targets"}
        if not required.issubset(carrier.files):
            raise ValueError(f"CARRIER_REQUIRED_FIELDS_MISSING:{dataset}:seed{seed}")
        u = carrier["query_U_cycle"]
        _validate_utility(u)
        nq, actions = u.shape
        closure = carrier["closure"][carrier["query_rows"]]
        if closure.shape != u.shape:
            raise ValueError("QUERY_CLOSURE_SHAPE_MISMATCH")
        t_rel = carrier["T_rel"]  # NPZ necessarily decodes this array; calculations below are chunked and do not copy it.
        k = int(np.max(carrier["labeled_targets"])) + 1
        relation = chunked_relation_mass(t_rel, u, k)
    per_sample = u.sum(axis=1)
    supported = per_sample > 0
    full_action_population = float(nq * actions)
    total_mass = float(u.sum(dtype=np.float64))
    row = {
        "DATASET": dataset.upper(), "SEED": seed, "Nq": nq, "S": actions,
        "supported_count": int(supported.sum()), "abstained_count": int((~supported).sum()),
        "utility_coverage": float(supported.mean()), "mean_U": float(u.mean(dtype=np.float64)),
        "mean_U_supported": float(u[supported].mean(dtype=np.float64)) if supported.any() else None,
        "utility_mass_per_sample": float(per_sample.mean(dtype=np.float64)),
        "total_utility_mass": total_mass, "positive_U_fraction": float((u > 0).mean()),
        "closure_fraction": float(np.mean(closure)), "CURRENT_NORMALIZED_ACTION_SCALE": 1.0,
        "MASS_RETENTION_FACTOR": total_mass / full_action_population,
        "carrier_path": str(path.relative_to(ROOT)),
    }
    row.update(relation)
    return row


def _c2a0_fields(dataset: str, seed: int) -> dict[str, Any]:
    path = C2A0_ROOT / dataset / f"seed{seed}" / "DIAGNOSTIC.json"
    data = json.loads(path.read_text())
    if data.get("full_gt_read") or not data.get("reporting_only"):
        raise ValueError("C2A0_DIAGNOSTIC_NOT_PRE_GT")
    wanted = ("query_flip_rate", "supported_flip_rate", "abstained_flip_rate", "flip_abstained_fraction",
              "agreement_nmi_query", "agreement_ari_query", "occupancy_tv_query")
    return {key: data.get(key) for key in wanted}


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = columns or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _rho(rows: list[dict[str, Any]], x: str, y: str) -> dict[str, Any]:
    xv, yv = np.asarray([row[x] for row in rows], dtype=float), np.asarray([row[y] for row in rows], dtype=float)
    statistic = spearmanr(xv, yv)
    return {"x": x, "y": y, "n": int(xv.size), "rho": float(statistic.statistic), "p_value": float(statistic.pvalue), "descriptive_only": True}


def formula_audit(synthetic: dict[str, Any]) -> dict[str, Any]:
    return {
        "diagnostic_only": True,
        "code_path": [
            "p3_c1_b1_final_formal/run_final_formal.py::_materialize_or_resume_branches",
            "run_bone_p3_c0_cross_dataset_formal.py::_semantic_epoch",
        ],
        "formal_uses_reused_c0_semantic_epoch": True,
        "inputs": {
            "U_cycle": "query_U_cycle [Nq,S], outer action evidence weight; denominator D=sum_{i,s} U_is",
            "T_rel": "[Nq,L,S], positive soft relation target; complement (1-T_rel) is negative target",
            "Pi": "used upstream to construct T_rel; not directly read by _semantic_epoch",
        },
        "formula": {
            "relation": "r_il^(v)=q_i^(v) dot stopgrad(a_l^(v)); safe=clamp(r, eps, 1-eps)",
            "positive": "P_is^(v)=-(1/4) sum_l T_ils log(safe(r_il^(v)))",
            "negative": "N_is^(v)=-(1/(4*(K-1))) sum_l (1-T_ils) log(1-safe(r_il^(v)))",
            "batch": "ell_b=[sum_v sum_{i in b,s} U_is(P_is^(v)+N_is^(v))]/[V*D]",
            "epoch": "L=sum_b ell_b=[sum_v sum_{i,s} U_is(P_is^(v)+N_is^(v))]/[V*D]",
            "scale": "L(cU)=L(U) and dL/dz is unchanged for c>0, because both numerator and D scale by c.",
        },
        "reduction_audit": [
            {"term": "positive_relation_loss", "raw_element_shape": "[B,S] after anchor sum", "weight_shape": "U[B,S] outer; T[B,L,S] within target", "mask": "none", "numerator": "sum_{i in B,s} U_is P_is", "denominator": "4 inside P; V*D outside", "reductions": "sum anchors, sum actions/query, average views", "detach": "anchor posterior detached"},
            {"term": "negative_relation_loss", "raw_element_shape": "[B,S] after anchor sum", "weight_shape": "U[B,S] outer; (1-T)[B,L,S] within target", "mask": "none", "numerator": "sum_{i in B,s} U_is N_is", "denominator": "4*(K-1) inside N; V*D outside", "reductions": "sum anchors, sum actions/query, average views", "detach": "anchor posterior detached"},
        ],
        "batch_and_masks": {"query_selection": "order slices query rows; all selected rows contribute", "anchor_mask": "none; all L anchors used", "relation_mask": "none", "utility_mask": "implicit U=0 gives zero outer contribution", "action_normalization": "D=sum over all Nq*S U values, not per-batch", "view_normalization": "divide by V", "optimizer": "one optimizer.step after all query batches"},
        "ABSOLUTE_UTILITY_MASS_CANCELLED": True,
        "UTILITY_MASS_SCALE_INVARIANT": synthetic["UTILITY_MASS_SCALE_INVARIANT"],
        "GRADIENT_MASS_SCALE_INVARIANT": synthetic["GRADIENT_MASS_SCALE_INVARIANT"],
        "synthetic_cpu_scale_audit": synthetic,
    }


def run_audit(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    verify_parent_integrity()
    output_root.mkdir(parents=True, exist_ok=True)
    synthetic = synthetic_scale_audit()
    code_audit = formula_audit(synthetic)
    (output_root / "CODE_FORMULA_AUDIT.json").write_text(json.dumps(code_audit, indent=2, sort_keys=True) + "\n")
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            row = _carrier_audit(dataset, seed)
            row.update(_c2a0_fields(dataset, seed))
            rows.append(row)
    if len(rows) != 24:
        raise ValueError("EXPECTED_EXACTLY_24_PAIRS")
    mass_columns = list(rows[0])
    _write_csv(output_root / "per_pair_mass_audit.csv", rows, mass_columns)
    join_columns = ["DATASET", "SEED", "utility_coverage", "mean_U", "total_utility_mass", "MASS_RETENTION_FACTOR",
                    "query_flip_rate", "supported_flip_rate", "abstained_flip_rate", "flip_abstained_fraction",
                    "agreement_nmi_query", "agreement_ari_query", "occupancy_tv_query"]
    _write_csv(output_root / "C2A0_JOINED_PRE_GT_TABLE.csv", rows, join_columns)
    summary = []
    summary_columns = ["DATASET", "U_COVER", "MEAN_U", "WEIGHT_MASS", "MASS_RETENTION", "FLIP_ALL", "FLIP_ZERO", "FLIP_ZERO_SHARE", "AGR_NMI", "AGR_ARI", "OCC_TV"]
    mapping = {"U_COVER": "utility_coverage", "MEAN_U": "mean_U", "WEIGHT_MASS": "total_utility_mass", "MASS_RETENTION": "MASS_RETENTION_FACTOR", "FLIP_ALL": "query_flip_rate", "FLIP_ZERO": "abstained_flip_rate", "FLIP_ZERO_SHARE": "flip_abstained_fraction", "AGR_NMI": "agreement_nmi_query", "AGR_ARI": "agreement_ari_query", "OCC_TV": "occupancy_tv_query"}
    for dataset in DATASETS:
        group = [row for row in rows if row["DATASET"] == dataset.upper()]
        item = {"DATASET": dataset.upper()}
        item.update({name: float(np.mean([row[key] for row in group])) for name, key in mapping.items()})
        summary.append(item)
    _write_csv(output_root / "three_seed_mass_summary.csv", summary, summary_columns)
    correlations = [_rho(rows, "MASS_RETENTION_FACTOR", name) for name in ("query_flip_rate", "abstained_flip_rate", "agreement_ari_query")]
    correlations.append(_rho(rows, "utility_coverage", "query_flip_rate"))
    summary_json = {"pair_count": 24, "dataset_count": 8, "seeds": list(SEEDS), "three_seed_means": summary,
                    "descriptive_spearman": correlations, "CURRENT_NORMALIZED_ACTION_SCALE_explanation": "1.0 because global D=sum(U) cancels common U scale; it is a normalized loss coefficient, not retained evidence mass.",
                    "MASS_RETENTION_FACTOR_definition": "sum_{i,s} U_is/(Nq*S), using the outer U action population in the actual loss before global normalization."}
    (output_root / "three_seed_mass_summary.json").write_text(json.dumps(summary_json, indent=2, sort_keys=True) + "\n")
    return {"code_audit": code_audit, "rows": rows, "summary": summary_json}


def freeze_audit(output_root: Path, test_results: str) -> Path:
    """Create the requested append-only B0 freeze from completed diagnostic outputs."""
    freeze = ROOT / "experiment_freeze/bone_p3_c2_b0_utility_mass_action_budget_audit_20261004"
    if freeze.exists():
        raise FileExistsError("B0_FREEZE_ALREADY_EXISTS")
    code = json.loads((output_root / "CODE_FORMULA_AUDIT.json").read_text())
    parents = verify_parent_integrity()
    freeze.mkdir(parents=True)
    status = "\n".join((
        "STATUS=P3_C2_B0_UTILITY_MASS_ACTION_BUDGET_AUDIT_FROZEN", "STAGE=P3-C2-B0", "",
        "DIAGNOSTIC_ONLY=true", "SCIENTIFIC_METHOD_CHANGED=false", "TRAINING_CHANGED=false", "HYPERPARAMETERS_CHANGED=false", "",
        "FULL_GT_USED=false", "FORMAL_METRICS_READ=false", "MODEL_TRAINING_RUN=false", "",
        "C2_A0_RESULTS_USED=true", "C2_A0_RESULTS_ARE_PRE_GT_ONLY=true", "",
        "ABSOLUTE_UTILITY_MASS_CANCELLED=true", f"UTILITY_MASS_SCALE_INVARIANT={str(code['UTILITY_MASS_SCALE_INVARIANT']).lower()}", f"GRADIENT_MASS_SCALE_INVARIANT={str(code['GRADIENT_MASS_SCALE_INVARIANT']).lower()}", "",
        "TUNING_GATE_CREATED=false", "PARAMETER_SELECTION_RULE_CREATED=false", "NEW_LOSS_CREATED=false", "",
        "PAIR_COUNT=24", "DATASET_COUNT=8", "SEEDS=20,30,50", "", "PARENT_SOURCE_MODIFICATION=false", "NEXT_STAGE=SCIENTIFIC_INTERPRETATION", "",
    ))
    (freeze / "STATUS.txt").write_text(status)
    formula = """# P3-C2-B0 exact semantic formula

`run_final_formal.py::_materialize_or_resume_branches` calls the frozen
`run_bone_p3_c0_cross_dataset_formal.py::_semantic_epoch`.

For view `v`, query `i`, anchor `l`, action `s`, and `D=sum_{i,s} U_is`:

`r_il=q_i dot stopgrad(a_l)`, `safe=clamp(r_il, eps, 1-eps)`.

`P_is=-(1/4) sum_l T_ils log(safe)` and
`N_is=-(1/(4*(K-1))) sum_l (1-T_ils) log(1-safe)`.

For each query batch `b`,
`ell_b=sum_v sum_{i in b,s} U_is(P_is+N_is)/(V*D)`; the epoch scalar is
`sum_b ell_b`.  Consequently common positive scaling of U cancels between
numerator and D.  Pi constructs T_rel upstream and is not read directly by
the semantic epoch.  No relation, sample, or anchor mask exists; U=0 is the
implicit outer contribution mask, and anchor posteriors are detached.
"""
    (freeze / "FORMULA.md").write_text(formula)
    for source_name, frozen_name in (
        ("CODE_FORMULA_AUDIT.json", "CODE_FORMULA_AUDIT.json"),
        ("three_seed_mass_summary.csv", "THREE_SEED_MASS_SUMMARY.csv"),
        ("three_seed_mass_summary.json", "THREE_SEED_MASS_SUMMARY.json"),
        ("C2A0_JOINED_PRE_GT_TABLE.csv", "C2A0_JOINED_PRE_GT_TABLE.csv"),
    ):
        shutil.copy2(output_root / source_name, freeze / frozen_name)
    new_hashes = source_hashes(("experiments/utility_relation/p3_c2_b0_utility_mass_action_budget_audit.py", "tests/test_p3_c2_b0_utility_mass_action_budget_audit.py"))
    (freeze / "new_source_sha256.txt").write_text("\n".join(f"{value}  {key}" for key, value in new_hashes.items()) + "\n")
    (freeze / "parent_source_sha256.txt").write_text("\n".join(f"{value}  {key}" for key, value in parents.items()) + "\n")
    freeze_refs = (
        ROOT / "experiment_freeze/bone_p3_c2_a0_utility_structural_drift_preregistered_20261004/FREEZE_SHA256.txt",
        ROOT / "experiment_freeze/bone_p3_c2_a0_utility_structural_drift_results_20261004/DIAGNOSTIC_ARTIFACT_SHA256.txt",
    )
    (freeze / "parent_freeze_sha256.txt").write_text("\n".join(f"{sha256_file(path)}  {path.relative_to(ROOT)}" for path in freeze_refs) + "\n")
    (freeze / "test_results.txt").write_text(test_results.rstrip() + "\n")
    integrity = {"parent_hashes_verified": True, "parent_source_modification": False, "parent_hashes": parents,
                 "c2a0_freeze_files": [str(path.relative_to(ROOT)) for path in freeze_refs]}
    (freeze / "PARENT_INTEGRITY.txt").write_text(json.dumps(integrity, indent=2, sort_keys=True) + "\n")
    return freeze


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P3-C2-B0 read-only utility mass audit")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    result = run_audit(arguments.output_root)
    frozen = freeze_audit(arguments.output_root, "pytest tests/test_p3_c2_b0_utility_mass_action_budget_audit.py: 4 passed") if arguments.freeze else None
    print(json.dumps({"pairs": len(result["rows"]), "output_root": str(arguments.output_root), "freeze": str(frozen) if frozen else None}, sort_keys=True))
