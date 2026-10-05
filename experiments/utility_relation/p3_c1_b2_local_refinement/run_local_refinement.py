"""B2 sequential local-grid runner, delegating all science to frozen B1 primitives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.utility_relation.p3_c1_b1_optuna import objective_a1, trial_runner_a1
from experiments.utility_relation.p3_c1_b1_optuna import formal_candidate_evaluator_r2 as frozen_evaluator
from experiments.utility_relation.p3_c1_b1_optuna.formal_base_reference import load_verified_base_reference
from experiments.utility_relation.p3_c1_b1_optuna.formal_tuning_context_a2 import build_formal_tuning_context_a2

from . import candidate_reuse
from .protocol import Candidate, DATASETS, DEV_SEEDS, FOLDS, candidates

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "outputs/bone_p3/p3_c1_b2_targeted_local_refinement"


def _new_candidate_result(context, candidate: Candidate) -> dict:
    bases = {seed: load_verified_base_reference(context, seed) for seed in DEV_SEEDS}
    results = {}
    for seed in DEV_SEEDS:
        carrier = context.carrier(seed)
        payloads = trial_runner_a1.build_fold_semantics_a1(carrier.arrays, carrier.adapter, carrier.folds)
        for fold in FOLDS:
            results[(seed, fold)] = frozen_evaluator._run_branch(carrier, fold, candidate, "cuda:0", payloads[fold])
    deltas, branches = {}, {}
    for seed in DEV_SEEDS:
        for fold in FOLDS:
            base = float(bases[seed].J_BASE_A if fold == "A" else bases[seed].J_BASE_B)
            result = results[(seed, fold)]
            ours = float(result["J_OURS"])
            delta = objective_a1.paired_delta_j(base, ours)
            deltas[(seed, fold)] = delta
            branches[f"{seed}{fold}"] = {"J_BASE": base, "J_OURS": ours, "DeltaJ": delta,
                "semantic_steps": result["semantic_steps"], "native_epochs": result["native_epochs"],
                "initial_state_sha256": result["initial_state_sha256"], "full_gt_loaded": False}
    aggregate = objective_a1.aggregate_candidate_scores(deltas)
    return {"candidate_origin": "NEW_B2_RUN", "candidate_path": None, "protocol_fingerprint": context.protocol_fingerprint,
            "carrier_state_sha256": {str(seed): context.carrier(seed).state_file_sha256 for seed in DEV_SEEDS},
            "carrier_pre_gt_sha256": {str(seed): context.carrier(seed).pre_gt_sha256 for seed in DEV_SEEDS},
            "branches": branches, "mean_delta_J": aggregate["mean_delta_J"], "worst_delta_J": aggregate["worst_delta_J"],
            "positive_count": aggregate["positive_count"], "full_gt_used": False}


def run_dataset(dataset: str, output_root: Path = OUTPUT_ROOT) -> list[dict]:
    if dataset not in DATASETS:
        raise ValueError("B2_UNREGISTERED_DATASET")
    preflight = candidate_reuse.preflight(dataset, candidates(dataset))
    context = build_formal_tuning_context_a2(dataset, candidate_reuse.B1_ROOT)
    target = output_root / dataset.lower()
    target.mkdir(parents=True, exist_ok=True)
    records = []
    by_key = {(entry["semantic_lr"], entry["semantic_exposure"]): entry for entry in preflight["entries"]}
    for candidate in candidates(dataset):
        destination = target / f"lr_{candidate.semantic_lr:.17g}__es_{candidate.semantic_exposure}.json"
        if destination.exists():
            record = json.loads(destination.read_text())
        else:
            plan = by_key[(candidate.semantic_lr, candidate.semantic_exposure)]
            if plan["reusable"]:
                source = plan["provenance"]
                evaluation = source.pop("evaluation")
                result = {**source, "branches": {key.replace("seed", "").replace("_fold", ""): value for key, value in evaluation["folds"].items()},
                          "mean_delta_J": evaluation["mean_delta_j"], "worst_delta_J": evaluation["worst_delta_j"], "positive_count": evaluation["positive_count"], "full_gt_used": False}
            else:
                result = _new_candidate_result(context, candidate)
            record = {"dataset": dataset, "semantic_lr": candidate.semantic_lr, "semantic_exposure": candidate.semantic_exposure,
                      "completion_state": "COMPLETE", **result}
            destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        records.append({"candidate": candidate, **record})
    if len(records) != len(candidates(dataset)) or any(item["completion_state"] != "COMPLETE" for item in records):
        raise RuntimeError("B2_INCOMPLETE_CANDIDATE_SET")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps({"completed": len(run_dataset(args.dataset, args.output_root)), "dataset": args.dataset}))
