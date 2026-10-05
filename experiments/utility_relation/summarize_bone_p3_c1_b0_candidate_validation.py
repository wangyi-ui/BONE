"""
Three-seed summarizer for one held-out validation dataset.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from experiments.utility_relation import (
    bone_p3_c0_cross_dataset_formal_protocol as c0_protocol,
)
from experiments.utility_relation import (
    bone_p3_c1_b0_candidate_protocol as candidate,
)
from experiments.utility_relation import (
    run_bone_p3_c1_b0_candidate_validation as runner,
)


def summarize(
    dataset: str,
) -> dict:

    rows = []

    for seed in candidate.SEEDS:

        path = (
            runner.root_for(
                dataset,
                seed,
            )
            / "postseal_metrics.json"
        )

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        rows.append(
            json.loads(
                path.read_text()
            )
        )

    metrics = (
        "ACC",
        "NMI",
        "Purity",
        "ARI",
    )

    summary = {
        "dataset":
            dataset,

        "semantic_lr":
            candidate.SEMANTIC_LR,

        "refinement_budget":
            candidate.REFINEMENT_BUDGET,

        "hyperparameters_selected_on":
            "ALOI_only",

        "seeds":
            list(
                candidate.SEEDS
            ),

        "metrics":
            {},
    }

    print()
    print("=" * 112)
    print(
        "P3-C1-B0 HELD-OUT VALIDATION SUMMARY"
    )
    print(
        "dataset =",
        dataset,
    )
    print("=" * 112)

    for metric in metrics:

        base = np.asarray(
            [
                row["BASE"][metric]
                for row in rows
            ],
            dtype=np.float64,
        )

        ours = np.asarray(
            [
                row["OURS"][metric]
                for row in rows
            ],
            dtype=np.float64,
        )

        delta = (
            ours - base
        )

        info = {
            "BASE_mean":
                float(
                    base.mean()
                ),

            "OURS_mean":
                float(
                    ours.mean()
                ),

            "delta_mean":
                float(
                    delta.mean()
                ),

            "delta_std":
                float(
                    delta.std()
                ),

            "worst_seed_delta":
                float(
                    delta.min()
                ),

            "positive_seed_count":
                int(
                    np.sum(
                        delta > 0
                    )
                ),

            "seed20_delta":
                float(
                    delta[0]
                ),

            "seed30_delta":
                float(
                    delta[1]
                ),

            "seed50_delta":
                float(
                    delta[2]
                ),
        }

        summary[
            "metrics"
        ][metric] = info

        print(
            f"{metric:7s} "
            f"BASE={info['BASE_mean']:.8f} "
            f"OURS={info['OURS_mean']:.8f} "
            f"delta={info['delta_mean']:+.8f} "
            f"worst={info['worst_seed_delta']:+.8f} "
            f"positive={info['positive_seed_count']}/3 "
            f"[20={info['seed20_delta']:+.8f}, "
            f"30={info['seed30_delta']:+.8f}, "
            f"50={info['seed50_delta']:+.8f}]"
        )

    out = (
        candidate.OUTPUT_ROOT
        / c0_protocol.dataset_slug(
            dataset
        )
        / "candidate_validation_summary.json"
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print(
        "P3_C1_B0_DATASET_SUMMARY_COMPLETE"
    )

    return summary


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=candidate.VALIDATION_DATASETS,
        required=True,
    )

    args = parser.parse_args()

    summarize(
        args.dataset
    )


if __name__ == "__main__":
    main()
