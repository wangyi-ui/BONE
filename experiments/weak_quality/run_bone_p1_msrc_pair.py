"""Paired native-BONE runner for the fixed MSRC weak-quality artifact.

This module deliberately has no import-time execution.  Its CLI requires --execute,
so materialization and tests cannot accidentally start the formal paired experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from experiments.weak_quality.bone_p1_msrc_protocol import FEATURE_DIMS, K, N, V
from method.amv import amv
from method.bone import BONE
from utils.data_loader import RandDatasetMV
from utils.method_util import get_cluster_metrics
from utils.seed_util import set_seed


EPOCHS = 100
BATCH_SIZE = 128
N_ANCHOR = 3
LAYER = 5
LAMBDA = 0.1
LR = 0.01
DEVICE = torch.device("cpu")
DEFAULT_ARTIFACT = Path("outputs/bone_p1/msrc_snr2p5_exact2of4_seed20/bone_p1_msrc_corruption.npz")


def load_arm_views(artifact_path: Path | str, prefix: str) -> List[np.ndarray]:
    """Load [N, D_v] numeric arm inputs without touching labels."""
    if prefix not in {"x_clean", "x_corrupt"}:
        raise ValueError("prefix must be x_clean or x_corrupt")
    with np.load(artifact_path, allow_pickle=False) as archive:
        views = [np.ascontiguousarray(archive[f"{prefix}_v{view_index}"]) for view_index in range(V)]
    for view_index, view in enumerate(views):
        if view.shape != (N, FEATURE_DIMS[view_index]) or not np.isfinite(view).all():
            raise ValueError(f"invalid {prefix}_v{view_index} in artifact")
    return views


def _amv_container(views: Sequence[np.ndarray]) -> SimpleNamespace:
    """Adapt artifact [N,D] arrays to AMV's native [D,N] read-only interface."""
    return SimpleNamespace(data=[view.T for view in views], m_list=list(FEATURE_DIMS), v=V, k=K)


def _train_arm(views: Sequence[np.ndarray], training_seed: int) -> np.ndarray:
    """Train one native BONE arm and return predicted clusters; labels never enter here."""
    # Reset every stochastic source at the start of each arm to keep the pair exact.
    set_seed(training_seed)
    native_data = _amv_container(views)
    anchors = amv(native_data, m=N_ANCHOR, ret_anchor=True, max_iter=100, seed=training_seed,
                  info=True, info_iter=False)
    # RandDatasetMV requires a label field, so feed fixed dummy values rather than GT labels.
    dummy_labels = np.zeros(N, dtype=np.float32)
    train_loader = DataLoader(
        dataset=RandDatasetMV(native_data.data, dummy_labels, device=DEVICE),
        batch_size=BATCH_SIZE,
        generator=torch.Generator().manual_seed(training_seed),
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )
    model = BONE(m_list=list(FEATURE_DIMS), k=K, a=N_ANCHOR, v=V, anchor_init=anchors,
                 layer=LAYER, lamb=LAMBDA, seed=training_seed, device=DEVICE)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    for _epoch in range(EPOCHS):
        model.train()
        for data_batch, _dummy_label_batch in train_loader:
            optimizer.zero_grad()
            result = model(data_batch)
            result["loss"].backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        result = model([torch.from_numpy(view).to(device=DEVICE, dtype=torch.float32) for view in views], test=True)
    return result["label"].cpu().numpy()


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    acc, nmi, purity = get_cluster_metrics(labels.copy(), predictions, nmi=False)
    return {"acc": float(acc), "nmi": float(nmi), "purity": float(purity)}


def run_paired_experiment(
    artifact_path: Path | str, output_dir: Path | str, training_seed: int = 20
) -> Dict[str, object]:
    """Run exactly CLEAN_BASE and CORRUPT_BASE, then use GT only for final metrics."""
    artifact_path, output_dir = Path(artifact_path), Path(output_dir)
    clean_predictions = _train_arm(load_arm_views(artifact_path, "x_clean"), training_seed)
    corrupt_predictions = _train_arm(load_arm_views(artifact_path, "x_corrupt"), training_seed)
    # GT labels are read only after both arms' optimization has completed.
    with np.load(artifact_path, allow_pickle=False) as archive:
        labels = np.ascontiguousarray(archive["labels"])
    clean_metrics = _metrics(labels, clean_predictions)
    corrupt_metrics = _metrics(labels, corrupt_predictions)
    deltas = {name: clean_metrics[name] - corrupt_metrics[name] for name in ("acc", "nmi", "purity")}
    if deltas["acc"] >= 0.03 and (deltas["nmi"] >= 0.01 or deltas["purity"] >= 0.01):
        verdict = "PASS"
    elif deltas["acc"] <= 0.0:
        verdict = "FAIL_CLOSED"
    else:
        verdict = "INCONCLUSIVE"
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "clean": output_dir / "clean_metrics.json",
        "corrupt": output_dir / "corrupt_metrics.json",
        "summary": output_dir / "paired_summary.json",
        "audit": output_dir / "run_audit.json",
    }
    files["clean"].write_text(json.dumps(clean_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files["corrupt"].write_text(json.dumps(corrupt_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"delta_acc": deltas["acc"], "delta_nmi": deltas["nmi"], "delta_purity": deltas["purity"],
               "verdict": verdict}
    files["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_audit = {
        "arms": ["CLEAN_BASE", "CORRUPT_BASE"],
        "only_input_difference": "x_clean_v* versus x_corrupt_v*",
        "training_seed_reset_per_arm": training_seed,
        "dataloader": {"generator_seed": training_seed, "num_workers": 0, "shuffle": True, "drop_last": True},
        "native_bone": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "n_anchor": N_ANCHOR, "layer": LAYER,
                        "lambda": LAMBDA, "lr": LR, "device": "cpu"},
        "artifact_standardized_input_used_without_reprocessing": True,
        "gt_labels_used_only_after_training_for_metrics": True,
    }
    files["audit"].write_text(json.dumps(run_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"clean": clean_metrics, "corrupt": corrupt_metrics, "summary": summary, "files": files}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, choices=[20, 30, 50], default=20)
    parser.add_argument("--execute", action="store_true", help="run the two formal training arms")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.execute:
        parser.error("refusing to start formal training without --execute")
    result = run_paired_experiment(args.artifact, args.output_dir, args.training_seed)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
