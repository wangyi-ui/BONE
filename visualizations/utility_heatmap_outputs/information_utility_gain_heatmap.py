#!/usr/bin/env python3
"""Render the supplied 48 information-utility gains without transforming them."""
from __future__ import annotations

import csv
import io
import os
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

# Verbatim values supplied in the request; this is the only data source.
SOURCE_CSV = """Dataset,SNR_dB,Delta_ACC,Delta_NMI,Delta_ARI
BDGP,0,-0.1467,0.0135,-0.2100
BDGP,2.5,0.2667,0.5006,0.4424
BDGP,5,0.0267,0.0804,0.0679
BDGP,10,0.0533,0.1499,0.1322
COIL20,0,0.9259,0.8188,1.2415
COIL20,2.5,0.7176,0.5638,1.0285
COIL20,5,0.5324,0.3563,0.6638
COIL20,10,0.4398,0.6964,0.9504
MNIST-USPS,0,0.0333,0.0884,0.0722
MNIST-USPS,2.5,0.1200,0.2961,0.2602
MNIST-USPS,5,0.0067,0.0565,0.0128
MNIST-USPS,10,0.0467,0.1357,0.1024
CCV,0,-0.2412,-0.1638,-0.1239
CCV,2.5,0.4872,0.1903,0.2523
CCV,5,0.6053,0.4451,0.3293
CCV,10,0.4823,0.0677,0.1579
"""
DATASETS = ("BDGP", "COIL20", "MNIST-USPS", "CCV")
SNR = ("0", "2.5", "5", "10")
METRICS = ("ACC", "NMI", "ARI")


def load_and_validate():
    rows = list(csv.DictReader(io.StringIO(SOURCE_CSV)))
    assert len(DATASETS) == 4 and len(SNR) == 4 and len(METRICS) == 3
    assert len(rows) == 16
    assert [row["Dataset"] for row in rows] == [d for d in DATASETS for _ in SNR]
    assert [row["SNR_dB"] for row in rows] == list(SNR) * 4
    matrices = {}
    for metric in METRICS:
        raw = [row[f"Delta_{metric}"] for row in rows]
        assert len(raw) == 16 and all(len(s.split(".")[1]) == 4 for s in raw)
        matrices[metric] = np.array([float(s) for s in raw], dtype=float).reshape(4, 4)
        assert np.isfinite(matrices[metric]).all()
        assert [f"{v:.4f}" for v in matrices[metric].flat] == raw
    assert sum(a.size for a in matrices.values()) == 48
    assert max(abs(a).max() for a in matrices.values()) <= 1.3
    assert matrices["ACC"][0, 0] == -0.1467
    assert matrices["ACC"][3, 0] == -0.2412
    assert matrices["ARI"][1, 0] == 1.2415
    return matrices


def signed_four(value):
    return f"{value:+.4f}".replace("-", "−")


def ink_color(rgba):
    # WCAG relative luminance; pick black/white for readable annotations.
    rgb = np.asarray(rgba[:3])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = float(np.dot(linear, (0.2126, 0.7152, 0.0722)))
    return "black" if luminance > 0.179 else "white"


def main():
    matrices = load_and_validate()
    csv_path = OUTPUT_DIR / "information_utility_gain_values.csv"
    csv_path.write_text(SOURCE_CSV, encoding="utf-8", newline="")
    assert csv_path.read_text(encoding="utf-8") == SOURCE_CSV

    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # editable text in SVG
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }):
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0), dpi=150)
        fig.subplots_adjust(left=0.075, right=0.906, bottom=0.18, top=0.87, wspace=0.39)
        cmap = plt.get_cmap("RdBu")  # negative red; zero white; positive blue
        norm = Normalize(vmin=-1.3, vmax=1.3)
        for index, (ax, metric) in enumerate(zip(axes, METRICS)):
            data = matrices[metric]
            # pcolormesh exports vector cells in PDF and SVG; no interpolation.
            ax.pcolormesh(np.arange(5), np.arange(5), data, cmap=cmap, norm=norm,
                          edgecolors="white", linewidth=0.5, shading="flat")
            ax.set_xlim(0, 4)
            ax.set_ylim(4, 0)
            ax.set_aspect("equal")
            ax.set_xticks(np.arange(4) + 0.5, [f"{s} dB" for s in SNR])
            ax.set_yticks(np.arange(4) + 0.5, DATASETS)
            ax.tick_params(length=0, pad=5)
            ax.set_xlabel("SNR (dB)", labelpad=8)
            ax.set_title(f"({chr(97 + index)})  Δ{metric}", pad=11, fontweight="semibold")
            for spine in ax.spines.values():
                spine.set_visible(False)
            for row in range(4):
                for col in range(4):
                    value = data[row, col]
                    ax.text(col + 0.5, row + 0.5, signed_four(value),
                            ha="center", va="center", fontsize=8.3,
                            color=ink_color(cmap(norm(value))), fontweight="medium")
        colorbar_ax = fig.add_axes((0.925, 0.235, 0.014, 0.595))
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = fig.colorbar(mapper, cax=colorbar_ax,
                                ticks=[-1.3, -0.65, 0, 0.65, 1.3])
        colorbar.ax.set_yticklabels(["−1.30", "−0.65", "0", "+0.65", "+1.30"])
        colorbar.set_label("OURS − UNIFORM-U", rotation=90, labelpad=10)
        colorbar.outline.set_linewidth(0.5)
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(OUTPUT_DIR / f"information_utility_gain_heatmap.{suffix}",
                        dpi=600, facecolor="white")
        plt.close(fig)
    print("Validated 4 datasets × 4 SNR conditions × 3 metrics = 48 supplied values.")
    print("Wrote CSV and PNG (600 DPI), vector PDF, editable SVG.")


if __name__ == "__main__":
    main()
