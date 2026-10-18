#!/usr/bin/env python3
"""Render the supplied gain data as an unannotated, shared-scale heatmap."""
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from information_utility_gain_heatmap import load_and_validate, DATASETS, SNR, METRICS
import numpy as np


def main():
    data = load_and_validate()  # Checks all 48 supplied values; does not transform them.
    with plt.rc_context({
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.titlesize': 11, 'axes.labelsize': 9,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    }):
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0), dpi=150)
        fig.subplots_adjust(left=.075, right=.906, bottom=.18, top=.87, wspace=.39)
        cmap = plt.get_cmap('RdBu')
        norm = Normalize(vmin=-1.3, vmax=1.3)
        for i, (ax, metric) in enumerate(zip(axes, METRICS)):
            ax.pcolormesh(np.arange(5), np.arange(5), data[metric],
                          cmap=cmap, norm=norm, shading='flat',
                          edgecolors='white', linewidth=.5)
            ax.set(xlim=(0, 4), ylim=(4, 0), aspect='equal')
            ax.set_xticks(np.arange(4) + .5, [f'{s} dB' for s in SNR])
            ax.set_yticks(np.arange(4) + .5, DATASETS)
            ax.tick_params(length=0, pad=5)
            ax.set_xlabel('SNR (dB)', labelpad=8)
            ax.set_title(f'({chr(97+i)})  Δ{metric}', pad=11,
                         fontweight='semibold')
            for spine in ax.spines.values():
                spine.set_visible(False)
        cax = fig.add_axes((.925, .235, .014, .595))
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                          cax=cax, ticks=[-1.3, -.65, 0, .65, 1.3])
        cb.ax.set_yticklabels(['−1.30', '−0.65', '0', '+0.65', '+1.30'])
        cb.set_label('OURS − UNIFORM-U', rotation=90, labelpad=10)
        cb.outline.set_linewidth(.5)
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(OUT / f'information_utility_gain_heatmap_no_values.{ext}',
                        dpi=600, facecolor='white')
        plt.close(fig)
    print('Saved no-value PNG, PDF, and SVG from the 48 validated values.')


if __name__ == '__main__':
    main()
