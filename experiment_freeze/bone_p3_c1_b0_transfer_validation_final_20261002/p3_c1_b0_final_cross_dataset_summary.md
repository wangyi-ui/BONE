# P3-C1-B0 Final Cross-Dataset Summary

- Tuning dataset: ALOI
- Held-out datasets: Caltech-5V, BDGP, HandWrite, COIL20, MNIST-USPS, CCV
- Frozen candidate: semantic_lr=1e-5, refinement_budget=5
- Held-out dataset retuning: NO

| Dataset | Role | ΔACC | ΔNMI | ΔPurity | ΔARI | ACC +seeds | NMI +seeds | Purity +seeds | ARI +seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ALOI | tuning | +0.01564815 | +0.01055056 | +0.01327160 | +0.01726837 | 3/3 | 3/3 | 3/3 | 3/3 |
| Caltech-5V | held_out | +0.00142857 | +0.00096573 | +0.00095238 | +0.00122233 | 2/3 | 1/3 | 2/3 | 1/3 |
| BDGP | held_out | +0.00013333 | +0.00008372 | +0.00013333 | +0.00015005 | 1/3 | 1/3 | 1/3 | 1/3 |
| HandWrite | held_out | -0.00016667 | +0.00098305 | -0.00016667 | -0.00027566 | 1/3 | 2/3 | 1/3 | 1/3 |
| COIL20 | held_out | -0.00324074 | -0.00468363 | -0.00532407 | -0.00768814 | 2/3 | 1/3 | 1/3 | 1/3 |
| MNIST-USPS | held_out | +0.00166667 | +0.00333980 | +0.00166667 | +0.00354645 | 2/3 | 2/3 | 2/3 | 2/3 |
| CCV | held_out | -0.00334662 | -0.00170863 | -0.00172253 | -0.00005675 | 0/3 | 0/3 | 1/3 | 1/3 |

## Frozen conclusion

The ALOI-selected configuration is not a universally optimal cross-dataset configuration.

The shared U_cycle/T_rel mechanism shows dataset-dependent sensitivity to semantic intervention strength and duration.

Next stage: P3-C1-B1 dataset-specific semantic exposure selection with a shared search space and shared selection rule.
