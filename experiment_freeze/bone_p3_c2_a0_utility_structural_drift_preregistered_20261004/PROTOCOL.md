# P3-C2-A0 Utility--Structural Drift Diagnostic

This is a descriptive, GT-free, read-only diagnostic over the already frozen
P3-C1-B1 final formal PRE-GT artifacts. It does not train, alter predictions,
alter hyperparameters, alter `U_cycle`, alter `T_rel`, or create any tuning
gate.

The primary population is `query_sample_ids`. BASE and OURS hard predictions
are joined by explicit sample ID; artifact row order is never trusted.

Cluster IDs are aligned by maximum-overlap rectangular Hungarian matching over
the complete shared sample set, mapping OURS clusters to BASE clusters. Any
unmatched OURS cluster receives a deterministic sentinel, counts as changed,
and remains included in occupancy mass.

Utility mass is exactly `sum(query_U_cycle[i, :])`. A sample is supported iff
this mass is strictly greater than zero and abstained iff it is exactly zero.
No epsilon, normalized support rule, threshold, or coverage gate is used.

The phase-B output is restricted to GT-free clustering agreement, flip,
utility-conditioned, rank-quartile, Spearman, and occupancy descriptions.

