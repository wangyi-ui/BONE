# P3-C1-B1-A1 Utility-Abstention Validation Amendment

## Scope and timing

The original P3-C1-B1 preregistration remains immutable. This versioned
amendment was triggered during pre-formal-tuning engineering feasibility,
before any Optuna study and before any formal ground-truth evaluation. No full
GT or ACC, NMI, Purity, or ARI value was inspected in forming this amendment.

The amendment applies identically to all eight registered datasets:
HandWrite, BDGP, MSRC, COIL20, Caltech-5V, MNIST-USPS, ALOI, and CCV. It is not
an MSRC-specific search rule.

## Unchanged scientific definitions

`U_cycle`, `T_rel`, the action space, sparse label identities, deterministic
two-fold split, learning-rate and exposure grids, dev and formal seeds,
carrier epochs, refinement horizon, four-way aggregation, tie-break, and
boundary-extension rule are unchanged. No epsilon, threshold, replacement
distribution, fallback prediction, or U recomputation is introduced.

Sparse labels continue to determine WHAT; frozen `U_cycle` continues to
determine WHETHER.

## Exact abstention equation

For validation anchor `i`, define `M_i = sum_s U_carrier[i,s]`. When `M_i>0`,
the original objective is unchanged:

`J_i = sum_{s:U_i,s>0} U_i,s log(K Pi_i,s,y_i) / M_i`.

When `M_i==0`, define exactly `J_i=0.0`. This is utility-conditioned
abstention: frozen utility supplies no valid action evidence. It states neither
that the prediction is correct nor that it is incorrect.

Both BASE and OURS apply this same rule, so an abstained anchor contributes
zero to both sides and hence zero to paired `DeltaJ=J_OURS-J_BASE`.

## Fold denominator and coverage

The fold score remains `J_fold=(1/K_val) sum_i J_i`. An abstained anchor stays
in the `K_val` denominator; the denominator is never replaced by the supported
anchor count.

Each fold records supported-anchor count, abstained-anchor count, and
`coverage=supported_anchor_count/K_val`. Coverage is audit-only: it is not an
Optuna objective, second objective, tie-break, parameter filter, or tunable
threshold.

## Class-level feasibility gate

For each dataset, development seed, and class, the two frozen sparse anchors
must satisfy `(M_1>0) OR (M_2>0)`. If both are zero, evaluation fails closed
with `CLASS_ZERO_UTILITY_EVIDENCE`. The protocol may not redraw labels, skip a
class, alter sparse IDs, add epsilon, or use full GT.

Apart from the exact zero-mass validation semantics above, every original B1
tuning, search, aggregation, selection, tie, and boundary rule is unchanged.
