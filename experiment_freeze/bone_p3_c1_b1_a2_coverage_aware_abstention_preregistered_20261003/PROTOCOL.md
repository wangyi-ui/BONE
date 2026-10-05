# P3-C1-B1-A2 protocol

For sparse anchor `i`, retain the frozen utility mass:

`M_i = sum_s U_cycle[i,s]`.

There is no epsilon, fallback, threshold, unweighted substitution, label
reselection, or anchor reselection.

The frozen A1 validation objective remains unchanged. If `M_i > 0`, the
utility-weighted log relation score is used. If `M_i == 0`, that anchor's
contribution is exactly zero. Every validation fold denominator remains the
dataset class count `K`, including when an entire class is utility-abstained.

For each class `c` with its two frozen sparse anchors `L_c`, define the
audit-only indicator:

`S_c = 1[max_{i in L_c} M_i > 0]`.

`S_c == 0` names a `UTILITY_ABSTAINED_CLASS`. It does not remove the class,
change WHAT, gate the class signature, alter T_rel, or affect search,
selection, candidate filtering, or dataset filtering.

The only A2 feasibility rule is evaluated independently per dev seed:

`sum_i 1[M_i > 0] > 0`.

If false, validation fails closed with
`NO_UTILITY_SUPPORTED_VALIDATION_EVIDENCE`. No percentage or class-coverage
threshold exists.

All training and search constants remain frozen: GridSampler over the same
5x5 semantic-LR/exposure grid, dev seeds 11 and 37, 20 native refinement
epochs, `n_jobs=1`, unchanged A1 objective, unchanged tie break, and unchanged
boundary-extension protocol.

