# A2 scientific feasibility amendment

A2 was introduced after pre-GT utility-coverage diagnostics revealed that the
A1 requirement of at least one utility-supported sparse anchor in every class
becomes structurally restrictive for large-K datasets.

The amendment does not relax U_cycle, does not alter the semantic target, and
does not introduce a coverage threshold. Instead, it makes the feasibility
rule consistent with the already-preregistered A1 abstention semantics:
zero-utility validation evidence contributes zero, while the class-balanced
denominator remains fixed at K.

The two frozen sparse anchors of every class continue to define WHAT through
the unchanged unweighted class-signature construction, even when both anchors
have zero utility. U_cycle continues to determine WHETHER validation evidence
is supported. No class is deleted, no objective is renormalized, and no label
or anchor is replaced.

The amendment was defined without consulting ACC, NMI, ARI, or full ground
truth. ALOI and CCV pre-GT diagnostics motivated the feasibility revision but
were not used to tune a threshold; no threshold exists.

This is a scientific protocol change limited specifically to the formal
pre-tuning feasibility condition. The objective, search space, selection
rule, U_cycle, T_rel, WHAT semantics, and training schedule are unchanged.

