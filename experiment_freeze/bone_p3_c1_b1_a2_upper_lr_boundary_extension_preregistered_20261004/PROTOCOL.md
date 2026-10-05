# P3-C1-B1-A2 Preregistered Upper-LR Boundary Extension

This freeze implements and tests the already-preregistered, single-stage
upper-learning-rate boundary rule. It does not execute any real candidate.

The upper-boundary extension is not a post-hoc response invented after
observing BDGP. The preregistered rule states that if any dataset's provisional
optimum reaches the upper main-grid LR boundary `3e-4`, the same `LR=1e-3`
exposure grid is added to all eight datasets before formal GT evaluation.
BDGP reached that boundary. No dataset reached the lower `3e-6` boundary, so
the lower `1e-6` extension is not activated.

The activated grid is exactly
`semantic_lr=1e-3` crossed with
`semantic_exposure in {1,3,5,10,20}`: five candidates per dataset and forty
candidates total. It is uniform across HandWrite, BDGP, MSRC, COIL20,
Caltech-5V, MNIST-USPS, ALOI, and CCV. There is no adaptive search and no
second extension beyond `1e-3`, even if `1e-3` later wins.

The execution route is frozen as A2 context and feasibility, preserving R3
numeric class-order canonicalization, then verified existing BASE references,
frozen candidate evaluator R2, frozen `trial_runner_a1`, and frozen
`objective_a1`. BASE and the original 25 candidates are reused without
retraining. Extension persistence is isolated under
`boundary_extension_upper_1e3/study_boundary_ext.db` and uses an exact
five-point Optuna GridSampler with `n_jobs=1` and resumable SQLite state.

After a future real extension run, the finalizer requires exactly 25 original
and 5 extension candidates. It applies the unchanged rounded lexicographic
rule: maximum 12-decimal rounded mean Delta J, then maximum 12-decimal rounded
worst Delta J, then maximum positive count, then smaller semantic exposure,
then smaller semantic LR.

The extension uses no GT-based criterion and is performed exactly once. No
ACC, NMI, ARI, or full GT may enter selection. The evaluated pool is extended
from 25 to 30, so the accurate status is
`PREREGISTERED_SEARCH_SPACE_EXTENSION_ACTIVATED=true`,
`POSTHOC_SEARCH_SPACE_CHANGE=false`, and
`SCIENTIFIC_PROTOCOL_CHANGED=false`.
