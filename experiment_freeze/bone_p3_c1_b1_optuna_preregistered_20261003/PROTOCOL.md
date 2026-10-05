# P3-C1-B1 Dataset-Specific Semantic Exposure Selection

Status: preregistered CPU-only scientific protocol.  This stage creates no
training task, reads no data file or full ground truth, imports no Optuna, and
uses no GPU.  Its future sampler metadata is `GridSampler` only.

## Scope and invariant search space

The exact datasets are HandWrite, BDGP, MSRC, COIL20, Caltech-5V, MNIST-USPS,
ALOI, and CCV.  MSRC has the same scientific candidate space as every other
dataset.  All candidates use 100 carrier native epochs followed by a fixed
20-native-epoch refinement horizon.  The main grid is the Cartesian product of
semantic learning rates `(3e-6, 1e-5, 3e-5, 1e-4, 3e-4)` and semantic exposure
`(1, 3, 5, 10, 20)`, hence 25 candidates per dataset.  Seeds are not
hyperparameters and may never be searched.

Dev seeds are `(11, 37)`, formal seeds are `(20, 30, 50)`, and the two folds
are `A` and `B`; the dev and formal seed sets do not overlap.  Scores are
rounded to 12 decimal places only for tie-breaking.

For exposure `E_s`, refinement consists of `E_s` epochs of Semantic Phase A
then Native Phase B, followed by `20-E_s` Native-Phase-B-only epochs.  Thus
`E_s=1` is `1*(S->N)+19*N`; `3` is `3*(S->N)+17*N`; `5` is
`5*(S->N)+15*N`; `10` is `10*(S->N)+10*N`; and `20` is `20*(S->N)`.
Unregistered values, `E_s<=0`, and `E_s>20` fail closed.

## Sparse-label two-fold validation

Only the frozen two sparse labeled rows per class and their sparse labels are
permitted.  Full GT and unlabeled GT are prohibited.  Within each class the
two rows are ordered by raw file row id as `l_c^(1)<l_c^(2)`.  Fold A trains on
every `l_c^(1)` and validates on every `l_c^(2)`; Fold B reverses them.  Each
fold has exactly one training and one validation anchor per class; train and
validation are disjoint; across folds every sparse row occurs once in each
role.  The split is deterministic and input-order invariant.  Duplicate raw
ids, unequal row/label shapes, any class other than exactly two labels, and a
missing expected class fail closed.  A canonical SHA256 split audit records
future split provenance.

## Frozen WHAT semantics

For fold training anchor `l`, the anchor signature is
`a_l=(1/S)*sum_s p_gen(l,s)`, with no U weighting.  Each class carrier is
`C_f[c]=mean_{l in class c training anchors} a_l`; at one anchor per class it
is simply `a_l`.  Define `E[i,s,c]=<p_gen(i,s),C_f[c]>`,
`D[i,s]=sum_c E[i,s,c]`, `Pi[i,s,c]=E[i,s,c]/D[i,s]`, and
`T_rel[i,l,s]=Pi[i,s,y_l]`.  There is no epsilon, temperature, or extra
softmax.  `U_cycle` is frozen/detached.  A fold's validation anchor label must
never enter its training `T_rel` construction.

## UCSV validation objective and paired gain

For `U_carrier_val[K_val,S]`, `Pi_val[K_val,S,K]`, and sparse
`val_targets[K_val]`, where `K_val=K`, compute for validation sample `i`:

`J_i = sum_s U_carrier[i,s]*log(K*Pi_val[i,s,y_i]) / sum_s U_carrier[i,s]`.

The fold score is `J_{r,f}=(1/K_val)*sum_i J_i`.  Carrier utility is frozen
and cannot be trial-updated.  If any validation utility mass is nonpositive,
the policy is `ZERO_UTILITY_POLICY=FAIL_CLOSED`: no epsilon, skipping, class
removal, unweighted substitute, or objective replacement is allowed.  A
nonfinite input/output fails closed.  A nonpositive true probability at a
positive-utility action fails closed.  Zero-utility actions are omitted from
the logarithm so `0*log(0)` cannot create NaN.  Chance probability `1/K`
gives exactly `log(K*(1/K))=0`.

For each dataset/dev-seed/fold, BASE starts from the same frozen carrier and
runs 20 native refinement epochs.  OURS(h) runs the registered exposure
schedule.  The paired score is `DeltaJ[r,f,h]=J_OURS[r,f,h]-J_BASE[r,f]`.
For each candidate its four dev scores are `(11,A),(11,B),(37,A),(37,B)`;
the primary future objective maximizes their mean.  Also record their minimum
as `worst_delta_J` and the number strictly positive as `positive_count` in
`0..4`.  Feasibility is checked for every dataset/dev-seed/fold validation
sample before tuning; any zero utility mass fails that dataset's B1 tuning.

## Deterministic selection and boundaries

Candidates rank lexicographically by: maximum rounded `mean_delta_J`; then
maximum rounded `worst_delta_J`; then maximum `positive_count`; then smaller
`E_s`; then smaller semantic learning rate.  No human selection is allowed.
Absolute OURS J is not the selection objective.

After the primary grid and before any formal GT evaluation, if any of the
eight provisional best rates is the lower boundary `3e-6`, all eight datasets
share one added `1e-6` rate crossed with every registered exposure.  If any is
the upper boundary `3e-4`, all eight share one added `1e-3` rate crossed with
every exposure.  Both may be added together.  There is exactly one extension
stage and no dataset-specific extension.  If a provisional or final best
exposure is 20, record `SEMANTIC_EXPOSURE_UPPER_BOUND_HIT=true`; do not add
30 or 40 because that would change the fixed refinement horizon.

Full GT, ACC, NMI, Purity, and ARI are forbidden as a training signal,
validation objective, model-selection rule, hyperparameter-selection rule,
early-stopping rule, or tie-break.  The only supervision permitted for B1
selection is the frozen two sparse labels per class.
