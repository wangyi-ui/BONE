# P3-C1-B1-A2 Final Hyperparameter Freeze

The final hyperparameters were selected from exactly 30 candidates
per dataset:

- 25 preregistered main-grid candidates;
- 5 preregistered upper-LR boundary-extension candidates.

The frozen lexicographic rule was:

1. maximize round(mean_delta_j, 12);
2. maximize round(worst_delta_j, 12);
3. maximize positive_count;
4. prefer smaller semantic_exposure;
5. prefer smaller semantic_lr.

The boundary-extension stage was executed exactly once.

No second boundary extension is allowed.

No full GT, ACC, NMI, or ARI was used for hyperparameter selection.

After this freeze, formal seed20/30/50 evaluation may not be used
to retune semantic_lr or semantic_exposure.
