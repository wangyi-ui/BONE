# P3-C1-B1 GPU Bridge Engineering Smoke

This append-only bridge connects the frozen B1 protocol to existing P3-B0 and
P3-C0 primitives.  It reuses frozen raw-view loading, sparse/corruption input
loading, raw-ID preprocessing, BONE construction, native loader/epoch,
semantic epoch, posterior extraction, action construction, and U-cycle
calculation.  No parent source was changed.

The only real run was Caltech-5V, dev seed 11, semantic learning rate `1e-5`,
and semantic exposure 5 on `cuda:0`.  The carrier completed 100 native epochs.
BASE, Fold-A OURS, and Fold-B OURS were isolated clones of the identical
carrier state.  BASE completed 20 native epochs.  Each OURS branch completed
five Semantic-to-Native epochs followed by fifteen native-only epochs.

The frozen sparse two-fold splitter supplied one train and one validation
anchor per class.  Validation anchors were excluded from semantic-training
rows.  Carrier U, class signatures, training Pi, and T_rel were frozen before
refinement.  Final validation scores called the frozen B1 `validation_j` and
`paired_delta_j` functions.  No four-score aggregation was fabricated because
seed 37 was not run.

Both zero-utility gates passed, all J/DeltaJ values were finite, and the result
was declared an engineering pipeline pass independently of DeltaJ sign.  Full
ground truth and clustering metrics were not loaded or computed.  No Optuna
study, grid search, second candidate, second seed, or other dataset was run.
