# Formal Optuna Runner R1 orchestration amendment

Root cause: the frozen runner created the Optuna study before materializing
the two required BASE references.  Trial 0 therefore failed at
`BASE_REFERENCE_MISSING`, before candidate refinement.  This is an
implementation-orchestration defect, not a scientific protocol failure.

R1 is append-only.  The new wrapper first builds the read-only formal context,
executes the frozen A1 feasibility gate, materializes and re-verifies BASE for
dev seeds 11 and 37, and only then creates a versioned Optuna namespace at
`study_r1.db` with study name `p3_c1_b1_a1__<dataset_slug>__main_r1`.

The existing `study.db` is never opened for writing, deleted, resumed, or
edited.  Candidate evaluation, GridSampler, SQLite resume, and cache semantics
continue to use the frozen implementation.  No scientific constants, folds,
U_cycle, T_rel, search space, or A1 semantics changed.

R1 tests use only CPU/mock/tmp_path artifacts.  No BASE training, candidate
refinement, GPU run, formal seed, full GT, or production `study_r1.db` was
created.
