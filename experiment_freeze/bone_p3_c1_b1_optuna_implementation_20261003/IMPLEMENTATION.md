# P3-C1-B1 Optuna Infrastructure Implementation Freeze

This implementation layer is append-only and imports the preregistered
`protocol` and `objective` modules as the sole scientific source of truth.  It
does not duplicate the dataset registry, candidate grid, seeds, horizon,
tie-break rule, or boundary-extension rule.

`study_schema.py` provides validated registered candidate records, canonical
keys, and four-fold paired-gain evaluations.  Aggregation delegates to frozen
`objective.aggregate_candidate_scores`.

`optuna_driver.py` uses Optuna `GridSampler` with only `semantic_lr` and
`semantic_exposure`, fixed `n_jobs=1`, maximize direction, deterministic study
names, SQLite storage, `load_if_exists=True`, persisted protocol fingerprints,
resume support, coverage auditing, and final selection via frozen
`objective.select_best_candidate` rather than `study.best_trial`.

`audit.py` exports JSON-compatible GT-free study audit records and delegates
boundary-extension and exposure-20 detection to frozen protocol helpers.
`summarize.py` exports only completed trial information and the preregistered
selection result; its best-parameter export records sparse dev/formal seeds and
`full_gt_used_for_selection=false`.

All tests used deterministic injected synthetic evaluators and pytest temporary
SQLite paths.  No dataset was opened, no carrier/model was constructed, no
training or formal tuning occurred, and no CUDA API was used.

Validation result: preregistered tests `30 passed`; new implementation tests
`11 passed`; combined B1 tests `41 passed`.  The SQLite test completed 7 new
trials, reopened the same study, completed all 25 unique candidates, and
matched the selected candidate from an uninterrupted 25-trial study.
