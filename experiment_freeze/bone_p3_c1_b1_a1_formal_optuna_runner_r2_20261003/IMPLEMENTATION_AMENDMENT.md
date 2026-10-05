# Formal Optuna Runner R2 candidate-evaluator import amendment

Root cause: the frozen `formal_candidate_evaluator.py` calls
`trial_runner_a1.build_fold_semantics_a1(...)` but omitted the corresponding
module import. R1 Trial 0 therefore failed with `NameError` before any candidate
refinement. This is an implementation dependency defect, not a scientific
protocol failure.

R2 is append-only. `formal_candidate_evaluator_r2.py` explicitly imports
`trial_runner_a1`, binds that single missing dependency into the frozen
evaluator module, and delegates all evaluation and cache behavior to the
frozen implementation. It does not reimplement `CandidateEvaluation`, A1,
fold construction, training schedules, U_cycle, T_rel, sparse-label behavior,
or aggregation.

`run_formal_optuna_tuning_r2.py` preserves the R1 ordering: build the context
and run its A1 feasibility gate; bootstrap and re-verify BASE seeds 11 and 37;
then create or load the isolated `study_r2.db` namespace named
`p3_c1_b1_a1__<dataset_slug>__main_r2`. The naming rule applies uniformly to
all eight datasets. Historical `study.db` and `study_r1.db` are never opened by
the R2 runner.

Tests use only CPU, mocks, and `tmp_path`. They cover the real frozen
fold-builder resolution path, four branches (11A, 11B, 37A, 37B), frozen
`CandidateEvaluation` semantics, verified BASE reuse without retraining and
with byte-identical artifacts, clean R2 study creation, exact 25-grid coverage,
7-to-25 SQLite resume, candidate-cache reuse, and no-GT contracts.

No real BASE training, candidate refinement, GPU run, formal-seed run,
production Optuna run, production `study_r2.db`, or full GT access occurred.

