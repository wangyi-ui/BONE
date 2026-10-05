# P3-C1-B1-A1 Amended MSRC Engineering Smoke

This freeze records one MSRC/dev-seed11 post-carrier refinement smoke for the
fixed candidate `semantic_lr=1e-5`, `semantic_exposure=5`. It reused the
already-completed 100-epoch carrier and did not run carrier training.

The frozen B1 runner could not be used directly because it rejects a
zero-utility validation anchor before refinement and hardcodes the original
objective in its evaluation path. The added `trial_runner_a1.py` is therefore
a thin orchestration wrapper. It calls the frozen isolated schedule, model
loader, native epoch, semantic epoch, and validation-posterior primitives. It
changes only final validation to the preregistered `objective_a1.py` rule.

Class feasibility was evaluated from the stored carrier `U_cycle` before
refinement and passed for all seven classes. BASE, OURS Fold A, and OURS Fold
B independently loaded the same carrier state and used isolated optimizers.
BASE ran 20 native epochs. Each OURS branch ran 5 semantic steps and 20 native
epochs under the frozen paired schedule.

Fold A retained all seven validation class slots: six supported anchors and
one abstention, for audit coverage 6/7. Raw row 175 was the class-5 validation
anchor, had utility mass zero, and contributed exactly zero to both BASE and
OURS. Fold B retained raw row 175 as the class-5 training anchor; its WHAT
signature remained the unweighted mean over stored `p_gen`, while all seven
validation anchors were supported.

Coverage was audit-only. DeltaJ sign was not a pass condition. No Optuna
study, formal tuning, seed37, other candidate, other dataset, formal seed,
full-GT load, or formal evaluator was used. Stored `U_cycle`, `T_rel`, sparse
labels, and both original carrier files were not modified.
