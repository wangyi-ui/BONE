# P3-C1-B1-A1 Formal Optuna Tuning Runner Implementation

This freeze records an append-only, CPU/mock-tested implementation for the
formal development search.  It adds a read-only carrier context, the A1
class-feasibility gate, one cached 20-native-epoch BASE reference per dev seed,
four isolated candidate branches, a 25-point GridSampler/SQLite runner, cached
resume behavior, preregistered final selection, and read-only audit output.

Scientific constants are imported from the frozen protocol, amendment,
objective, and study-schema modules.  The implementation does not alter
WHAT/T_rel, frozen U_cycle, sparse folds, refinement horizon, tie-break, or
boundary-extension policy.  It never reads full GT and has no ACC/NMI/Purity/ARI
path.

The pre-implementation gate verified the R1 materializer freeze, all historical
manifests, and 16/16 formal carriers.  New tests pass 18/18.  Applicable frozen
regression passes 160 tests; two historical phase-only assertions are
deselected: one requiring the formal_tuning namespace to be absent and one
requiring carrier_adapter.py/trial_runner.py to be absent.  Those assertions
predate the already-frozen carrier and R1 implementation stages; neither test
nor any frozen source was modified.

No real BASE run, candidate refinement, Optuna study, 25-grid, formal seed, or
full-GT operation was executed.
