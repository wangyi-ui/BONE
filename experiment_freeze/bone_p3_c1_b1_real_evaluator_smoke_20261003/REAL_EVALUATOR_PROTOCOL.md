# P3-C1-B1 Real Evaluator Integration Smoke

This append-only stage ran exactly one new real path: Caltech-5V, development
seed 37, semantic learning rate `1e-5`, and exposure 5 on CUDA device 0.  It
created a distinct 100-native-epoch seed37 carrier and isolated BASE, Fold-A,
and Fold-B branches.  BASE ran 20 native epochs; each OURS branch ran five
Semantic-to-Native epochs followed by fifteen native-only epochs.

Seed11 was not rerun.  Its two DeltaJ values were loaded from the verified
immutable GPU-bridge freeze summary whose hash matches the summary entry in
the frozen smoke-result manifest.  The new seed37 results and frozen seed11
results were passed to the pure real-evaluator layer.

The real-evaluator validates dataset, candidate, seeds, all epoch counts,
initial-state parity, zero-utility gates, finite DeltaJ values, and the GT
firewall.  It constructs the frozen `CandidateEvaluation` schema and delegates
mean, worst, and positive-count aggregation to the frozen objective helper.
No selection helper is called because only one candidate exists.

All CPU contract tests passed.  Both seed37 zero-utility gates passed, all
seed37 J/DeltaJ values were finite, initial BASE/A/B state hashes matched, and
the four-score CandidateEvaluation was completed.  DeltaJ sign was not used
as a pass criterion.

No Optuna study, formal tuning, additional candidate, other dataset, MSRC run,
formal seed, full ground truth, or clustering metric was used.
