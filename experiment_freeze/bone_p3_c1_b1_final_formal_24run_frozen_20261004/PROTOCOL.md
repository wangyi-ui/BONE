# P3-C1-B1 Final Formal 24-Run Freeze

This freeze records the reporting-only formal evaluation of the
frozen P3-C1-B1 method on eight datasets and three training seeds
(20, 30, 50), for exactly 24 formal runs.

Hyperparameters were selected before full-GT formal evaluation using
the frozen sparse-label validation protocol.

Dataset-specific semantic_lr and semantic_exposure are treated as
optimization hyperparameters, not as additional method modules.

The tuning machinery is therefore an experimental hyperparameter
selection protocol rather than part of the core information-utility
architecture.

All formal runs use the previously frozen dataset-specific parameters.
Full GT is accessed only after PRE_GT_SEAL and only for reporting
ACC/NMI/ARI.

Formal metrics may not be used to retune semantic_lr or
semantic_exposure for this frozen experiment.

The core scientific mechanism remains:

Sparse labels determine WHAT;
U_cycle determines WHETHER;

followed by utility-conditioned semantic action and native structural
consolidation.
