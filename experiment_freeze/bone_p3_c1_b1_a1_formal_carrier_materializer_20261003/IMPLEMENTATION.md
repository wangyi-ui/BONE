# P3-C1-B1-A1 Formal Development Carrier Materializer

This CPU-tested implementation defines the future one-command-one-carrier
workflow for the Cartesian product imported from the frozen protocol: eight
datasets and development seeds 11 and 37, for 16 carriers. It did not create
or train any real formal carrier during this stage.

The router delegates the seven generic datasets to frozen
`carrier_adapter.py` and MSRC to frozen `msrc_adapter.py`; it contains no BONE
training implementation. The active protocol identity is imported from the A1
amendment. Dataset, seed, and carrier-epoch scientific constants are imported
from the original B1 protocol.

Each invocation accepts exactly one dataset and one development seed. It uses
a unique staging directory and publishes `carrier/` atomically only after the
model state, pre-GT artifact, formula/shape audit, hashes, and audit file pass.
`COMPLETE.json` is written last. A verified complete carrier is skipped;
complete hash drift fails closed and can never be overwritten. Incomplete
target/staging content fails by default and can be replaced only through the
explicit target-scoped option. The production CLI additionally locks its
output root to the registered `formal_tuning/` namespace; pytest exercises the
internal orchestration only under temporary directories.

The stored pre-GT artifact is checked for the required bridge tensors, unique
sample IDs, exactly two sparse labels per class, finite tensors, canonical
action count, and frozen U formula parity. Zero-mass samples are audited rather
than rejected because A1 supplies validation abstention semantics. No full GT,
refinement, Optuna study, formal seed evaluation, or clustering metric path is
present.

The read-only inventory expects all 16 imported identities and reports ready
only when every expected carrier has a valid COMPLETE marker, matching hashes,
valid protocol audit, and `full_gt_loaded=false`. Unexpected paths cannot
substitute for a missing expected identity.
