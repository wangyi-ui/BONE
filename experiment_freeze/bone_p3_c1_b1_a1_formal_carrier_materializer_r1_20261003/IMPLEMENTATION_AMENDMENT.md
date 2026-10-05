# P3-C1-B1-A1 Formal Carrier Materializer R1

## Root cause and scope

The frozen MSRC adapter was correctly restricted to seed11 for its original
engineering-smoke stage. The first formal materializer routed both development
seeds directly to that smoke adapter, so MSRC seed37 failed before training
with `ONLY_DEV_SEED11_IS_ALLOWED`. This was an implementation routing
incompatibility, not a scientific protocol failure.

R1 adds a versioned formal MSRC adapter whose legal seeds are imported from
the frozen `protocol.DEV_SEEDS`. It reuses the frozen MSRC input loader,
dataset configuration, preprocessing, corruption and sparse-label artifacts,
BONE/native training primitives, action construction, and carrier-array
materialization. Removing the smoke-only execution restriction from the formal
route changes no formula or scientific constant.

The R1 materializer is thin composition over the frozen materializer. During
one serialized invocation it replaces only the MSRC adapter extension point
and adds the formal adapter source hash to MSRC provenance. Generic datasets,
atomic staging, COMPLETE creation and verification, hashes, resume/skip,
no-GT rules, and audits remain the frozen implementation.

The R1 CLI still accepts exactly one dataset and one dev seed. The frozen
inventory CLI uses `--output`; R1 documentation and tests never use
`--output-json`.

## Existing carriers

Before and after implementation, the formal inventory is exactly five valid
COMPLETE carriers and eleven missing carriers. All 20 files belonging to the
five carriers retained byte-identical SHA-256 values. No real carrier was
trained, created, removed, replaced, or overwritten by this amendment.

## Test note

All 11 new R1 tests pass. Of the requested frozen regression selection, 102
tests pass. One historical implementation-stage environment assertion expects
the real `formal_tuning/` directory not to exist; it necessarily fails now
that five authorized formal carriers exist. The raw failure was preserved,
and the applicable regression rerun records `102 passed, 1 deselected`. The
old test and the existing carriers were not modified to manufacture a pass.
