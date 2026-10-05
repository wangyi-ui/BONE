# P3-C1-B1 Final Formal Runner Protocol

## Entry-point decision

The frozen P3-C0 runner is not the final entry point because it excludes MSRC,
fixes semantic LR at `1e-4`, and applies semantic action in all 20 refinement
epochs. It also does not bind itself to `FINAL_HYPERPARAMETERS.json`.

The new entry points are append-only thin orchestration under
`experiments.utility_relation.p3_c1_b1_final_formal`.

## Reused frozen components

- P3-C0 lower-level native epoch, semantic epoch, model construction, state
  hashing, prediction, and raw-view input primitives;
- P3-B0 generic action/posterior/U/T carrier construction;
- P3-C1-B1 A2 coverage-aware feasibility validation;
- R3 numeric sparse-label ordering semantics;
- the P3-C0 post-seal ACC/NMI/ARI metric implementation.

No loss, objective, action space, U_cycle, T_rel, WHAT rule, carrier horizon,
or native schedule was redefined.

## Final parameter binding

The only parameter source is the byte-identical file
`experiment_freeze/bone_p3_c1_b1_a2_final_hyperparameters_20261004/FINAL_HYPERPARAMETERS.json`
with SHA256
`1e296aa4ea8cd562857044c1dca8432f854d32dc67868002f240af2c0d53b9d0`.
The runner offers no semantic-LR or semantic-exposure CLI options. A missing,
malformed, differently hashed, non-frozen, or incomplete parameter freeze is
rejected.

## MSRC route

MSRC is explicitly dispatched through
`p3_c1_b1_optuna.msrc_adapter._load_frozen_msrc_inputs`, preserving the legacy
exactly-2-of-4 corruption and protocol-equivalent P3-B0 carrier semantics. It
then uses the same frozen lower-level carrier/refinement primitives as the
other datasets. It is not added to the historical C0 argparse surface.

## BASE and OURS

Both arms clone the identical 100-native-epoch carrier state and use the same
dataset, training seed, frozen corruption realization, native optimizer setup,
20-epoch refinement horizon, and native sample order.

BASE performs 20 native-only refinement epochs.

OURS performs semantic action followed by native consolidation in the first
frozen `semantic_exposure` epochs, then native-only consolidation for the
remaining `20 - semantic_exposure` epochs. `semantic_lr` affects only the
semantic optimizer.

## GT isolation

`run_final_formal.py` never requests `Y` and seals predictions and all pre-GT
artifacts before completion. `evaluate_final_formal.py` first verifies the
complete seal, frozen parameter binding, runner sources, and frozen parents;
only then does it request `Y`. ACC, NMI, ARI, and their OURS-minus-BASE deltas
are reporting-only and cannot update the parameter freeze.

## Resume and isolation

Outputs are isolated under
`outputs/bone_p3/p3_c1_b1_final_formal/<dataset_slug>/seed<seed>`.
Completed carrier and paired-branch stages have hash-verified markers. A sealed
COMPLETE run is verified and skipped. A configuration-only interruption can
resume. Unsealed partial model artifacts fail closed rather than being silently
overwritten.

## CLI

```text
python -m experiments.utility_relation.p3_c1_b1_final_formal.run_final_formal \
  --dataset <one-of-8> --training-seed <20|30|50> --device cuda:0 \
  --output-root <path> --execute
```

No real GPU training was run while implementing, testing, preflighting, or
freezing this runner.

