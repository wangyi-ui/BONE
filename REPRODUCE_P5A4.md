# BONE Reproduction Checkpoint

## Current milestone

**P5-A4 held-out validation complete.**

This is NOT the final paper release.

Completed:
- core method implementation;
- Information Utility / U_cycle;
- sparse-label T_rel semantic direction;
- utility-conditioned semantic relation purification;
- native structural consolidation;
- 8-dataset held-out validation;
- unseen training seeds 149 / 163 / 179.

Current held-out decision:

`PARTIAL_REPLICATION`

Pending:
- ablation studies;
- parameter sensitivity analysis;
- final paper tables;
- supplementary experiments / complexity analysis if required.

## Reproduction

### 1. Environment

Use:

`reproducibility/environment_history.yml`

and:

`reproducibility/pip_freeze.txt`

as environment references.

### 2. Raw datasets

Raw dataset files are NOT committed to GitHub.

Place the original datasets at the paths expected by the BONE loaders.

### 3. Restore frozen weak-quality inputs

Run:

`bash reproducibility/restore_frozen_inputs.sh`

This restores the exact sparse-label splits and corruption masks.

### 4. Regenerate MSRC P1 artifact

After the original MSRC dataset is available:

`bash reproducibility/regenerate_msrc_p1.sh`

The regenerated artifact must match the frozen SHA256.

### 5. Final fixed protocol

See:

`reproducibility/P5_A1_PROTOCOL.json`

Held-out seeds:

`149, 163, 179`

No retuning is permitted when reproducing P5-A4.

### 6. Final runner

`experiments/utility_relation/run_bone_p5_a2_fixed_config_heldout.py`

### 7. Final evaluator

`experiments/utility_relation/evaluate_bone_p5_a4_fixed_config_heldout.py`

### 8. Frozen result reference

See:

`reproducibility/results/`
