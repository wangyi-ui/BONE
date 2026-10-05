# P3-C3-A0 UCAB Two-Tier Evaluation Protocol

## 1. Target datasets

ALOI
CCV


## 2. Historical paired seeds

20
30
50

These seeds have already been used to inspect CURRENT
post-seal clustering performance.

They are retained because they provide the cleanest matched
paired comparison:

same dataset
same corruption
same training seed
same frozen optimizer/LR/exposure
CURRENT vs UCAB

They are NOT treated as unseen confirmatory evidence.

Their role is:

DESCRIPTIVE MATCHED PAIRED EVIDENCE.


## 3. Unseen confirmatory seeds

140
150
170

These seeds are frozen before UCAB implementation and before
any new UCAB GT result is inspected.

Their role is:

PRIMARY CONFIRMATORY PASS/FAIL EVIDENCE.

No confirmatory seed may be replaced after preregistration.


## 4. Historical paired arms

Primary historical paired comparison:

CURRENT
TRUE_BUDGET

CURRENT must preserve the exact frozen pre-UCAB semantic
objective.

TRUE_BUDGET differs only through the frozen UCAB action-budget
mechanism.


## 5. Confirmatory arms

CURRENT
TRUE_BUDGET
MEAN_BUDGET
SHUFFLE_BUDGET


## 6. Frozen optimizer settings

CCV:

optimizer = Adam
semantic_lr = 1e-5
semantic_exposure = 3


ALOI:

optimizer = Adam
semantic_lr = 1e-4
semantic_exposure = 1


No optimizer, LR or exposure search may be reopened.


## 7. GT-read firewall

After implementation and engineering validation:

ALL unseen confirmatory pre-GT runs must be completed and
sealed before ANY new UCAB GT clustering metric is read.

In particular:

Do not inspect UCAB ACC/NMI/ARI on historical seeds 20/30/50
and then modify the method before running 140/150/170.

Once implementation is frozen, the same implementation must
be used for both seed groups.


## 8. Primary confirmatory clustering-rescue gate

Only unseen confirmatory seeds 140/150/170 determine this gate.

Across:

2 datasets x 3 seeds = 6 paired comparisons,

TRUE_BUDGET - CURRENT must satisfy ALL:

1. Macro mean DeltaACC > 0.

2. At least 4 / 6 seed-level DeltaACC values > 0.

3. ALOI mean DeltaACC > 0.

4. CCV mean DeltaACC > 0.


## 9. Utility-specific mechanism gate

Again using only unseen confirmatory seeds:

TRUE_BUDGET - MEAN_BUDGET:

- macro mean DeltaACC > 0
- at least 4 / 6 positive paired DeltaACC values

AND

TRUE_BUDGET - SHUFFLE_BUDGET:

- macro mean DeltaACC > 0
- at least 4 / 6 positive paired DeltaACC values


## 10. Secondary consistency gate

Using unseen confirmatory seeds:

Among ACC, NMI and ARI,

at least 2 / 3 metrics must have positive macro mean:

TRUE_BUDGET - CURRENT.

ACC must satisfy the primary gate regardless.


## 11. Scientific PASS

P3-C3-A0 UCAB passes only if:

PRIMARY CONFIRMATORY CLUSTERING-RESCUE GATE = PASS

AND

UTILITY-SPECIFIC MECHANISM GATE = PASS

AND

SECONDARY CONSISTENCY GATE = PASS.


## 12. Historical paired results

20/30/50 are reported separately as matched historical
paired evidence.

They may strengthen or weaken interpretation, but they cannot
override the unseen confirmatory pass/fail decision.


## 13. Fail-closed restrictions

After preregistration, no result-driven modification of:

- seed set
- budget formula
- optimizer
- semantic LR
- semantic exposure
- corruption
- sparse labels
- arm definitions
- pass/fail gates

is permitted.
