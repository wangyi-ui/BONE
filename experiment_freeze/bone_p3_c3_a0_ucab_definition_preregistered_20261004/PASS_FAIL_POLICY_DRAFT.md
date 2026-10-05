# P3-C3-A0 Pass/Fail Policy Draft

This file freezes the scientific direction but NOT the final
confirmatory seeds. The final gate will be completed only after
the unseen-seed provenance audit.

Two logically separate gates are required.

## Mechanism gate

TRUE_BUDGET must outperform both:

MEAN_BUDGET
SHUFFLE_BUDGET

under the preregistered aggregation rule.

Otherwise any gain cannot be attributed specifically to
sample-conditioned information utility.


## Clustering-rescue gate

The final confirmatory protocol will require improvement over
CURRENT on unseen training seeds.

ACC is the primary clustering metric.

NMI and ARI are secondary metrics.

Exact seed count and aggregation thresholds are intentionally
NOT frozen here and must be preregistered after the unseen-seed
provenance audit and before any UCAB GT evaluation.


## Fail-closed rule

No post-GT:

- seed replacement
- LR search
- exposure search
- optimizer search
- budget-function search
- threshold search

is permitted.
