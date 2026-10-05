# Comparison Policy

## Source-level comparator

UCAB is developed from the frozen pre-UCAB semantic method.

The historical semantic implementation is NOT edited.

CURRENT must retain the exact old semantic-loss mathematics.

B1/B5 optimization-search infrastructure is not part of the
new scientific method.


## Parameter-level comparator

A fair comparison requires CURRENT and all UCAB arms to use
the same frozen semantic optimizer, LR and exposure within
each dataset.

For the currently targeted datasets:

CCV:
optimizer = Adam
semantic_lr = 1e-5
semantic_exposure = 3

ALOI:
optimizer = Adam
semantic_lr = 1e-4
semantic_exposure = 1

These values are frozen from pre-GT selection and B5 finalization.

The historical default lr=1e-4 / exposure=20 is NOT the
primary comparator because that would confound the method
change with parameter changes.


## Scientific isolation

Between CURRENT and UCAB arms, the only permitted scientific
difference is the absolute action-budget mechanism.

The following remain identical:

- carrier initialization
- corruption realization
- sparse labeled IDs
- U_cycle
- T_rel
- relation primitive
- anchor detach
- semantic optimizer family
- semantic LR
- semantic exposure
- semantic ordering
- native minibatch ordering
- native optimizer
- native objective
- semantic-then-native phase order
