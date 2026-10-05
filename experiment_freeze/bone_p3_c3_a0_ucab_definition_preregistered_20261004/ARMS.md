# Frozen Experimental Arms

Only the following four GPU arms are allowed.

## CURRENT

Exact frozen pre-UCAB semantic objective.

No action-budget modification.

This arm MUST call or exactly reproduce the frozen legacy
semantic implementation.

CURRENT is the primary method comparator.


## TRUE_BUDGET

Uses:

alpha_{is} = U_{is} / sum_t U_{it}

and:

B_i =
1 - [product_s (1-U_{is})]^(1/S)

with zero-mass samples receiving alpha=0 and B=0.

This is the proposed UCAB arm.


## MEAN_BUDGET

Let:

I+ = {i : sum_s U_{is} > 0}

and:

B_bar =
mean_{i in I+}(B_i_TRUE)

Then:

B_i_MEAN =
B_bar, i in I+
0,     otherwise

alpha remains the true sample-specific alpha.

Purpose:

Control for global semantic-action magnitude while removing
sample-specific budget information.


## SHUFFLE_BUDGET

Within I+ only, deterministically derange the TRUE budget values:

B_i_SHUFFLE = B_{pi(i)}_TRUE

Requirements:

- no fixed points;
- exact same supported sample set;
- exact same budget multiset;
- exact same mean;
- exact same variance;
- deterministic mapping from frozen seed/protocol;
- alpha remains attached to the original sample.

Purpose:

Destroy sample-budget correspondence without changing
the budget distribution.


## Forbidden arms / modifications

No:

- learned budget network
- sigmoid temperature search
- power gamma
- clipping search
- threshold
- top-k
- memory bank
- pseudo-label expansion
- additional CE
- simultaneous semantic/native additive objective
- optimizer-family search
- LR search
- exposure search
- native-objective modification
