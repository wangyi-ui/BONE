# P3-C3-A0 UCAB Frozen Mathematical Definition

## 1. Frozen current semantic objective

Let:

- i = query-sample index
- l = labeled-anchor index
- s = complementary-action index
- v = view index

Frozen tensors:

U_{is} = U_cycle[i,s]

T_{ils} = T_rel[i,l,s]

Both remain detached from gradient computation.

For view v, the existing relation primitive remains:

r^v_{il} = q^v_i · stopgrad(q^v_l)

The current semantic action loss can be represented as:

L_CURRENT
=
(1 / (V D))
sum_v sum_i sum_s U_{is} ell^v_{is}

where:

D = sum_i sum_s U_{is}

and ell^v_{is} is the existing positive-plus-negative
anchor-reduced semantic relation contribution.

The existing implementation therefore preserves relative
sample/action weighting but cancels common absolute scaling:

L_CURRENT(cU) = L_CURRENT(U), c > 0.

No part of L_CURRENT is modified in the CURRENT arm.


## 2. Within-sample relative action validity

For each query sample:

m_i = sum_s U_{is}

For m_i > 0:

alpha_{is} = U_{is} / m_i

For m_i = 0:

alpha_{is} = 0 for all s.

alpha answers WHICH semantic actions are relatively valid.


## 3. Utility-conditioned absolute action budget

The frozen parameter-free budget is:

B_i
=
1
-
[
product_s (1 - U_{is})
]^(1/S)

where S is the number of complementary actions.

U remains detached, therefore B is detached.

Properties:

1. 0 <= B_i <= 1.
2. all U_{is}=0 => B_i=0.
3. if all U_{is}=u, then B_i=u.
4. action-count normalization is intrinsic through exponent 1/S.
5. no trainable parameter is introduced.
6. B_i is generally nonlinear under U -> cU.


## 4. UCAB semantic objective

Define the sample-level semantic contribution:

ell_bar_i
=
(1/V)
sum_v
sum_s
alpha_{is} ell^v_{is}

Then:

L_UCAB
=
(1/N_q)
sum_i
B_i ell_bar_i

where N_q is the total number of query samples.

CRITICAL:

The denominator MUST NOT be sum_i B_i.

Otherwise the absolute action-budget effect would again
be normalized away.


## 5. Zero-utility rule

If m_i = 0:

alpha_{is}=0
B_i=0

Therefore the sample receives zero semantic intervention.

No uniform fallback is permitted.


## 6. Analytical linear-budget redundancy control

The following is NOT a formal GPU arm:

B_i_linear = mean_s U_{is}

Combined with normalized alpha, this reduces to:

L_linear = mean(U) * L_CURRENT

up to the frozen reduction convention.

Therefore mean-U alone is treated as an analytical
global-rescaling control and is not proposed as the UCAB mechanism.


## 7. Gradient contract

stopgrad(U_cycle) = U_cycle
stopgrad(T_rel) = T_rel
stopgrad(B) = B

No gradient may flow into:

- U_cycle
- T_rel
- B_i
- anchor posterior

Only the query/model parameters participating in the
existing semantic relation primitive receive semantic gradients.
