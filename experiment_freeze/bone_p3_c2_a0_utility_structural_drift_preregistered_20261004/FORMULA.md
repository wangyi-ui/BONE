# Frozen Formulae

For query sample `i` and action index `s`:

```text
M_i = sum_s query_U_cycle[i, s]
mean_utility_i = M_i / S
supported_i = (M_i > 0)
abstained_i = (M_i == 0)
```

Construct an OURS-to-BASE mapping using Hungarian maximum overlap on the full
shared ID population. For query samples:

```text
F_i = 1[BASE_i != aligned(OURS_i)]
query_flip_rate = mean_i(F_i)
supported_flip_rate = P(F=1 | M>0)
abstained_flip_rate = P(F=1 | M=0)
```

If a conditioning group is empty, its rate is `null`. If there are no flips,
both flip-source fractions are `null`. If a changed or stable set is empty,
its conditional utility means and their deltas are `null`.

Supported samples are sorted ascending by `(mean_utility_i, sample_id)` and
partitioned with `np.array_split(..., 4)`. The resulting Q1--Q4 rates are
descriptive only.

Spearman is calculated for `(mean_utility_i, F_i)` over all query samples and
supported query samples. Constant input returns `rho=null`, `p_value=null`,
`reason=CONSTANT_INPUT`.

GT-free agreement uses `agreement_nmi_*` and `agreement_ari_*` exclusively
for BASE versus OURS predictions. Query occupancy TV includes an unmatched
OURS sentinel category. Entropy is normalized by the cardinality of each
arm's effective support; a single category has normalized entropy zero.

