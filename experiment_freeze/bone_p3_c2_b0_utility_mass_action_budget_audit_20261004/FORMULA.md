# P3-C2-B0 exact semantic formula

`run_final_formal.py::_materialize_or_resume_branches` calls the frozen
`run_bone_p3_c0_cross_dataset_formal.py::_semantic_epoch`.

For view `v`, query `i`, anchor `l`, action `s`, and `D=sum_{i,s} U_is`:

`r_il=q_i dot stopgrad(a_l)`, `safe=clamp(r_il, eps, 1-eps)`.

`P_is=-(1/4) sum_l T_ils log(safe)` and
`N_is=-(1/(4*(K-1))) sum_l (1-T_ils) log(1-safe)`.

For each query batch `b`,
`ell_b=sum_v sum_{i in b,s} U_is(P_is+N_is)/(V*D)`; the epoch scalar is
`sum_b ell_b`.  Consequently common positive scaling of U cancels between
numerator and D.  Pi constructs T_rel upstream and is not read directly by
the semantic epoch.  No relation, sample, or anchor mask exists; U=0 is the
implicit outer contribution mask, and anchor posteriors are detached.
