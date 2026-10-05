# Formal Optuna Runner R3 numeric class-order amendment

Root cause: `sparse_cv.py` orders class keys by `repr`, so numeric class IDs
with two or more digits are placed lexicographically. COIL20 therefore reached
the frozen A1 fold builder in the order `0,1,10,...,19,2,...,9`. The raw sparse
split itself remains the correct numeric contiguous label set `0..19`. The
frozen `FOLD_CLASS_ORDER_MUST_BE_CONTIGUOUS` guard is correct because
`class_signature[c]` must represent class `c`.

R3 is append-only. `canonical_fold_order_r3.py` sorts paired
`(target, raw_row_id)` records by `int(target)` within each existing train and
validation membership. It fail-closes unless each side contains exactly one
anchor for every class in `range(K)`. Pair counters, raw-ID multisets, target
multisets, and order-independent pair hashes are checked before returning.
Targets are neither normalized nor re-encoded.

`formal_tuning_context_r3.py` first invokes the frozen context builder, then
creates frozen-dataclass copies whose only changed field is `folds`. Carrier
arrays, views, state paths, artifact hashes, protocol fingerprint, U_cycle,
p_gen, p_ver, and sample IDs are unchanged. Already numeric folds preserve the
original `SparseFold` object identity.

`run_formal_optuna_tuning_r3.py` uses the R3 context and otherwise delegates to
the R2 candidate evaluator, R1 BASE bootstrap, frozen GridSampler/objective,
candidate cache, and `CandidateEvaluation`. Its isolated namespace is
`study_r3.db` / `p3_c1_b1_a1__<dataset_slug>__main_r3` for all eight datasets.

The read-only eight-dataset audit found numeric-order identity for HandWrite,
BDGP, MSRC, Caltech-5V, and MNIST-USPS; numeric permutation is required for
COIL20, ALOI, and CCV. Every dataset passes fold class coverage and pair
preservation. ALOI and CCV separately retain their pre-existing fail-closed
`CLASS_ZERO_UTILITY_EVIDENCE` feasibility result; R3 does not bypass it.

No real BASE run, candidate refinement, GPU run, Optuna run, production
`study_r3.db`, sparse-split rewrite, or full-GT access occurred.

