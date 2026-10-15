# ALOI cluster allocation differences

Run from `/workspace/BONE`:

```sh
python visualizations/aloi_p6_1/allocation_difference/plot_allocation_difference.py
```

Requires the original input ZIP and the existing shared t-SNE coordinates in
`visualizations/aloi_p6_1/`. The script validates input hashes and sample order,
reads the same q_views, and reuses the stored coordinates without fitting t-SNE.
Hungarian maximum-overlap alignment to OURS is recomputed and checked against
the earlier color mappings. Original predictions remain unchanged.

All 10,800 samples appear in each panel, in original sample order, with the same
marker size, opacity, coordinates and axes. Gray means identical aligned cluster
assignment to OURS; red means different assignment. Plot overlap can obscure
individual points. Titles report the counts from all samples, not visible pixels.
No filtering, label tuning, training or ground-truth evaluation is performed.

红色仅表示预测簇分配发生变化，不代表该样本被正确纠正。

The JSON report records counts, fractions, label permutations, source hashes,
and output hashes. The script verifies unchanged source files and decodes the
PNG; the generated PDF was also successfully parsed with pdfinfo.

All original figures, frozen inputs, formal results and predictions are preserved.
