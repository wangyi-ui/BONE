# ALOI P6.1: seed-20 ablation t-SNE

Run from `/workspace/BONE`:

```sh
python visualizations/aloi_p6_1/plot_tsne.py
```

To reproduce from scratch, place the original uploaded
`ALOI_P6_1_seed20_visualization.zip` in this directory. The original input ZIP
is retained in the cloud workspace but is not committed to this branch.
Only NumPy arrays and prediction-seal metadata are read directly from it;
model checkpoints are not deserialized. No training or evaluator runs.

`q_views.npy` is reshaped from (10800, 4, 100) to (10800, 400), without
standardization, resampling or additional feature reduction. One t-SNE fit
(perplexity 30, random state 20, PCA initialization, automatic learning rate,
1000 iterations, Euclidean Barnes-Hut) supplies coordinates for every arm.
`shared_tsne_coordinates.npz` preserves coordinates and sample IDs. Reruns
reuse that embedding after checking input hashes and parameters.

All five prediction arrays and sample IDs are verified against the uploaded
prediction seal. Hungarian maximum-overlap matching aligns each arm's cluster
IDs to OURS for colors only. The original arrays remain untouched. Alignment
is a one-to-one permutation, not an evaluation against ground truth. No
true labels are provided or inferred. Color matching cannot make differing
partitions identical. With 100 colors, some hues are visually similar.

Outputs:
- `aloi_tsne_{BASE,UNIFORM_U,SHUFFLED_U,SHUFFLED_T,OURS}.{png,pdf}`
- `aloi_tsne_five_arm_comparison.{png,pdf}`
- `tsne_render_report.json`: input/output hashes and all color mappings
- `tsne_metadata.json`: t-SNE parameters and convergence information
- `tsne_run.log`: actual run output

Each panel uses all samples with the same point order, coordinate limits,
colors and aspect ratio. PDFs retain vector text and rasterize the scatter
points at 300 dpi for manageable size. PNG decoding and PDF headers are
checked after rendering. All new files and caches remain in this directory.

`ALOI_P6_1_tsne_figures.zip` contains all 12 PNG/PDF outputs for download.
The superseded score-comparison figures and local caches are not included
in this branch.

Existing training code, frozen experiments, formal results, metrics,
dependency declarations and lockfiles must remain unchanged. Use the existing
checkout, without creating a Git worktree.
