"""One shared t-SNE; prediction-only coloring with Hungarian alignment."""
import os
from pathlib import Path
OUT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME', str(OUT / '.cache'))
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import hashlib
import io
import json
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits
from PIL import Image

ARMS = ['BASE', 'UNIFORM_U', 'SHUFFLED_U', 'SHUFFLED_T', 'OURS']
sha = lambda data: hashlib.sha256(data).hexdigest()
archive = OUT / 'ALOI_P6_1_seed20_visualization.zip'
archive_hash = sha(archive.read_bytes())
with zipfile.ZipFile(archive) as z:
    seal = json.loads(z.read('PREDICTION_SEAL.json'))
    inputs = {}
    def load(name):
        raw = z.read(name)
        inputs[name] = sha(raw)
        return np.load(io.BytesIO(raw), allow_pickle=False)
    q = load('q_views.npy')
    ids = load('sample_ids.npy')
    predictions = {arm: load(f'{arm}/predictions.npy') for arm in ARMS}
    for item in seal['prediction_artifacts'].values():
        assert inputs[item['path']] == item['sha256'], item['path']
assert q.shape == (10800, 4, 100) and np.isfinite(q).all()
assert ids.shape == (10800,) and np.array_equal(np.sort(ids), np.arange(10800))
for labels in predictions.values():
    assert labels.shape == (10800,) and np.issubdtype(labels.dtype, np.integer)
    assert labels.min() >= 0 and labels.max() < 100

reference = predictions['OURS']
aligned = {}
mappings = {}
for arm, labels in predictions.items():
    counts = np.zeros((100, 100), dtype=np.int64)
    np.add.at(counts, (labels, reference), 1)
    rows, cols = linear_sum_assignment(-counts)
    mapping = np.empty(100, dtype=np.int64)
    mapping[rows] = cols
    if arm == 'OURS':
        mapping = np.arange(100)
    aligned[arm] = mapping[labels]
    mappings[arm] = {str(i): int(j) for i, j in enumerate(mapping)}
    print(arm, 'nonempty clusters:', len(np.unique(labels)),
          'color IDs changed:', int(np.sum(mapping != np.arange(100))), flush=True)

params = dict(n_components=2, perplexity=30.0, learning_rate='auto',
              init='pca', random_state=20, max_iter=1000,
              method='barnes_hut', angle=0.5, n_jobs=4, verbose=1)
cache = OUT / 'shared_tsne_coordinates.npz'
metadata = OUT / 'tsne_metadata.json'
key = {'inputs': inputs, 'parameters': params}
if cache.exists():
    assert metadata.exists() and json.loads(metadata.read_text())['cache_key'] == key
    with np.load(cache, allow_pickle=False) as data:
        xy = data['coordinates']
        assert np.array_equal(data['sample_ids'], ids)
    print('Reusing the single saved t-SNE embedding.', flush=True)
else:
    with threadpool_limits(limits=4):
        model = TSNE(**params)
        xy = model.fit_transform(q.reshape(10800, 400))
    assert xy.shape == (10800, 2) and np.isfinite(xy).all()
    np.savez_compressed(cache, coordinates=xy, sample_ids=ids)
    metadata.write_text(json.dumps({'cache_key': key, 'kl_divergence': float(model.kl_divergence_),
                                   'iterations': int(model.n_iter_)}, indent=2) + '\n')

# Same 100 colors, point order, limits and geometry in every panel.
colors = np.concatenate([plt.get_cmap(name)(np.linspace(0, 1, 20))
                         for name in ['tab20', 'tab20b', 'tab20c', 'turbo', 'gist_ncar']])
cmap = ListedColormap(colors)
order = np.random.default_rng(20).permutation(len(ids))
lo, hi = xy.min(axis=0), xy.max(axis=0)
pad = (hi - lo) * .04
outputs = []
def panel(ax, arm):
    ax.scatter(xy[order, 0], xy[order, 1], c=aligned[arm][order], cmap=cmap,
               vmin=-.5, vmax=99.5, s=3.5, alpha=.8, linewidths=0, rasterized=True)
    ax.set(xlim=(lo[0]-pad[0], hi[0]+pad[0]), ylim=(lo[1]-pad[1], hi[1]+pad[1]),
           aspect='equal', title=arm, xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_visible(False)
def save(fig, stem):
    for ext in ['png', 'pdf']:
        path = OUT / f'{stem}.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        if ext == 'png':
            with Image.open(path) as image:
                image.verify()
        else:
            assert path.read_bytes().startswith(b'%PDF-')
        outputs.append({'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path.read_bytes())})
    plt.close(fig)
with plt.rc_context({'font.size': 10, 'pdf.fonttype': 42}):
    for arm in ARMS:
        fig, ax = plt.subplots(figsize=(6, 6))
        panel(ax, arm)
        fig.suptitle('ALOI · seed 20 · shared t-SNE', fontsize=13)
        fig.text(.5, .02, 'Predicted clusters; colors aligned to OURS. No ground-truth labels.',
                 ha='center', fontsize=8)
        save(fig, f'aloi_tsne_{arm}')
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.8))
    for ax, arm in zip(axes, ARMS):
        panel(ax, arm)
    fig.suptitle('ALOI · seed 20 · one shared t-SNE of q_views (10800 × 400)', fontsize=14)
    fig.text(.5, .04, 'Predicted clusters only; Hungarian color alignment to OURS. Shared coordinates and axes; no ground-truth labels.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .09, 1, .91))
    save(fig, 'aloi_tsne_five_arm_comparison')
assert sha(archive.read_bytes()) == archive_hash
report = {'archive_sha256': archive_hash, 'input_sha256': inputs,
          'reference': 'OURS', 'alignment': 'Hungarian maximum sample overlap; permutation for coloring only',
          'cluster_mappings': mappings, 'tsne_fit_count': 1,
          'ground_truth_used': False, 'outputs': outputs}
(OUT / 'tsne_render_report.json').write_text(json.dumps(report, indent=2) + '\n')
print('Verified', len(outputs), 'PNG/PDF outputs; input ZIP unchanged.', flush=True)
