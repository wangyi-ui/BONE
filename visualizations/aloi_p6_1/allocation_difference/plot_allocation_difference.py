"""All-sample disagreement with OURS; no training, projection fit or evaluation."""
import os
from pathlib import Path
OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent
os.environ['MPLCONFIGDIR'] = str(OUT / '.mplconfig')
os.environ['XDG_CACHE_HOME'] = str(OUT / '.cache')
import hashlib
import io
import json
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from PIL import Image

sha = lambda data: hashlib.sha256(data).hexdigest()
archive = SOURCE / 'ALOI_P6_1_seed20_visualization.zip'
coordinates = SOURCE / 'shared_tsne_coordinates.npz'
# Preserve every existing source-level file, including the original figures.
before = {p.name: sha(p.read_bytes()) for p in SOURCE.iterdir() if p.is_file()}
previous = json.loads((SOURCE / 'tsne_render_report.json').read_text())
with zipfile.ZipFile(archive) as z:
    seal = json.loads(z.read('PREDICTION_SEAL.json'))
    def load(name):
        raw = z.read(name)
        assert sha(raw) == previous['input_sha256'][name], name
        return np.load(io.BytesIO(raw), allow_pickle=False)
    q = load('q_views.npy')
    ids = load('sample_ids.npy')
    arms = ['BASE', 'UNIFORM_U', 'SHUFFLED_U', 'SHUFFLED_T']
    predictions = {arm: load(f'{arm}/predictions.npy') for arm in arms + ['OURS']}
    for arm, labels in predictions.items():
        assert previous['input_sha256'][f'{arm}/predictions.npy'] == seal['prediction_artifacts'][arm]['sha256']
        assert labels.shape == (10800,) and np.issubdtype(labels.dtype, np.integer)
        assert labels.min() >= 0 and labels.max() < 100
assert q.shape == (10800, 4, 100) and np.isfinite(q).all()
assert np.array_equal(np.sort(ids), np.arange(10800))
with np.load(coordinates, allow_pickle=False) as data:
    assert np.array_equal(ids, data['sample_ids'])
    xy = data['coordinates']
assert xy.shape == (10800, 2) and np.isfinite(xy).all()
reference = predictions['OURS']
report = {'reference': 'OURS', 'samples_per_panel': 10800,
          'tsne_refitted': False, 'ground_truth_used': False,
          'coordinates_sha256': before[coordinates.name],
          'archive_sha256': before[archive.name], 'arms': {}}

font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
fig, axes = plt.subplots(2, 2, figsize=(12, 11), sharex=True, sharey=True)
lo, hi = xy.min(axis=0), xy.max(axis=0)
pad = (hi-lo)*.04
for ax, arm in zip(axes.flat, arms):
    labels = predictions[arm]
    contingency = np.zeros((100, 100), dtype=np.int64)
    np.add.at(contingency, (labels, reference), 1)
    rows, cols = linear_sum_assignment(-contingency)
    mapping = np.empty(100, dtype=np.int64)
    mapping[rows] = cols
    assert len(np.unique(mapping)) == 100
    assert {str(i): int(j) for i,j in enumerate(mapping)} == previous['cluster_mappings'][arm]
    changed = mapping[labels] != reference
    count = int(changed.sum())
    # Plot ALL points at identical size/opacity, in fixed sample order.
    colors = np.where(changed, '#d7191c', '#d9d9d9')
    ax.scatter(xy[:,0], xy[:,1], c=colors, s=7, alpha=1,
               edgecolors='none', rasterized=True)
    ax.set_title(f'{arm} vs OURS | Different: {count/len(ids):.2%} ({count}/10800)', fontsize=11)
    ax.set(xlim=(lo[0]-pad[0],hi[0]+pad[0]), ylim=(lo[1]-pad[1],hi[1]+pad[1]),
           aspect='equal', xticks=[], yticks=[])
    for spine in ax.spines.values(): spine.set_visible(False)
    report['arms'][arm] = {'different_count': count, 'different_fraction': count/len(ids),
                          'same_count': int((~changed).sum()),
                          'color_mapping': {str(i): int(j) for i,j in enumerate(mapping)}}
fig.suptitle('ALOI seed 20 · Cluster allocation differences on shared t-SNE', fontsize=15)
fig.legend(handles=[Line2D([],[],marker='o',linestyle='None',color=c,label=t)
                    for c,t in [('#d9d9d9','Same allocation as OURS'),('#d7191c','Different allocation from OURS')]],
           loc='lower center',bbox_to_anchor=(.5,.075),ncol=2,frameon=False)
fig.text(.5,.055,'All 10,800 samples per panel; Hungarian alignment to OURS; identical coordinates and axes.',
         ha='center',fontsize=10)
fig.text(.5,.03,'红色仅表示预测簇分配发生变化，不代表该样本被正确纠正。',
         ha='center',fontproperties=font,fontsize=11)
fig.text(.5,.01,'Red indicates a change in predicted cluster assignment, not evidence of a correct correction.',
         ha='center',fontsize=9)
fig.tight_layout(rect=(0,.11,1,.95))
report['outputs'] = {}
for ext in ['png','pdf']:
    path = OUT / f'aloi_allocation_difference.{ext}'
    fig.savefig(path,dpi=300,bbox_inches='tight')
    if ext=='png':
        with Image.open(path) as im: im.verify()
    else: assert path.read_bytes().startswith(b'%PDF-')
    report['outputs'][path.name]={'sha256':sha(path.read_bytes()),'bytes':path.stat().st_size}
plt.close(fig)
assert all(sha((SOURCE/name).read_bytes()) == digest for name,digest in before.items())
report['all_existing_source_files_unchanged'] = True
(OUT/'allocation_difference_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({a:{k:v for k,v in r.items() if k!='color_mapping'} for a,r in report['arms'].items()},indent=2))
print('Both outputs validated; all existing source files unchanged.')
