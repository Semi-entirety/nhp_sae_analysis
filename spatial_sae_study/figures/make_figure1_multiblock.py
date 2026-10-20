import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 180,
})
R = Path('/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/nhp_sae_analysis/results')
BLUE = '#577590'
ORANGE = '#f3722c'

# Blocks 0 and 2: means over three independently trained paired seeds.
early = pd.read_csv(R / 'b12_eval_seg/segment.csv')
rows = []
for block in (0, 2):
    cond = f'dino_b{block}_postLN'
    for arm in ('baseline', 'spatial'):
        d = early[(early.cond == cond) & (early.arm == arm)]
        for metric in ('gap', 'prec5'):
            rows.append({
                'block': block, 'arm': arm, 'metric': metric,
                'mean': d[metric].mean(), 'sd': d[metric].std(ddof=1),
                'n_seeds': len(d),
            })

# Blocks 4, 6, 8, 10, 11: one fixed spatial/baseline checkpoint pair, evaluated on
# the same 500 ADE20K validation images. No across-training-seed error bar.
for block in (4, 6, 8, 10, 11):
    d = pd.read_csv(R / f'seg_cmp_block{block}/summary_metrics.csv')
    d = d[(d.representation == 'Full') & d.method.isin(['baseline', 'spatial'])]
    for arm in ('baseline', 'spatial'):
        z = d[d.method == arm].iloc[0]
        rows += [
            {'block': block, 'arm': arm, 'metric': 'gap',
             'mean': z.gap_mean, 'sd': np.nan, 'n_seeds': 1},
            {'block': block, 'arm': arm, 'metric': 'prec5',
             'mean': z.precision_at_5_mean, 'sd': np.nan, 'n_seeds': 1},
        ]
D = pd.DataFrame(rows)

fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.65))
settings = [
    ('gap', 'A  Same-object grouping', 'Similarity gap (higher is better)'),
    ('prec5', 'B  Same-instance precision among 5 nearest patches',
     'Precision@5 (higher is better)'),
]
for ax, (metric, title, ylabel) in zip(axs, settings):
    for arm, color, label in [('baseline', BLUE, 'Baseline'),
                              ('spatial', ORANGE, 'Spatial SAE')]:
        d = D[(D.metric == metric) & (D.arm == arm)].sort_values('block')
        # Connecting line shows the observed block trend. All displayed block
        # summaries use the same filled marker style for visual continuity.
        ax.plot(d.block, d['mean'], color=color, lw=2, zorder=2)
        early_d = d[d.n_seeds == 3]
        deep_d = d[d.n_seeds == 1]
        ax.scatter(early_d.block, early_d['mean'], s=52, marker='o',
                   facecolors=color, edgecolors=color, linewidths=1.4,
                   label=label, zorder=4)
        ax.scatter(deep_d.block, deep_d['mean'], s=52, marker='o',
                   facecolors=color, edgecolors=color, linewidths=1.4,
                   zorder=4)
    ax.axvline(3, color='#aaaaaa', lw=.9, ls=':', zorder=1)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 11])
    ax.set_xlabel('DINOv2 block')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', alpha=.22)
    ax.spines[['top', 'right']].set_visible(False)

axs[0].legend(frameon=False, loc='best')
# Explain the two evidence levels once, without cluttering either panel.
fig.text(.5, -.015,
         'Blocks 0 and 2 show means over 3 training seeds; blocks 4, 6, 8, 10, and 11 show one fixed checkpoint pair.',
         ha='center', va='top', fontsize=9, color='#444444')
fig.suptitle('Segmentation-related readouts across DINOv2 blocks',
             fontweight='bold', y=1.02)
fig.tight_layout(rect=[0, .06, 1, 1])
fig.savefig('fig_main_results_clear.png', bbox_inches='tight', dpi=220)
plt.close(fig)

print(D.pivot_table(index=['block','n_seeds'], columns=['metric','arm'], values='mean'))
print('saved fig_main_results_clear.png')
