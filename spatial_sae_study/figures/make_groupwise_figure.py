import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
R=Path('/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/nhp_sae_analysis/results')
blocks=[0,2,4,6,8,10,11]; rows=[]
for b in blocks:
 s=pd.read_csv(R/f'seg_cmp_block{b}/summary_metrics.csv')
 p=pd.read_csv(R/f'groupwise_downstream_b{b}_s0/spatial_downstream_summary.csv')
 for g in ['G0','G1']:
  a=s[(s.method=='spatial')&(s.representation==g)].iloc[0]; c=s[(s.method=='baseline')&(s.representation==g)].iloc[0]
  x=p[(p.arm=='spatial')&(p.group==g)].iloc[0]; y=p[(p.arm=='baseline')&(p.group==g)].iloc[0]
  rows.append(dict(block=b,group=g,segment_alignment=a.gap_mean-c.gap_mean,
    precision_at_5=a.precision_at_5_mean-c.precision_at_5_mean,
    propagation_N8=x.prop_raw_N8-y.prop_raw_N8))
d=pd.DataFrame(rows); d.to_csv('groupwise_downstream_differences.csv',index=False)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(12,3.5),sharex=True)
for ax,(col,title) in zip(axs,[('segment_alignment','Segment alignment'),('precision_at_5','Same-instance precision@5'),('propagation_N8','Few-label propagation\n(8 labels per class)')]):
 for g,color in [('G0','#2878B5'),('G1','#D95319')]:
  q=d[d.group==g]; ax.plot(q.block,q[col],'-o',lw=2,ms=6,label=g,color=color)
 ax.axhline(0,color='0.35',lw=1,ls='--'); ax.set_title(title); ax.set_xlabel('DINOv2 block'); ax.set_xticks(blocks); ax.grid(axis='y',alpha=.2)
axs[0].set_ylabel('Spatial − baseline')
axs[0].legend(frameon=False)
fig.tight_layout(); fig.savefig('fig_groupwise_downstream.png',dpi=220,bbox_inches='tight')
