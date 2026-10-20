import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'figure.dpi':160})
R=Path('/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/nhp_sae_analysis/results')
BLUE='#577590'; ORANGE='#f3722c'; GREEN='#43aa8b'; RED='#d1495b'; GREY='#777777'
# Main result
seg=pd.read_csv(R/'backbone_eval_seg/segment.csv'); seg=seg[seg.cond.eq('dino_b0_postLN')]
prop=pd.read_csv(R/'b12_eval/propagation.csv'); prop=prop[prop.cond.eq('dino_b0_postLN')]
fvu=pd.read_csv(R/'whiten_shortcut/shortcut_g0_share.csv'); fvu=fvu[(fvu.cond.eq('dino_b0_postLN'))&(fvu.ebin.eq(0))]
fig,axs=plt.subplots(1,4,figsize=(13.2,3.35))
def paired(ax,d,val,title,ylabel):
    pv=d.pivot(index='seed',columns='arm',values=val)
    for _,row in pv.iterrows(): ax.plot([0,1],[row.baseline,row.spatial],'-o',color='#999999',lw=1.2,ms=4)
    ax.scatter([0,1],[pv.baseline.mean(),pv.spatial.mean()],s=90,c=[BLUE,ORANGE],edgecolor='black',zorder=4)
    ax.set_xticks([0,1],['Baseline','Spatial']); ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis='y',alpha=.2)
paired(axs[0],seg,'gap','A  Same-object grouping','Similarity gap (higher is better)'); axs[0].text(.5,.05,'diff +0.1005 ± 0.0081, p=.0021',transform=axs[0].transAxes,ha='center',fontsize=8)
paired(axs[1],prop,'prop_N8','B  Few-label propagation','Accuracy (higher is better)'); axs[1].text(.5,.05,'diff +0.0368 ± 0.0083, p=.016',transform=axs[1].transAxes,ha='center',fontsize=8)
paired(axs[2],fvu,'fvu','C  SAE reconstruction','FVU (lower is better)'); axs[2].text(.5,.05,'diff +0.0122 ± 0.0040, p=.034',transform=axs[2].transAxes,ha='center',fontsize=8)
x=np.arange(2); vals=[-0.054,-0.069]; err=[0.007,0.010]
axs[3].bar(x,vals,yerr=err,color=RED,alpha=.88,capsize=4); axs[3].axhline(0,color='black',lw=.8)
axs[3].set_xticks(x,['Exact match','Within 1 patch'],rotation=12); axs[3].set_ylabel('Spatial − baseline accuracy'); axs[3].set_title('D  Cross-view localization'); axs[3].grid(axis='y',alpha=.2)
fig.suptitle('Downstream and reconstruction results at DINOv2 block 0',fontweight='bold',y=1.02)
fig.tight_layout(); fig.savefig('fig_main_results_clear.png',bbox_inches='tight'); plt.close(fig)
# Code variance figure
z=np.load(R/'pca_curve/pca_curves.npz'); ks=np.arange(1,257)
def curves(arm,kind): return np.stack([z[f'dino_b0_postLN|{arm}|s{s}|{kind}'] for s in range(3)])
sp,bl=curves('spatial','raw'),curves('baseline','raw'); sn,bn=curves('spatial','null'),curves('baseline','null')
fig,axs=plt.subplots(1,2,figsize=(9.2,3.45))
for arr,c,l in [(bl,BLUE,'Baseline: observed'),(sp,ORANGE,'Spatial: observed')]:
 m=arr.mean(0); sd=arr.std(0,ddof=1); axs[0].plot(ks,m,color=c,label=l); axs[0].fill_between(ks,m-sd,m+sd,color=c,alpha=.14)
for arr,c,l in [(bn,BLUE,'Baseline: shuffled'),(sn,ORANGE,'Spatial: shuffled')]:
 axs[0].plot(ks,arr.mean(0),color=c,ls='--',lw=1.4,label=l)
axs[0].text(.98,.55,'Effective rank (observed / shuffled)\nSpatial 14.1 / 31.3\nBaseline 23.7 / 41.1',transform=axs[0].transAxes,ha='right',va='center',fontsize=8,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
axs[0].set(xlabel='Number of principal components',ylabel='Cumulative variance explained',title='A  Observed and shuffled PCA curves'); axs[0].legend(frameon=False,fontsize=8); axs[0].grid(alpha=.2)
excess=(sp-sn)-(bl-bn); m=excess.mean(0); sd=excess.std(0,ddof=1)
axs[1].plot(ks,m,color=GREEN); axs[1].fill_between(ks,m-sd,m+sd,color=GREEN,alpha=.18); axs[1].axhline(0,color='black',lw=.8)
axs[1].set(xlabel='Number of principal components',ylabel='Spatial − baseline\nafter each arm’s shuffled control',title='B  Difference after shuffled controls'); axs[1].grid(alpha=.2)
fig.suptitle(r'PCA summaries of the block-0 $G_0$ code',fontweight='bold',y=1.02)
fig.tight_layout(); fig.savefig('fig_code_variance_clear.png',bbox_inches='tight'); plt.close(fig)
# Mechanism / input differences
syn=pd.read_csv(R/'synth_a1_n8/paired_summary_per_seed.csv')
fig,axs=plt.subplots(1,5,figsize=(16.4,3.45))
for _,row in syn.iterrows(): axs[0].plot([0,1],[row.baseline,row.spatial],'-o',color='#999',lw=1,ms=3.5)
axs[0].scatter([0,1],[syn.baseline.mean(),syn.spatial.mean()],s=90,c=[BLUE,ORANGE],edgecolor='black',zorder=4)
axs[0].axhline(0,color='black',lw=.7); axs[0].set_xticks([0,1],['Baseline','Spatial']); axs[0].set_ylabel('Smooth − rough allocation'); axs[0].set_title('A  Synthetic factor allocation'); axs[0].grid(axis='y',alpha=.2)
axs[1].bar([0,1],[1.01,7.91],color=[GREEN,BLUE]); axs[1].set_xticks([0,1],['Synthetic','DINOv2 b0']); axs[1].set_ylabel('Smoothest / roughest tercile variance'); axs[1].set_ylim(0,8.6); axs[1].set_title('B  Channel-energy contrast'); axs[1].grid(axis='y',alpha=.2)
axs[2].bar([0,1],[.047,.347],color=[GREEN,BLUE]); axs[2].set_xticks([0,1],['Synthetic','DINOv2 b0']); axs[2].set_ylabel(r'Channel smoothness $p_{90}-p_{10}$'); axs[2].set_title('C  Smoothness range'); axs[2].grid(axis='y',alpha=.2)
axs[3].bar([0,1],[.229,.315],color=[GREEN,BLUE]); axs[3].set_xticks([0,1],['Synthetic','DINOv2 b0']); axs[3].set_ylabel('Mean per-channel PCA-50 participation'); axs[3].set_ylim(0,.4); axs[3].set_title('D  Leading-PC participation'); axs[3].grid(axis='y',alpha=.2)
w=pd.read_csv(R/'whiten_shortcut/shortcut_g0_share_summary.csv'); w=w[w.ebin.eq(0)]
raw=w[w.cond.eq('dino_b0_postLN')].iloc[0]; wh=w[w.cond.eq('dino_b0_whiten')].iloc[0]
xx=np.arange(2); width=.34
axs[4].bar(xx-width/2,[raw.baseline,wh.baseline],width,color=BLUE,label='Baseline',alpha=.78)
axs[4].bar(xx+width/2,[raw.spatial,wh.spatial],width,color=ORANGE,label='Spatial',alpha=.78)
wd=pd.read_csv(R/'whiten_shortcut/shortcut_g0_share.csv'); wd=wd[wd.ebin.eq(0)]
for j,cond in enumerate(['dino_b0_postLN','dino_b0_whiten']):
 for arm,off,c in [('baseline',-width/2,BLUE),('spatial',width/2,ORANGE)]:
  yy=wd[(wd.cond.eq(cond))&(wd.arm.eq(arm))].sort_values('seed').rho.values
  axs[4].scatter(np.full(len(yy),j+off),yy,s=22,color=c,edgecolor='white',linewidth=.5,zorder=4)
axs[4].set_xticks(xx,['Original','Whitened']); axs[4].set_ylabel('Smoothness–$G_0$ allocation correlation'); axs[4].set_title('E  Routing before and after whitening'); axs[4].legend(frameon=False,fontsize=8); axs[4].grid(axis='y',alpha=.2)
fig.suptitle('Synthetic and DINOv2 input structure and routing',fontweight='bold',y=1.02)
fig.tight_layout(); fig.savefig('fig_mechanism_clear.png',bbox_inches='tight'); plt.close(fig)
