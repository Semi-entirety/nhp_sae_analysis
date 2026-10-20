"""A3: image-level bootstrap for DINOv2-b0 substrate correlations.

The resampling unit is an IMAGE, never a channel.  Each bootstrap replicate
reconstructs the full 768x768 covariance from per-image sufficient statistics,
refits PCA-50, and recomputes energy, PCA participation, Spearman and partial
Spearman. This preserves the within-image/channel dependence structure.
"""
from __future__ import annotations
import argparse, io, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from scipy import stats

THIS=Path(__file__).resolve()
for p in [THIS.parent, THIS.parent.parent, THIS.parent.parent/'temporal-saes'/'dictionary_learning']:
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from new_spatial_train_triplet import BlockAwareDINOFeatureExtractor, build_transform
from dino_smoothness import autocorr_per_feature
from block_shortcut_sweep import partial_spearman

def ibytes(r):
    d=r['image']; return d['bytes'] if isinstance(d,dict) else d

def rho(a,b): return float(stats.spearmanr(a,b).statistic)

def metrics(ac, cov, k):
    cov=(cov+cov.T)/2
    vals, vecs=torch.linalg.eigh(cov)
    vals=vals.clamp_min(0)
    vals=vals[-k:]; vecs=vecs[:,-k:]
    energy=torch.diag(cov).clamp_min(1e-12)
    explained=(vecs.square()*vals[None,:]).sum(1)
    lr=(explained/energy).clamp(0,1)
    e=energy.cpu().numpy(); l=lr.cpu().numpy(); c=ac.cpu().numpy()
    ok=np.isfinite(c)&np.isfinite(e)&np.isfinite(l)
    ce,cl,el=rho(c[ok],e[ok]),rho(c[ok],l[ok]),rho(e[ok],l[ok])
    pcl,_=partial_spearman(cl,ce,el,int(ok.sum()))
    pce,_=partial_spearman(ce,cl,el,int(ok.sum()))
    return dict(rho_contig_energy=ce,rho_contig_lowrank=cl,rho_energy_lowrank=el,
                partial_contig_lowrank_given_energy=pcl,
                partial_contig_energy_given_lowrank=pce,
                pca50_evr=float(vals.sum()/torch.diag(cov).sum().clamp_min(1e-12)))

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--parquet',default='/n/home12/binxuwang/T-SAE-Spatial/imagenet_val_2k.parquet')
    a.add_argument('--dino_repo',required=True); a.add_argument('--outdir',required=True)
    a.add_argument('--n_images',type=int,default=200); a.add_argument('--n_boot',type=int,default=500)
    a.add_argument('--seed',type=int,default=20260909); a.add_argument('--batch',type=int,default=20)
    args=a.parse_args(); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    dev='cuda'; rows=pd.read_parquet(args.parquet).sample(args.n_images,random_state=0)
    T=build_transform('dinov2',224)
    imgs=[]
    for _,r in rows.iterrows(): imgs.append(T(Image.open(io.BytesIO(ibytes(r))).convert('RGB')))
    imgs=torch.stack(imgs); ex=BlockAwareDINOFeatureExtractor('dinov2_vitb14',args.dino_repo,dev,dino_block=0,norm=True)
    toks=[]
    with torch.no_grad():
        for i in range(0,len(imgs),args.batch): toks.append(ex.patch_tokens(imgs[i:i+args.batch].to(dev)).double().cpu())
    X=torch.cat(toks).double() # I,P,D CPU
    I,P,D=X.shape
    # Per-image autocorrelation; raw DINO channels are variable in all images.
    maps=X.permute(0,2,1).reshape(I,D,int(P**.5),int(P**.5)).float().to(dev)
    xc=maps-maps.mean((-2,-1),keepdim=True); v=xc.square().mean((-2,-1))
    h=(xc[:,:,:,:-1]*xc[:,:,:,1:]).mean((-2,-1)); w=(xc[:,:,:-1,:]*xc[:,:,1:,:]).mean((-2,-1))
    ac_i=(((h+w)/2)/(v+1e-12)).double().cpu()
    del maps,xc,v,h,w
    # Sufficient statistics keep full cross-channel covariance within every image.
    sums=X.sum(1).to(dev); cross=torch.einsum('ipd,ipe->ide',X.to(dev),X.to(dev))
    def from_counts(counts):
        ww=torch.as_tensor(counts,dtype=torch.double,device=dev)
        s=torch.einsum('i,id->d',ww,sums); q=torch.einsum('i,ide->de',ww,cross)
        n=float(counts.sum()*P); mu=s/n; cov=q/n-torch.outer(mu,mu)
        ac=torch.einsum('i,id->d',torch.as_tensor(counts,dtype=torch.double),ac_i)/float(counts.sum())
        return metrics(ac.to(dev),cov,50)
    full=from_counts(np.ones(I,dtype=int)); print('[full]',full,flush=True)
    rng=np.random.default_rng(args.seed); rec=[]
    for b in range(args.n_boot):
        idx=rng.integers(0,I,size=I); cnt=np.bincount(idx,minlength=I)
        rec.append(dict(bootstrap=b,**from_counts(cnt)))
        if (b+1)%25==0: print(f'[bootstrap] {b+1}/{args.n_boot}',flush=True)
    d=pd.DataFrame(rec); d.to_csv(out/'bootstrap_draws.csv',index=False)
    rows=[]
    for col,val in full.items():
        q=np.quantile(d[col],[.025,.5,.975]); rows.append(dict(metric=col,point=val,boot_median=q[1],ci_low=q[0],ci_high=q[2]))
    pd.DataFrame(rows).to_csv(out/'bootstrap_summary.csv',index=False)
    (out/'run_args.json').write_text(json.dumps(vars(args)|{'resampling_unit':'image','pca_refit_each_draw':True},indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__': main()
