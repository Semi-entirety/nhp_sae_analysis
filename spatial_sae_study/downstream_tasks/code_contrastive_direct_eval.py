"""Direct held-out tests of the spatial SAE contrastive objective on ADE20K."""
from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from scipy.stats import ttest_rel, wilcoxon
from datasets import load_dataset
HERE=Path(__file__).resolve().parent
REPO=Path('/n/home12/binxuwang/T-SAE-Spatial')
for p in [HERE,REPO/'scripts',REPO/'temporal-saes'/'dictionary_learning']:
 sys.path.insert(0,str(p))
from code_smoothness_same_vs_across import encode_images, bootstrap, signflip_p
from fvu_compare import BlockDINO, load_sae
from repro_utils import check_gpu_match, gpu_fingerprint

def rings(side,d):
 out=[]
 for a in range(side*side):
  y,x=divmod(a,side)
  cand=[]
  for yy in range(side):
   for xx in range(side):
    if max(abs(yy-y),abs(xx-x))==d: cand.append(yy*side+xx)
  out.append(cand)
 return out

def pair_indices(nimg,side,n_pairs,seed,dist):
 rr=rings(side,dist); rng=np.random.default_rng(seed+1009*dist)
 A=[]; P=[]; I=[]
 for i in range(nimg):
  for _ in range(n_pairs):
   a=int(rng.integers(side*side)); p=int(rng.choice(rr[a]))
   A.append(i*side*side+a); P.append(i*side*side+p); I.append(i)
 return np.array(A),np.array(P),np.array(I)

def image_reduce(v,img,nimg):
 return np.bincount(img,weights=np.asarray(v),minlength=nimg)/np.bincount(img,minlength=nimg)

def eval_group(z,group,device,batch_images,pairs_per_image,seed):
 n,h,w,d=z.shape; flat=z.reshape(n*h*w,d).astype(np.float32); rec=[]
 # training-aligned batches: 8 images x 64 pairs, same negative construction
 for start in range(0,n,batch_images):
  ni=min(batch_images,n-start)
  A,P,I=pair_indices(ni,h,pairs_per_image,seed+7919*start,1)
  A=A+start*h*w; P=P+start*h*w; I=I+start
  za=torch.from_numpy(flat[A]).to(device); zp=torch.from_numpy(flat[P]).to(device)
  for norm,label,temp in [(False,'raw',1.0),(True,'cosine',0.1)]:
   aa=F.normalize(za,dim=1) if norm else za; pp=F.normalize(zp,dim=1) if norm else zp
   logits=(aa@pp.T)/temp; lab=torch.arange(len(A),device=device)
   loss=.5*(F.cross_entropy(logits,lab,reduction='none')+F.cross_entropy(logits.T,lab,reduction='none'))
   acc=.5*((logits.argmax(1)==lab).float()+(logits.T.argmax(1)==lab).float())
   pos=logits.diag(); neg=(logits.sum(1)-pos)/(len(A)-1); margin=pos-neg
   for metric,val in [('infonce',-loss),('retrieval_top1',acc),('positive_logit',pos),('positive_minus_negative',margin)]:
    vals=image_reduce(val.detach().cpu().numpy(),I-start,ni)
    for j,v in enumerate(vals): rec.append(dict(image=start+j,group=group,test='training_aligned',variant=label,metric=metric,value=float(v)))
 # within-image spatial distance curves; fixed 512 sampled directed pairs per distance
 sims={}
 for dist in (1,2,4,8):
  A,P,I=pair_indices(n,h,512,seed+12345,dist)
  a=flat[A]; p=flat[P]; dot=(a*p).sum(1); cos=dot/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(p,axis=1),1e-8)
  sims[(dist,'raw')]=image_reduce(dot,I,n); sims[(dist,'cosine')]=image_reduce(cos,I,n)
  for variant,val in [('raw',sims[(dist,'raw')]),('cosine',sims[(dist,'cosine')])]:
   for i,v in enumerate(val): rec.append(dict(image=i,group=group,test='distance_similarity',variant=variant,metric=f'distance_{dist}',value=float(v)))
 for dist in (2,4,8):
  for variant in ('raw','cosine'):
   val=sims[(1,variant)]-sims[(dist,variant)]
   for i,v in enumerate(val): rec.append(dict(image=i,group=group,test='adjacent_margin',variant=variant,metric=f'd1_minus_d{dist}',value=float(v)))
 return rec

def paired(raw,nboot,nperm,seed):
 out=[]
 for key,d in raw.groupby(['group','test','variant','metric']):
  p=d.pivot(index='image',columns='arm',values='value').dropna()
  if not {'spatial','baseline'}<=set(p.columns): continue
  delta=(p.spatial-p.baseline).to_numpy(); mean,sd,lo,hi,bp=bootstrap(delta,nboot,seed+len(out))
  try: wp=float(wilcoxon(delta).pvalue)
  except ValueError: wp=1.
  out.append(dict(zip(['group','test','variant','metric'],key),n_images=len(delta),spatial_mean=p.spatial.mean(),baseline_mean=p.baseline.mean(),paired_diff=mean,paired_sd=sd,ci95_low=lo,ci95_high=hi,paired_t_p=ttest_rel(p.spatial,p.baseline).pvalue,wilcoxon_p=wp,signflip_p=signflip_p(delta,nperm,seed+1000+len(out)),bootstrap_p=bp))
 return pd.DataFrame(out)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--ckpt',action='append',required=True); ap.add_argument('--dino_block',type=int,required=True); ap.add_argument('--n_images',type=int,default=200); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--batch_images',type=int,default=8); ap.add_argument('--pairs_per_image',type=int,default=64); ap.add_argument('--n_boot',type=int,default=5000); ap.add_argument('--n_perm',type=int,default=100000); ap.add_argument('--device',default='cuda'); ap.add_argument('--outdir',required=True); args=ap.parse_args()
 out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True); print('[GPU]',gpu_fingerprint(args.device),flush=True)
 specs=[(x.split(':',1)[0],Path(x.split(':',1)[1])) for x in args.ckpt]; check_gpu_match([p.parent for _,p in specs])
 ds=load_dataset('1aurent/ADE20K',split='validation',cache_dir=os.environ.get('HF_HOME')); rng=np.random.default_rng(args.seed); idx=rng.choice(len(ds),min(args.n_images,len(ds)),replace=False); images=[ds[int(i)]['image'] for i in idx]
 dino=BlockDINO(args.device,dino_block=args.dino_block); rows=[]
 for arm,path in specs:
  ae,groups=load_sae(str(path),args.device); full=encode_images(ae,dino,images,16,args.device); g0=int(groups[0])
  for group,z in [('Full',full),('G0',full[...,:g0]),('G1',full[...,g0:])]:
   print(arm,group,z.shape,flush=True)
   q=eval_group(z,group,args.device,args.batch_images,args.pairs_per_image,args.seed+100*args.dino_block)
   for r in q:r['arm']=arm;r['block']=args.dino_block
   rows+=q
 raw=pd.DataFrame(rows); raw.to_csv(out/'contrastive_direct_raw.csv',index=False); pp=paired(raw,args.n_boot,args.n_perm,args.seed+47*args.dino_block); pp.insert(0,'block',args.dino_block); pp.to_csv(out/'contrastive_direct_paired.csv',index=False); json.dump(vars(args),open(out/'run_config.json','w'),indent=2); print(pp.to_string(index=False),flush=True)
if __name__=='__main__':main()
