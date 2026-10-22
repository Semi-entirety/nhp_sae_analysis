"""Adjacent cosine minus random-patch-in-batch cosine, ADE20K, paired arms."""
import argparse,os,sys,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import ttest_rel,wilcoxon
from datasets import load_dataset
REPO=Path('/n/home12/binxuwang/T-SAE-Spatial'); HERE=Path(__file__).resolve().parent
for p in [HERE,REPO/'scripts',REPO/'temporal-saes'/'dictionary_learning']: sys.path.insert(0,str(p))
from code_smoothness_same_vs_across import encode_images,bootstrap,signflip_p
from fvu_compare import BlockDINO,load_sae
from repro_utils import check_gpu_match,gpu_fingerprint

def adjacent_edges(side):
 a=[];p=[]
 for y in range(side):
  for x in range(side-1):a.append(y*side+x);p.append(y*side+x+1)
 for y in range(side-1):
  for x in range(side):a.append(y*side+x);p.append((y+1)*side+x)
 return np.array(a),np.array(p)
def cos(a,b):
 return (a*b).sum(1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-8)
def evaluate(codes,arm,group,batch_images,reps,seed):
 n,h,w,d=codes.shape; A,P=adjacent_edges(h); out=[]
 for bs in range(0,n,batch_images):
  be=min(n,bs+batch_images); pool=codes[bs:be].reshape(-1,d).astype(np.float32)
  for i in range(bs,be):
   z=codes[i].reshape(-1,d).astype(np.float32); same=float(cos(z[A],z[P]).mean()); null=[]
   local=i-bs
   for r in range(reps):
    rng=np.random.default_rng(seed+1000003*r+9176*i)
    q=rng.integers(0,len(pool),size=len(A))
    # Exclude the identical anchor and its designated positive when donor is target image.
    bad=(q==local*h*w+A)|(q==local*h*w+P)
    while bad.any():
     q[bad]=rng.integers(0,len(pool),size=int(bad.sum()));bad=(q==local*h*w+A)|(q==local*h*w+P)
    null.append(float(cos(z[A],pool[q]).mean()))
   across=float(np.mean(null));out.append(dict(image=i,arm=arm,group=group,same_image=same,batch_random=across,effect=same-across))
 return out
def paired(raw,nboot,nperm,seed):
 out=[]
 for group,d in raw.groupby('group'):
  for q in ['same_image','batch_random','effect']:
   p=d.pivot(index='image',columns='arm',values=q).dropna();delta=(p.spatial-p.baseline).to_numpy();m,sd,lo,hi,bp=bootstrap(delta,nboot,seed+len(out))
   try:wp=float(wilcoxon(delta).pvalue)
   except ValueError:wp=1.
   out.append(dict(group=group,quantity=q,n_images=len(delta),spatial_mean=p.spatial.mean(),baseline_mean=p.baseline.mean(),paired_diff=m,paired_sd=sd,ci95_low=lo,ci95_high=hi,paired_t_p=ttest_rel(p.spatial,p.baseline).pvalue,wilcoxon_p=wp,signflip_p=signflip_p(delta,nperm,seed+1000+len(out)),bootstrap_p=bp))
 return pd.DataFrame(out)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--ckpt',action='append',required=True);ap.add_argument('--dino_block',type=int,required=True);ap.add_argument('--n_images',type=int,default=200);ap.add_argument('--batch_images',type=int,default=8);ap.add_argument('--null_reps',type=int,default=10);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--n_boot',type=int,default=5000);ap.add_argument('--n_perm',type=int,default=100000);ap.add_argument('--device',default='cuda');ap.add_argument('--outdir',required=True);a=ap.parse_args()
 out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);print('[GPU]',gpu_fingerprint(a.device),flush=True);spec=[(x.split(':',1)[0],Path(x.split(':',1)[1])) for x in a.ckpt];check_gpu_match([p.parent for _,p in spec])
 ds=load_dataset('1aurent/ADE20K',split='validation',cache_dir=os.environ.get('HF_HOME'));rng=np.random.default_rng(a.seed);idx=rng.choice(len(ds),min(a.n_images,len(ds)),replace=False);images=[ds[int(i)]['image'] for i in idx];dino=BlockDINO(a.device,dino_block=a.dino_block);rows=[]
 for arm,path in spec:
  ae,groups=load_sae(str(path),a.device);full=encode_images(ae,dino,images,16,a.device);g0=int(groups[0])
  for group,z in [('Full',full),('G0',full[...,:g0]),('G1',full[...,g0:])]: rows+=evaluate(z,arm,group,a.batch_images,a.null_reps,a.seed+100*a.dino_block)
 raw=pd.DataFrame(rows);raw.insert(0,'block',a.dino_block);raw.to_csv(out/'batch_random_raw.csv',index=False);pp=paired(raw,a.n_boot,a.n_perm,a.seed+47*a.dino_block);pp.insert(0,'block',a.dino_block);pp.to_csv(out/'batch_random_paired.csv',index=False);json.dump(vars(a),open(out/'run_config.json','w'),indent=2);print(pp.to_string(index=False),flush=True)
if __name__=='__main__':main()
