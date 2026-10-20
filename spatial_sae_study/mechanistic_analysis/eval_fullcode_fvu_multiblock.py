from pathlib import Path
import argparse, io, sys
import pandas as pd, torch
from PIL import Image
ROOT=Path('/n/home12/binxuwang/T-SAE-Spatial')
for p in [ROOT/'scripts',ROOT/'temporal-saes'/'dictionary_learning']:
 sys.path.insert(0,str(p))
from arm_pipeline import pipeline_from_ckpt, assert_same_pipeline
from fvu_compare import load_sae

def ibyte(r):
 d=r['image']; return d['bytes'] if isinstance(d,dict) else d
@torch.no_grad()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--block',type=int,required=True); ap.add_argument('--n_images',type=int,default=200); ap.add_argument('--outdir',required=True); ap.add_argument('--device',default='cuda'); a=ap.parse_args()
 S=Path('/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/T-SAE-Spatial')
 if a.block==0: names={'spatial':'spatial_block0','baseline':'spatial_block0_baseline'}
 elif a.block==2: names={x:f'dino_b2_postLN_{x}_s0' for x in ['spatial','baseline']}
 else: names={'spatial':f'spatial_block{a.block}','baseline':f'spatial_block{a.block}_baseline'}
 pipes={x:pipeline_from_ckpt(S/n,a.device) for x,n in names.items()}; assert_same_pipeline(*pipes.values())
 aes={x:load_sae(str(S/n/'ae_latest.pt'),a.device)[0] for x,n in names.items()}
 df=pd.read_parquet(ROOT/'imagenet_val_2k.parquet').sample(a.n_images,random_state=0)
 imgs=[]
 for _,r in df.iterrows():
  try: imgs.append(Image.open(io.BytesIO(ibyte(r))).convert('RGB'))
  except: pass
 rows=[]
 for arm in ['spatial','baseline']:
  xs=[]; rs=[]
  for im in imgs:
   x=pipes[arm].tokens(im); z=aes[arm].encode(x,use_threshold=False); xh=aes[arm].decode(z)
   xs.append(x.cpu()); rs.append((x-xh).cpu())
  X=torch.cat(xs); R=torch.cat(rs); fvu=float(R.var(0).sum()/X.var(0).sum())
  rows.append(dict(block=a.block,arm=arm,n_images=len(imgs),n_tokens=len(X),fvu=fvu,explained_variance=1-fvu))
  print(rows[-1],flush=True)
 out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out/f'fvu_b{a.block}.csv',index=False)
if __name__=='__main__': main()
