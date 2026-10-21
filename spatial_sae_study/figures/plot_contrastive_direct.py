import csv,math,os
from xml.sax.saxutils import escape
R='/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/nhp_sae_analysis/results'; OUT='contrastive_direct_figures'; os.makedirs(OUT,exist_ok=True)
B=[0,2,4,6,8,10,11]; GROUPS=[('Full','#30343b'),('G0','#2a6fbb'),('G1','#e07a28')]; rows=[]
for b in B:
 for x in csv.DictReader(open(f'{R}/contrastive_direct_ade20k_b{b}_s0/contrastive_direct_paired.csv')): x['block']=b; rows.append(x)
def nice_ticks(lo,hi,n=5):
 span=max(hi-lo,1e-9); raw=span/n; base=10**math.floor(math.log10(raw)); st=min([1,2,2.5,5,10],key=lambda q:abs(q*base-raw))*base; v=math.ceil(lo/st)*st; out=[]
 while v<=hi+1e-12: out.append(v);v+=st
 return out
def fmt(v):
 if abs(v)>=100:return f'{v:.0f}'
 if abs(v)>=1:return f'{v:.1f}'
 if abs(v)>=.01:return f'{v:.3f}'
 return f'{v:.4f}'
def make(name,title,subtitle,test,variant,metrics,labels):
 cols=2; nr=math.ceil(len(metrics)/2); PW,PH=520,370; W,H=PW*cols,PH*nr+80; S=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/>',f'<text x="{W/2}" y="30" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700">{escape(title)}</text>',f'<text x="{W/2}" y="55" text-anchor="middle" font-family="Arial" font-size="13" fill="#555">{escape(subtitle)}</text>']
 for k,(met,lab) in enumerate(zip(metrics,labels)):
  row,col=divmod(k,2); ox=col*PW; oy=70+row*PH; L,Rr,T,Bb=78,24,46,62; pw=PW-L-Rr; ph=PH-T-Bb
  dat={}; vals=[]
  for g,c in GROUPS:
   seq=[]
   for b in B:
    x=next(q for q in rows if q['block']==b and q['group']==g and q['test']==test and q['variant']==variant and q['metric']==met); v=float(x['paired_diff']);lo=float(x['ci95_low']);hi=float(x['ci95_high']);seq.append((b,v,lo,hi));vals += [lo,hi]
   dat[g]=seq
  lo=min(vals+[0]);hi=max(vals+[0]);pad=.12*max(hi-lo,1e-5);lo-=pad;hi+=pad
  xx=lambda b:ox+L+B.index(b)*pw/(len(B)-1); yy=lambda v:oy+T+(hi-v)*ph/(hi-lo)
  S.append(f'<text x="{ox+PW/2}" y="{oy+23}" text-anchor="middle" font-family="Arial" font-size="17" font-weight="600">{escape(lab)}</text>')
  for tv in nice_ticks(lo,hi):
   y=yy(tv);S += [f'<line x1="{ox+L}" y1="{y:.1f}" x2="{ox+PW-Rr}" y2="{y:.1f}" stroke="#e5e7eb"/>',f'<text x="{ox+L-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{fmt(tv)}</text>']
  if lo<=0<=hi:S.append(f'<line x1="{ox+L}" y1="{yy(0):.1f}" x2="{ox+PW-Rr}" y2="{yy(0):.1f}" stroke="#777" stroke-dasharray="6,5"/>')
  S += [f'<line x1="{ox+L}" y1="{oy+T}" x2="{ox+L}" y2="{oy+PH-Bb}" stroke="#222"/>',f'<line x1="{ox+L}" y1="{oy+PH-Bb}" x2="{ox+PW-Rr}" y2="{oy+PH-Bb}" stroke="#222"/>']
  for b in B:S.append(f'<text x="{xx(b):.1f}" y="{oy+PH-Bb+22}" text-anchor="middle" font-family="Arial" font-size="11">{b}</text>')
  S.append(f'<text x="{ox+PW/2}" y="{oy+PH-18}" text-anchor="middle" font-family="Arial" font-size="13">DINOv2 block</text>')
  for gi,(g,c) in enumerate(GROUPS):
   off=(-6,0,6)[gi];pts=' '.join(f'{xx(b)+off:.1f},{yy(v):.1f}' for b,v,_,_ in dat[g]);S.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.2"/>')
   for b,v,a,z in dat[g]:
    x=xx(b)+off;S += [f'<line x1="{x:.1f}" y1="{yy(a):.1f}" x2="{x:.1f}" y2="{yy(z):.1f}" stroke="{c}"/>',f'<circle cx="{x:.1f}" cy="{yy(v):.1f}" r="4.5" fill="{c}" stroke="white"/>']
  if k==0:
   for gi,(g,c) in enumerate(GROUPS):
    x=ox+L+gi*100; y=oy+T+15;S += [f'<circle cx="{x}" cy="{y}" r="4" fill="{c}"/>',f'<text x="{x+9}" y="{y+4}" font-family="Arial" font-size="11">{g}</text>']
 S.append('</svg>');open(f'{OUT}/{name}.svg','w').write('\n'.join(S))
make('training_aligned_raw','Held-out training-aligned contrastive test — raw dot product','Spatial − Baseline; positive favors Spatial; bars are 95% image-bootstrap CI','training_aligned','raw',['infonce','positive_logit','positive_minus_negative','retrieval_top1'],['Negative InfoNCE loss (higher is better)','Positive-pair logit','Positive − mean-negative logit','Top-1 neighbor retrieval'])
make('training_aligned_cosine','Held-out contrastive test — normalized cosine','Spatial − Baseline; normalized codes, temperature 0.1','training_aligned','cosine',['infonce','positive_logit','positive_minus_negative','retrieval_top1'],['Negative InfoNCE loss (higher is better)','Positive-pair logit','Positive − mean-negative logit','Top-1 neighbor retrieval'])
make('within_image_cosine','Within-image adjacent-vs-distant margin — cosine','Spatial − Baseline; positive means Spatial has a steeper spatial similarity gradient','adjacent_margin','cosine',['d1_minus_d2','d1_minus_d4','d1_minus_d8'],['Distance 1 − distance 2','Distance 1 − distance 4','Distance 1 − distance 8'])
make('within_image_raw','Within-image adjacent-vs-distant margin — raw dot product','Spatial − Baseline; positive means Spatial has a steeper spatial similarity gradient','adjacent_margin','raw',['d1_minus_d2','d1_minus_d4','d1_minus_d8'],['Distance 1 − distance 2','Distance 1 − distance 4','Distance 1 − distance 8'])
