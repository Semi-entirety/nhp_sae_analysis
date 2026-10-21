import csv, os, math
from xml.sax.saxutils import escape
ROOT='/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/nhp_sae_analysis/results'
OUT='smoothness_effect_figures'
BLOCKS=[0,2,4,6,8,10,11]
METRICS=[('cosine','Cosine smoothness'),('fourier','Fourier smoothness'),('wavelet','Wavelet smoothness'),('multiscale','Multiscale smoothness'),('lipschitz','Lipschitz smoothness')]
GROUPS=[('Full','#30343b'),('G0','#2a6fbb'),('G1','#e07a28')]
os.makedirs(OUT,exist_ok=True)
rows=[]
for b in BLOCKS:
 p=f'{ROOT}/code_smoothness_b{b}_s0/smoothness_spatial_vs_baseline_paired.csv'
 with open(p,newline='') as f:
  for r in csv.DictReader(f):
   if r['quantity']=='effect': rows.append(r)

def fmt(v):
 return f'{v:.3f}' if abs(v)>=.01 else f'{v:.4f}'

def svg_for(metric,title):
 data={}
 vals=[]
 for g,_ in GROUPS:
  seq=[]
  for b in BLOCKS:
   rr=next(r for r in rows if int(r['block'])==b and r['group']==g and r['metric']==metric)
   y=float(rr['paired_diff']); lo=float(rr['ci95_low']); hi=float(rr['ci95_high'])
   seq.append((b,y,lo,hi)); vals += [lo,hi]
  data[g]=seq
 lo=min(vals+[0]); hi=max(vals+[0]); span=max(hi-lo,.01); pad=.14*span
 lo-=pad; hi+=pad
 W,H=920,610; L,R,T,B=112,42,95,92; pw=W-L-R; ph=H-T-B
 x=lambda b:L+BLOCKS.index(b)*pw/(len(BLOCKS)-1)
 offsets={'Full':-7,'G0':0,'G1':7}
 y=lambda v:T+(hi-v)*ph/(hi-lo)
 # ticks
 step_raw=(hi-lo)/6
 base=10**math.floor(math.log10(step_raw))
 step=min([1,2,2.5,5,10], key=lambda q:abs(q*base-step_raw))*base
 t0=math.ceil(lo/step)*step
 ticks=[]; v=t0
 while v<=hi+1e-12: ticks.append(v); v+=step
 s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
 '<rect width="100%" height="100%" fill="white"/>',
 f'<text x="{W/2}" y="38" text-anchor="middle" font-family="Arial" font-size="25" font-weight="700">{escape(title)}: Spatial − Baseline effect</text>',
 f'<text x="{W/2}" y="66" text-anchor="middle" font-family="Arial" font-size="14" fill="#555">Positive values indicate greater within-image smoothness for the spatial SAE</text>']
 # grid/ticks
 for tv in ticks:
  yy=y(tv); s.append(f'<line x1="{L}" y1="{yy:.1f}" x2="{W-R}" y2="{yy:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
  s.append(f'<text x="{L-12}" y="{yy+5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#444">{fmt(tv)}</text>')
 if lo<=0<=hi:
  yy=y(0); s.append(f'<line x1="{L}" y1="{yy:.1f}" x2="{W-R}" y2="{yy:.1f}" stroke="#777" stroke-width="1.7" stroke-dasharray="7,6"/>')
 # axes
 s += [f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="#222" stroke-width="1.5"/>',f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="#222" stroke-width="1.5"/>']
 for b in BLOCKS:
  xx=x(b); s.append(f'<text x="{xx:.1f}" y="{H-B+28}" text-anchor="middle" font-family="Arial" font-size="14">{b}</text>')
 s.append(f'<text x="{(L+W-R)/2}" y="{H-28}" text-anchor="middle" font-family="Arial" font-size="16">DINOv2 block</text>')
 s.append(f'<text x="28" y="{(T+H-B)/2}" transform="rotate(-90 28 {(T+H-B)/2})" text-anchor="middle" font-family="Arial" font-size="16">Effect difference (Spatial − Baseline)</text>')
 # errorbars + lines + filled points
 for g,c in GROUPS:
  pts=' '.join(f'{x(b)+offsets[g]:.1f},{y(v):.1f}' for b,v,_,_ in data[g])
  s.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.5"/>')
  for b,v,a,z in data[g]:
   xx=x(b)+offsets[g]; ya=y(a); yz=y(z); yy=y(v)
   s += [f'<line x1="{xx:.1f}" y1="{ya:.1f}" x2="{xx:.1f}" y2="{yz:.1f}" stroke="{c}" stroke-width="1.5"/>',f'<line x1="{xx-5:.1f}" y1="{ya:.1f}" x2="{xx+5:.1f}" y2="{ya:.1f}" stroke="{c}" stroke-width="1.5"/>',f'<line x1="{xx-5:.1f}" y1="{yz:.1f}" x2="{xx+5:.1f}" y2="{yz:.1f}" stroke="{c}" stroke-width="1.5"/>',f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5.2" fill="{c}" stroke="white" stroke-width="1.2"/>']
 # legend
 lx=L+12
 for i,(g,c) in enumerate(GROUPS):
  xx=lx+i*120; s += [f'<line x1="{xx}" y1="{T+17}" x2="{xx+28}" y2="{T+17}" stroke="{c}" stroke-width="2.5"/>',f'<circle cx="{xx+14}" cy="{T+17}" r="5" fill="{c}"/>',f'<text x="{xx+36}" y="{T+22}" font-family="Arial" font-size="14">{g}</text>']
 s.append(f'<text x="{W-R}" y="{H-15}" text-anchor="end" font-family="Arial" font-size="11" fill="#666">Points: paired mean differences; bars: 95% image-bootstrap CI</text>')
 s.append('</svg>')
 path=f'{OUT}/smoothness_effect_{metric}.svg'
 open(path,'w').write('\n'.join(s))
 return path
for m,t in METRICS: print(svg_for(m,t))
