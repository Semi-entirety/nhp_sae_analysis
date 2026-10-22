import csv,os,math
from xml.sax.saxutils import escape
R='/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/nhp_sae_analysis/results';B=[0,2,4,6,8,10,11]; groups=['Full','G0','G1']; series=[('Across-image mosaic','#8b5cf6'),('Batch-random patch','#0f9d8a')]
data={}
for g in groups:
 for b in B:
  a=next(x for x in csv.DictReader(open(f'{R}/code_smoothness_ade20k_b{b}_s0/smoothness_spatial_vs_baseline_paired.csv')) if x['group']==g and x['metric']=='cosine' and x['quantity']=='effect')
  q=next(x for x in csv.DictReader(open(f'{R}/batch_random_effect_ade20k_b{b}_s0/batch_random_paired.csv')) if x['group']==g and x['quantity']=='effect')
  data[(g,b,'Across-image mosaic')]=tuple(map(float,[a['paired_diff'],a['ci95_low'],a['ci95_high']]))
  data[(g,b,'Batch-random patch')]=tuple(map(float,[q['paired_diff'],q['ci95_low'],q['ci95_high']]))
W,H=1200,430;S=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/>','<text x="600" y="32" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700">Effect difference depends on the negative control</text>','<text x="600" y="56" text-anchor="middle" font-family="Arial" font-size="13" fill="#555">ADE20K cosine: [(adjacent − control) Spatial] − [(adjacent − control) Baseline]</text>']
for j,g in enumerate(groups):
 ox=j*400;L,Rr,T,Bb=72,25,92,60;pw=400-L-Rr;ph=H-T-Bb;vals=[]
 for nm,_ in series:
  for b in B:vals += list(data[(g,b,nm)][1:])
 lo=min(vals+[0]);hi=max(vals+[0]);pad=.12*(hi-lo);lo-=pad;hi+=pad
 x=lambda b:ox+L+B.index(b)*pw/6;y=lambda v:T+(hi-v)*ph/(hi-lo)
 S.append(f'<text x="{ox+200}" y="82" text-anchor="middle" font-family="Arial" font-size="18" font-weight="600">{g}</text>')
 for k in range(6):
  v=lo+k*(hi-lo)/5;yy=y(v);S += [f'<line x1="{ox+L}" y1="{yy:.1f}" x2="{ox+400-Rr}" y2="{yy:.1f}" stroke="#e5e7eb"/>',f'<text x="{ox+L-8}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{v:.3f}</text>']
 S.append(f'<line x1="{ox+L}" y1="{y(0):.1f}" x2="{ox+400-Rr}" y2="{y(0):.1f}" stroke="#777" stroke-dasharray="6,5"/>')
 for b in B:S.append(f'<text x="{x(b):.1f}" y="{H-Bb+22}" text-anchor="middle" font-family="Arial" font-size="11">{b}</text>')
 S += [f'<line x1="{ox+L}" y1="{T}" x2="{ox+L}" y2="{H-Bb}" stroke="#222"/>',f'<line x1="{ox+L}" y1="{H-Bb}" x2="{ox+400-Rr}" y2="{H-Bb}" stroke="#222"/>',f'<text x="{ox+220}" y="{H-14}" text-anchor="middle" font-family="Arial" font-size="13">DINOv2 block</text>']
 for si,(nm,c) in enumerate(series):
  off=(-5,5)[si];pts=' '.join(f'{x(b)+off:.1f},{y(data[(g,b,nm)][0]):.1f}' for b in B);S.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.4"/>')
  for b in B:
   v,a,z=data[(g,b,nm)];xx=x(b)+off;S += [f'<line x1="{xx:.1f}" y1="{y(a):.1f}" x2="{xx:.1f}" y2="{y(z):.1f}" stroke="{c}"/>',f'<circle cx="{xx:.1f}" cy="{y(v):.1f}" r="4.5" fill="{c}" stroke="white"/>']
 if j==0:
  for si,(nm,c) in enumerate(series):
   xx=ox+L+si*155;yy=T+15;S += [f'<circle cx="{xx}" cy="{yy}" r="4" fill="{c}"/>',f'<text x="{xx+9}" y="{yy+4}" font-family="Arial" font-size="11">{nm}</text>']
S.append('</svg>');os.makedirs('batch_random_figures',exist_ok=True);open('batch_random_figures/control_comparison.svg','w').write('\n'.join(S))
