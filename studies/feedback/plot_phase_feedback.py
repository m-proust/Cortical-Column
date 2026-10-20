"""Feedback off/on phase curves from existing fits; no alpha-amplitude splits."""
from pathlib import Path
import json,os
import numpy as np
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(Path.home()/'.cache'/'mpl_feedback_protocol'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
DATA=REPO/'saved_trials/cosyne/feedback_protocol'
OUT=DATA/'phase_curves';OUT.mkdir(exist_ok=True)                 # arrays, summary, methods
FIG=REPO/'figures/cosyne/feedback_protocol';FIG.mkdir(parents=True,exist_ok=True)
m=json.loads((DATA/'metrics.json').read_text())['probes']
seeds=m['seeds'];grid=np.linspace(-np.pi,np.pi,361);deg=np.degrees(grid)
boot=np.random.default_rng(38192).integers(0,len(seeds),(5000,len(seeds)))
arrays={};summary={}
for ref,label,filename in [('source','L6 alpha phase','feedback_vs_no_feedback.png'),('local','Local alpha phase','feedback_vs_no_feedback_local_phase.png')]:
 fig,axs=plt.subplots(1,2,figsize=(11,5.2),sharey=True,layout='constrained')
 for ax,L in zip(axs,['L4C','L23']):
  for c,color,name in [(0,'#356b9b','Feedback off'),(1,'#c75435','Feedback on')]:
   rows=[next(r for r in m['responses'] if r['seed']==s and r['layer']==L and r['feedback']==c)[ref] for s in seeds]
   curves=np.array([r['mean']+r['cos']*np.cos(grid)+r['sin']*np.sin(grid) for r in rows])
   mean=curves.mean(0);ci=np.quantile(curves[boot].mean(1),[.025,.975],axis=0)
   arrays[f'{ref}_{L}_{c}_curves']=curves
   ax.fill_between(deg,*ci,color=color,alpha=.16,linewidth=0)
   ax.plot(deg,mean,color=color,lw=2.6,label=name)
   summary[f'{ref}_{L}_{c}']={'mean_response':float(np.mean([r['mean'] for r in rows])),
       'mean_individual_modulation':float(np.mean([r['amplitude'] for r in rows])),
       'mean_curve_modulation':float(np.hypot(np.mean([r['cos'] for r in rows]),np.mean([r['sin'] for r in rows])))}
  ax.set(title='Layer 2/3' if L=='L23' else 'Layer 4C',xlabel=f'{label} (degrees)',xticks=[-180,-90,0,90,180],xlim=(-180,180))
  ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
 axs[0].set_ylabel('Evoked response (pulse − sham, Hz/cell)');axs[0].legend(frameon=False,loc='upper left')
 fig.suptitle('Excitability across alpha phase: feedback off versus on\nFeedforward maintained in both conditions',fontsize=13)
 fig.supxlabel('25 pA, 10 ms probes · response 0–15 ms · 6 networks × 96 onsets per condition\nLines: mean network fits; shading: pointwise 95% network-bootstrap intervals',fontsize=9)
 fig.savefig(FIG/filename,dpi=190);fig.savefig(FIG/filename.replace('.png','.svg'));plt.close(fig)
np.savez_compressed(OUT/'phase_curve_arrays.npz',phase_degrees=deg,seeds=seeds,**arrays)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
(OUT/'METHODS.md').write_text('''# Feedback off versus on: excitability against alpha phase

No high/low-alpha split is used. All 96 probe onsets from each of six networks enter each condition's phase fit. Responses are 25 pA/10 ms pulse minus identical-state sham, counted over 0–15 ms. Phase is the retrospective 7–16 Hz Hilbert phase of the unpulsed E-population rate. The primary plot uses each condition's L6 phase; the companion uses the receiving layer's phase.

Curves reuse the original per-network fits: intercept + cosine × cos(phase) + sine × sin(phase). Lines average the six fitted curves with equal network weight. Shading is a pointwise 95% bootstrap interval across networks, using 5000 resamples. Identical resample indices are used for the paired conditions. These are not simultaneous confidence bands or a test based on individual onsets. No phase alignment to each curve's preferred phase is performed.

Both arms maintain feedforward input from time zero; feedback is added at 2000 ms in the on arm. Probe onsets begin at 4000 ms. The ordinate retains absolute response differences; responses are not normalized or rate matched. A mean curve can have less modulation than the mean of individual fitted amplitudes when preferred phases differ across networks. Both quantities are retained in summary.json.
''')
print('Saved feedback/no-feedback curves using all onsets.')
