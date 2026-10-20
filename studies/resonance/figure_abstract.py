"""Two panels for the abstract: predicted against measured peak (baseline + 14 held-out timing changes),
and the phase budget of the E-SOM loop at its closing frequency. Colours follow cosyne_abstract.tex.

    ~/Desktop/venv/bin/python studies/resonance/figure_abstract.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import freqmech as fm

NAVY, BLUE, LIGHT, LORANGE, ORANGE, GREY = '#002255', '#2F6DB5', '#8FBCE6', '#FFB27A', '#FF6D1E', '#8C8C8C'
FIG = os.path.join(os.path.dirname(os.path.dirname(fm.HERE)), 'figures', 'frequency_mechanism')
r = json.load(open(os.path.join(fm.OUT, 'L6_results.json')))
rows = [x for x in r['held_out'] if 'SS' not in x['condition']]
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'serif'})
fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.3), layout='constrained', gridspec_kw={'width_ratios': [1, 1.25]})

groups = [('SOM decay 10-40 ms', NAVY, 'o', lambda c: 'tauSOM' in c), ('loop delay', BLUE, '^', lambda c: 'loopdelay' in c)]
for lab, col, mk, sel in groups:
    rr = [x for x in rows if sel(x['condition'])]
    m = np.array([x['measured_hz'] for x in rr]); p = np.array([x['predicted_hz'] for x in rr])
    ax[0].errorbar(m, p, xerr=np.abs(np.array([x['measured_ci'] for x in rr]).T - m),
                   yerr=np.abs(np.array([x['predicted_ci'] for x in rr]).T - p),
                   fmt=mk, ms=3.5, color=col, lw=0.7, label=lab)
b = r['baseline']
ax[0].plot(b['measured_peak_hz_cached'], b['predicted_peak_hz'], 's', ms=5, color=ORANGE, label='baseline', zorder=5)
ax[0].plot([7, 17], [7, 17], color=GREY, lw=0.6, zorder=0)
err = np.array([x['predicted_hz'] - x['measured_hz'] for x in rows])
ax[0].text(0.97, 0.05, f'RMS error {np.sqrt((err ** 2).mean()):.2f} Hz', transform=ax[0].transAxes, ha='right', fontsize=7)
ax[0].set(xlabel='Measured peak (Hz)', ylabel='Predicted peak (Hz)', xlim=(7, 17), ylim=(7, 17), aspect='equal')
ax[0].legend(frameon=False, fontsize=6.5, loc='upper left', handletextpad=0.2, borderpad=0)

pb = r['phase_budget']
stages = [('SOM$\\to$E synapse (GABA decay)', 'SOM->E synapse (GABA decay)', ORANGE), ('SOM cells', 'SOM neurons', LORANGE),
          ('E cells', 'E neurons', NAVY), ('E$\\to$SOM synapse', 'E->SOM synapse (AMPA+NMDA)', BLUE),
          ('conduction delays', None, LIGHT)]
left = 0
for lab, key, col in stages:
    v = pb[key] if key else pb['E->SOM delay'] + pb['SOM->E delay']
    ax[1].barh(0, v, left=left, color=col, edgecolor='white', lw=0.6, height=0.5, label=lab)
    ax[1].text(left + v / 2, 0, f'{v:.0f}°', ha='center', va='center', fontsize=7,
               color='white' if col in (NAVY, BLUE, ORANGE) else 'black')
    left += v
ax[1].axvline(180, color='k', lw=0.6, ls=':')
ax[1].set(xlim=(0, 190), ylim=(-0.45, 1.25), yticks=[], xticks=[0, 45, 90, 135, 180],
          xlabel=f"Phase lag around the loop at {pb['frequency_hz']:.1f} Hz (deg)")
ax[1].spines['left'].set_visible(False)
ax[1].legend(frameon=False, fontsize=6.5, loc='upper left', ncol=2, handlelength=1, columnspacing=1, borderpad=0)
fig.savefig(os.path.join(FIG, 'abstract_frequency_panels.pdf')); fig.savefig(os.path.join(FIG, 'abstract_frequency_panels.png'), dpi=220)
print('side paths and self-coupling (not drawn):', round(pb['other pathways through PV, VIP'] + pb['E self-coupling'] + pb['SOM self-coupling'], 1), 'deg; drawn total', round(left, 1))
