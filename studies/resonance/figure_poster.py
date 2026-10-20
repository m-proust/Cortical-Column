"""Poster panels: Nyquist plot of the E-SOM loop (baseline and +10 ms delay per arm) and the phase budget
at the closing frequency. Colours follow cosyne_abstract.tex.

    ~/Desktop/venv/bin/python studies/resonance/figure_poster.py

G(f) = -L(f), where the network's characteristic function is proportional to 1 - L = 1 + G: the critical
point is -1. Both curves use the responses measured at baseline; +10 ms only changes the delay terms.
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
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'serif'})

runs = fm.load('L6'); meta = runs[0][1]; pops = meta['pops']; fr = np.array(meta['freqs'])
N = np.mean([fm.susceptibility(z, m)[1] for z, m in runs], 0)
meta = dict(meta, rates=dict(zip(pops, np.mean([np.asarray(z['rate_closed'][2000:], float).mean(0) for z, _ in runs], 0))))
ES, SE = ('E', 'SOM'), ('SOM', 'E')
base = fm.model(meta, fr, N)
slow = fm.model(meta, fr, N, {'delay_shift': {ES: 10, SE: 10}})
f = base['f']
info = {}

# ------------------------------------------------------------------ Nyquist
fig, ax = plt.subplots(figsize=(3.6, 3.4), layout='constrained')
sel = (f >= 5) & (f <= 50)
for pr, col, lab in [(base, NAVY, 'baseline'), (slow, ORANGE, '+10 ms delay per arm')]:
    G = -pr['L']
    ax.plot(G.real[sel], G.imag[sel], color=col, lw=1.3, label=lab)
    zp = pr['zero_phase'][0]
    k = int(np.argmin(np.abs(f - zp)))
    ax.plot(G.real[k], 0, 'o', color=col, ms=4.5, zorder=5)
    info[lab] = {'closing_hz': zp, 'gain': float(abs(G[k])), 'peak_hz': pr['peak']}
    ax.annotate(f'{zp:.1f} Hz\ngain {abs(G[k]):.2f}', (G.real[k], 0), xytext=(-4, 8) if col == NAVY else (5, -6),
                textcoords='offset points', ha='center' if col == NAVY else 'left', va='bottom' if col == NAVY else 'top', fontsize=7, color=col)
G = -base['L']; kp = int(np.argmin(np.abs(f - base['peak'])))
ax.plot(G.real[kp], G.imag[kp], 's', color=NAVY, ms=4, mfc='white', zorder=5)
ax.annotate(f"spectral peak\n{base['peak']:.1f} Hz", (G.real[kp], G.imag[kp]), xytext=(14, -24), textcoords='offset points',
            fontsize=7, color=NAVY, ha='left', arrowprops=dict(arrowstyle='-', color=NAVY, lw=0.5))
ax.plot([-1, G.real[kp]], [0, G.imag[kp]], color=GREY, lw=0.6, ls=':')
for hz in [8, 10, 20, 30]:                       # direction of increasing frequency
    k = int(np.argmin(np.abs(f - hz)))
    ax.annotate(f'{hz}', (G.real[k], G.imag[k]), xytext=(3, 2) if hz > 15 else (-9, -2), textcoords='offset points', fontsize=6, color=GREY)
    ax.plot(G.real[k], G.imag[k], '.', color=NAVY, ms=3)
ax.plot(-1, 0, 'x', color='k', ms=7, mew=1.5, zorder=6)
ax.annotate('$-1$', (-1, 0), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=8)
ax.axhline(0, color=GREY, lw=0.4); ax.axvline(0, color=GREY, lw=0.4)
ax.set(xlabel='Re $G(f)$', ylabel='Im $G(f)$', aspect='equal', xlim=(-1.9, 0.9), ylim=(-1.45, 0.95), title='Loop gain of the E–SOM loop')
ax.legend(frameon=False, fontsize=7, loc='lower right', title='grey numbers: Hz', title_fontsize=6)
fig.text(0.5, -0.01, 'Distance to $-1$ sets amplification;\ncrossing beyond $-1$ means self-oscillation.', ha='center', va='top', fontsize=7.5)
fig.savefig(os.path.join(FIG, 'poster_nyquist.pdf'), bbox_inches='tight'); fig.savefig(os.path.join(FIG, 'poster_nyquist.png'), dpi=220, bbox_inches='tight')
plt.close(fig)
info['distance_to_minus1_at_peak'] = float(abs(1 + G[kp])); info['distance_to_minus1_at_closing'] = float(1 - info['baseline']['gain'])

# ------------------------------------------------------------------ phase budget
pb = json.load(open(os.path.join(fm.OUT, 'L6_results.json')))['phase_budget']
stages = [('SOM→E synaptic decay', pb['SOM->E synapse (GABA decay)'], ORANGE, 'white'), ('SOM response', pb['SOM neurons'], LORANGE, 'black'),
          ('E response', pb['E neurons'], NAVY, 'white'), ('E→SOM synapse', pb['E->SOM synapse (AMPA+NMDA)'], BLUE, 'white'),
          ('delays', pb['E->SOM delay'] + pb['SOM->E delay'], LIGHT, 'black')]
side = pb['other pathways through PV, VIP'] + pb['E self-coupling'] + pb['SOM self-coupling']
fig, ax = plt.subplots(figsize=(6.2, 1.7), layout='constrained')
left = 0
for lab, v, col, tc in stages:
    ax.barh(0, v, left=left, color=col, edgecolor='white', lw=0.8, height=0.55)
    ax.text(left + v / 2, 0, f'{v:.0f}°', ha='center', va='center', fontsize=8, color=tc)
    ax.text(left + v / 2, 0.36, lab, ha='center', va='bottom', fontsize=7, rotation=0 if v > 30 else 25)
    left += v
ax.barh(-0.5, side, left=left, color='white', edgecolor=GREY, hatch='////', lw=0.6, height=0.3)
ax.text(left + side - 2, -0.5, f'side paths (PV, VIP, E→E) {side:.0f}°', ha='right', va='center', fontsize=7, color='#444444')
ax.axvline(180, color='k', lw=0.7, ls=':')
ax.set(xlim=(0, 188), ylim=(-0.75, 0.95), yticks=[], xticks=[0, 45, 90, 135, 180], xlabel='Phase lag (deg)')
ax.set_title(f"Phase lags at the closing frequency ({pb['frequency_hz']:.1f} Hz)", fontsize=9)
ax.spines['left'].set_visible(False)
fig.savefig(os.path.join(FIG, 'poster_phase_budget.pdf')); fig.savefig(os.path.join(FIG, 'poster_phase_budget.png'), dpi=220)
info['segments_deg'] = {s[0]: round(s[1], 1) for s in stages} | {'side paths': round(side, 1), 'total': round(left + side, 1)}
print(json.dumps(info, indent=1))
