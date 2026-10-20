"""Predicted amplification of the loop against the measured E-rate spectrum, for baseline and held-out
timing changes. Amplification = 1 / |1 + G(f)|^2 (G: E-SOM loop gain, responses measured at baseline;
each condition only changes the synaptic decay or delay terms). Colours follow cosyne_abstract.tex.

    ~/Desktop/venv/bin/python studies/resonance/figure_amplification.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import freqmech as fm
from measured import spectra

NAVY, ORANGE, GREY = '#002255', '#FF6D1E', '#8C8C8C'
FIG = os.path.join(os.path.dirname(os.path.dirname(fm.HERE)), 'figures', 'frequency_mechanism')
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'serif'})
runs = fm.load('L6'); meta = runs[0][1]; pops = meta['pops']; fr = np.array(meta['freqs'])
N = np.mean([fm.susceptibility(z, m)[1] for z, m in runs], 0)
meta = dict(meta, rates=dict(zip(pops, np.mean([np.asarray(z['rate_closed'][2000:], float).mean(0) for z, _ in runs], 0))))
ES, SE = ('E', 'SOM'), ('SOM', 'E')
conds = [('SOM→E decay 10 ms', 'L6_tauSOME10', {'tau': {SE: 10}, 'gain': {SE: 2.0}}),
         ('SOM→E decay 15 ms', 'L6_tauSOME15', {'tau': {SE: 15}, 'gain': {SE: 20 / 15}}),
         ('baseline (20 ms)', 'L6', None),
         ('SOM→E decay 30 ms', 'L6_tauSOME30', {'tau': {SE: 30}, 'gain': {SE: 20 / 30}}),
         ('SOM→E decay 40 ms', 'L6_tauSOME40', {'tau': {SE: 40}, 'gain': {SE: 0.5}}),
         ('loop delays ×0.5', 'L6_loopdelayx0.5', {'delay_scale': {ES: 0.5, SE: 0.5}}),
         ('loop delays ×2', 'L6_loopdelayx2', {'delay_scale': {ES: 2, SE: 2}}),
         ('+2.5 ms per arm', 'L6_loopdelay+2.5', {'delay_shift': {ES: 2.5, SE: 2.5}}),
         ('+5 ms per arm', 'L6_loopdelay+5', {'delay_shift': {ES: 5, SE: 5}}),
         ('+10 ms per arm', 'L6_loopdelay+10', {'delay_shift': {ES: 10, SE: 10}})]
fig, axes = plt.subplots(2, 5, figsize=(10, 3.9), layout='constrained', sharex=True, sharey=True)
for ax, (title, cond, mods) in zip(axes.flat, conds):
    pr = fm.model(meta, fr, N, mods); f = pr['f']
    amp = 1 / np.abs(1 - pr['L']) ** 2
    fa = fm.peak(f, amp); fs = pr['peak']
    fm_, P, _ = spectra(cond); m = P.mean(0); fp = fm.peak(fm_, m)
    sel = (fm_ >= 3) & (fm_ <= 40); g = (f >= 3) & (f <= 40)
    lo, hi = np.percentile(P[:, sel] / m[sel].max(), [16, 84], axis=0)
    ax.fill_between(fm_[sel], lo, hi, color=NAVY, alpha=0.15, lw=0)
    ax.plot(fm_[sel], m[sel] / m[sel].max(), color=NAVY, lw=1.1, label='spiking network (E-rate spectrum)')
    ax.plot(f[g], amp[g] / amp[g].max(), color=ORANGE, lw=1.1, label='theory: loop amplification')
    ax.axvline(fp, color=NAVY, lw=0.6, ls=':'); ax.axvline(fa, color=ORANGE, lw=0.6, ls=':')
    gain = abs(pr['L'][np.argmin(abs(f - pr['zero_phase'][0]))])
    ax.set_title(title + ('\n(theory: loop unstable)' if gain >= 1 else '\n'), fontsize=8)
    ax.text(0.97, 0.95, f'theory {fa:.1f} Hz', transform=ax.transAxes, ha='right', va='top', fontsize=7, color=ORANGE)
    ax.text(0.97, 0.86, f'network {fp:.1f} Hz', transform=ax.transAxes, ha='right', va='top', fontsize=7, color=NAVY)
    ax.set(xscale='log', xlim=(3, 40)); ax.set_xticks([5, 10, 20, 40]); ax.set_xticklabels(['5', '10', '20', '40']); ax.minorticks_off()
    print(f'{cond:18s} amplification peak {fa:5.2f}  full-spectrum prediction {fs:5.2f}  measured {fp:5.2f}  gain at closing {abs(pr["L"][np.argmin(abs(f - pr["zero_phase"][0]))]):.2f}')
for ax in axes[1]: ax.set_xlabel('Frequency (Hz)')
for ax in axes[:, 0]: ax.set_ylabel('Power (normalised)')
fig.savefig(os.path.join(FIG, 'poster_amplification.pdf')); fig.savefig(os.path.join(FIG, 'poster_amplification.png'), dpi=200)
