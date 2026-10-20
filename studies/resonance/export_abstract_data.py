"""Data for the abstract's frequency panel (Fig. 2E): network and theory E-rate spectra for three SOM->E
decays, written as a pgfplots table into the abstract's figures/data folder.

    ~/Desktop/venv/bin/python studies/resonance/export_abstract_data.py
"""
import os
import numpy as np
import freqmech as fm
from measured import spectra

OUT = os.path.expanduser('~/Desktop/cosyne_abstract/figures/data/freq_theory.dat')
runs = fm.load('L6'); meta = runs[0][1]; pops = meta['pops']; fr = np.array(meta['freqs'])
N = np.mean([fm.susceptibility(z, m)[1] for z, m in runs], 0)
meta = dict(meta, rates=dict(zip(pops, np.mean([np.asarray(z['rate_closed'][2000:], float).mean(0) for z, _ in runs], 0))))
SE = ('SOM', 'E'); iE = pops.index('E')
cols, head = [], ['f']
for tau, cond in [(10, 'L6_tauSOME10'), (20, 'L6'), (40, 'L6_tauSOME40')]:
    f, P, _ = spectra(cond); sel = (f >= 3) & (f <= 40); m = P.mean(0)[sel]
    pr = fm.model(meta, fr, N, None if tau == 20 else {'tau': {SE: tau}, 'gain': {SE: 20 / tau}})
    s = np.interp(f[sel], pr['f'], pr['S'][:, iE])
    if not cols:
        cols.append(f[sel])
    cols += [m / m.max(), s / s.max()]; head += [f'net{tau}', f'th{tau}']
    print(tau, 'ms: network peak', round(fm.peak(f, P.mean(0)), 2), 'theory peak', round(pr['peak'], 2), 'n', len(P))
np.savetxt(OUT, np.column_stack(cols), header=' '.join(head), comments='', fmt='%.5g')
print('wrote', OUT)
