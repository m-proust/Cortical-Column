"""Measured peak frequencies of cached SNN runs (minimal_config/experiments/fig1/_runs), for comparison
with the theory. Raw E-rate spectrum averaged over networks; bootstrap over networks for the interval."""
import glob
import os

import numpy as np

from freqmech import HERE, peak

RUNS = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'minimal_config', 'experiments', 'fig1', '_runs')


def spectra(cond):
    files = sorted(glob.glob(os.path.join(RUNS, glob.escape(cond) + '_s[0-9]*.npz')))
    zz = [np.load(f) for f in files]
    if not zz:
        return None, None, {}
    rates = {k[5:]: float(np.mean([z[k] for z in zz])) for k in zz[0].files if k.startswith('rate_')}
    return zz[0]['f_rate'], np.array([z['psd_E'] for z in zz]), rates


def measured_peak(cond, nboot=2000, seed=0):
    f, P, rates = spectra(cond)
    if f is None:
        return None
    rng = np.random.default_rng(seed)
    boots = [peak(f, P[rng.integers(0, len(P), len(P))].mean(0)) for _ in range(nboot)]
    # prominence: peak power over the mean of the spectrum 5 Hz either side (dB)
    m = P.mean(0)
    fp = peak(f, m)
    prom = np.nan
    if np.isfinite(fp):
        side = m[(np.abs(f - (fp - 5)) < 1) | (np.abs(f - (fp + 5)) < 1)].mean()
        prom = 10 * np.log10(m[np.argmin(np.abs(f - fp))] / side)
    return {'peak': fp, 'ci': np.nanpercentile(boots, [2.5, 97.5]).tolist(), 'n': len(P),
            'prom_db': float(prom), 'rates': rates, 'f': f, 'psd': m}
