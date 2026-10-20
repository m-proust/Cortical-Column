"""The one alpha metric used in every Cosyne figure.

E-population rate (1 ms bins) -> Welch PSD (2 s segments, 75% overlap, 0.5 Hz bins,
the same as minimal_config/run_spectrum.py) -> straight-line fit of log10 power vs
log10 frequency over 2-80 Hz. Everything is the residual above that fit, in dB:
    peak_hz / peak_db   largest residual in 2-80 Hz
    alpha_db            mean residual over 8-13 Hz
"""
import numpy as np
from scipy.signal import welch

FIT_BAND = (2.0, 80.0)
ALPHA_BAND = (8.0, 13.0)


def rate_psd(spike_times_ms, t0_ms, t1_ms, nperseg=2000):
    """PSD of a population's spike count in 1 ms bins between t0 and t1."""
    t = np.asarray(spike_times_ms, dtype=float)
    x, _ = np.histogram(t, bins=np.arange(t0_ms, t1_ms + 1, 1.0))
    x = x.astype(float)
    nperseg = min(nperseg, len(x))
    return welch(x - x.mean(), fs=1000, nperseg=nperseg, noverlap=int(0.75 * nperseg))


def residual_db(f, psd):
    """(f, residual in dB above the 1/f fit) restricted to FIT_BAND."""
    m = (f >= FIT_BAND[0]) & (f <= FIT_BAND[1])
    f, p = f[m], psd[m]
    slope, icpt = np.polyfit(np.log10(f), np.log10(p), 1)
    return f, 10 * (np.log10(p) - (slope * np.log10(f) + icpt))


def alpha_metrics(f, psd):
    ff, r = residual_db(f, psd)
    k = int(np.argmax(r))
    a = (ff >= ALPHA_BAND[0]) & (ff <= ALPHA_BAND[1])
    return {'peak_hz': float(ff[k]), 'peak_db': float(r[k]), 'alpha_db': float(r[a].mean())}
