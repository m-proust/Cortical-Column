"""Linear-response theory of an isolated layer, built from measured neuronal susceptibilities.

Rates r (Hz) of the populations obey, for small fluctuations,
    r(f) = M(f) r(f) + noise,      M[i, k] = sum_X  N[i, X](f) * G[i, k, X](f)
N[i, X]  measured (replay_susceptibility.py): rate change of population i per nS of conductance
         of type X (A = AMPA, N = NMDA, I = GABA) common to its cells, at the intact operating state.
G[i, k, X] exact from the model's parameters: mean conductance of type X in population i per Hz of
         population k = in-degree * jump * tau / (1 + i w tau) * <exp(-i w delay)>.
Indices are [frequency, target, source]. Fourier convention exp(-i w t): a lag has negative phase.
"""
import glob
import json
import os

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, 'saved_trials', 'resonance')     # replay runs, results json, analytic/
TYPES = ['A', 'N', 'I']
TAU_MS = {'AMPA': {'E': 5.0, 'PV': 1.0, 'SOM': 2.0, 'VIP': 2.0}, 'NMDA': 100.0,
          'PV': 6.0, 'SOM': 20.0, 'VIP': 8.0}          # config/config.py
DT_MS = 0.1
FGRID = np.arange(1.0, 50.0001, 0.05)


# ----------------------------------------------------------------------------- measurement
def load(cond, tag=''):
    files = sorted(glob.glob(os.path.join(OUT, f'{cond}_s[0-9]*{tag}.npz')))
    files = [f for f in files if (tag or 'amp' not in os.path.basename(f))]
    return [(np.load(f), json.loads(str(np.load(f)['metadata']))) for f in files]


def susceptibility(z, meta, frozen=False):
    """N[f, pop, type] in Hz/nS and its standard error over probe periods."""
    fr = np.array(meta['freqs'])
    P, s0, n = meta['period_ms'], meta['settle_ms'], meta['n_period']
    t = (np.arange(P) + 0.5) / 1000.0
    e = np.exp(-2j * np.pi * fr[:, None] * t[None]) / P * 2          # [f, t]
    u = np.asarray(z['probe_nS'][s0:s0 + P], float)                    # one period, [t, 3*pop]
    U = e @ u                                                          # [f, 3*pop]
    npop = len(meta['pops'])
    N = np.zeros((len(fr), npop, 3), complex)
    se = np.zeros((len(fr), npop, 3))
    pre = 'rate_frozen_' if frozen else 'rate_'
    for k, x in enumerate(TYPES):
        d = (np.asarray(z[pre + x + '+'], float) - np.asarray(z[pre + x + '-'], float)) / 2
        d = d[s0:s0 + n * P].reshape(n, P, npop)
        Y = np.einsum('ft,ntp->nfp', e, d)                             # [period, f, pop]
        H = Y / U[None, :, k::3]
        N[:, :, k] = H.mean(0)
        se[:, :, k] = np.abs(H - H.mean(0)).std(0) / np.sqrt(n)
    return fr, N, se


def interp(fr, N, f=FGRID):
    """Susceptibility on a dense grid (shape-preserving cubic through the measured lines)."""
    flat = N.reshape(len(fr), -1)
    out = np.array([PchipInterpolator(fr, c.real)(f) + 1j * PchipInterpolator(fr, c.imag)(f) for c in flat.T])
    return out.T.reshape((len(f),) + N.shape[1:])


def smooth(fr, N, order=6):
    """Low-order polynomial in f for real and imaginary parts: N is smooth, the lines are noisy."""
    x = (fr - 25.0) / 25.0
    flat = N.reshape(len(fr), -1)
    out = np.array([np.polyval(np.polyfit(x, c.real, order), x) + 1j * np.polyval(np.polyfit(x, c.imag, order), x)
                    for c in flat.T])
    return out.T.reshape(N.shape)


# ----------------------------------------------------------------------------- theory
def delay_factor(hist, f, shift_ms=0.0, scale=1.0):
    h = np.asarray(hist, float)
    d = (np.arange(len(h)) * DT_MS * scale + shift_ms) / 1000.0
    return (h[None] * np.exp(-2j * np.pi * f[:, None] * d[None])).sum(1) / h.sum()


def couplings(meta, f=FGRID, mods=None):
    """G[f, target, source, type] in nS per Hz. mods changes parameters analytically:
        tau[(pre, post)] = ms         synaptic decay of one pathway (inhibitory) or tau['NMDA'] / ('AMPA', post)
        gain[(pre, post)] = factor    conductance scale (all receptor types of that pathway)
        delay_shift[(pre, post)] = ms, delay_scale[(pre, post)] = factor
        add[(pre, post)] = (K, jump_nS, like)  new pathway; delays copied from pathway `like`
        remove = [(pre, post)]
        no_filter = [(pre, post)]     instantaneous synapse with the same charge
    """
    mods = mods or {}
    pops, cc, wiring = meta['pops'], meta['conductance'], dict(meta['wiring'])
    w = 2j * np.pi * f
    G = np.zeros((len(f), len(pops), len(pops), 3), complex)
    paths = {tuple(k.split('_')): v for k, v in wiring.items()}
    extra = {}
    for (pre, post), (K, jump, like) in mods.get('add', {}).items():
        paths[(pre, post)] = {'K': K, 'delay_hist': paths[like]['delay_hist']}
        extra[(pre, post)] = jump
    for (pre, post), wi in paths.items():
        if (pre, post) in mods.get('remove', []) or pre not in pops or post not in pops:
            continue
        i, k = pops.index(post), pops.index(pre)
        D = delay_factor(wi['delay_hist'], f, mods.get('delay_shift', {}).get((pre, post), 0.0),
                         mods.get('delay_scale', {}).get((pre, post), 1.0))
        gain = mods.get('gain', {}).get((pre, post), 1.0) * wi['K']
        name = f'{pre}_{post}'

        def kern(tau_ms):
            tau = tau_ms / 1000.0
            return tau if (pre, post) in mods.get('no_filter', []) else tau / (1 + w * tau)
        if pre == 'E':
            ta = mods.get('tau', {}).get(('AMPA', post), TAU_MS['AMPA'][post])
            tn = mods.get('tau', {}).get('NMDA', TAU_MS['NMDA'])
            G[:, i, k, 0] = gain * extra.get((pre, post), cc.get(name + '_AMPA', 0.01)) * kern(ta) * D
            G[:, i, k, 1] = gain * cc.get(name + '_NMDA', 0.0) * mods.get('nmda_gain', 1.0) * kern(tn) * D
        else:
            ti = mods.get('tau', {}).get((pre, post), TAU_MS[pre])
            G[:, i, k, 2] = gain * extra.get((pre, post), cc.get(name, 0.02)) * kern(ti) * D
    return G


def loop_matrix(Nf, G):
    """M[f, target, source] from N[f, target, type] and G[f, target, source, type]."""
    return np.einsum('fix,fikx->fik', Nf, G)


def resolvent(M):
    return np.linalg.inv(np.eye(M.shape[-1])[None] - M)


def spectrum(M, q):
    """Rate spectra S[f, pop] for independent noise of spectrum q[pop] (or q[f, pop]) in each population."""
    R = resolvent(M)
    q = np.broadcast_to(np.asarray(q, float), (M.shape[0], M.shape[1]))
    return np.einsum('fik,fk->fi', np.abs(R) ** 2, q)


def reduce(M, keep):
    """Eliminate the other populations exactly (their feedback is kept inside the reduced terms)."""
    n = M.shape[-1]
    keep = np.asarray(keep)
    hid = np.array([i for i in range(n) if i not in keep])
    A = M[:, keep[:, None], keep]
    if len(hid) == 0:
        return A
    B, C, D = M[:, keep[:, None], hid], M[:, hid[:, None], keep], M[:, hid[:, None], hid]
    return A + B @ np.linalg.inv(np.eye(len(hid))[None] - D) @ C


def es_loop(M, pops):
    """Effective E-SOM loop gain L(f): det(I - M) is proportional to 1 - L."""
    m = reduce(M, [pops.index('E'), pops.index('SOM')])
    return m[:, 0, 1] * m[:, 1, 0] / ((1 - m[:, 0, 0]) * (1 - m[:, 1, 1])), m


def peak(f, p, lo=3.0, hi=35.0, sigma_hz=0.5):
    """Largest local maximum of a spectrum in [lo, hi] Hz (parabolic refinement); nan if at an edge."""
    df = f[1] - f[0]
    s = gaussian_filter1d(np.asarray(p, float), sigma_hz / df) if sigma_hz else np.asarray(p, float)
    idx = np.flatnonzero((f >= lo) & (f <= hi))
    k = idx[np.argmax(s[idx])]
    if k in (idx[0], idx[-1]):
        return np.nan
    y = np.log(np.maximum(s[k - 1:k + 2], 1e-300))
    den = y[0] - 2 * y[1] + y[2]
    d = 0.5 * (y[0] - y[2]) / den if den < 0 else 0.0
    return float(f[k] + np.clip(d, -0.5, 0.5) * df)


def zero_phase(f, L, lo=4.0, hi=35.0):
    """Frequencies in [lo, hi] where arg L crosses 0 downward (the loop's phase condition)."""
    ph = np.unwrap(np.angle(L))
    ph -= 2 * np.pi * np.round(ph[np.argmin(np.abs(f - 10))] / (2 * np.pi))
    out = []
    for k in np.flatnonzero((ph[:-1] > 0) & (ph[1:] <= 0)):
        if lo <= f[k] <= hi:
            out.append(float(f[k] + (f[k + 1] - f[k]) * ph[k] / (ph[k] - ph[k + 1])))
    return out


def model(meta, fr, N, mods=None, f=FGRID):
    """Everything derived from one susceptibility set: M, E-rate spectrum, loop gain, peak."""
    pops = meta['pops']
    Nf = interp(fr, N, f)
    M = loop_matrix(Nf, couplings(meta, f, mods))
    q = np.array([meta['rates'][p] / meta['neuron_counts'][p] for p in pops])
    S = spectrum(M, q)
    out = {'f': f, 'M': M, 'S': S, 'pops': pops, 'peak': peak(f, S[:, pops.index('E')])}
    if 'SOM' in pops:
        out['L'], out['m'] = es_loop(M, pops)
        out['zero_phase'] = zero_phase(f, out['L'])
    return out
