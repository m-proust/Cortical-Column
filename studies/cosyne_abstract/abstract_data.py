"""Cosyne abstract: rebuild the data tables of the model panels from saved_trials/cosyne/.

Writes the plain-text tables that the pgfplots figures of ~/Desktop/cosyne_abstract/cosyne_abstract.tex
read (figures/data/*.dat). Colours, legends and labels live in the .tex, so these tables are the
complete content of each curve. plot_abstract_panels.py draws the same panels with matplotlib.

    ~/Desktop/venv/bin/python studies/cosyne_abstract/abstract_data.py [--out DIR] [--check]

--out    where to write the tables (default figures/cosyne_abstract/data/). To update the abstract
         itself, pass --out ~/Desktop/cosyne_abstract/figures/data
--check  compare every table with the abstract's copy and report the largest difference

Panel -> table -> saved runs (all under saved_trials/cosyne/):
  Fig 2A  iso_lfp.dat       isolated layers, bipolar LFP in dB above the 1/f fit   isolated_layers/{L23,L4AB,L4C@own,L5,L6}_s*.npz
  Fig 2B  reduced.dat       isolated L6 reduced to E+SOM / E+SOM+PV / E+PV, E rate  reduced_L6/
  Fig 2C  transplant.dat    L2/3 alone vs with L6's PV->E and SOM->PV, E rate       isolated_layers/{L23,fx_L23+PE+SP}_s*.npz
  Fig 2D  finite.dat        per-cell alpha power vs N, by shared-input fraction    finite_size/{independent,shared}/
  Fig 2E  freq_theory.dat   E-rate spectra, network vs linear theory, SOM->E decay 10/20/40 ms
                                                                                  replay_susceptibility/ + isolated_layers/{L6,L6_tauSOME10,L6_tauSOME40}
  Fig 3A  replay_L4C.dat    isolated L4C: no input / L6 timing removed / real L6 spikes   isolated_layers/L4C@own + layer_replay/L6_to_L4C@own_*
          replay_L23.dat    isolated L2/3: same with L4C spikes                          isolated_layers/L23 + layer_replay/L4C_to_L23_*
  Fig 3B  gating_amp.dat    L4C response to a 25 pA pulse vs local alpha phase, weakest vs
                            strongest third of ongoing alpha amplitude, 95% bootstrap  pulse_gating_L4C/
  Fig 3D  gating_fb.dat     L2/3 response to a 25 pA pulse vs L6 alpha phase, feedback off/on
                                                                                  feedback_protocol/phase_curves/phase_curve_arrays.npz
                            (that file is built from feedback_protocol/data by
                             studies/feedback/analyze.py, then plot_phase_feedback.py)
Not rebuilt here: Fig 1 (traced from slide images, see make_data.py in the abstract folder), Fig 3C (feedback
depth-frequency map, an image cropped from the CSID slides; its data are feedback_protocol/*_spectral_arrays.npz).

Same computations as ~/Desktop/cosyne_abstract/make_data.py; only the data locations changed.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'studies', 'phase_gating'))
sys.path.insert(0, os.path.join(ROOT, 'studies', 'resonance'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import residual_db

SAVED = os.path.join(ROOT, 'saved_trials', 'cosyne')
ISO = os.path.join(SAVED, 'isolated_layers')
RED = os.path.join(SAVED, 'reduced_L6')
REP = os.path.join(SAVED, 'layer_replay')
ABSTRACT_DATA = os.path.expanduser('~/Desktop/cosyne_abstract/figures/data')
N_E_L6, N_IN = 2040, 45                     # L6 E cells; Poisson sources per E cell


def runs(d, cond):
    return [np.load(f) for f in sorted(glob.glob(os.path.join(d, glob.escape(cond) + '_s*.npz')))]


def mean_resid(zs, lfp=False):
    """Mean power above the 1/f fit (dB): E rate, or the in-layer bipolar LFP channel (index 3, z = -0.415 mm)."""
    get = (lambda z: (z['f_lfp'], z['psd_lfp'][3])) if lfp else (lambda z: (z['f_rate'], z['psd_E']))
    f = residual_db(*get(zs[0]))[0]
    return f, np.mean([residual_db(*get(z))[1] for z in zs], 0)


class Writer:
    def __init__(self, out):
        self.out = out
        os.makedirs(out, exist_ok=True)
        self.names = []

    def table(self, name, header, *cols):
        np.savetxt(os.path.join(self.out, name), np.column_stack(cols), header=' '.join(header), comments='', fmt='%.5g')
        self.names.append(name)
        print(f'wrote {name:16s} {len(cols[0])} rows')

    def spectra(self, name, conds, lfp=False, fmax=90.0):
        cols, f = [], None
        for _, d, cond in conds:
            zs = runs(d, cond)
            assert zs, f'no runs for {cond} in {d}'
            f, r = mean_resid(zs, lfp)
            cols.append(r)
        k = (f >= 1.5) & (f <= fmax)
        self.table(name, ['f'] + [c[0] for c in conds], f[k], *[c[k] for c in cols])


# ---------------------------------------------------------------- Fig. 2
def fig2(w):
    w.spectra('iso_lfp.dat', [('L23', ISO, 'L23'), ('L4AB', ISO, 'L4AB'), ('L4C', ISO, 'L4C@own'), ('L5', ISO, 'L5'),
                              ('L6', ISO, 'L6')], lfp=True)
    w.spectra('reduced.dat', [('ESOM', RED, 'L6_E+SOM'), ('ESOMPV', RED, 'L6_E+SOM+PV'), ('EPV', RED, 'L6_E+PV')])
    w.spectra('transplant.dat', [('alone', ISO, 'L23'), ('swap', ISO, 'fx_L23+PE+SP')])

    # finite size: per-cell E-rate PSD averaged over 8-13 Hz (as archive/analysis/cosyne/finite_size_shared.py)
    R = {}
    for fn in glob.glob(os.path.join(SAVED, 'finite_size', '*', 'L6_Nx*.npz')):
        m = re.search(r'Nx([\d.]+)(?:_sh(\d+))?_s(\d+)\.npz', fn)
        k, n = float(m.group(1)), int(m.group(2) or 0)
        z = np.load(fn)
        f = z['f_rate']
        psd = z['psd_E'] * 1e6 / (N_E_L6 * k) ** 2
        R.setdefault((n, k), []).append(psd[(f >= 8) & (f <= 13)].mean())
    ns, ks = sorted({n for n, _ in R}), sorted({k for _, k in R})
    N = np.array([N_E_L6 * k for k in ks])
    cols = [[np.mean(R[(n, k)]) for k in ks] for n in ns]
    w.table('finite.dat', ['N'] + [f's{round(100 * n / N_IN)}' for n in ns] + ['invN'], N, *cols,
            cols[0][ks.index(1.0)] * N_E_L6 / N)

    # frequency theory (as studies/resonance/export_abstract_data.py)
    import freqmech as fm
    import measured
    fm.OUT = os.path.join(SAVED, 'replay_susceptibility')
    measured.RUNS = ISO
    rr = fm.load('L6'); meta = rr[0][1]; pops = meta['pops']; fr = np.array(meta['freqs'])
    Nsus = np.mean([fm.susceptibility(z, m)[1] for z, m in rr], 0)
    meta = dict(meta, rates=dict(zip(pops, np.mean([np.asarray(z['rate_closed'][2000:], float).mean(0) for z, _ in rr], 0))))
    SE, iE = ('SOM', 'E'), pops.index('E')
    cols, head = [], ['f']
    for tau, cond in [(10, 'L6_tauSOME10'), (20, 'L6'), (40, 'L6_tauSOME40')]:
        f, P, _ = measured.spectra(cond); sel = (f >= 3) & (f <= 40); m = P.mean(0)[sel]
        pr = fm.model(meta, fr, Nsus, None if tau == 20 else {'tau': {SE: tau}, 'gain': {SE: 20 / tau}})
        s = np.interp(f[sel], pr['f'], pr['S'][:, iE])
        if not cols:
            cols.append(f[sel])
        cols += [m / m.max(), s / s.max()]; head += [f'net{tau}', f'th{tau}']
        print(f'  SOM->E decay {tau} ms: network peak {fm.peak(f, P.mean(0)):.2f} Hz, theory {pr["peak"]:.2f} Hz, {len(P)} networks')
    w.table('freq_theory.dat', head, *cols)


# ---------------------------------------------------------------- Fig. 3
def fig3(w):
    for tag, src, tgt in [('L4C', 'L6', 'L4C@own'), ('L23', 'L4C', 'L23')]:
        w.spectra(f'replay_{tag}.dat', [('none', ISO, tgt), ('poisson', REP, f'{src}_to_{tgt}_poisson'),
                                        ('replay', REP, f'{src}_to_{tgt}_replay')])
    import phase_x_amplitude as px
    import plot_key_figure as kf
    d, doses = px.load(os.path.join('cosyne', 'pulse_gating_L4C'), 'L4C')
    seeds, ph = d['seed'], d['ph_own']
    terc = np.digitize(px.within_rank(d['amp_own'], seeds), [-1 / 6, 1 / 6])
    y = d['pulse'][:, list(doses).index(25)] - d['sham']
    cols = []
    for t in (0, 2):                                                       # weakest and strongest third of ongoing alpha
        gx, gm, lo, hi = kf.smooth_with_band(y[terc == t], ph[terc == t], seeds[terc == t])
        cols += [gm, lo, hi]
    w.table('gating_amp.dat', ['deg', 'weak', 'weaklo', 'weakhi', 'strong', 'stronglo', 'stronghi'], gx, *cols)

    z = np.load(os.path.join(SAVED, 'feedback_protocol', 'phase_curves', 'phase_curve_arrays.npz'))
    deg = z['phase_degrees'][::3]
    boot = np.random.default_rng(38192).integers(0, 6, (5000, 6))          # same resamples as plot_phase_feedback.py
    cols = []
    for c in (0, 1):
        cur = z[f'source_L23_{c}_curves'][:, ::3]
        lo, hi = np.quantile(cur[boot].mean(1), [.025, .975], axis=0)
        cols += [cur.mean(0), lo, hi]
    w.table('gating_fb.dat', ['deg', 'off', 'offlo', 'offhi', 'on', 'onlo', 'onhi'], deg, *cols)


def check(w):
    print(f'\ncomparison with {ABSTRACT_DATA}:')
    for name in w.names:
        a = np.loadtxt(os.path.join(w.out, name), skiprows=1)
        b = np.loadtxt(os.path.join(ABSTRACT_DATA, name), skiprows=1)
        if a.shape != b.shape:
            print(f'  {name:16s} DIFFERENT SHAPE {a.shape} vs {b.shape}')
            continue
        d = np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))
        print(f'  {name:16s} max relative difference {d:.1e}' + ('  (identical to 5 significant digits)' if d < 1e-4 else ''))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'figures', 'cosyne_abstract', 'data'))
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    w = Writer(os.path.expanduser(a.out))
    fig2(w); fig3(w)
    if a.check:
        check(w)
