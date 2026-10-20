"""Cosyne reviewer checks: (1) adaptation as a hidden slow element, (2) sensitivity of which layer
expresses alpha to the assumed PV->E, SOM->PV and VIP->SOM strengths.

Runs: minimal_config/experiments/reviewer/_runs/<cond>_s<seed>.npz (run_reviewer.py); baseline L6 /
L23 from fig1/_runs (same seeds). Metric: studies/cosyne_abstract/metrics.py (E-rate dB above the 1/f fit;
alpha_db = mean over 8-13 Hz). Peak frequency = largest residual of the seed-averaged spectrum in
4-30 Hz. CIs: 95% percentile bootstrap over networks.

    ~/Desktop/venv/bin/python studies/cosyne_abstract/reviewer_checks.py
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import alpha_metrics, residual_db
from minimal_config.experiments.conditions import MAP_KG, TUNED_N, INTACT_RATES_HZ

EXP = os.path.join(ROOT, 'minimal_config', 'experiments')
RUNS = os.path.join(EXP, 'reviewer', '_runs')
FIG1 = os.path.join(EXP, 'fig1', '_runs')
OUT = os.path.join(ROOT, 'figures', 'cosyne', 'reviewer')
SEEDS = [58925] + list(range(1, 10))
POPS = ['E', 'PV', 'SOM', 'VIP']
RNG = np.random.default_rng(1)
NBOOT = 5000


def load(cond):
    d = FIG1 if cond in ('L6', 'L23') else RUNS
    zs = [np.load(f) for s in SEEDS for f in [os.path.join(d, f'{cond}_s{s}.npz')] if os.path.exists(f)]
    if not zs and cond == 'L6_E+SOM':                   # reduced circuits ran elsewhere
        zs = [np.load(f) for f in sorted(glob.glob(os.path.join(EXP, '**', f'{cond}_s*.npz'), recursive=True))]
    return zs


def peak_hz(f, P):
    ff, r = residual_db(f, P)
    m = (ff >= 4) & (ff <= 30)
    return float(ff[m][np.argmax(r[m])])


def summarise(cond):
    zs = load(cond)
    if not zs:
        return None
    f = zs[0]['f_rate']
    P = np.array([z['psd_E'] for z in zs])
    a = np.array([alpha_metrics(f, p)['alpha_db'] for p in P])
    boot_a = [a[RNG.integers(0, len(a), len(a))].mean() for _ in range(NBOOT)]
    boot_f = [peak_hz(f, P[RNG.integers(0, len(P), len(P))].mean(0)) for _ in range(NBOOT)]
    rates = {p: float(np.mean([z[f'rate_{p}'] for z in zs])) for p in POPS if f'rate_{p}' in zs[0].files}
    f_res, res = residual_db(f, P.mean(0))
    return dict(cond=cond, n=len(zs), alpha_db=float(a.mean()), alpha_ci=np.percentile(boot_a, [2.5, 97.5]).tolist(),
                peak_hz=peak_hz(f, P.mean(0)), peak_ci=np.percentile(boot_f, [2.5, 97.5]).tolist(),
                rates=rates, f=f_res, residual=res)


def adaptation():
    conds = ['L6', 'L6_noadaptE', 'L6_fastadapt', 'L6_meanwE', 'L6_meanwSOM', 'L6_meanw',
             'L6_meanw_leak', 'L6_E+SOM', 'L6_E+SOM_meanw']
    rows = [r for r in map(summarise, conds) if r]
    print('\n=== (1) adaptation, isolated L6, rate-matched ===')
    for r in rows:
        print(f"{r['cond']:16s} n={r['n']:2d}  alpha {r['alpha_db']:5.2f} dB [{r['alpha_ci'][0]:.2f},{r['alpha_ci'][1]:.2f}]"
              f"  peak {r['peak_hz']:5.2f} Hz [{r['peak_ci'][0]:.2f},{r['peak_ci'][1]:.2f}]  "
              + ' '.join(f"{p} {v:.2f}" for p, v in r['rates'].items()))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for r in rows:
        k = 1 if 'E+SOM' in r['cond'] else 0
        m = (r['f'] >= 2) & (r['f'] <= 40)
        ax[k].semilogx(r['f'][m], r['residual'][m], label=f"{r['cond']} ({r['alpha_db']:.1f} dB, {r['peak_hz']:.1f} Hz)")
    for a_, t in zip(ax, ['Isolated L6', 'L6 E + SOM']):
        a_.axvspan(8, 13, color='0.9'); a_.axhline(0, color='0.6', lw=0.5)
        a_.set(xlabel='Frequency (Hz)', ylabel='E-rate power (dB above 1/f)', title=t)
        a_.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'adaptation_spectra.png'), dpi=200)
    return rows


def sens_map():
    tuned = json.load(open(TUNED_N))
    um_file = os.path.join(EXP, 'reviewer', 'unmatched.txt')
    failed = set(open(um_file).read().split()) if os.path.exists(um_file) else set()
    print('\n=== (2) sensitivity map: alpha_db (peak Hz) per PV->E (rows) x SOM->PV (cols) ===')
    grids = {}
    for base in ['L6', 'L23']:
        for vs in MAP_KG['VS']:
            A = np.full((5, 5), np.nan); F = np.full((5, 5), np.nan); M = np.zeros((5, 5), bool)
            for i, pe in enumerate(MAP_KG['PE']):
                for j, sp in enumerate(MAP_KG['SP']):
                    c = f'map_{base}_PE{pe:g}_SP{sp:g}_VS{vs:g}'
                    M[i, j] = c not in failed or c in tuned
                    r = summarise(c)
                    if r and r['n'] >= 5:
                        A[i, j], F[i, j] = r['alpha_db'], r['peak_hz']
            grids[(base, vs)] = (A, F, M)
            print(f'\n{base} circuit, VIP->SOM {vs} nS  (SP across: {MAP_KG["SP"]})')
            for i, pe in enumerate(MAP_KG['PE']):
                print(f'  PE {pe:5.1f} ' + '  '.join('  unmatchable ' if not M[i, j] else ('   pending   ' if np.isnan(A[i, j]) else
                                                     f'{A[i, j]:5.2f} ({F[i, j]:4.1f})') for j in range(5)))
    fig, axs = plt.subplots(2, 2, figsize=(7.5, 6.5), sharex=True, sharey=True)
    vmax = np.nanmax([np.nanmax(g[0]) for g in grids.values() if np.isfinite(g[0]).any()] + [1])
    for (base, vs), ax in zip(grids, axs.ravel()):
        A, F, M = grids[(base, vs)]
        im = ax.imshow(A, origin='lower', cmap='magma', vmin=0, vmax=vmax, aspect='auto')
        for i in range(5):
            for j in range(5):
                txt = 'x' if not M[i, j] else ('' if np.isnan(A[i, j]) else f'{A[i, j]:.1f}\n{F[i, j]:.0f} Hz')
                ax.text(j, i, txt, ha='center', va='center', fontsize=6,
                        color='k' if np.nan_to_num(A[i, j]) > 0.6 * vmax else 'w')
        for (pe, sp, mk) in [(19.6, 11.6, 'L6'), (34.2, 37.2, 'L23')]:
            ax.add_patch(plt.Rectangle((MAP_KG['SP'].index(sp) - .5, MAP_KG['PE'].index(pe) - .5), 1, 1,
                                       fill=False, ec='cyan' if mk == 'L6' else 'lime', lw=1.5))
        ax.set_title(f'{base} circuit, VIP->SOM {vs} nS', fontsize=8)
        ax.set_xticks(range(5), [f'{v:g}' for v in MAP_KG['SP']]); ax.set_yticks(range(5), [f'{v:g}' for v in MAP_KG['PE']])
    for ax in axs[1]: ax.set_xlabel('SOM->PV per-cell input (nS)')
    for ax in axs[:, 0]: ax.set_ylabel('PV->E per-cell input (nS)')
    fig.colorbar(im, ax=axs, label='alpha, dB above 1/f (8-13 Hz)', shrink=0.7)
    fig.suptitle('cyan = L6 values, green = L23 values; x = not rate-matchable', fontsize=8)
    fig.savefig(os.path.join(OUT, 'sensitivity_map.png'), dpi=200)
    np.savez(os.path.join(OUT, 'sensitivity_map.npz'), **{f'{b}_VS{v:g}_{k}': g[n] for (b, v), g in grids.items()
                                                          for n, k in enumerate(['alpha', 'peak', 'matched'])},
             PE=MAP_KG['PE'], SP=MAP_KG['SP'])
    return grids


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ['adapt', 'map']
    if 'adapt' in which:
        adaptation()
    if 'map' in which:
        sens_map()
