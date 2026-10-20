"""Predict the L6 peak frequency from measured susceptibilities, test it on held-out sweeps, and
decompose it into the phase lag of each stage of the E-SOM loop.

    ~/Desktop/venv/bin/python studies/resonance/analyse.py [condition]

Writes saved_trials/resonance/<condition>_results.json and figures into figures/frequency_mechanism/.
"""
import json
import os
import sys

import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import freqmech as fm
from measured import measured_peak

COND = sys.argv[1] if len(sys.argv) > 1 else 'L6'
SMOOTH = os.environ.get('SMOOTH', '0') == '1'      # polynomial smoothing of the susceptibilities (default: none)
TAG = '_smooth' if SMOOTH else ''
FIG = os.path.join(os.path.dirname(os.path.dirname(fm.HERE)), 'figures', 'frequency_mechanism')
os.makedirs(FIG, exist_ok=True)
f = fm.FGRID

runs = fm.load(COND)
assert runs, 'no runs for ' + COND
meta = runs[0][1]
pops = meta['pops']
iE, iS = pops.index('E'), pops.index('SOM')
for z, m in runs:
    assert all(all(v.values()) for v in m['exact'].values()), 'replay not exact'
    m['rates'] = dict(zip(pops, np.asarray(z['rate_closed'][m['settle_ms']:], float).mean(0)))
fr = np.array(meta['freqs'])
Ns = np.array([fm.susceptibility(z, m)[1] for z, m in runs])            # [seed, f, pop, type]
has_frozen = 'rate_frozen_A+' in runs[0][0].files
Nfz = np.array([fm.susceptibility(z, m, frozen=True)[1] for z, m in runs]) if has_frozen else None
rates = {p: float(np.mean([m['rates'][p] for _, m in runs])) for p in pops}
meta = dict(meta, rates=rates)
# in-degrees and delays averaged over networks
for name in meta['wiring']:
    meta['wiring'][name]['K'] = float(np.mean([m['wiring'][name]['K'] for _, m in runs]))


def predict(N, mods=None):
    return fm.model(meta, fr, fm.smooth(fr, N) if SMOOTH else N, mods)


def boot_peak(mods=None, n=200, seed=0):
    rng = np.random.default_rng(seed)
    out = [predict(Ns[rng.integers(0, len(Ns), len(Ns))].mean(0), mods)['peak'] for _ in range(n)]
    return np.nanpercentile(out, [2.5, 97.5]).tolist()


N = Ns.mean(0)
base = predict(N)
res = {'condition': COND, 'n_networks': len(runs), 'rates_hz': rates}

# ---- measured spectrum of the same closed runs
P = []
for z, m in runs:
    x = np.asarray(z['rate_closed'][m['settle_ms']:, iE], float)
    fw, p = welch(x - x.mean(), fs=1000, nperseg=2000, noverlap=1500)
    P.append(p)
P = np.array(P)
rng = np.random.default_rng(1)
bp = [fm.peak(fw, P[rng.integers(0, len(P), len(P))].mean(0)) for _ in range(1000)]
res['baseline'] = {'predicted_peak_hz': base['peak'], 'predicted_ci': boot_peak(),
                   'measured_peak_hz_same_runs': fm.peak(fw, P.mean(0)),
                   'measured_ci_same_runs': np.nanpercentile(bp, [2.5, 97.5]).tolist(),
                   'loop_phase_zero_hz': base['zero_phase']}
cached = measured_peak(COND)
if cached:
    res['baseline']['measured_peak_hz_cached'] = cached['peak']
    res['baseline']['measured_ci_cached'] = cached['ci']
print(json.dumps(res['baseline'], indent=1))

# ---- held-out interventions: parameters changed analytically, susceptibilities kept from baseline
tests = {}
if COND == 'L6':
    ES, SE = ('E', 'SOM'), ('SOM', 'E')
    for t in [10, 15, 30, 40]:
        tests[f'L6_tauSOME{t}'] = {'tau': {SE: t}, 'gain': {SE: 20 / t}}
        tests[f'L6_tauSOM{t}'] = {'tau': {('SOM', p): t for p in ['E', 'PV', 'VIP']},
                                  'gain': {('SOM', p): 20 / t for p in ['E', 'PV', 'VIP']}}
    for d in [2.5, 5, 10]:
        tests[f'L6_loopdelay+{d:g}'] = {'delay_shift': {ES: d, SE: d}}
    for s in [0.5, 1.5, 2]:
        tests[f'L6_loopdelayx{s:g}'] = {'delay_scale': {ES: s, SE: s}}
    for g in [0.2, 0.44, 1, 1.95]:
        tests[f'L6_SS{g:g}'] = {'add': {('SOM', 'SOM'): (33.0, g, ('SOM', 'E'))}}
rows = []
for name, mods in tests.items():
    m = measured_peak(name)
    pr = predict(N, mods)
    rows.append({'condition': name, 'predicted_hz': pr['peak'], 'predicted_ci': boot_peak(mods, 100),
                 'measured_hz': m['peak'], 'measured_ci': m['ci'], 'measured_prom_db': m['prom_db'], 'n': m['n']})
    print(f"{name:20s} predicted {pr['peak']:6.2f}  measured {m['peak']:6.2f} [{m['ci'][0]:.2f}, {m['ci'][1]:.2f}]  prom {m['prom_db']:.1f} dB")
res['held_out'] = rows

# ---- phase budget of the E-SOM loop at the predicted peak
def budget(Nx, mods=None, at=None):
    """Phase lag (deg, positive = lag) of each stage of the loop at frequency `at` (default: where the
    loop phase crosses zero nearest the spectral peak)."""
    pr = predict(Nx, mods)
    zp = pr['zero_phase']
    if at is None:
        at = min(zp, key=lambda x: abs(x - pr['peak'])) if zp and np.isfinite(pr['peak']) else pr['peak']
    k = int(np.argmin(np.abs(f - at)))
    Nf = fm.interp(fr, fm.smooth(fr, Nx) if SMOOTH else Nx)[k]                      # [pop, type]
    G = fm.couplings(meta, f, mods)[k]                           # [target, source, type]
    w = 2 * np.pi * f[k]
    deg = lambda c: float(-np.degrees(np.angle(c)))
    es = meta['wiring']['E_SOM']; se = meta['wiring']['SOM_E']
    mm = mods or {}
    D_es = fm.delay_factor(es['delay_hist'], f[k:k + 1], mm.get('delay_shift', {}).get(('E', 'SOM'), 0), mm.get('delay_scale', {}).get(('E', 'SOM'), 1))[0]
    D_se = fm.delay_factor(se['delay_hist'], f[k:k + 1], mm.get('delay_shift', {}).get(('SOM', 'E'), 0), mm.get('delay_scale', {}).get(('SOM', 'E'), 1))[0]
    m_SE_dir = (Nf[iS, 0] * G[iS, iE, 0] + Nf[iS, 1] * G[iS, iE, 1])      # E -> SOM arm
    m_ES_dir = Nf[iE, 2] * G[iE, iS, 2]                                    # SOM -> E arm
    syn_SE = m_SE_dir / (Nf[iS, 0] * D_es)                                 # AMPA + NMDA mixture, relative to the AMPA response
    syn_ES = G[iE, iS, 2] / D_se
    m = pr['m'][k]
    b = {'frequency_hz': float(f[k]), 'peak_hz': pr['peak'],
         'E->SOM delay': deg(D_es), 'E->SOM synapse (AMPA+NMDA)': deg(syn_SE), 'SOM neurons': deg(Nf[iS, 0]),
         'SOM->E delay': deg(D_se), 'SOM->E synapse (GABA decay)': deg(syn_ES), 'E neurons': deg(-Nf[iE, 2]),
         'other pathways through PV, VIP': deg(m[0, 1] * m[1, 0] / (m_ES_dir * m_SE_dir)),
         'E self-coupling': deg(1 / (1 - m[0, 0])), 'SOM self-coupling': deg(1 / (1 - m[1, 1])),
         'loop_gain_abs': float(abs(pr['L'][k]))}
    b['total'] = sum(v for kk, v in b.items() if kk not in ('frequency_hz', 'peak_hz', 'loop_gain_abs'))
    return b


res['phase_budget'] = budget(N)
print(json.dumps(res['phase_budget'], indent=1))
if has_frozen:
    Nz = Nfz.mean(0)
    k = int(np.argmin(np.abs(fr - round(res['phase_budget']['frequency_hz']))))
    res['adaptation'] = {
        'peak_without_spike_adaptation_feedback_hz': predict(Nz)['peak'],
        'phase_lead_from_adaptation_deg': {p: {x: float(np.degrees(np.angle(N[k, i, j] / Nz[k, i, j])))
                                              for j, x in enumerate(fm.TYPES)} for i, p in enumerate(pops)},
        'gain_ratio_with_over_without': {p: {x: float(abs(N[k, i, j] / Nz[k, i, j]))
                                             for j, x in enumerate(fm.TYPES)} for i, p in enumerate(pops)}}
    print(json.dumps(res['adaptation'], indent=1))

# ---- counterfactuals inside the model: remove one source of lag at a time
cf = {'no E-SOM loop delays': {'delay_scale': {('E', 'SOM'): 0.0, ('SOM', 'E'): 0.0}},
      'instantaneous SOM->E synapse': {'no_filter': [('SOM', 'E')]},
      'no NMDA on E->SOM (AMPA only, same charge)': None,
      'no PV->E': {'remove': [('PV', 'E')]}, 'no VIP->SOM': {'remove': [('VIP', 'SOM')]},
      'no E->E': {'remove': [('E', 'E')]}}
res['counterfactual_peaks_hz'] = {}
for name, mods in cf.items():
    if mods is None:
        continue
    res['counterfactual_peaks_hz'][name] = predict(N, mods)['peak']
# local sensitivities (Hz per unit change), by finite differences in the model
sens = {}
for name, lo, hi, dx in [('SOM->E decay (per ms)', {'tau': {('SOM', 'E'): 19}, 'gain': {('SOM', 'E'): 20 / 19}},
                          {'tau': {('SOM', 'E'): 21}, 'gain': {('SOM', 'E'): 20 / 21}}, 2.0),
                         ('E->SOM delay (per ms)', {'delay_shift': {('E', 'SOM'): -0.5}}, {'delay_shift': {('E', 'SOM'): 0.5}}, 1.0),
                         ('SOM->E delay (per ms)', {'delay_shift': {('SOM', 'E'): -0.5}}, {'delay_shift': {('SOM', 'E'): 0.5}}, 1.0),
                         ('NMDA decay (per 10 ms, same charge)', {'tau': {'NMDA': 95}, 'nmda_gain': 100 / 95},
                          {'tau': {'NMDA': 105}, 'nmda_gain': 100 / 105}, 1.0),
                         ('E->SOM strength (per +10%)', {'gain': {('E', 'SOM'): 0.95}}, {'gain': {('E', 'SOM'): 1.05}}, 1.0),
                         ('SOM->E strength (per +10%)', {'gain': {('SOM', 'E'): 0.95}}, {'gain': {('SOM', 'E'): 1.05}}, 1.0),
                         ('PV->E strength (per +10%)', {'gain': {('PV', 'E'): 0.95}}, {'gain': {('PV', 'E'): 1.05}}, 1.0),
                         ('E->E strength (per +10%)', {'gain': {('E', 'E'): 0.95}}, {'gain': {('E', 'E'): 1.05}}, 1.0)]:
    sens[name] = (predict(N, hi)['peak'] - predict(N, lo)['peak']) / dx
res['sensitivity_hz'] = sens
print(json.dumps(res['counterfactual_peaks_hz'], indent=1)); print(json.dumps(sens, indent=1))

json.dump(res, open(os.path.join(fm.OUT, f'{COND}{TAG}_results.json'), 'w'), indent=1)
np.savez(os.path.join(fm.OUT, f'{COND}{TAG}_model.npz'), f=f, S=base['S'], L=base['L'], M=base['M'], fr=fr, N=N,
         N_seeds=Ns, **({'N_frozen': Nfz.mean(0)} if has_frozen else {}), f_meas=fw, psd_meas=P)

# ---- figures
NAVY, ORANGE, GREY = '#002255', '#FF6D1E', '#8C8C8C'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

fig, ax = plt.subplots(2, 3, figsize=(10, 5.2), layout='constrained')
panels = [(iE, 2, 'E cells, inhibitory conductance', -1), (iS, 0, 'SOM cells, AMPA conductance', 1), (iS, 1, 'SOM cells, NMDA conductance', 1)]
for c, (i, x, title, sg) in enumerate(panels):
    sem = np.abs(Ns[:, :, i, x] - N[:, i, x]).std(0) / np.sqrt(len(Ns))
    ax[0, c].errorbar(fr, np.abs(N[:, i, x]), sem, color=NAVY, lw=1, marker='o', ms=2.5, label='intact')
    ax[1, c].plot(fr, np.degrees(np.angle(sg * N[:, i, x])), color=NAVY, lw=1, marker='o', ms=2.5)
    if has_frozen:
        ax[0, c].plot(fr, np.abs(Nfz.mean(0)[:, i, x]), color=ORANGE, lw=1, marker='o', ms=2.5, label='spike adaptation replayed')
        ax[1, c].plot(fr, np.degrees(np.angle(sg * Nfz.mean(0)[:, i, x])), color=ORANGE, lw=1, marker='o', ms=2.5)
    ax[0, c].set(title=title, xscale='log'); ax[1, c].set(xscale='log', xlabel='Frequency (Hz)')
    ax[1, c].axhline(0, color=GREY, lw=0.5)
ax[0, 0].set_ylabel('Gain (Hz per nS)'); ax[1, 0].set_ylabel('Phase (deg, negative = lag)'); ax[0, 0].legend(frameon=False, fontsize=7)
fig.savefig(os.path.join(FIG, f'{COND}{TAG}_susceptibility.png'), dpi=160); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), layout='constrained')
sel = (fw >= 2) & (fw <= 45)
ax[0].plot(fw[sel], P.mean(0)[sel] / P.mean(0)[sel].max(), color=NAVY, label='spiking network')
s = base['S'][:, iE]; g = (f >= 2) & (f <= 45)
ax[0].plot(f[g], s[g] / s[g].max(), color=ORANGE, label='theory, nothing fitted')
ax[0].set(xscale='log', xlabel='Frequency (Hz)', ylabel='E-rate power (normalised)', title='Baseline spectrum'); ax[0].legend(frameon=False, fontsize=7, loc='upper left')
if rows:
    good = [r for r in rows if r['measured_prom_db'] >= 3 and np.isfinite(r['predicted_hz'])]
    weak = [r for r in rows if r not in good and np.isfinite(r['predicted_hz'])]
    for rr, col, lab in [(good, NAVY, 'clear peak'), (weak, GREY, 'weak peak (< 3 dB)')]:
        if rr:
            ax[1].errorbar([r['measured_hz'] for r in rr], [r['predicted_hz'] for r in rr],
                           xerr=np.abs(np.array([r['measured_ci'] for r in rr]).T - [r['measured_hz'] for r in rr]),
                           yerr=np.abs(np.array([r['predicted_ci'] for r in rr]).T - [r['predicted_hz'] for r in rr]),
                           fmt='o', ms=4, color=col, lw=0.8, label=lab)
    ax[1].plot(res['baseline'].get('measured_peak_hz_cached', np.nan), base['peak'], 's', color=ORANGE, ms=6, label='baseline')
    ax[1].plot([6, 24], [6, 24], color=GREY, lw=0.6)
    ax[1].set(xlabel='Measured peak (Hz)', ylabel='Predicted peak (Hz)', title='Held-out timing changes'); ax[1].legend(frameon=False, fontsize=7)
b = res['phase_budget']
keys = [k for k in b if k not in ('frequency_hz', 'peak_hz', 'loop_gain_abs', 'total')]
ax[2].barh(range(len(keys)), [b[k] for k in keys], color=[ORANGE if b[k] > 0 else NAVY for k in keys])
ax[2].set_yticks(range(len(keys))); ax[2].set_yticklabels(keys, fontsize=7); ax[2].invert_yaxis()
ax[2].axvline(0, color='k', lw=0.5)
ax[2].set(xlabel='Phase lag where the loop closes (deg)', title=f"Lags sum to {b['total']:.0f} deg at {b['frequency_hz']:.1f} Hz")
fig.savefig(os.path.join(FIG, f'{COND}{TAG}_frequency_theory.png'), dpi=160); plt.close(fig)
print('figures in', FIG)
