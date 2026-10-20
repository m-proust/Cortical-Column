"""Checks on the L6 frequency prediction: estimator choice by cross-validation (no held-out peaks used),
prediction intervals, error against rate change, sensitivity to the VIP response."""
import json, os
import numpy as np
import freqmech as fm
from measured import measured_peak, spectra

runs = fm.load('L6'); meta = runs[0][1]; pops = meta['pops']; fr = np.array(meta['freqs'])
Ns = np.array([fm.susceptibility(z, m)[1] for z, m in runs])
rates = dict(zip(pops, np.mean([np.asarray(z['rate_closed'][2000:], float).mean(0) for z, _ in runs], 0)))
meta = dict(meta, rates=rates)
out = {}

# 1. estimator: leave one network out; predict its raw susceptibility from the other four
band = (fr >= 4) & (fr <= 30)
ests = {'unsmoothed mean': lambda X: X, **{f'polynomial order {o}': (lambda X, o=o: fm.smooth(fr, X, o)) for o in [4, 6, 8, 10, 12]}}
cv = {}
for name, est in ests.items():
    e = []
    for i in range(len(Ns)):
        tr = est(np.delete(Ns, i, 0).mean(0))
        e.append(np.linalg.norm((tr - Ns[i])[band]) ** 2 / np.linalg.norm(Ns[i][band]) ** 2)
    fit = est(Ns.mean(0))
    cv[name] = {'cv_relative_sq_error': float(np.mean(e)), 'sem': float(np.std(e) / np.sqrt(len(e))),
                'misfit_to_mean_relative_sq': float(np.linalg.norm((fit - Ns.mean(0))[band]) ** 2 / np.linalg.norm(Ns.mean(0)[band]) ** 2),
                'baseline_peak': fm.model(meta, fr, fit)['peak']}
    print(f"{name:22s} CV error {cv[name]['cv_relative_sq_error']:.4f} +- {cv[name]['sem']:.4f}   misfit to mean {cv[name]['misfit_to_mean_relative_sq']:.4f}   baseline peak {cv[name]['baseline_peak']:.2f}")
out['estimator_cv'] = cv
# where does the order-6 polynomial misfit? residual / sem for the three loop responses
iE, iS = pops.index('E'), pops.index('SOM'); N = Ns.mean(0); sem = np.abs(Ns - N).std(0) / np.sqrt(len(Ns))
res6 = np.abs(fm.smooth(fr, N, 6) - N) / sem
for lab, (i, x) in {'E inh': (iE, 2), 'SOM AMPA': (iS, 0), 'SOM NMDA': (iS, 1)}.items():
    print('order-6 misfit in SEM units,', lab, ' 1-6 Hz:', np.round(res6[fr <= 6, i, x].mean(), 1), ' 7-30 Hz:', np.round(res6[(fr > 6) & (fr <= 30), i, x].mean(), 1))

# 2. prediction intervals and joint coverage; 3. error against rate change
r = json.load(open(os.path.join(fm.OUT, 'L6_results.json')))
rows = [x for x in r['held_out'] if 'SS' not in x['condition']]
base_rates = measured_peak('L6', nboot=10)['rates']
tab = []
for x in rows:
    m = measured_peak(x['condition'], nboot=10)
    pw = (x['predicted_ci'][1] - x['predicted_ci'][0]) / 3.92; mw = (x['measured_ci'][1] - x['measured_ci'][0]) / 3.92
    err = x['predicted_hz'] - x['measured_hz']
    tab.append({'condition': x['condition'], 'err': err, 'z': err / np.hypot(pw, mw),
                'dSOM_pct': 100 * (m['rates']['SOM'] / base_rates['SOM'] - 1), 'dPV_pct': 100 * (m['rates']['PV'] / base_rates['PV'] - 1)})
    print(f"{x['condition']:18s} err {err:+.2f} Hz  z {tab[-1]['z']:+.1f}  SOM rate {tab[-1]['dSOM_pct']:+.1f}%  PV {tab[-1]['dPV_pct']:+.1f}%")
err = np.array([t['err'] for t in tab]); z = np.array([t['z'] for t in tab]); ds = np.array([t['dSOM_pct'] for t in tab])
out['coverage'] = {'n': len(tab), 'within_joint_95': int((np.abs(z) < 1.96).sum()), 'rmse': float(np.sqrt((err ** 2).mean())), 'bias': float(err.mean()),
                   'corr_error_vs_SOM_rate_change': float(np.corrcoef(err, ds)[0, 1]), 'slope_hz_per_pct': float(np.polyfit(ds, err, 1)[0]),
                   'intercept_hz': float(np.polyfit(ds, err, 1)[1])}
out['table'] = tab
print(json.dumps(out['coverage'], indent=1))

# 4. VIP response wrong by 15% in gain, or 15 deg in phase
iV = pops.index('VIP'); vip = {}
for lab, fac in [('gain x0.85', 0.85), ('gain x1.15', 1.15), ('phase +15 deg', np.exp(1j * np.radians(15))), ('phase -15 deg', np.exp(-1j * np.radians(15)))]:
    X = N.copy(); X[:, iV] *= fac
    vip[lab] = fm.model(meta, fr, X)['peak'] - fm.model(meta, fr, N)['peak']
out['vip_response_error_shift_hz'] = vip; print(vip)
json.dump(out, open(os.path.join(fm.OUT, 'L6_checks.json'), 'w'), indent=1)
