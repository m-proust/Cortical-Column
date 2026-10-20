"""Quasi-renewal AdEx population: adiabatic hazard from the EIF Fokker-Planck, w(s) after the last spike exact,
older spikes + subthreshold a-term as mean-field feedback.  chi = nu0*G(w)/(1-P0(w)),
G = int ds chi_EIF(w; mu - w(s)) [S0(s) - int_s^inf P0(x) e^{-iw(x-s)} dx]."""
import json, os, time, warnings, sys; warnings.filterwarnings('ignore')
import numpy as np
from eif_theory import *
ROOT='/Users/mathildeproust/Desktop/Cortical-Column'

def eif_grid(cells, p, wgrid, freqs, scale, rate_pop):
    """Phi[cell, k] (Hz) and chi[cell, k, f] (Hz/pA) of the EIF with a static extra current -w."""
    c = cells[p]; Vbar = np.full(len(c['C']), -58.0)
    for _ in range(2):
        gtot, tau, sig2, _ = population_response(cells, p, freqs, scale, Vbar)
        gA, gN, gI = c['gm']['A'], c['gm']['N'], c['gm']['I']
        wa = c['a'] * (Vbar - c['EL'])
        Vbar = (gA * EE + gN * B(Vbar) * EE + gI * EI + c['gL'] * c['EL'] - wa - c['b']*c['tw']*1e-3*rate_pop) / gtot
    Vfree = (gA * EE + gN * B(Vbar) * EE + gI * EI + c['gL'] * c['EL'] - wa) / gtot
    Phi, Chi = [], []
    wv = 2 * np.pi * np.asarray(freqs) * 1e-3
    for wk in wgrid:
        F = lambda V, wk=wk: (c['gL'] * (c['EL'] - V) + c['gL'] * c['DT'] * np.exp((V - VT) / c['DT'])
                              + gA * (EE - V) + gN * B(V) * (EE - V) + gI * (EI - V) - wa - wk) / gtot
        if len(wv) == 0:
            r0, r1 = threshold_integrate(F, tau, sig2, np.full(len(tau), c['Vcut']), c['tref'], wv, dV=0.02)
            Phi.append(r0); Chi.append(r1); continue
        r0, r1, rg = threshold_integrate(F, tau, sig2, np.full(len(tau), c['Vcut']), c['tref'], wv, dV=0.02,
                                         shunt=(Vfree - wk / gtot, gtot))
        chiI = r1 / gtot[:, None]                                    # Hz/pA
        Vb = (Vfree - wk / gtot)[:, None]
        Bn = B(Vb); sN = Bn * (1 - (EE - Vb) * 0.062 * (1 - Bn))
        # Hz/nS for AMPA, NMDA, GABA probes: current at mean driving force + shunt at mean voltage
        Chi.append(np.stack([chiI * (EE - Vb) + rg, chiI * (EE - Vb) * Bn + sN * rg, chiI * (EI - Vb) + rg, chiI], -1))
        Phi.append(r0)
    Chi = np.array(Chi)
    return np.array(Phi).T, np.moveaxis(Chi, 0, 1), gtot, tau, wa

def renewal(cells, p, freqs, scale, rate_pop, n_w=28, smax=4.0, ds=1e-3):
    c = cells[p]; n = len(c['C'])
    bmax = 3.0 * c['b'].max() + 1
    wgrid = np.r_[0, np.geomspace(0.5, bmax, n_w - 1)] if bmax > 1 else np.array([0.0])
    Phi, Chi, gtot, tau, wa = eif_grid(cells, p, wgrid, freqs, scale, rate_pop)
    s = np.arange(0, smax, ds)
    tw = c['tw'][:, None] * 1e-3
    wafter = c['b'].copy()
    for _ in range(25):
        wsv = wafter[:, None] * np.exp(-s[None] / tw)                    # [cell, s]
        lam = np.array([np.exp(np.interp(wsv[i], wgrid, np.log(np.maximum(Phi[i], 1e-12)))) for i in range(n)])
        S0 = np.exp(-np.cumsum(lam, 1) * ds); P0 = lam * S0
        L = (P0 * np.exp(-s[None] / tw)).sum(1) * ds
        new = c['b'] / np.maximum(1 - L, 1e-6)
        if np.allclose(new, wafter, rtol=1e-4): break
        wafter = new
    nu = 1 / (S0.sum(1) * ds)
    out = dict(nu=nu, wafter=wafter, wgrid=wgrid, gtot=gtot, tau=tau)
    if freqs is None or len(freqs) == 0: return out
    om = 2 * np.pi * np.asarray(freqs)
    chi_pop = np.zeros((n, len(om), 4), complex)
    for i in range(n):
      for ch in [3, 0, 1, 2]:
        # chi_EIF at age s: interpolate over w (real & imag)
        cs = np.array([np.interp(wsv[i], wgrid, Chi[i, :, f, ch].real) + 1j * np.interp(wsv[i], wgrid, Chi[i, :, f, ch].imag)
                       for f in range(len(om))])                        # [f, s]
        E = np.exp(-1j * om[:, None] * s[None])                          # [f, s]
        # T(s) = int_s^inf P0(x) e^{-iw(x-s)} dx  via reversed cumulative sum
        Q = P0[i][None] * E
        T = np.flip(np.cumsum(np.flip(Q, 1), 1), 1) * ds / E
        G = (cs * (S0[i][None] - T)).sum(1) * ds
        P0h = Q.sum(1) * ds
        chir = nu[i] * G / (1 - P0h)
        # mean-field feedback of OLDER spikes' w (beyond the next spike) and of the subthreshold a-term
        Y = (np.exp(-s / (tw[i, 0]))[None] * (1 - S0[i])[None] * E).sum(1) * ds
        Z = 1 / (gtot[i] * (1 + 1j * om * 1e-3 * tau[i]))
        if ch == 3:
            K = c['a'][i] * Z / (1 + 1j * om * tw[i, 0]) + c['b'][i] * Y * chir
        chi_pop[i, :, ch] = chir / (1 + K)
    out['chi'] = chi_pop
    return out

if __name__ == '__main__':
    d = np.load(os.path.join(fm.OUT, 'analytic', 'Nmeas.npz')); fr, N = d['fr'], d['N']
    runs = [r for r in fm.load('L6') if 'rate_frozen_A+' in r[0].files]; meta = runs[0][1]
    rates = json.load(open(os.path.join(fm.OUT, 'L6_results.json')))['rates_hz']
    cells = sample_cells(meta, rates, n=int(sys.argv[2]) if len(sys.argv) > 2 else 80)
    res = {}
    for j, p in enumerate(sys.argv[1].split(',')):
        j = meta['pops'].index(p); t = time.time()
        r1 = renewal(cells, p, [], 1.0, rates[p])['nu'].mean()
        lo, hi = 0.5, 3.0
        for _ in range(11):
            mid = np.sqrt(lo * hi); r = renewal(cells, p, [], mid, rates[p])['nu'].mean()
            lo, hi = (mid, hi) if r < rates[p] else (lo, mid)
        sc = np.sqrt(lo * hi); o = renewal(cells, p, fr, sc, rates[p]); chi = o['chi'].mean(0)[:, 3]; res_N = o['chi'].mean(0)[:, :3]
        ma = (N[:, j, 0] - N[:, j, 2]) / 80
        res[p] = dict(scale=sc, chi=chi, N=res_N, ma=ma, nu=o['nu'].mean(), rate_campbell=r1)
        print(f'\n== {p}: Campbell-only rate {r1:.2f} Hz (meas {rates[p]:.2f}); noise scale {sc:.2f}; w_after {np.median(o["wafter"]):.0f} pA ({time.time()-t:.0f}s)')
        print('  f  | renewal theory | measured   (|chi| Hz/nA, phase)')
        for i in [0, 2, 4, 7, 9, 11, 14, 19, 24, 29, 34, 39]:
            f = lambda z: f'{abs(z)*1e3:6.0f} {np.degrees(np.angle(z)):5.0f}'
            print(f'{fr[i]:4.0f} | {f(chi[i])} | {f(ma[i])}')
        np.save(os.path.join(fm.OUT, 'analytic', f'renewal_{p}.npy'), res[p], allow_pickle=True)
