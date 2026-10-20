"""Parameter-only linear response of the L6 AdEx populations (no simulation).

Per cell (heterogeneity sampled exactly as add_heterogeneity_to_layer + binomial in-degrees):
  mean conductances  g_X = sum_src K * rate * jump * tau         (Campbell, mean)
  voltage variance   sum_X (E_X - V)^2 Var(g_X)/g_tot^2 * tau_X/(tau_X + tau_eff)
                     Var(g_X) = sum_src K * rate * jump^2 * tau/2  (Campbell, variance)
  EIF Fokker-Planck in the effective-time-constant approximation, white noise with the same voltage
  variance; stationary rate and current-modulation response by Richardson's threshold integration.
Adaptation: w = a(V-EL) (subthreshold) + b*spikes, each low-passed by tau_w (mean-field feedback).
"""
import json, sys
import numpy as np

ROOT = '/Users/mathildeproust/Desktop/Cortical-Column'
sys.path.insert(0, ROOT + '/studies/resonance')
import freqmech as fm

IP = {'E': dict(a=4, b=100, DT=2, C=97, gL=4.2, tw=100, EL=-65, tref=5),
      'PV': dict(a=0, b=0, DT=.5, C=38, gL=3.8, tw=50, EL=-68, tref=2),
      'SOM': dict(a=4, b=50, DT=1.5, C=92, gL=4.3, tw=180, EL=-63, tref=5),
      'VIP': dict(a=2, b=40, DT=2, C=70, gL=3.3, tw=150, EL=-61, tref=5)}
VT, VRE, EE, EI = -50.0, -65.0, 0.0, -80.0
TAU_A = {'E': 5., 'PV': 1., 'SOM': 2., 'VIP': 2.}
TAU_I = {'PV': 6., 'SOM': 20., 'VIP': 8.}
SLOPE = False
EXT_J, EXT_RATE = 1.25, 5.0                     # nS, Hz (Poisson AMPA)


def B(v):
    return 1.0 / (1.0 + 0.28 * np.exp(-0.062 * v))


def sample_cells(meta, rates, n=300, seed=0):
    rng = np.random.default_rng(seed)
    cells = {}
    for p in meta['pops']:
        q = IP[p]
        vary = lambda x, s: x * np.clip(1 + rng.standard_normal(n) * s, .5, 1.5)
        c = dict(C=vary(q['C'], .15), gL=vary(q['gL'], .12), tw=vary(q['tw'], .15), b=vary(q['b'], .20),
                 a=vary(q['a'], .15), EL=q['EL'] + 2 * rng.standard_normal(n), DT=vary(q['DT'], .10),
                 Vcut=VT + 5 * q['DT'], tref=q['tref'])
        # mean and variance of each conductance type (nS, nS^2); Poisson in-degree is fixed per cell
        gm = {'A': np.full(n, meta['poisson_N'][p] * EXT_RATE * EXT_J * TAU_A[p] * 1e-3),
              'N': np.zeros(n), 'I': np.zeros(n)}
        gv = {'A': np.full(n, meta['poisson_N'][p] * EXT_RATE * EXT_J ** 2 * TAU_A[p] * 1e-3 / 2),
              'N': np.zeros(n), 'I': np.zeros(n)}
        tv = {'A': TAU_A[p], 'N': 100., 'I': 0.0}    # I: variance-weighted tau, filled below
        iw = 0.0
        for name, wi in meta['wiring'].items():
            pre, post = name.split('_')
            if post != p:
                continue
            npre = meta['neuron_counts'][pre]
            K = rng.binomial(npre, min(wi['K'] / npre, 1.0), n).astype(float)
            r = rates[pre]
            cc = meta['conductance']
            if pre == 'E':
                for typ, key, tau in [('A', '_AMPA', TAU_A[p]), ('N', '_NMDA', 100.)]:
                    J = cc.get(name + key, 0.0 if key == '_NMDA' else 0.01)
                    gm[typ] += K * r * J * tau * 1e-3
                    gv[typ] += K * r * J ** 2 * tau * 1e-3 / 2
            else:
                J, tau = cc.get(name, 0.02), TAU_I[pre]
                gm['I'] += K * r * J * tau * 1e-3
                v = K * r * J ** 2 * tau * 1e-3 / 2
                gv['I'] += v
                iw += v * tau
        tv['I'] = iw / np.maximum(gv['I'], 1e-30)
        c.update(gm=gm, gv=gv, tv=tv)
        cells[p] = c
    return cells


def threshold_integrate(F, tau, sig2, Vcut, tref, w_list, dV=0.01, Vlb=-100.0, shunt=None):
    """F(V) drift in mV (array over cells, function of V); tau, sig2 per cell.
    Returns r0 (Hz) and r1/mu1 (Hz/mV) for each angular frequency in w_list (rad/ms)."""
    V = np.arange(Vlb, Vcut[0] + dV / 2, dV)
    ncell = len(tau)
    w = np.asarray(w_list)[None, :]
    P0 = np.zeros(ncell); J0 = np.ones(ncell)
    PA = np.zeros((ncell, w.shape[1]), complex); JA = np.ones_like(PA) + 0j
    PB = np.zeros_like(PA); JB = np.zeros_like(PA)
    PC = np.zeros_like(PA); JC = np.zeros_like(PA)
    sumP0 = np.zeros(ncell)
    H = (tau / sig2)[:, None]
    for k in range(len(V) - 1, 0, -1):
        G = F(V[k]) / sig2
        e = np.exp(-np.clip(G * dV, -50, 700))
        ph = np.where(np.abs(G * dV) > 1e-8, (1 - e) / np.where(G == 0, 1, G), dV)
        G2, e2, ph2 = G[:, None], e[:, None], ph[:, None]
        # stationary
        P0n = P0 * e + (tau / sig2) * J0 * ph
        # r-driven part (J=1 at threshold) and mu-driven part (source P0/sig2)
        PAn = PA * e2 + H * JA * ph2
        PBn = PB * e2 + H * JB * ph2 - (P0[:, None] / sig2[:, None]) * ph2
        if shunt is not None:
            Vb, gt = shunt
            PCn = PC * e2 + H * JC * ph2 - (P0 * (Vb - V[k]) / gt / sig2)[:, None] * ph2
            JC = JC + 1j * w * dV * PC
            PC = PCn
        JA = JA + 1j * w * dV * PA
        JB = JB + 1j * w * dV * PB
        P0, PA, PB = P0n, PAn, PBn
        sumP0 += P0 * dV
        if V[k] > VRE >= V[k - 1]:              # reset: flux re-injected (delayed by tref)
            J0 = J0 - 1.0
            JA = JA - np.exp(-1j * w * tref)
    # P in units of (1/ms per unit flux) * ms... normalise: r0 = 1/(int P0 + tref) in 1/ms
    r0 = 1.0 / (sumP0 + tref)
    # boundary: total flux -> 0 at Vlb:  r1*JA + mu1*JB = 0
    resp = -JB / JA                                      # r1/mu1 in units of r0 ... scaled below
    if shunt is not None:
        return r0 * 1e3, resp * r0[:, None] * 1e3, -JC / JA * r0[:, None] * 1e3
    return r0 * 1e3, resp * r0[:, None] * 1e3


def population_response(cells, p, freqs, sigma_scale=1.0, Vbar=None):
    c = cells[p]
    n = len(c['C'])
    Vbar = Vbar if Vbar is not None else np.full(n, -58.0)
    gA, gN, gI = c['gm']['A'], c['gm']['N'], c['gm']['I']
    gchord = c['gL'] + gA + gN * B(Vbar) + gI
    slope = B(Vbar) * (1 - (EE - Vbar) * 0.062 * (1 - B(Vbar)))         # NMDA slope conductance factor
    gexp = c['gL'] * np.exp((Vbar - VT) / c['DT'])                     # EIF spike-current slope
    gtot = gchord if not SLOPE else c['gL'] + gA + gN * slope + gI - gexp
    tau = c['C'] / gtot                                   # ms
    var = sum((dr - Vbar) ** 2 * c['gv'][X] / gtot ** 2 * c['tv'][X] / (c['tv'][X] + tau)
              for X, dr in [('A', EE - 0 * Vbar), ('N', (EE - Vbar) * B(Vbar) + Vbar), ('I', EI + 0 * Vbar)])
    sig2 = var * sigma_scale ** 2
    return gtot, tau, sig2, c


def solve(cells, p, freqs, rate_pop, sigma_scale=1.0, wmean=None):
    c = cells[p]
    Vbar = np.full(len(c['C']), -58.0)
    for _ in range(2):                                     # Vbar -> conductances -> Vbar (mean voltage)
        gtot, tau, sig2, _ = population_response(cells, p, freqs, sigma_scale, Vbar)
        w0 = c['a'] * (Vbar - c['EL']) + c['b'] * c['tw'] * 1e-3 * rate_pop if wmean is None else wmean
        gA, gN, gI = c['gm']['A'], c['gm']['N'], c['gm']['I']

        def F(V):
            I = (c['gL'] * (c['EL'] - V) + c['gL'] * c['DT'] * np.exp((V - VT) / c['DT'])
                 + gA * (EE - V) + gN * B(V) * (EE - V) + gI * (EI - V) - w0)
            return I / gtot
        Vbar_new = c['EL'] + (gA * EE + gN * B(Vbar) * EE + gI * EI + c['gL'] * c['EL'] - w0) / gtot - c['EL']
        Vbar = Vbar_new
    w = 2 * np.pi * np.asarray(freqs) * 1e-3
    r0, r1 = threshold_integrate(F, tau, sig2, np.full(len(tau), c['Vcut']), c['tref'], w)
    chi_cell = r1 / gtot[:, None]                          # Hz/pA (mu1 = I1/gtot)
    return dict(r0=r0, chi_eif=chi_cell, gtot=gtot, tau=tau, sig=np.sqrt(sig2), Vbar=Vbar, w0=w0)


def adapt(chi_eif, c, freqs, gtot, tau, mode):
    """mode 'frozen': subthreshold a-term only; 'full': + mean-field spike-triggered b-term."""
    wm = 2 * np.pi * np.asarray(freqs) * 1e-3
    tw = c['tw'][:, None]
    Z = 1 / (gtot[:, None] * (1 + 1j * wm * tau[:, None]))           # mV/pA, mean voltage per current
    K = c['a'][:, None] * Z / (1 + 1j * wm * tw)
    if mode == 'full':
        K = K + c['b'][:, None] * tw * 1e-3 * chi_eif / (1 + 1j * wm * tw)
    return chi_eif / (1 + K)
