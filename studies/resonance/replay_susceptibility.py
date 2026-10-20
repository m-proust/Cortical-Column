"""Neuronal susceptibility of each population at its intact operating state (isolated layer).

    ~/Desktop/venv/bin/python studies/resonance/replay_susceptibility.py L6 out.npz --seed 1

1. Run the closed network and record every spike.
2. Replay those spikes through the same synapses and delays (recurrent synapses off), with the
   same background Poisson stream: every cell gets exactly its baseline input and the network is
   open loop. The 'exact' branch checks that the spikes are reproduced.
3. On top of the replay, add a small conductance common to all cells of a population
   (AMPA-like, NMDA-like or GABA-like; a periodic multisine) with sign + and -. The rate change
   divided by the probe is the susceptibility N(f) in Hz/nS, with the operating state untouched.
4. Same again with each cell's spike-triggered adaptation (w += b) driven by its BASELINE spikes
   instead of its own ('frozen' branches): the susceptibility without adaptation feedback, at the
   same operating state.

The three inhibitory conductances enter the membrane equation identically, so one GABA-like probe
serves PV, SOM and VIP inputs.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

ap = argparse.ArgumentParser()
ap.add_argument('condition')
ap.add_argument('out')
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--T', type=float, default=60.0, help='measured seconds per branch (after 2 s settle)')
ap.add_argument('--amp', type=float, default=10.0, help='probe rms, in pA of equivalent current at -58 mV')
ap.add_argument('--no-frozen', action='store_true', help='skip the frozen-adaptation branches')
a = ap.parse_args()

import brian2 as b2
from brian2 import ms, nS, pA, siemens
b2.BrianLogger.log_level_error()

from minimal_config.config import make_config, add_tonic_inhibition
from src.column import CorticalColumn
from src.analysis import add_heterogeneity_to_layer

SETTLE = 2000                       # ms discarded at the start of every branch
PERIOD = 1000                       # ms, probe period
FREQS = np.r_[np.arange(1, 31), np.arange(32, 51, 2)].astype(float)
TYPES = ['A', 'N', 'I']             # AMPA-like, NMDA-like, GABA-like
V_REF = -58.0                       # mV, only used to choose probe sizes
DRIVE_MV = {'A': 0 - V_REF, 'N': (0 - V_REF) / (1 + 0.28 * np.exp(-0.062 * V_REF)), 'I': V_REF + 80}

start = time.time()
cfg = make_config(a.condition)
pops = list(next(iter(cfg['layers'].values()))['neuron_counts'])

# --- probe conductances and switchable spike-triggered adaptation, in every population ---
for j, p in enumerate(pops):
    eq = cfg['models']['equations'][p]
    for pat, rep in [(r'\+ gE_AMPA\*\(Ee - v\)', '+ (gE_AMPA + gp_A)*(Ee - v)'),
                     (r'\+ gE_NMDA\*\(Ee - v\)/', '+ (gE_NMDA + gp_N)*(Ee - v)/'),
                     (r'- \(gPV \+ gSOM \+ gVIP\)\*\(v - Ei\)', '- (gPV + gSOM + gVIP + gp_I)*(v - Ei)')]:
        eq, n = re.subn(pat, rep, eq, count=1)
        assert n == 1, (p, pat)
    eq += ''.join(f'\n        gp_{x} = s{x}*probe(t, {3 * j + k}) : siemens\n        s{x} : 1 (shared)'
                  for k, x in enumerate(TYPES))
    eq += '\n        adapt_on : 1 (shared)\n'
    cfg['models']['equations'][p] = eq
cfg['models']['reset'] = cfg['models']['reset'].replace('w+=b', 'w+=b*adapt_on')
assert 'adapt_on' in cfg['models']['reset']

n_period = int(round(a.T * 1000 / PERIOD))
T = SETTLE + n_period * PERIOD
rng = np.random.default_rng(1000 + a.seed)
tt = np.arange(PERIOD) / 1000.0
wave = np.zeros((PERIOD, 3 * len(pops)))
for c in range(wave.shape[1]):
    ph = rng.uniform(0, 2 * np.pi, len(FREQS))
    w = np.cos(2 * np.pi * tt[:, None] * FREQS[None] + ph).sum(1) * np.sqrt(2 / len(FREQS))
    wave[:, c] = w * a.amp / DRIVE_MV[TYPES[c % 3]]            # pA / mV = nS
probe_nS = np.tile(wave, (T // PERIOD + 1, 1))[:T + 1]
cfg['models']['common_namespace']['probe'] = b2.TimedArray(probe_nS * nS, dt=1 * ms)

b2.start_scope()
np.random.seed(a.seed); b2.seed(a.seed)
b2.defaultclock.dt = cfg['simulation']['DT']
dt_ms = float(b2.defaultclock.dt / ms)
col = CorticalColumn(column_id=0, config=cfg)
(layer,) = col.layers.values()
add_heterogeneity_to_layer(layer, cfg)
for ob in list(col.network.objects):
    if isinstance(ob, (b2.StateMonitor, b2.PopulationRateMonitor)):
        col.network.remove(ob)
add_tonic_inhibition(col, cfg)
groups = layer.neuron_groups
for g in groups.values():
    g.sA = 0; g.sN = 0; g.sI = 0; g.adapt_on = 1

# --- replay machinery: spike sources, copies of every recurrent synapse, adaptation replay ---
cc = next(iter(cfg['layers'].values()))['conductance']
sources = {p: b2.SpikeGeneratorGroup(len(g), [], [] * ms, name=f'replay_source_{p}') for p, g in groups.items()}
replicas, wiring = [], {}
for name, syn in layer.synapses.items():
    pre, post = name.split('_')
    if pre == 'E':
        event = f'gE_AMPA_post += {cc.get(name + "_AMPA", 0.01)}*nS'
        if cc.get(name + '_NMDA', 0) > 0:
            event += f'\ngE_NMDA_post += {cc[name + "_NMDA"]}*nS'
    else:
        event = f'g{pre}_post += {cc.get(name, 0.02)}*nS'
    rep = b2.Synapses(sources[pre], groups[post], on_pre=event, name='replay_syn_' + name)
    rep.connect(i=np.asarray(syn.i[:]), j=np.asarray(syn.j[:]))
    rep.delay = syn.delay[:]
    rep.active = False
    replicas.append(rep)
    d = np.asarray(syn.delay[:] / ms)
    wiring[name] = {'K': len(d) / len(groups[post]),
                    'K_sd': float(np.std(np.bincount(np.asarray(syn.j[:]), minlength=len(groups[post])))),
                    'delay_hist': np.bincount(np.rint(d / dt_ms).astype(int)).tolist()}
adapt_syn = []
for p, g in groups.items():
    s = b2.Synapses(sources[p], g, on_pre='w_post += b_post', name='replay_adapt_' + p)
    s.connect(j='i')
    s.active = False
    adapt_syn.append(s)
for g in sources.values():
    g.active = False
col.network.add(*sources.values(), *replicas, *adapt_syn)

STATE_VARS = ['v', 'w', 'gE_AMPA', 'gE_NMDA', 'gPV', 'gSOM', 'gVIP']
diag = {p: b2.StateMonitor(g, STATE_VARS, record=np.arange(min(64, len(g))), dt=5 * ms, name='diag_' + p)
        for p, g in groups.items()}
col.network.add(*diag.values())
col.network.store('init')


def extract():
    rate, spikes = [], {}
    for p, g in groups.items():
        m = layer.monitors[p + '_spikes']
        ts, ix = np.array(m.t / ms), np.array(m.i)
        spikes[p] = (ts, ix)
        rate.append(np.histogram(ts, np.arange(T + 1))[0] * 1000.0 / len(g))
    return np.array(rate).T.astype(np.float32), spikes


b2.seed(a.seed + 19000)
col.network.run(T * ms)
r0, spikes0 = extract()
stats = {p: {v: [float(np.mean(np.asarray(getattr(diag[p], v))[:, SETTLE // 5:])),
                 float(np.std(np.asarray(getattr(diag[p], v))[:, SETTLE // 5:].mean(0)))] for v in STATE_VARS}
         for p in pops}
print('closed', dict(zip(pops, np.round(r0[SETTLE:].mean(0), 3))), f'{time.time() - start:.0f}s', flush=True)


def same(sa, sb):
    return bool(np.array_equal(np.rint(sa[0] / dt_ms), np.rint(sb[0] / dt_ms)) and np.array_equal(sa[1], sb[1]))


branches = [('exact', None, 0, 1)] + [(f'{x}{s}', x, sg, 1) for x in TYPES for s, sg in [('+', 1), ('-', -1)]]
if not a.no_frozen:
    branches += [('frozen_exact', None, 0, 0)] + [(f'frozen_{x}{s}', x, sg, 0) for x in TYPES for s, sg in [('+', 1), ('-', -1)]]
rates, exact = {'closed': r0}, {}
for name, typ, sign, adapt in branches:
    col.network.restore('init', restore_random_state=False)
    b2.seed(a.seed + 19000)
    for syn in layer.synapses.values():
        syn.active = False
    for rep in replicas:
        rep.active = True
    for s in adapt_syn:
        s.active = (adapt == 0)
    for p, g in sources.items():
        ts, ix = spikes0[p]
        ts = np.rint(ts / dt_ms) * dt_ms
        order = np.lexsort((ix, ts))
        g.active = True
        g.set_spikes(ix[order], ts[order] * ms, sorted=True)
    for g in groups.values():
        g.sA = 0; g.sN = 0; g.sI = 0; g.adapt_on = adapt
        if typ:
            setattr(g, 's' + typ, sign)
    col.network.run(T * ms)
    r, sp = extract()
    rates[name] = r
    if sign == 0:
        exact[name] = {p: same(sp[p], spikes0[p]) for p in pops}
    print(name, dict(zip(pops, np.round(r[SETTLE:].mean(0), 3))), exact.get(name, ''), f'{time.time() - start:.0f}s', flush=True)

meta = dict(vars(a), pops=pops, freqs=FREQS.tolist(), settle_ms=SETTLE, period_ms=PERIOD, n_period=n_period,
            types=TYPES, exact=exact, state_stats=stats, wiring=wiring, conductance=cc,
            neuron_counts={p: len(g) for p, g in groups.items()},
            poisson_N={p: int(v['N']) for p, v in next(iter(cfg['layers'].values()))['poisson_inputs'].items()},
            tonic_inh=cfg['minimal'].get('tonic_inh', {}), elapsed_s=time.time() - start)
np.savez_compressed(a.out, probe_nS=probe_nS[:T].astype(np.float32), metadata=json.dumps(meta),
                    **{'rate_' + k: v for k, v in rates.items()})
print('SAVED', a.out, flush=True)
