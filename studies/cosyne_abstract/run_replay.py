"""Cosyne abstract Fig. 3A (was 'Fig 2c' while drafting): drive an isolated layer with another layer's recorded spikes, or with
rate-matched Poisson trains (same per-cell mean rate, no timing).

    ~/Desktop/venv/bin/python studies/cosyne_abstract/run_replay.py <target_cond> <source_layer> \
        {replay|poisson} <out.npz> [--seed 58925] [--segment 0]
    ~/Desktop/venv/bin/python studies/cosyne_abstract/run_replay.py <target_cond> <source_layer> match

Target: a minimal_config condition (e.g. 'L4C@own', 'L23', 'L6'); its layer is named by the
condition's base layer for the inter-layer wiring. Source spikes: the resting intact column
(saved_trials/exp6_alpha_gamma_pv, hand-tuned, network seed 58881, 15 s trials); segment k
is trial k, 1000-11500 ms, shifted to start at 0. Wiring: every (source, target) pathway of
config/config.py with its p, g (AMPA+NMDA / GABA) and delay rule, as in src/column.py.

REPLAY_SOURCES=rest uses saved_trials/cosyne/rest_sources/rest_s<101+k>.npz instead (20 independent
11.5 s recordings, studies/cosyne_abstract/record_rest.py), with its own tuned-N key. The abstract
uses REPLAY_SOURCES=rest (run_layer_replay.sh sets it).

Copy of minimal_config/experiments/run_replay.py (now archive/minimal_config/experiments/); only the paths changed.

Rate matching ('match'): tune the target's background Poisson N with the POISSON source
(segment 0) to the target's intact rates, saved in tuned_N.json as 'replay:<src>-><cond>'.
Both replay and poisson then use that N, so the mean input is identical and only the
source's spike timing differs.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MINI = os.path.join(ROOT, 'minimal_config')
sys.path.insert(0, ROOT)

SRC_FILES = sorted(glob.glob(os.path.join(ROOT, 'saved_trials', 'exp6_alpha_gamma_pv', '*.npz')))
# 'rest' set: studies/cosyne_abstract/record_rest.py, segment k = seed REST_SEEDS[k] (11.5 s each)
REST_SEEDS = list(range(101, 121))
SOURCE_SET = os.environ.get('REPLAY_SOURCES', 'exp6')
SEG_T0, T_MS, SKIP_MS = 1000.0, 10500.0, 500.0
POPS = ['E', 'PV', 'SOM', 'VIP']


def key(target, source):
    return f'replay:{source}->{target}' + ('' if SOURCE_SET == 'exp6' else f':{SOURCE_SET}')


def source_spikes(source, segment):
    if SOURCE_SET == 'rest':
        z = np.load(os.path.join(ROOT, 'saved_trials', 'cosyne', 'rest_sources', f'rest_s{REST_SEEDS[segment]}.npz'))
        out = {}
        for p in POPS:
            t, i = z[f'{source}_{p}_t'].astype(float), z[f'{source}_{p}_i'].astype(int)
            m = (t >= SEG_T0) & (t < SEG_T0 + T_MS)
            out[p] = (t[m] - SEG_T0, i[m])
        return out
    z = np.load(SRC_FILES[segment], allow_pickle=True)
    sd = z['spike_data'].item()[source]
    out = {}
    for p in POPS:
        t = np.asarray(sd[f'{p}_spikes']['times_ms'], float)
        i = np.asarray(sd[f'{p}_spikes']['spike_indices'], int)
        m = (t >= SEG_T0) & (t < SEG_T0 + T_MS)
        out[p] = (t[m] - SEG_T0, i[m])
    return out


def simulate(target, source, mode, seed, segment, poisson_N=None, T=T_MS, spikes=True):
    import brian2 as b2
    import cleo
    from brian2 import ms, Hz, nS, StateMonitor, SpikeGeneratorGroup, PoissonGroup, Synapses
    from config.config import CONFIG as FULL
    from minimal_config.config import make_config
    from minimal_config.experiments.conditions import CONDITIONS, TUNED_N
    from src.column import CorticalColumn
    from src.analysis import add_heterogeneity_to_layer

    tgt_layer = CONDITIONS[target]['base'].replace('@own', '')
    if poisson_N is None:
        poisson_N = json.load(open(TUNED_N))[key(target, source)]['poisson_N']
    cfg = make_config(target, poisson_N)
    np.random.seed(seed); b2.seed(seed); cleo.utilities.set_seed(seed)
    b2.start_scope()
    b2.defaultclock.dt = cfg['simulation']['DT']
    col = CorticalColumn(column_id=0, config=cfg)
    for layer in col.layers.values():
        add_heterogeneity_to_layer(layer, cfg)
    for obj in list(col.network.objects):
        if isinstance(obj, StateMonitor):
            col.network.remove(obj)
    (layer,) = col.layers.values()

    n_src = FULL['layers'][source]['neuron_counts']
    spk = source_spikes(source, segment)
    groups = {}
    for p in POPS:
        t, i = spk[p]
        if mode == 'replay':
            groups[p] = SpikeGeneratorGroup(n_src[p], i, t * ms)
        else:
            src_rates = np.bincount(i, minlength=n_src[p]) / (T_MS / 1000.0)
            groups[p] = PoissonGroup(n_src[p], rates=src_rates * Hz)
    col.network.add(*groups.values())

    conns = FULL['inter_layer_connections'].get((source, tgt_layer), {})
    conds = FULL['inter_layer_conductances'].get((source, tgt_layer), {})
    for conn, prob in conns.items():
        if prob <= 0:
            continue
        pre, post = conn.split('_')
        d_mean, d_std = col._get_inter_layer_delay_params(pre, post, source, tgt_layer)
        d_mean, d_std = d_mean / ms, d_std / ms
        if pre == 'E':
            on_pre = f"gE_AMPA_post += {conds.get(conn + '_AMPA', 0.01)}*nS"
            if conds.get(conn + '_NMDA', 0.0) > 0:
                on_pre += f"\ngE_NMDA_post += {conds[conn + '_NMDA']}*nS"
        else:
            on_pre = f"g{pre}_post += {conds.get(conn, 0.02)}*nS"
        syn = Synapses(groups[pre], layer.neuron_groups[post], on_pre=on_pre)
        syn.connect(p=float(prob))
        syn.delay = f'{d_mean}*ms + clip(randn()*{d_std}, -{d_std}*0.5, {d_std}*2)*ms'
        col.network.add(syn)

    col.network.run(T * ms)
    from scipy.signal import welch
    out = {'T': T, 'seed': seed, 'segment': segment, 'mode': mode, 'source': source, 'target': target,
           'source_set': SOURCE_SET}
    for p, g in layer.neuron_groups.items():
        t = layer.monitors[f'{p}_spikes'].t / ms
        tt = t[t > SKIP_MS]
        out[f'rate_{p}'] = len(tt) / len(g) / ((T - SKIP_MS) / 1000)
        x, _ = np.histogram(tt, bins=np.arange(SKIP_MS, T + 1, 1.0))
        f_r, out[f'psd_{p}'] = welch(x - x.mean(), fs=1000, nperseg=2000, noverlap=1500)
        if spikes:
            out[f'spk_t_{p}'] = np.asarray(t, np.float32)
            out[f'spk_i_{p}'] = np.asarray(layer.monitors[f'{p}_spikes'].i, np.int32)
            out[f'n_{p}'] = len(g)
    out['f_rate'] = f_r
    # source drive actually delivered (for the record): E-rate PSD of the source E population
    t, _ = spk['E']
    x, _ = np.histogram(t[t > SKIP_MS], bins=np.arange(SKIP_MS, T + 1, 1.0))
    _, out['psd_srcE'] = welch(x - x.mean(), fs=1000, nperseg=2000, noverlap=1500)
    return out


def match(target, source, tol=0.08, max_iter=20):
    from minimal_config.match_rates import next_N
    from minimal_config.experiments.conditions import build, save_tuned
    _, spec = build(target)
    goal = spec['target_rates_hz']
    N = {p: int(n) for p, n in spec['poisson_N'].items()}
    hist = {p: [] for p in N}
    tried, results = [], []
    for it in range(max_iter):
        r = subprocess.run([sys.executable, os.path.abspath(__file__), '--worker', target, source,
                            json.dumps(N)], capture_output=True, text=True, cwd=ROOT)
        line = [l for l in r.stdout.splitlines() if l.startswith('RATES ')]
        if not line:
            raise RuntimeError(r.stderr[-3000:])
        rates = json.loads(line[-1][6:])
        err = {p: rates[p] / goal[p] - 1 for p in N}
        tried.append(dict(N)); results.append((max(abs(e) for e in err.values()), dict(N), rates))
        print(f'it{it:2d} N={N}  ' + '  '.join(f'{p} {rates[p]:.2f} ({err[p]:+.0%})' for p in N), flush=True)
        if all(abs(e) < tol for e in err.values()):
            break
        N = next_N(N, rates, goal, hist, err, tol, tried)
        if N is None:
            break
    best = min(results, key=lambda x: x[0])
    if best[0] < tol:
        save_tuned(key(target, source), best[1], best[2])
        print(f'MATCHED (max error {best[0]:.1%})')
    else:
        print(f'NOT MATCHED: best {best[0]:.1%} at N={best[1]} rates={best[2]}')
        sys.exit(1)


if __name__ == '__main__':
    if sys.argv[1] == '--worker':
        o = simulate(sys.argv[2], sys.argv[3], 'poisson', 58925, 0, json.loads(sys.argv[4]),
                     T=3000.0, spikes=False)
        print('RATES ' + json.dumps({p: float(o[f'rate_{p}']) for p in POPS}))
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument('target'); ap.add_argument('source')
    ap.add_argument('mode', choices=['replay', 'poisson', 'match'])
    ap.add_argument('out', nargs='?')
    ap.add_argument('--seed', type=int, default=58925)
    ap.add_argument('--segment', type=int, default=0)
    a = ap.parse_args()
    if a.mode == 'match':
        match(a.target, a.source)
    else:
        o = simulate(a.target, a.source, a.mode, a.seed, a.segment)
        np.savez(a.out, **o)
        print('saved', a.out, {p: round(float(o[f'rate_{p}']), 2) for p in POPS})
