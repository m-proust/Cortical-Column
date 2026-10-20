"""Rate matching: tune the Poisson N of each population until it fires at its target.

    ~/Desktop/venv/bin/python minimal_config/match_rates.py <condition> [--tol 0.08] [--max-iter 20]

<condition> is a name from experiments/conditions.py (e.g. L4C+L6_VIP-SOM).
Targets are that condition's target rates. Each iteration runs a fresh 3 s simulation (seed
58925, rates over 500-3000 ms) and updates all four N together, because the
populations drive each other (raising PV lowers E, E drives SOM, ...).
When every population is within --tol, the N and achieved rates are written
to experiments/tuned_N.json. If it does not converge, the best setting found is
reported and nothing is written.

Rate noise between single runs is ~5%, so tolerances below ~6% will not converge.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
T_MS, SKIP_MS, SEED = 3000.0, 500.0, 58925


def simulate(exp, poisson_N):
    """Worker, run in a fresh process: population rates for the given Poisson N."""
    import brian2 as b2
    from brian2 import ms, StateMonitor
    sys.path.insert(0, ROOT)
    from minimal_config.config import make_config, add_tonic_inhibition
    from src.column import CorticalColumn
    from src.analysis import add_heterogeneity_to_layer

    cfg = make_config(exp, poisson_N)
    np.random.seed(SEED); b2.seed(SEED); b2.start_scope()
    b2.defaultclock.dt = cfg['simulation']['DT']
    col = CorticalColumn(column_id=0, config=cfg)
    for layer in col.layers.values():
        add_heterogeneity_to_layer(layer, cfg)
    for obj in list(col.network.objects):
        if isinstance(obj, StateMonitor):
            col.network.remove(obj)
    add_tonic_inhibition(col, cfg)
    col.network.run(T_MS * ms)
    (layer,) = col.layers.values()
    return {p: float(np.sum(layer.monitors[f'{p}_spikes'].t / ms > SKIP_MS) / len(g)
                     / ((T_MS - SKIP_MS) / 1000))
            for p, g in layer.neuron_groups.items()}


def run(exp, poisson_N):
    out = subprocess.run([sys.executable, os.path.abspath(__file__), '--worker', exp,
                          json.dumps(poisson_N)], capture_output=True, text=True, cwd=ROOT)
    lines = [l for l in out.stdout.splitlines() if l.startswith('RATES ')]
    if not lines:
        raise RuntimeError(f'simulation failed:\n{out.stderr[-3000:]}')
    return json.loads(lines[-1][6:])


def next_N(N, rates, target, hist, err, tol, tried):
    """Damped secant step on log(rate) vs N, per population, rounded to integers."""
    new = {}
    for p in N:
        hist[p].append((N[p], np.log(max(rates[p], 0.02))))
        (n1, y1) = hist[p][-1]
        slope = None
        if len(hist[p]) >= 2 and hist[p][-2][0] != n1:
            slope = (y1 - hist[p][-2][1]) / (n1 - hist[p][-2][0])
        if slope is None or slope <= 0.005:
            slope = 2.5 / N[p]                      # default: rate ~ N^2.5 near threshold
        step = float(np.clip((np.log(target[p]) - y1) / slope, -0.35 * N[p], 0.5 * N[p]))
        if abs(err[p]) >= tol and abs(step) < 1:
            step = np.sign(step)
        new[p] = max(1, int(round(N[p] + 0.8 * step)))
    if new == N or new in tried:                   # stuck on the integer grid: nudge the worst one
        worst = max(err, key=lambda p: abs(err[p]))
        new = dict(N); new[worst] = max(1, new[worst] + (-1 if err[worst] > 0 else 1))
        if new == N or new in tried:
            return None
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('condition')
    ap.add_argument('--tol', type=float, default=0.08)
    ap.add_argument('--max-iter', type=int, default=20)
    a = ap.parse_args()
    sys.path.insert(0, ROOT)
    from minimal_config.config import load
    from minimal_config.experiments.conditions import save_tuned
    exp = a.condition
    _, spec = load(exp)
    target = spec['target_rates_hz']
    fixed = {p: int(spec['poisson_N'][p]) for p in spec.get('fixed_pops', [])}
    N = {p: int(n) for p, n in spec['poisson_N'].items() if p not in fixed}
    hist = {p: [] for p in N}
    tried, results = [], []

    print(f'matching {exp}\ntargets {target}')
    for it in range(a.max_iter):
        rates = run(exp, dict(N, **fixed))
        err = {p: rates[p] / target[p] - 1 for p in N}
        tried.append(dict(N))
        results.append((max(abs(e) for e in err.values()), dict(N), rates))
        print(f'it{it:2d} N={N} fixed={fixed}  ' + '  '.join(f'{p} {rates[p]:.2f} ({rates[p] / target[p] - 1:+.0%})' for p in rates),
              flush=True)
        if all(abs(e) < a.tol for e in err.values()):
            break
        N = next_N(N, rates, target, hist, err, a.tol, tried)
        if N is None:
            break

    best_err, best_N, best_rates = min(results, key=lambda r: r[0])
    if best_err < a.tol:
        save_tuned(exp, dict(best_N, **fixed), best_rates)
        print(f'MATCHED (max error {best_err:.1%}); saved N for {exp} in experiments/tuned_N.json')
    else:
        print(f'NOT MATCHED: best max error {best_err:.1%} at N={best_N}; nothing saved.\n'
              f'Raise --max-iter, or put that N in tuned_N.json and rerun from there.')
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        print('RATES ' + json.dumps(simulate(sys.argv[2], json.loads(sys.argv[3]))))
    else:
        main()
