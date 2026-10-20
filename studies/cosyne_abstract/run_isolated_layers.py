"""Cosyne abstract, isolated-layer runs (Fig. 2A, 2C, 2E network spectra; Fig. 3A "no input").

    ~/Desktop/venv/bin/python studies/cosyne_abstract/run_isolated_layers.py [--set layers|sweeps] [--jobs 6]
        [--only fx_L23+ES ...]

Per condition (run in parallel over conditions): match_rates.py if the condition has no
tuned N yet, then minimal_config/run_spectrum.py --spikes for each seed into
saved_trials/cosyne/isolated_layers/<cond>_s<seed>.npz.
Existing outputs are skipped, so the command can be re-run to resume or to add seeds.
Unmatched conditions are not simulated; they are listed in isolated_layers/unmatched.txt.

Copy of minimal_config/experiments/run_fig1.py with the output moved to saved_trials/cosyne and the
condition/seed sets of the abstract built in (--set):
  layers  L23 L4AB L4C@own L5 L6 fx_L23+PE+SP, seeds 58925 1..19 (20 networks)
  sweeps  the SOM decay / loop delay / SOM->SOM held-out conditions, seeds 1..10 (SOM->SOM: 58925 1..4)
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MINI = os.path.join(ROOT, 'minimal_config')
sys.path.insert(0, ROOT)
from minimal_config.experiments.conditions import CONDITIONS, TUNED_N

OUT = os.path.join(ROOT, 'saved_trials', 'cosyne', 'isolated_layers')
RUNS = OUT
LOGS = os.path.join(OUT, 'logs')
PY = sys.executable


def tuned(name):
    return name in json.load(open(TUNED_N))


def do_condition(name, seeds):
    log = open(os.path.join(LOGS, f'{name}.log'), 'a')
    if not tuned(name):
        r = subprocess.run([PY, os.path.join(MINI, 'match_rates.py'), name, '--max-iter', '20'],
                           stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        if r.returncode != 0 or not tuned(name):
            with open(os.path.join(OUT, 'unmatched.txt'), 'a') as fh:
                fh.write(name + '\n')
            return name, 'UNMATCHED'
    for s in seeds:
        out = os.path.join(RUNS, f'{name}_s{s}.npz')
        if os.path.exists(out):
            continue
        subprocess.run([PY, os.path.join(MINI, 'run_spectrum.py'), name, out, '--seed', str(s), '--spikes'],
                       stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    return name, 'done'


LAYERS = ['L23', 'L4AB', 'L4C@own', 'L5', 'L6', 'fx_L23+PE+SP']
SWEEPS = ([f'L6_tauSOME{t}' for t in (10, 15, 30, 40)] + [f'L6_tauSOM{t}' for t in (10, 15, 30, 40)]
          + [f'L6_loopdelay+{d}' for d in ('2.5', '5', '10')] + [f'L6_loopdelayx{s}' for s in ('0.5', '1.5', '2')])
SOM_SOM = [f'L6_SS{g}' for g in ('0.2', '0.44', '1', '1.95')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=['layers', 'sweeps'], default='layers')
    ap.add_argument('--seeds', type=int, nargs='+', help='override the set\'s seeds')
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--only', nargs='+', help='override the set\'s conditions')
    a = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True); os.makedirs(LOGS, exist_ok=True)
    if a.set == 'layers':
        jobs = [(n, a.seeds or [58925] + list(range(1, 20))) for n in (a.only or LAYERS)]
    else:
        jobs = [(n, a.seeds or list(range(1, 11))) for n in (a.only or SWEEPS)]
        if not a.only:
            jobs += [(n, a.seeds or [58925, 1, 2, 3, 4]) for n in SOM_SOM]
    with ThreadPoolExecutor(a.jobs) as ex:
        for name, status in ex.map(lambda j: do_condition(*j), jobs):
            print(f'{name:28s} {status}', flush=True)


if __name__ == '__main__':
    main()
