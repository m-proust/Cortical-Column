"""Cosyne reviewer checks (adaptation, sensitivity map): rate-match each condition, then simulate it per seed.

    ~/Desktop/venv/bin/python studies/cosyne_abstract/run_reviewer.py [--seeds 58925 1 2 3 4] [--jobs 6]
        [--only fx_L23+ES ...]

Per condition (run in parallel over conditions): match_rates.py if the condition has no
tuned N yet, then run_spectrum.py --spikes for each seed into reviewer/_runs/<cond>_s<seed>.npz.
Existing outputs are skipped, so the command can be re-run to resume or to add seeds.
Unmatched conditions are not simulated; they are listed in reviewer/unmatched.txt.
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

OUT = os.path.join(MINI, 'experiments', 'reviewer')
RUNS = os.path.join(OUT, '_runs')
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[58925, 1, 2, 3, 4])
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--only', nargs='+')
    a = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True); os.makedirs(LOGS, exist_ok=True)
    names = [n for x in a.only for n in x.split()] if a.only else ([c for c in CONDITIONS if c.startswith('L6_') and 'adapt' in c] + [c for c in CONDITIONS if c.startswith('map_')])
    with ThreadPoolExecutor(a.jobs) as ex:
        for name, status in ex.map(lambda n: do_condition(n, a.seeds), names):
            print(f'{name:28s} {status}', flush=True)


if __name__ == '__main__':
    main()
