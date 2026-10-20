"""Cosyne abstract Fig. 3A, step 1: record resting spike trains of every population of the intact
hand-tuned column (config/config.py), to be replayed into isolated layers by run_replay.py.
One 11.5 s recording per seed; the network and the noise both follow the seed. The abstract used
seeds 101..120.

    ~/Desktop/venv/bin/python studies/cosyne_abstract/record_rest.py            # all 20, skips existing
    ~/Desktop/venv/bin/python studies/cosyne_abstract/record_rest.py 101 102
Saves saved_trials/cosyne/rest_sources/rest_s<seed>.npz with <layer>_<pop>_t (ms), _i, _n.
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _ROOT)
import numpy as np
import brian2 as b2
from brian2 import ms, StateMonitor
import cleo

from config.config import CONFIG
from src.column import CorticalColumn
from src.analysis import add_heterogeneity_to_layer

T_MS = 11500.0
OUT = os.path.join(_ROOT, 'saved_trials', 'cosyne', 'rest_sources')


def main(seed):
    os.makedirs(OUT, exist_ok=True)
    np.random.seed(seed); b2.seed(seed); cleo.utilities.set_seed(seed)
    b2.start_scope()
    b2.defaultclock.dt = CONFIG['simulation']['DT']
    col = CorticalColumn(column_id=0, config=CONFIG)
    for layer in col.layers.values():
        add_heterogeneity_to_layer(layer, CONFIG)
    for obj in list(col.network.objects):
        if isinstance(obj, StateMonitor):
            col.network.remove(obj)
    col.network.run(T_MS * ms)
    out = {'seed': seed, 'T': T_MS}
    for L, lay in col.layers.items():
        for p, g in lay.neuron_groups.items():
            mon = lay.monitors[f'{p}_spikes']
            out[f'{L}_{p}_t'] = np.asarray(mon.t / ms, np.float32)
            out[f'{L}_{p}_i'] = np.asarray(mon.i, np.int32)
            out[f'{L}_{p}_n'] = len(g)
    fn = os.path.join(OUT, f'rest_s{seed}.npz')
    np.savez(fn, **out)
    print('saved', fn)


if __name__ == '__main__':
    for seed in [int(x) for x in sys.argv[1:]] or range(101, 121):
        if not os.path.exists(os.path.join(OUT, f'rest_s{seed}.npz')):
            main(seed)
