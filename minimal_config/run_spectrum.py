"""Simulate one minimal-config experiment at baseline and save its spectra.

    ~/Desktop/venv/bin/python minimal_config/run_spectrum.py <condition> <out.npz> [--T 10500] [--seed 58925] [--spikes]

Saves, for the whole run after a 500 ms transient:
  - population rates (Hz) and firing-rate PSDs of E, PV, SOM, VIP (1 ms bins)
  - bipolar LFP PSDs of every channel (kernel method, as in main.py)
Welch uses 2 s segments with 75% overlap (0.5 Hz resolution), so a 10 s run
averages ~17 segments; main.py's 1.5 s window is too short to compare peaks.
"""
import argparse
import os
import sys

import numpy as np
from scipy.signal import welch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import brian2 as b2
import cleo
from brian2 import ms, StateMonitor

from minimal_config.config import make_config, add_tonic_inhibition
from src.column import CorticalColumn
from src.analysis import add_heterogeneity_to_layer, compute_bipolar_lfp
from tools.lfp_kernel import calculate_lfp_kernel_method

SKIP_MS = 500.0

ap = argparse.ArgumentParser()
ap.add_argument('condition', help='name in experiments/conditions.py')
ap.add_argument('out')
ap.add_argument('--T', type=float, default=10500.0)
ap.add_argument('--seed', type=int, default=58925)
ap.add_argument('--spikes', action='store_true', help='also save every spike (for CV / correlations)')
ap.add_argument('--shared-E', type=int, default=0,
                help="replace this many of each E cell's Poisson sources by sources common to all E cells "
                     '(same mean and variance of the drive per cell; input correlation = shared-E / N)')
a = ap.parse_args()

cfg = make_config(a.condition)
(_lname, _lcfg), = cfg['layers'].items()
if a.shared_E:
    _lcfg['poisson_inputs']['E']['N'] -= a.shared_E
    assert _lcfg['poisson_inputs']['E']['N'] >= 0
np.random.seed(a.seed); b2.seed(a.seed); cleo.utilities.set_seed(a.seed)
b2.start_scope()
b2.defaultclock.dt = cfg['simulation']['DT']
col = CorticalColumn(column_id=0, config=cfg)
for layer in col.layers.values():
    add_heterogeneity_to_layer(layer, cfg)
for obj in list(col.network.objects):
    if isinstance(obj, StateMonitor):
        col.network.remove(obj)
if a.shared_E:                                   # one pool, all-to-all: every E cell gets the same trains
    _pool = b2.PoissonGroup(a.shared_E, _lcfg['input_rate'])
    _syn = b2.Synapses(_pool, col.layers[_lname].neuron_groups['E'], on_pre='gE_AMPA_post += w_shared',
                       namespace={'w_shared': cfg['synapses']['Q']['EXT_AMPA']})
    _syn.connect()
    col.network.add(_pool, _syn)
add_tonic_inhibition(col, cfg)
col.network.run(a.T * ms)

(name, layer), = col.layers.items()
out = {'T': a.T, 'seed': a.seed, 'shared_E': a.shared_E}
for p, g in layer.neuron_groups.items():
    t = layer.monitors[f'{p}_spikes'].t / ms
    t = t[t > SKIP_MS]
    out[f'rate_{p}'] = len(t) / len(g) / ((a.T - SKIP_MS) / 1000)
    x, _ = np.histogram(t, bins=np.arange(SKIP_MS, a.T + 1, 1.0))
    f_r, out[f'psd_{p}'] = welch(x - x.mean(), fs=1000, nperseg=2000, noverlap=1500)
    if a.spikes:
        mon = layer.monitors[f'{p}_spikes']
        out[f'spk_t_{p}'] = np.asarray(mon.t / ms, dtype=np.float32)
        out[f'spk_i_{p}'] = np.asarray(mon.i, dtype=np.int32)
        out[f'n_{p}'] = len(g)
out['f_rate'] = f_r

spikes = {name: {k: v for k, v in layer.monitors.items() if 'spikes' in k}}
lfp, t_lfp = calculate_lfp_kernel_method(spikes, {name: layer.neuron_groups}, cfg['layers'],
                                         cfg['electrode_positions'], sim_duration_ms=a.T)
bip, labels, depths = compute_bipolar_lfp(lfp, cfg['electrode_positions'])
fs = 1000.0 / (t_lfp[1] - t_lfp[0])
start = int(SKIP_MS / (t_lfp[1] - t_lfp[0]))
psds = []
for k in bip:
    f_l, P = welch(bip[k][start:], fs=fs, nperseg=int(2 * fs), noverlap=int(1.5 * fs))
    keep = f_l <= 100
    psds.append(P[keep])
out['f_lfp'] = f_l[keep]
out['psd_lfp'] = np.array(psds)
out['lfp_depths'] = np.array([depths[k] for k in bip])
np.savez(a.out, **out)
print('saved', a.out, {p: round(float(out[f'rate_{p}']), 2) for p in layer.neuron_groups})
