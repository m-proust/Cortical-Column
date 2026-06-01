"""Raster + population rate plots for trial_XXX.npz files, styled identically
to main.py (uses plot_raster and plot_rate from src.visualization).

Wraps the saved numpy spike/rate data in tiny shim objects so the existing
Brian2-monitor-based plotting code works unchanged.
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from brian2 import second, ms, Hz

from config.config import CONFIG
from src.visualization import plot_raster, plot_rate


class _SpikeMonShim:
    """Mimics brian2 SpikeMonitor: exposes .t (Quantity) and .i (array)."""
    def __init__(self, times_ms, indices):
        self.t = np.asarray(times_ms) * ms
        self.i = np.asarray(indices)


class _RateMonShim:
    """Mimics brian2 PopulationRateMonitor: .t (Quantity), .smooth_rate(...)."""
    def __init__(self, t_ms, rate_hz):
        self._t_ms = np.asarray(t_ms)
        self._rate_hz = np.asarray(rate_hz)
        self.t = self._t_ms * ms

    def smooth_rate(self, window='flat', width=10 * ms):
        width_ms = float(width / ms)
        if len(self._t_ms) < 2 or width_ms <= 0:
            return self._rate_hz * Hz
        dt_ms = float(np.mean(np.diff(self._t_ms)))
        n = max(1, int(round(width_ms / dt_ms)))
        if window == 'flat':
            kernel = np.ones(n) / n
        elif window == 'gaussian':
            sigma_pts = max(1.0, n / 2.0)
            half = int(np.ceil(3 * sigma_pts))
            ks = np.arange(-half, half + 1)
            kernel = np.exp(-0.5 * (ks / sigma_pts) ** 2)
            kernel /= kernel.sum()
        else:
            kernel = np.ones(n) / n
        smoothed = np.convolve(self._rate_hz, kernel, mode='same')
        return smoothed * Hz


def _build_monitor_dicts(data):
    spike_data = data['spike_data'].item()
    rate_data = data['rate_data'].item()

    spike_monitors = {}
    rate_monitors = {}
    for layer_name, pops in spike_data.items():
        spike_monitors[layer_name] = {}
        for pop_key, payload in pops.items():
            payload = payload.item() if hasattr(payload, 'item') else payload
            spike_monitors[layer_name][pop_key] = _SpikeMonShim(
                payload['times_ms'], payload['spike_indices'])

    for layer_name, pops in rate_data.items():
        rate_monitors[layer_name] = {}
        for pop_key, payload in pops.items():
            payload = payload.item() if hasattr(payload, 'item') else payload
            rate_monitors[layer_name][pop_key] = _RateMonShim(
                payload['t_ms'], payload['rate_hz'])

    return spike_monitors, rate_monitors


def plot_trial(fname):
    data = np.load(fname, allow_pickle=True)
    spike_monitors, rate_monitors = _build_monitor_dicts(data)
    baseline_time = float(data['baseline_ms'])
    stim_time_ms = float(data['post_ms']) if 'post_ms' in data.files \
        else float(data['time_array_ms'][-1]) - baseline_time

    seed = int(data['seed']) if 'seed' in data.files else -1
    trial_id = int(data['trial_id']) if 'trial_id' in data.files else -1
    tag = f"{os.path.basename(fname)}  trial={trial_id}  seed={seed}"

    fig_r = plot_raster(spike_monitors, baseline_time, stim_time_ms,
                        CONFIG['layers'])
    fig_r.suptitle(tag, fontsize=12)
    fig_r.tight_layout()

    fig_h = plot_rate(rate_monitors, CONFIG['layers'],
                      baseline_time, stim_time_ms * ms,
                      smooth_window=15 * ms, ylim_max=80, show_stats=True)
    fig_h.suptitle(tag, fontsize=12)
    fig_h.tight_layout()

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', type=str,
                        default='results/trials_19_05_2')
    parser.add_argument('--trials', type=int, nargs='+', default=None,
                        help='Specific trial indices (default: first 3).')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.base_path, 'trial_*.npz')))
    if not files:
        raise FileNotFoundError(f"no trial_*.npz under {args.base_path}")

    if args.trials is None:
        chosen = files[:3]
    else:
        chosen = []
        for idx in args.trials:
            cand = os.path.join(args.base_path, f"trial_{idx:03d}.npz")
            if os.path.exists(cand):
                chosen.append(cand)
            else:
                print(f"  skipped: {cand} not found")

    for fname in chosen:
        print(f"Plotting {fname}")
        plot_trial(fname)
