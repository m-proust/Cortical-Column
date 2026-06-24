"""Spike-LFP phase locking (PPC) spectrum per layer for one saved trial.

Run: path/to/venv/bin/python analysis/coupling/ppc_computation.py
Edit the settings block below to change the trial, analysis window or frequency grid.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfiltfilt
from brian2 import second
from config.config import CONFIG

# ---- settings ----
TRIAL_NPZ = "results/trial_12_06_12s_stimuli/trial_000.npz"
T_DISCARD = 2.0          # s skipped at start
T_ANALYSIS = 10.0        # s analysed
MIN_SPIKES = 5           # min spikes/neuron to include in PPC
BW_HALF = 5.0            # bandpass half-width (Hz)
FREQ_RANGE = (5, 120)
FREQ_STEP = 2.0
MAX_E_NEURONS = 1000     # subsample E to keep runtime down
OUT_DIR = "figures/ppc"

COLORS = {'E': '#2E8B57', 'PV': '#C0392B', 'SOM': '#1F4E96', 'VIP': '#D4A017'}


def bandpass(signal, fs, low, high, order=3):
    nyq = fs / 2.0
    f_lo, f_hi = max(low, 1.0), min(high, nyq - 1.0)
    if f_hi <= f_lo:
        return np.zeros_like(signal)
    sos = butter(order, [f_lo / nyq, f_hi / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, signal)


def build_spike_trains(times_ms, spike_indices):
    if len(spike_indices) == 0:
        return {}
    return {idx: times_ms[spike_indices == idx] / 1000.0
            for idx in np.unique(spike_indices)}


def normalize_lfp(lfp, dt):
    lfp = lfp - np.mean(lfp)
    return lfp / np.sqrt(np.sum(lfp**2) * dt)


def get_spike_phases_per_neuron(spike_trains, t_lfp, inst_phase, t_discard, t_end):
    neuron_phases = {}
    for nid, st in spike_trains.items():
        st_window = st[(st >= t_discard) & (st <= t_end)]
        if len(st_window) < 2:
            continue
        idx = np.clip(np.searchsorted(t_lfp, st_window), 0, len(inst_phase) - 1)
        neuron_phases[nid] = inst_phase[idx]
    return neuron_phases


def compute_ppc(phases):
    """Pairwise phase consistency: bias-free coupling estimate in [-1, 1]."""
    N = len(phases)
    if N < 2:
        return np.nan
    R_squared = (np.sum(np.cos(phases))**2 + np.sum(np.sin(phases))**2) / N**2
    return (N * R_squared - 1.0) / (N - 1.0)


def compute_ppc_spectrum(spike_trains, lfp_normalized, t_lfp, fs, t_discard, t_end,
                         freq_range, freq_step, bw, min_spikes, max_neurons=None):
    neuron_ids = list(spike_trains.keys())
    if max_neurons is not None and len(neuron_ids) > max_neurons:
        neuron_ids = list(np.random.choice(neuron_ids, max_neurons, replace=False))
        spike_trains = {nid: spike_trains[nid] for nid in neuron_ids}

    freqs = np.arange(freq_range[0], freq_range[1] + freq_step, freq_step)
    ppc_mean = np.full(len(freqs), np.nan)
    ppc_sem = np.full(len(freqs), np.nan)

    for fi, fc in enumerate(freqs):
        lfp_bp = bandpass(lfp_normalized, fs, fc - bw, fc + bw)
        if np.all(lfp_bp == 0):
            continue
        phase_signal = np.angle(hilbert(lfp_bp))
        neuron_phases = get_spike_phases_per_neuron(
            spike_trains, t_lfp, phase_signal, t_discard, t_end)

        ppc_vals = [compute_ppc(ph) for ph in neuron_phases.values()
                    if len(ph) >= min_spikes]
        ppc_vals = [v for v in ppc_vals if not np.isnan(v)]
        if ppc_vals:
            ppc_mean[fi] = np.mean(ppc_vals)
            ppc_sem[fi] = np.std(ppc_vals) / np.sqrt(len(ppc_vals))

    return freqs, ppc_mean, ppc_sem


def cell_type_from_key(key):
    upper = key.upper()
    for ct in ['VIP', 'SOM', 'PV']:
        if ct in upper:
            return ct
    head = key.split('_')[0]
    return 'E' if 'E' in head.upper() and len(head) <= 2 else head


# ---- load trial ----
data = np.load(TRIAL_NPZ, allow_pickle=True)
spike_data = (data["spike_data"].item() if data["spike_data"].size == 1
              else data["spike_data"])
lfp_full_data = (data["lfp_full"].item() if data["lfp_full"].size == 1
                 else data["lfp_full"])
dt_sec = float(CONFIG['simulation']['DT'] / second)
fs = 1.0 / dt_sec
print(f"dt = {dt_sec*1000} ms, fs = {fs:.0f} Hz")

all_spike_trains = {
    layer: {mon: build_spike_trains(sd["times_ms"], sd["spike_indices"])
            for mon, sd in layer_mons.items()}
    for layer, layer_mons in spike_data.items()
}

# ---- compute PPC spectra ----
t_end = T_DISCARD + T_ANALYSIS
results = {}
for layer_name in all_spike_trains:
    results[layer_name] = {}
    lfp_full = lfp_full_data[layer_name]
    t_full = np.arange(len(lfp_full)) * dt_sec
    mask = (t_full >= T_DISCARD) & (t_full <= t_end)
    lfp_normalized = normalize_lfp(lfp_full[mask], dt_sec)
    t_lfp = t_full[mask]

    for key, trains in all_spike_trains[layer_name].items():
        ct = cell_type_from_key(key)
        max_n = MAX_E_NEURONS if ct == 'E' else None
        results[layer_name][ct] = compute_ppc_spectrum(
            trains, lfp_normalized, t_lfp, fs, T_DISCARD, t_end,
            FREQ_RANGE, FREQ_STEP, BW_HALF, MIN_SPIKES, max_n)
    print(f"  done {layer_name}")

# ---- plot ----
os.makedirs(OUT_DIR, exist_ok=True)
layers = list(results.keys())
fig, axes = plt.subplots(len(layers), 1, figsize=(10, 5 * len(layers)), squeeze=False)
for i, layer in enumerate(layers):
    ax = axes[i, 0]
    for ct in ['E', 'PV', 'SOM', 'VIP']:
        if ct not in results[layer]:
            continue
        freqs, ppc_mean, ppc_sem = results[layer][ct]
        valid = ~np.isnan(ppc_mean)
        if not np.any(valid):
            continue
        ax.plot(freqs[valid], ppc_mean[valid], color=COLORS[ct], linewidth=2, label=ct)
        ax.fill_between(freqs[valid], ppc_mean[valid] - ppc_sem[valid],
                        ppc_mean[valid] + ppc_sem[valid], color=COLORS[ct], alpha=0.15)
    ax.set_ylabel('Spike-LFP phase locking (PPC)', fontsize=12)
    ax.set_xlim(FREQ_RANGE)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11)
    ax.set_title(layer, fontsize=13, fontstyle='italic')
    if i == len(layers) - 1:
        ax.set_xlabel('Frequency (Hz)', fontsize=12)

plt.tight_layout()
out = os.path.join(OUT_DIR, "ppc_spectrum.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"saved {out}")

# ---- gamma-band PPC summary ----
for layer_name in results:
    print(f"\n{layer_name}:")
    for ct in ['PV', 'E', 'SOM', 'VIP']:
        if ct not in results[layer_name]:
            continue
        freqs, ppc_mean, _ = results[layer_name][ct]
        vals = ppc_mean[(freqs >= 30) & (freqs <= 50)]
        if np.any(~np.isnan(vals)):
            print(f"  {ct}: mean={np.nanmean(vals):.6f}, peak={np.nanmax(vals):.6f}")
