"""
Compute trial-averaged Pairwise Phase Consistency (PPC) from saved trial .npz files.

Usage:
    python analyze_ppc_trials.py results/13_03_LGN_gratings_2

Loads all trial_*.npz in the given directory, computes per-neuron PPC at each
frequency for each cell type and layer, averages across trials, and plots
mean +/- SEM for baseline vs stimulus.
"""

import os
import sys
import glob
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


# ── Layer config (must match config2.py) ──────────────────────────────────────
LAYER_Z_RANGES = {
    'L23':  (0.45, 1.10),
    'L4AB': (0.14, 0.45),
    'L4C':  (-0.14, 0.14),
    'L5':   (-0.34, -0.14),
    'L6':   (-0.62, -0.34),
}
CELL_TYPES = ['E', 'PV', 'SOM', 'VIP']
COLORS = {'E': '#1f77b4', 'PV': '#d62728', 'SOM': '#2ca02c', 'VIP': '#ff7f0e'}


# ── PPC helpers ───────────────────────────────────────────────────────────────
def _get_inst_phase(lfp_signal, fs, freq, bandwidth=4.0):
    f_lo = max(freq - bandwidth / 2, 0.5)
    f_hi = freq + bandwidth / 2
    nyq = fs / 2.0
    if f_hi >= nyq:
        return None
    b, a = butter(3, [f_lo / nyq, f_hi / nyq], btype='band')
    filtered = filtfilt(b, a, lfp_signal)
    analytic = hilbert(filtered)
    return np.angle(analytic)


def compute_ppc_per_neuron_from_arrays(
    spike_times_ms, spike_neuron_ids,
    time_window, lfp_signal, time_array,
    fs, freq, min_spikes_per_neuron=5, bandwidth=4.0
):
    """
    Compute per-neuron weighted-average PPC from raw arrays (not Brian2 monitors).
    """
    inst_phase = _get_inst_phase(lfp_signal, fs, freq, bandwidth)
    if inst_phase is None:
        return np.nan

    t_start, t_end = time_window
    mask = (spike_times_ms >= t_start) & (spike_times_ms < t_end)
    times = spike_times_ms[mask]
    neuron_ids = spike_neuron_ids[mask]

    if len(times) < 2:
        return np.nan

    spike_samples = np.searchsorted(time_array, times)
    valid = (spike_samples >= 0) & (spike_samples < len(inst_phase))
    spike_samples = spike_samples[valid]
    neuron_ids = neuron_ids[valid]
    phases_all = inst_phase[spike_samples]

    unique_neurons = np.unique(neuron_ids)
    weighted_ppc_sum = 0.0
    weight_sum = 0

    for nid in unique_neurons:
        neuron_mask = neuron_ids == nid
        phases = phases_all[neuron_mask]
        n = len(phases)
        if n < min_spikes_per_neuron:
            continue
        resultant = np.sum(np.exp(1j * phases))
        ppc_i = (np.abs(resultant) ** 2 - n) / (n * (n - 1))
        w = n * (n - 1)
        weighted_ppc_sum += ppc_i * w
        weight_sum += w

    if weight_sum == 0:
        return np.nan
    return weighted_ppc_sum / weight_sum


def _find_closest_bipolar_channel(layer_z_range, channel_depths):
    layer_center = (layer_z_range[0] + layer_z_range[1]) / 2.0
    distances = [abs(d - layer_center) for d in channel_depths]
    return int(np.argmin(distances))


# ── Main analysis ─────────────────────────────────────────────────────────────
def compute_ppc_all_trials(
    trial_dir,
    freq_range=(1, 100),
    n_freqs=50,
    fs=10000,
    min_spikes_per_neuron=5,
    bandwidth=4.0,
    transient_skip=500,
):
    """
    Load all trials and compute PPC spectra.

    Returns
    -------
    freqs : 1D array
    ppc_all : dict[layer][cell_type][condition] -> (n_trials, n_freqs) array
    """
    trial_files = sorted(glob.glob(os.path.join(trial_dir, 'trial_*.npz')))
    if not trial_files:
        raise FileNotFoundError(f"No trial_*.npz found in {trial_dir}")

    freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
    n_trials = len(trial_files)
    layer_names = list(LAYER_Z_RANGES.keys())

    # Pre-allocate: ppc_all[layer][cell_type][condition] = (n_trials, n_freqs)
    ppc_all = {}
    for layer in layer_names:
        ppc_all[layer] = {}
        for ct in CELL_TYPES:
            ppc_all[layer][ct] = {
                'baseline': np.full((n_trials, n_freqs), np.nan),
                'stimulus': np.full((n_trials, n_freqs), np.nan),
            }

    for ti, fpath in enumerate(trial_files):
        print(f"  Processing {os.path.basename(fpath)} ({ti+1}/{n_trials})...")
        data = np.load(fpath, allow_pickle=True)

        time_array = data['time_array_ms']
        channel_depths = data['channel_depths']
        bipolar_matrix = data['bipolar_matrix']
        spike_data = data['spike_data'].item()  # dict
        baseline_ms = float(data['baseline_ms'])
        stim_ms = float(data['post_ms'])

        # Time windows (skip transient)
        base_start = transient_skip
        base_end = baseline_ms
        stim_start = baseline_ms + transient_skip
        stim_end = baseline_ms + stim_ms

        base_mask = (time_array >= base_start) & (time_array < base_end)
        stim_mask = (time_array >= stim_start) & (time_array < stim_end)

        time_base = time_array[base_mask]
        time_stim = time_array[stim_mask]

        if len(time_base) < 100 or len(time_stim) < 100:
            continue

        for layer in layer_names:
            z_range = LAYER_Z_RANGES[layer]
            ch_idx = _find_closest_bipolar_channel(z_range, channel_depths)
            lfp_base = bipolar_matrix[ch_idx][base_mask]
            lfp_stim = bipolar_matrix[ch_idx][stim_mask]

            if layer not in spike_data:
                continue

            for ct in CELL_TYPES:
                spike_key = f'{ct}_spikes'
                if spike_key not in spike_data[layer]:
                    continue

                sd = spike_data[layer][spike_key]
                spike_times = sd['times_ms']
                spike_ids = sd['spike_indices']

                for fi, f in enumerate(freqs):
                    ppc_all[layer][ct]['baseline'][ti, fi] = (
                        compute_ppc_per_neuron_from_arrays(
                            spike_times, spike_ids,
                            (base_start, base_end), lfp_base, time_base,
                            fs, f, min_spikes_per_neuron, bandwidth
                        )
                    )
                    ppc_all[layer][ct]['stimulus'][ti, fi] = (
                        compute_ppc_per_neuron_from_arrays(
                            spike_times, spike_ids,
                            (stim_start, stim_end), lfp_stim, time_stim,
                            fs, f, min_spikes_per_neuron, bandwidth
                        )
                    )

    return freqs, ppc_all, n_trials


def plot_ppc_trials(freqs, ppc_all, n_trials, smooth_sigma=1.5):
    """
    Plot trial-averaged PPC with SEM shading.
    """
    layer_names = list(LAYER_Z_RANGES.keys())
    n_layers = len(layer_names)

    fig, axes = plt.subplots(n_layers, 2, figsize=(14, 3 * n_layers),
                             sharex=True, sharey=False)
    if n_layers == 1:
        axes = axes[np.newaxis, :]

    for row, layer in enumerate(layer_names):
        for ct in CELL_TYPES:
            for col, cond in enumerate(['baseline', 'stimulus']):
                mat = ppc_all[layer][ct][cond]  # (n_trials, n_freqs)

                # Mean and SEM across trials (ignoring NaN)
                with np.errstate(all='ignore'):
                    mean_ppc = np.nanmean(mat, axis=0)
                    n_valid = np.sum(~np.isnan(mat), axis=0)
                    sem_ppc = np.nanstd(mat, axis=0) / np.sqrt(np.maximum(n_valid, 1))

                # Smooth
                if smooth_sigma > 0:
                    valid = ~np.isnan(mean_ppc)
                    if np.sum(valid) > 3:
                        mean_ppc[valid] = gaussian_filter1d(mean_ppc[valid], smooth_sigma)
                        sem_ppc[valid] = gaussian_filter1d(sem_ppc[valid], smooth_sigma)

                axes[row, col].plot(freqs, mean_ppc, color=COLORS[ct],
                                    label=ct, linewidth=1.5, alpha=0.85)
                axes[row, col].fill_between(
                    freqs, mean_ppc - sem_ppc, mean_ppc + sem_ppc,
                    color=COLORS[ct], alpha=0.15
                )

        axes[row, 0].set_title(f'{layer} — Baseline', fontsize=10)
        axes[row, 1].set_title(f'{layer} — Stimulus', fontsize=10)

        for col in range(2):
            axes[row, col].set_ylabel('PPC', fontsize=9)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend(fontsize=8, loc='upper right')
            axes[row, col].set_ylim(bottom=-0.01)

    axes[-1, 0].set_xlabel('Frequency (Hz)', fontsize=11)
    axes[-1, 1].set_xlabel('Frequency (Hz)', fontsize=11)
    fig.suptitle(f'Trial-averaged PPC (n={n_trials} trials, Vinck et al. 2010)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_ppc_trials.py <trial_dir>")
        print("Example: python analyze_ppc_trials.py results/13_03_LGN_gratings_2")
        sys.exit(1)

    trial_dir = sys.argv[1]
    print(f"Loading trials from {trial_dir}...")

    freqs, ppc_all, n_trials = compute_ppc_all_trials(
        trial_dir,
        freq_range=(1, 100),
        n_freqs=50,
        fs=10000,
        min_spikes_per_neuron=5,
        bandwidth=4.0,
    )

    print(f"Plotting PPC averaged over {n_trials} trials...")
    fig = plot_ppc_trials(freqs, ppc_all, n_trials)
    plt.show()
