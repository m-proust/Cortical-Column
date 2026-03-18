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


def compute_ppc_pooled_from_arrays(
    spike_times_ms,
    time_window, lfp_signal, time_array,
    fs, freq, min_spikes=10, bandwidth=4.0
):
    """
    Pooled-spike PPC: pool all spikes from the population (ignoring neuron
    identity) and compute PPC from all pairwise phase differences.

    Uses the Vinck et al. (2010) formula on the full spike set:
        PPC = (|R|^2 - N) / (N*(N-1))
    where R = sum(exp(j*theta)) across ALL spikes.

    This avoids the sparse-neuron problem for low-firing-rate populations
    like E cells, where most individual neurons have <5 spikes.
    """
    inst_phase = _get_inst_phase(lfp_signal, fs, freq, bandwidth)
    if inst_phase is None:
        return np.nan

    t_start, t_end = time_window
    mask = (spike_times_ms >= t_start) & (spike_times_ms < t_end)
    times = spike_times_ms[mask]

    if len(times) < min_spikes:
        return np.nan

    spike_samples = np.searchsorted(time_array, times)
    valid = (spike_samples >= 0) & (spike_samples < len(inst_phase))
    spike_samples = spike_samples[valid]
    phases = inst_phase[spike_samples]
    n = len(phases)

    if n < min_spikes:
        return np.nan

    resultant = np.sum(np.exp(1j * phases))
    ppc = (np.abs(resultant) ** 2 - n) / (n * (n - 1))
    return ppc


def compute_ppc_diagnostics_from_arrays(
    spike_times_ms, spike_neuron_ids,
    time_window, lfp_signal, time_array,
    fs, freq, min_spikes_per_neuron=5, bandwidth=4.0
):
    """
    Diagnostic version: returns PPC, MRL, mean phase, n_neurons_used,
    mean_spikes_per_neuron, and per-neuron PPC distribution.
    """
    inst_phase = _get_inst_phase(lfp_signal, fs, freq, bandwidth)
    if inst_phase is None:
        return {'ppc': np.nan, 'mrl': np.nan, 'mean_phase': np.nan,
                'n_neurons': 0, 'mean_spikes': 0, 'ppc_per_neuron': []}

    t_start, t_end = time_window
    mask = (spike_times_ms >= t_start) & (spike_times_ms < t_end)
    times = spike_times_ms[mask]
    neuron_ids = spike_neuron_ids[mask]

    if len(times) < 2:
        return {'ppc': np.nan, 'mrl': np.nan, 'mean_phase': np.nan,
                'n_neurons': 0, 'mean_spikes': 0, 'ppc_per_neuron': []}

    spike_samples = np.searchsorted(time_array, times)
    valid = (spike_samples >= 0) & (spike_samples < len(inst_phase))
    spike_samples = spike_samples[valid]
    neuron_ids = neuron_ids[valid]
    phases_all = inst_phase[spike_samples]

    # Population-level MRL (across all spikes, ignoring neuron identity)
    all_resultant = np.sum(np.exp(1j * phases_all))
    mrl = np.abs(all_resultant) / len(phases_all)
    mean_phase = np.angle(all_resultant)

    unique_neurons = np.unique(neuron_ids)
    weighted_ppc_sum = 0.0
    weight_sum = 0
    ppc_per_neuron = []
    spikes_per_neuron = []

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
        ppc_per_neuron.append(ppc_i)
        spikes_per_neuron.append(n)

    ppc = weighted_ppc_sum / weight_sum if weight_sum > 0 else np.nan
    n_used = len(ppc_per_neuron)
    mean_spk = np.mean(spikes_per_neuron) if spikes_per_neuron else 0

    return {
        'ppc': ppc,
        'mrl': mrl,
        'mean_phase': mean_phase,
        'n_neurons': n_used,
        'mean_spikes': mean_spk,
        'ppc_per_neuron': np.array(ppc_per_neuron),
        'spikes_per_neuron': np.array(spikes_per_neuron) if spikes_per_neuron else np.array([]),
    }


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


def compute_ppc_all_trials_mazzoni(
    trial_dir,
    freq_range=(1, 100),
    n_freqs=50,
    fs=10000,
    min_spikes_per_neuron=5,
    bandwidth=4.0,
    transient_skip=500,
):
    """
    Like compute_ppc_all_trials but uses the per-layer Mazzoni LFP proxy
    instead of bipolar LFP.

    Returns
    -------
    freqs : 1D array
    ppc_all : dict[layer][cell_type][condition] -> (n_trials, n_freqs) array
    n_trials : int
    """
    trial_files = sorted(glob.glob(os.path.join(trial_dir, 'trial_*.npz')))
    if not trial_files:
        raise FileNotFoundError(f"No trial_*.npz found in {trial_dir}")

    freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
    n_trials = len(trial_files)
    layer_names = list(LAYER_Z_RANGES.keys())

    ppc_all = {}
    for layer in layer_names:
        ppc_all[layer] = {}
        for ct in CELL_TYPES:
            ppc_all[layer][ct] = {
                'baseline': np.full((n_trials, n_freqs), np.nan),
                'stimulus': np.full((n_trials, n_freqs), np.nan),
            }

    for ti, fpath in enumerate(trial_files):
        print(f"  [Mazzoni] Processing {os.path.basename(fpath)} ({ti+1}/{n_trials})...")
        data = np.load(fpath, allow_pickle=True)

        # Check that Mazzoni data exists
        if 'mazzoni_lfp_matrix' not in data or 'mazzoni_layer_names' not in data:
            print(f"    Skipping {os.path.basename(fpath)}: no Mazzoni LFP data.")
            continue

        mazzoni_layer_names = list(data['mazzoni_layer_names'])
        mazzoni_matrix = data['mazzoni_lfp_matrix']  # (n_layers, n_time)
        mazzoni_time = data['mazzoni_time_ms']
        spike_data = data['spike_data'].item()
        baseline_ms = float(data['baseline_ms'])
        stim_ms = float(data['post_ms'])

        base_start = transient_skip
        base_end = baseline_ms
        stim_start = baseline_ms + transient_skip
        stim_end = baseline_ms + stim_ms

        base_mask = (mazzoni_time >= base_start) & (mazzoni_time < base_end)
        stim_mask = (mazzoni_time >= stim_start) & (mazzoni_time < stim_end)

        time_base = mazzoni_time[base_mask]
        time_stim = mazzoni_time[stim_mask]

        if len(time_base) < 100 or len(time_stim) < 100:
            continue

        for layer in layer_names:
            if layer not in mazzoni_layer_names:
                continue
            maz_idx = mazzoni_layer_names.index(layer)
            lfp_base = mazzoni_matrix[maz_idx][base_mask]
            lfp_stim = mazzoni_matrix[maz_idx][stim_mask]

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


def compute_ppc_concat_trials_mazzoni(
    trial_dir,
    freq_range=(1, 100),
    n_freqs=50,
    fs=10000,
    min_spikes_per_neuron=5,
    bandwidth=4.0,
    transient_skip=500,
):
    """
    Concatenate spike phases across trials for each neuron, then compute
    per-neuron PPC on the concatenated data.

    For each (layer, cell_type, condition, frequency), this:
      1. Extracts the instantaneous LFP phase for each trial
      2. For each spike, looks up its phase in that trial's LFP
      3. Collects all (neuron_id, phase) pairs across trials
      4. Computes per-neuron PPC on the concatenated phases

    Returns
    -------
    freqs : 1D array
    ppc_concat : dict[layer][cell_type][condition] -> 1D array (n_freqs,)
    n_trials : int
    """
    trial_files = sorted(glob.glob(os.path.join(trial_dir, 'trial_*.npz')))
    if not trial_files:
        raise FileNotFoundError(f"No trial_*.npz found in {trial_dir}")

    freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
    n_trials = len(trial_files)
    layer_names = list(LAYER_Z_RANGES.keys())

    # Accumulate phases: phases_acc[layer][ct][cond][freq_idx] = list of (neuron_id, phase)
    phases_acc = {}
    for layer in layer_names:
        phases_acc[layer] = {}
        for ct in CELL_TYPES:
            phases_acc[layer][ct] = {
                'baseline': [[] for _ in range(len(freqs))],
                'stimulus': [[] for _ in range(len(freqs))],
            }

    for ti, fpath in enumerate(trial_files):
        print(f"  [Concat] Processing {os.path.basename(fpath)} ({ti+1}/{n_trials})...")
        data = np.load(fpath, allow_pickle=True)

        if 'mazzoni_lfp_matrix' not in data or 'mazzoni_layer_names' not in data:
            continue

        mazzoni_layer_names = list(data['mazzoni_layer_names'])
        mazzoni_matrix = data['mazzoni_lfp_matrix']
        mazzoni_time = data['mazzoni_time_ms']
        spike_data = data['spike_data'].item()
        baseline_ms = float(data['baseline_ms'])
        stim_ms = float(data['post_ms'])

        base_start = transient_skip
        base_end = baseline_ms
        stim_start = baseline_ms + transient_skip
        stim_end = baseline_ms + stim_ms

        for cond, t_start, t_end in [('baseline', base_start, base_end),
                                      ('stimulus', stim_start, stim_end)]:
            t_mask = (mazzoni_time >= t_start) & (mazzoni_time < t_end)
            t_arr = mazzoni_time[t_mask]
            if len(t_arr) < 100:
                continue

            for layer in layer_names:
                if layer not in mazzoni_layer_names:
                    continue
                maz_idx = mazzoni_layer_names.index(layer)
                lfp = mazzoni_matrix[maz_idx][t_mask]

                if layer not in spike_data:
                    continue

                # Pre-compute inst_phase for each frequency
                for fi, f in enumerate(freqs):
                    inst_phase = _get_inst_phase(lfp, fs, f, bandwidth)
                    if inst_phase is None:
                        continue

                    for ct in CELL_TYPES:
                        spike_key = f'{ct}_spikes'
                        if spike_key not in spike_data[layer]:
                            continue

                        sd = spike_data[layer][spike_key]
                        spike_times = sd['times_ms']
                        spike_ids = sd['spike_indices']

                        mask = (spike_times >= t_start) & (spike_times < t_end)
                        times = spike_times[mask]
                        nids = spike_ids[mask]

                        if len(times) == 0:
                            continue

                        spike_samples = np.searchsorted(t_arr, times)
                        valid = (spike_samples >= 0) & (spike_samples < len(inst_phase))
                        spike_samples = spike_samples[valid]
                        nids = nids[valid]
                        ph = inst_phase[spike_samples]

                        # Append (neuron_id, phase) pairs
                        pairs = np.column_stack([nids, ph])
                        phases_acc[layer][ct][cond][fi].append(pairs)

    # Now compute per-neuron PPC from concatenated phases
    ppc_concat = {}
    for layer in layer_names:
        ppc_concat[layer] = {}
        for ct in CELL_TYPES:
            ppc_concat[layer][ct] = {
                'baseline': np.full(len(freqs), np.nan),
                'stimulus': np.full(len(freqs), np.nan),
            }
            for cond in ['baseline', 'stimulus']:
                for fi in range(len(freqs)):
                    chunks = phases_acc[layer][ct][cond][fi]
                    if not chunks:
                        continue
                    all_pairs = np.vstack(chunks)  # (N_total, 2)
                    neuron_ids = all_pairs[:, 0].astype(int)
                    phases_all = all_pairs[:, 1]

                    unique_neurons = np.unique(neuron_ids)
                    weighted_ppc_sum = 0.0
                    weight_sum = 0

                    for nid in unique_neurons:
                        nmask = neuron_ids == nid
                        phases = phases_all[nmask]
                        n = len(phases)
                        if n < min_spikes_per_neuron:
                            continue
                        resultant = np.sum(np.exp(1j * phases))
                        ppc_i = (np.abs(resultant) ** 2 - n) / (n * (n - 1))
                        w = n * (n - 1)
                        weighted_ppc_sum += ppc_i * w
                        weight_sum += w

                    if weight_sum > 0:
                        ppc_concat[layer][ct][cond][fi] = weighted_ppc_sum / weight_sum

    # Print summary stats
    print(f"\n{'='*70}")
    print(f"CONCATENATED PPC SUMMARY ({n_trials} trials)")
    print(f"{'='*70}")
    for cond in ['baseline', 'stimulus']:
        print(f"\n--- {cond.upper()} ---")
        for layer in layer_names:
            for ct in CELL_TYPES:
                vals = ppc_concat[layer][ct][cond]
                if np.all(np.isnan(vals)):
                    continue
                peak_idx = np.nanargmax(vals)
                print(f"  {layer:5s} {ct:3s}: peak PPC={vals[peak_idx]:.4f} "
                      f"@ {freqs[peak_idx]:.1f} Hz")

    return freqs, ppc_concat, n_trials


def compute_ppc_concat_trials_kernel(
    trial_dir,
    freq_range=(1, 100),
    n_freqs=50,
    fs=10000,
    min_spikes_per_neuron=5,
    bandwidth=4.0,
    transient_skip=500,
):
    """
    Same as compute_ppc_concat_trials_mazzoni but uses the kernel-based
    bipolar LFP instead of the Mazzoni proxy.
    """
    trial_files = sorted(glob.glob(os.path.join(trial_dir, 'trial_*.npz')))
    if not trial_files:
        raise FileNotFoundError(f"No trial_*.npz found in {trial_dir}")

    freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
    n_trials = len(trial_files)
    layer_names = list(LAYER_Z_RANGES.keys())

    phases_acc = {}
    for layer in layer_names:
        phases_acc[layer] = {}
        for ct in CELL_TYPES:
            phases_acc[layer][ct] = {
                'baseline': [[] for _ in range(len(freqs))],
                'stimulus': [[] for _ in range(len(freqs))],
            }

    for ti, fpath in enumerate(trial_files):
        print(f"  [Concat kernel] Processing {os.path.basename(fpath)} ({ti+1}/{n_trials})...")
        data = np.load(fpath, allow_pickle=True)

        time_array = data['time_array_ms']
        channel_depths = data['channel_depths']
        bipolar_matrix = data['bipolar_matrix']
        spike_data = data['spike_data'].item()
        baseline_ms = float(data['baseline_ms'])
        stim_ms = float(data['post_ms'])

        base_start = transient_skip
        base_end = baseline_ms
        stim_start = baseline_ms + transient_skip
        stim_end = baseline_ms + stim_ms

        for cond, t_start, t_end in [('baseline', base_start, base_end),
                                      ('stimulus', stim_start, stim_end)]:
            t_mask = (time_array >= t_start) & (time_array < t_end)
            t_arr = time_array[t_mask]
            if len(t_arr) < 100:
                continue

            for layer in layer_names:
                z_range = LAYER_Z_RANGES[layer]
                ch_idx = _find_closest_bipolar_channel(z_range, channel_depths)
                lfp = bipolar_matrix[ch_idx][t_mask]

                if layer not in spike_data:
                    continue

                for fi, f in enumerate(freqs):
                    inst_phase = _get_inst_phase(lfp, fs, f, bandwidth)
                    if inst_phase is None:
                        continue

                    for ct in CELL_TYPES:
                        spike_key = f'{ct}_spikes'
                        if spike_key not in spike_data[layer]:
                            continue

                        sd = spike_data[layer][spike_key]
                        spike_times = sd['times_ms']
                        spike_ids = sd['spike_indices']

                        mask = (spike_times >= t_start) & (spike_times < t_end)
                        times = spike_times[mask]
                        nids = spike_ids[mask]

                        if len(times) == 0:
                            continue

                        spike_samples = np.searchsorted(t_arr, times)
                        valid = (spike_samples >= 0) & (spike_samples < len(inst_phase))
                        spike_samples = spike_samples[valid]
                        nids = nids[valid]
                        ph = inst_phase[spike_samples]

                        pairs = np.column_stack([nids, ph])
                        phases_acc[layer][ct][cond][fi].append(pairs)

    # Compute per-neuron PPC from concatenated phases
    ppc_concat = {}
    for layer in layer_names:
        ppc_concat[layer] = {}
        for ct in CELL_TYPES:
            ppc_concat[layer][ct] = {
                'baseline': np.full(len(freqs), np.nan),
                'stimulus': np.full(len(freqs), np.nan),
            }
            for cond in ['baseline', 'stimulus']:
                for fi in range(len(freqs)):
                    chunks = phases_acc[layer][ct][cond][fi]
                    if not chunks:
                        continue
                    all_pairs = np.vstack(chunks)
                    neuron_ids = all_pairs[:, 0].astype(int)
                    phases_all = all_pairs[:, 1]

                    unique_neurons = np.unique(neuron_ids)
                    weighted_ppc_sum = 0.0
                    weight_sum = 0

                    for nid in unique_neurons:
                        nmask = neuron_ids == nid
                        phases = phases_all[nmask]
                        n = len(phases)
                        if n < min_spikes_per_neuron:
                            continue
                        resultant = np.sum(np.exp(1j * phases))
                        ppc_i = (np.abs(resultant) ** 2 - n) / (n * (n - 1))
                        w = n * (n - 1)
                        weighted_ppc_sum += ppc_i * w
                        weight_sum += w

                    if weight_sum > 0:
                        ppc_concat[layer][ct][cond][fi] = weighted_ppc_sum / weight_sum

    print(f"\n{'='*70}")
    print(f"CONCATENATED PPC SUMMARY — KERNEL LFP ({n_trials} trials)")
    print(f"{'='*70}")
    for cond in ['baseline', 'stimulus']:
        print(f"\n--- {cond.upper()} ---")
        for layer in layer_names:
            for ct in CELL_TYPES:
                vals = ppc_concat[layer][ct][cond]
                if np.all(np.isnan(vals)):
                    continue
                peak_idx = np.nanargmax(vals)
                print(f"  {layer:5s} {ct:3s}: peak PPC={vals[peak_idx]:.4f} "
                      f"@ {freqs[peak_idx]:.1f} Hz")

    return freqs, ppc_concat, n_trials


def plot_ppc_concat(freqs, ppc_maz, ppc_ker, n_trials, smooth_sigma=1.5):
    """
    Side-by-side comparison: Mazzoni LFP (left) vs Kernel/bipolar LFP (right)
    for concatenated-trial PPC.
    """
    layer_names = list(LAYER_Z_RANGES.keys())
    n_layers = len(layer_names)

    fig, axes = plt.subplots(n_layers, 4, figsize=(22, 3 * n_layers),
                             sharex=True, sharey=False)
    if n_layers == 1:
        axes = axes[np.newaxis, :]

    titles = ['Mazzoni — Baseline', 'Mazzoni — Stimulus',
              'Kernel — Baseline', 'Kernel — Stimulus']

    for row, layer in enumerate(layer_names):
        for ct in CELL_TYPES:
            for col, (ppc_src, cond) in enumerate([
                (ppc_maz, 'baseline'), (ppc_maz, 'stimulus'),
                (ppc_ker, 'baseline'), (ppc_ker, 'stimulus'),
            ]):
                ppc = ppc_src[layer][ct][cond].copy()

                if smooth_sigma > 0:
                    valid = ~np.isnan(ppc)
                    if np.sum(valid) > 3:
                        ppc[valid] = gaussian_filter1d(ppc[valid], smooth_sigma)

                axes[row, col].plot(freqs, ppc, color=COLORS[ct],
                                    label=ct, linewidth=1.5, alpha=0.85)

        for col in range(4):
            axes[row, col].set_ylabel(f'{layer}', fontsize=9)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend(fontsize=7, loc='upper right')
            axes[row, col].set_ylim(bottom=-0.01)

        if row == 0:
            for col, t in enumerate(titles):
                axes[row, col].set_title(t, fontsize=10)

    for col in range(4):
        axes[-1, col].set_xlabel('Frequency (Hz)', fontsize=10)

    fig.suptitle(f'PPC — Concatenated across {n_trials} trials: Mazzoni vs Kernel LFP',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def plot_ppc_concat_mazzoni(freqs, ppc_concat, n_trials, smooth_sigma=1.5):
    """
    Plot concatenated-trial PPC (single curve per cell type, no SEM since
    it's a single estimate from all trials combined).
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
                ppc = ppc_concat[layer][ct][cond].copy()

                if smooth_sigma > 0:
                    valid = ~np.isnan(ppc)
                    if np.sum(valid) > 3:
                        ppc[valid] = gaussian_filter1d(ppc[valid], smooth_sigma)

                axes[row, col].plot(freqs, ppc, color=COLORS[ct],
                                    label=ct, linewidth=1.5, alpha=0.85)

        axes[row, 0].set_title(f'{layer} — Baseline', fontsize=10)
        axes[row, 1].set_title(f'{layer} — Stimulus', fontsize=10)

        for col in range(2):
            axes[row, col].set_ylabel('PPC', fontsize=9)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend(fontsize=8, loc='upper right')
            axes[row, col].set_ylim(bottom=-0.01)

    axes[-1, 0].set_xlabel('Frequency (Hz)', fontsize=11)
    axes[-1, 1].set_xlabel('Frequency (Hz)', fontsize=11)
    fig.suptitle(f'PPC — Concatenated across {n_trials} trials (Mazzoni LFP)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def compute_ppc_pooled_all_trials_mazzoni(
    trial_dir,
    freq_range=(1, 100),
    n_freqs=50,
    fs=10000,
    min_spikes=10,
    bandwidth=4.0,
    transient_skip=500,
):
    """
    Pooled-spike PPC using Mazzoni LFP. Pools all spikes from each cell type
    per layer (ignoring neuron identity) and computes a single PPC value.

    This avoids the sparse-neuron bias that makes per-neuron PPC unreliable
    for low-firing-rate populations like E cells.
    """
    trial_files = sorted(glob.glob(os.path.join(trial_dir, 'trial_*.npz')))
    if not trial_files:
        raise FileNotFoundError(f"No trial_*.npz found in {trial_dir}")

    freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
    n_trials = len(trial_files)
    layer_names = list(LAYER_Z_RANGES.keys())

    ppc_all = {}
    for layer in layer_names:
        ppc_all[layer] = {}
        for ct in CELL_TYPES:
            ppc_all[layer][ct] = {
                'baseline': np.full((n_trials, n_freqs), np.nan),
                'stimulus': np.full((n_trials, n_freqs), np.nan),
            }

    for ti, fpath in enumerate(trial_files):
        print(f"  [Pooled Mazzoni] Processing {os.path.basename(fpath)} ({ti+1}/{n_trials})...")
        data = np.load(fpath, allow_pickle=True)

        if 'mazzoni_lfp_matrix' not in data or 'mazzoni_layer_names' not in data:
            print(f"    Skipping {os.path.basename(fpath)}: no Mazzoni LFP data.")
            continue

        mazzoni_layer_names = list(data['mazzoni_layer_names'])
        mazzoni_matrix = data['mazzoni_lfp_matrix']
        mazzoni_time = data['mazzoni_time_ms']
        spike_data = data['spike_data'].item()
        baseline_ms = float(data['baseline_ms'])
        stim_ms = float(data['post_ms'])

        base_start = transient_skip
        base_end = baseline_ms
        stim_start = baseline_ms + transient_skip
        stim_end = baseline_ms + stim_ms

        base_mask = (mazzoni_time >= base_start) & (mazzoni_time < base_end)
        stim_mask = (mazzoni_time >= stim_start) & (mazzoni_time < stim_end)

        time_base = mazzoni_time[base_mask]
        time_stim = mazzoni_time[stim_mask]

        if len(time_base) < 100 or len(time_stim) < 100:
            continue

        for layer in layer_names:
            if layer not in mazzoni_layer_names:
                continue
            maz_idx = mazzoni_layer_names.index(layer)
            lfp_base = mazzoni_matrix[maz_idx][base_mask]
            lfp_stim = mazzoni_matrix[maz_idx][stim_mask]

            if layer not in spike_data:
                continue

            for ct in CELL_TYPES:
                spike_key = f'{ct}_spikes'
                if spike_key not in spike_data[layer]:
                    continue

                sd = spike_data[layer][spike_key]
                spike_times = sd['times_ms']

                for fi, f in enumerate(freqs):
                    ppc_all[layer][ct]['baseline'][ti, fi] = (
                        compute_ppc_pooled_from_arrays(
                            spike_times,
                            (base_start, base_end), lfp_base, time_base,
                            fs, f, min_spikes, bandwidth
                        )
                    )
                    ppc_all[layer][ct]['stimulus'][ti, fi] = (
                        compute_ppc_pooled_from_arrays(
                            spike_times,
                            (stim_start, stim_end), lfp_stim, time_stim,
                            fs, f, min_spikes, bandwidth
                        )
                    )

    return freqs, ppc_all, n_trials


def plot_ppc_trials_pooled_mazzoni(freqs, ppc_all, n_trials, smooth_sigma=1.5):
    """
    Plot trial-averaged pooled PPC (Mazzoni LFP).
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
                mat = ppc_all[layer][ct][cond]

                with np.errstate(all='ignore'):
                    mean_ppc = np.nanmean(mat, axis=0)
                    n_valid = np.sum(~np.isnan(mat), axis=0)
                    sem_ppc = np.nanstd(mat, axis=0) / np.sqrt(np.maximum(n_valid, 1))

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
            axes[row, col].set_ylabel('PPC (pooled)', fontsize=9)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend(fontsize=8, loc='upper right')
            axes[row, col].set_ylim(bottom=-0.01)

    axes[-1, 0].set_xlabel('Frequency (Hz)', fontsize=11)
    axes[-1, 1].set_xlabel('Frequency (Hz)', fontsize=11)
    fig.suptitle(f'Trial-averaged POOLED PPC — Mazzoni LFP (n={n_trials} trials)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


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


def plot_ppc_trials_mazzoni(freqs, ppc_all, n_trials, smooth_sigma=1.5):
    """
    Plot trial-averaged PPC computed with the per-layer Mazzoni LFP.
    Same layout as plot_ppc_trials.
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
                mat = ppc_all[layer][ct][cond]

                with np.errstate(all='ignore'):
                    mean_ppc = np.nanmean(mat, axis=0)
                    n_valid = np.sum(~np.isnan(mat), axis=0)
                    sem_ppc = np.nanstd(mat, axis=0) / np.sqrt(np.maximum(n_valid, 1))

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
    fig.suptitle(f'Trial-averaged PPC — Mazzoni LFP (n={n_trials} trials)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def run_diagnostics(trial_dir, diag_freqs=(10, 30, 50), fs=10000,
                     min_spikes_per_neuron=5, bandwidth=4.0, transient_skip=500):
    """
    Run PPC diagnostics at a few key frequencies for the first trial.
    Prints per-cell-type stats and plots per-neuron PPC distributions.
    """
    trial_files = sorted(glob.glob(os.path.join(trial_dir, 'trial_*.npz')))
    if not trial_files:
        print("No trial files found.")
        return

    fpath = trial_files[0]
    print(f"\n{'='*70}")
    print(f"PPC DIAGNOSTICS — {os.path.basename(fpath)}")
    print(f"{'='*70}")

    data = np.load(fpath, allow_pickle=True)
    time_array = data['time_array_ms']
    spike_data = data['spike_data'].item()
    baseline_ms = float(data['baseline_ms'])
    stim_ms = float(data['post_ms'])

    # Use Mazzoni LFP if available, else bipolar
    if 'mazzoni_lfp_matrix' in data:
        use_mazzoni = True
        mazzoni_layer_names = list(data['mazzoni_layer_names'])
        mazzoni_matrix = data['mazzoni_lfp_matrix']
        mazzoni_time = data['mazzoni_time_ms']
        print("Using Mazzoni per-layer LFP for diagnostics.\n")
    else:
        use_mazzoni = False
        channel_depths = data['channel_depths']
        bipolar_matrix = data['bipolar_matrix']
        print("Using bipolar LFP for diagnostics.\n")

    layer_names = list(LAYER_Z_RANGES.keys())

    # Collect diagnostics for plotting
    all_diag = {}  # (layer, ct, cond, freq) -> diagnostics dict

    for cond, (t_start, t_end) in [('baseline', (transient_skip, baseline_ms)),
                                     ('stimulus', (baseline_ms + transient_skip,
                                                   baseline_ms + stim_ms))]:
        if use_mazzoni:
            t_mask = (mazzoni_time >= t_start) & (mazzoni_time < t_end)
            t_arr = mazzoni_time[t_mask]
        else:
            t_mask = (time_array >= t_start) & (time_array < t_end)
            t_arr = time_array[t_mask]

        if len(t_arr) < 100:
            continue

        print(f"\n--- {cond.upper()} ({t_start:.0f} - {t_end:.0f} ms) ---")

        for layer in layer_names:
            if use_mazzoni:
                if layer not in mazzoni_layer_names:
                    continue
                maz_idx = mazzoni_layer_names.index(layer)
                lfp = mazzoni_matrix[maz_idx][t_mask]
            else:
                z_range = LAYER_Z_RANGES[layer]
                ch_idx = _find_closest_bipolar_channel(z_range, channel_depths)
                lfp = bipolar_matrix[ch_idx][t_mask]

            if layer not in spike_data:
                continue

            for ct in CELL_TYPES:
                spike_key = f'{ct}_spikes'
                if spike_key not in spike_data[layer]:
                    continue

                sd = spike_data[layer][spike_key]
                spike_times = sd['times_ms']
                spike_ids = sd['spike_indices']

                # Count total spikes in window
                in_win = (spike_times >= t_start) & (spike_times < t_end)
                n_total_spikes = np.sum(in_win)
                n_unique = len(np.unique(spike_ids[in_win])) if n_total_spikes > 0 else 0

                for freq in diag_freqs:
                    diag = compute_ppc_diagnostics_from_arrays(
                        spike_times, spike_ids,
                        (t_start, t_end), lfp, t_arr,
                        fs, freq, min_spikes_per_neuron, bandwidth
                    )
                    all_diag[(layer, ct, cond, freq)] = diag

                    print(f"  {layer:5s} {ct:3s} @ {freq:3d}Hz: "
                          f"PPC={diag['ppc']:+.4f}  "
                          f"MRL={diag['mrl']:.4f}  "
                          f"phase={np.degrees(diag['mean_phase']):+6.1f}°  "
                          f"neurons={diag['n_neurons']:4d}/{n_unique:4d}  "
                          f"spk/neuron={diag['mean_spikes']:.1f}")

    # Plot per-neuron PPC distributions for each cell type
    print(f"\n{'='*70}")
    print("Plotting per-neuron PPC distributions...")
    print(f"{'='*70}\n")

    n_freqs_diag = len(diag_freqs)
    fig, axes = plt.subplots(len(layer_names), n_freqs_diag,
                              figsize=(5 * n_freqs_diag, 3 * len(layer_names)),
                              squeeze=False)

    for row, layer in enumerate(layer_names):
        for col, freq in enumerate(diag_freqs):
            ax = axes[row, col]
            for ct in CELL_TYPES:
                key = (layer, ct, 'stimulus', freq)
                if key not in all_diag:
                    continue
                ppc_dist = all_diag[key]['ppc_per_neuron']
                if len(ppc_dist) == 0:
                    continue
                # Histogram of per-neuron PPC
                bins = np.linspace(-0.05, 0.3, 40)
                ax.hist(ppc_dist, bins=bins, color=COLORS[ct], alpha=0.4,
                        label=f"{ct} (n={len(ppc_dist)}, "
                              f"med={np.median(ppc_dist):.3f})",
                        density=True)
                ax.axvline(np.median(ppc_dist), color=COLORS[ct],
                           linestyle='--', linewidth=1.5)

            ax.set_title(f'{layer} — {freq} Hz (stimulus)', fontsize=9)
            ax.set_xlabel('PPC per neuron', fontsize=8)
            ax.legend(fontsize=6, loc='upper right')
            ax.grid(True, alpha=0.3)

    fig.suptitle('Per-neuron PPC distributions (stimulus period)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Also plot MRL vs PPC scatter
    fig2, axes2 = plt.subplots(1, n_freqs_diag,
                                figsize=(5 * n_freqs_diag, 4), squeeze=False)
    for col, freq in enumerate(diag_freqs):
        ax = axes2[0, col]
        for ct in CELL_TYPES:
            ppcs = []
            mrls = []
            for layer in layer_names:
                key = (layer, ct, 'stimulus', freq)
                if key in all_diag and not np.isnan(all_diag[key]['ppc']):
                    ppcs.append(all_diag[key]['ppc'])
                    mrls.append(all_diag[key]['mrl'])
            if ppcs:
                ax.scatter(ppcs, mrls, color=COLORS[ct], label=ct, s=60,
                           edgecolors='black', linewidth=0.5)
        ax.set_xlabel('PPC (weighted avg)', fontsize=10)
        ax.set_ylabel('MRL (all spikes)', fontsize=10)
        ax.set_title(f'{freq} Hz — stimulus', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        # diagonal reference
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=0.8)

    fig2.suptitle('PPC vs MRL across layers (stimulus)', fontsize=13,
                  fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    return all_diag


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_ppc_trials.py <trial_dir>")
        print("       python analyze_ppc_trials.py <trial_dir> --diag")
        print("       python analyze_ppc_trials.py <trial_dir> --concat")
        print("       python analyze_ppc_trials.py <trial_dir> --pooled")
        print("Example: python analyze_ppc_trials.py results/13_03_LGN_gratings_2")
        sys.exit(1)

    trial_dir = sys.argv[1]
    run_diag = '--diag' in sys.argv

    run_pooled = '--pooled' in sys.argv
    run_concat = '--concat' in sys.argv

    if run_diag:
        all_diag = run_diagnostics(trial_dir, diag_freqs=(10, 30, 50))
        plt.show()
    elif run_concat:
        ppc_args = dict(freq_range=(1, 100), n_freqs=50, fs=10000,
                        min_spikes_per_neuron=5, bandwidth=4.0)

        print(f"Computing CONCATENATED per-neuron PPC (Mazzoni LFP) from {trial_dir}...")
        freqs_c, ppc_maz, n_trials_c = compute_ppc_concat_trials_mazzoni(
            trial_dir, **ppc_args)

        print(f"\nComputing CONCATENATED per-neuron PPC (Kernel LFP) from {trial_dir}...")
        freqs_k, ppc_ker, n_trials_k = compute_ppc_concat_trials_kernel(
            trial_dir, **ppc_args)

        print(f"\nPlotting side-by-side comparison ({n_trials_c} trials)...")
        fig_cmp = plot_ppc_concat(freqs_c, ppc_maz, ppc_ker, n_trials_c)
        fig_maz = plot_ppc_concat_mazzoni(freqs_c, ppc_maz, n_trials_c)
        plt.show()
    elif run_pooled:
        print(f"Computing POOLED PPC (Mazzoni LFP) from {trial_dir}...")
        freqs_p, ppc_pooled, n_trials_p = compute_ppc_pooled_all_trials_mazzoni(
            trial_dir,
            freq_range=(1, 100),
            n_freqs=50,
            fs=10000,
            min_spikes=10,
            bandwidth=4.0,
        )
        print(f"Plotting pooled PPC averaged over {n_trials_p} trials...")
        fig_p = plot_ppc_trials_pooled_mazzoni(freqs_p, ppc_pooled, n_trials_p)
        plt.show()
    else:
        print(f"Loading trials from {trial_dir}...")

        freqs, ppc_all, n_trials = compute_ppc_all_trials(
            trial_dir,
            freq_range=(1, 100),
            n_freqs=50,
            fs=10000,
            min_spikes_per_neuron=5,
            bandwidth=4.0,
        )

        print(f"Plotting PPC (bipolar LFP) averaged over {n_trials} trials...")
        fig1 = plot_ppc_trials(freqs, ppc_all, n_trials)

        print(f"Computing PPC with Mazzoni per-layer LFP...")
        freqs_m, ppc_all_m, n_trials_m = compute_ppc_all_trials_mazzoni(
            trial_dir,
            freq_range=(1, 100),
            n_freqs=50,
            fs=10000,
            min_spikes_per_neuron=5,
            bandwidth=4.0,
        )

        print(f"Plotting PPC (Mazzoni LFP) averaged over {n_trials_m} trials...")
        fig2 = plot_ppc_trials_mazzoni(freqs_m, ppc_all_m, n_trials_m)
        plt.show()
