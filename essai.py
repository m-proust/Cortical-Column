
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt, detrend
from scipy.stats import circmean, circstd
import seaborn as sns



def compute_ppc(phases):
    """
    Pairwise Phase Consistency (Vinck et al., 2010).

    PPC = (2 / (n*(n-1))) * sum_{j<k} cos(theta_j - theta_k)

    This is equivalent to:
    PPC = (n * R^2 - 1) / (n - 1)
    where R is the mean resultant length, which is faster to compute.

    Parameters
    ----------
    phases : array-like, shape (n_spikes,)
        Phase angles in radians at spike times.

    Returns
    -------
    ppc : float
        Pairwise phase consistency. Range: [-1, 1].
        0 = no phase locking, 1 = perfect locking.
    """
    n = len(phases)
    if n < 2:
        return np.nan

    # Mean resultant length (fast computation)
    C = np.sum(np.cos(phases))
    S = np.sum(np.sin(phases))
    R2 = (C**2 + S**2) / n**2

    # PPC (unbiased estimator)
    ppc = (n * R2 - 1) / (n - 1)
    return ppc


def compute_ppc_spectrum(spike_times_ms, lfp_signal, lfp_time_ms,
                         freqs=None, bandwidth=4.0, fs=10000,
                         min_spikes=20):
    """
    Compute PPC as a function of frequency for one neuron population
    relative to one LFP channel.

    For each frequency, the LFP is bandpass filtered, the instantaneous
    phase is extracted via Hilbert transform, and PPC is computed from
    the phases at spike times.

    Parameters
    ----------
    spike_times_ms : array
        Spike times in ms.
    lfp_signal : array
        LFP time series (raw, will be filtered).
    lfp_time_ms : array
        Time vector for LFP in ms.
    freqs : array or None
        Center frequencies for analysis. Default: 5-100 Hz in 2 Hz steps.
    bandwidth : float
        Filter bandwidth (Hz) around each center frequency.
    fs : float
        Sampling rate in Hz.
    min_spikes : int
        Minimum spikes required to compute PPC.

    Returns
    -------
    freqs : array
        Frequencies analyzed.
    ppc_values : array
        PPC at each frequency.
    mean_phase : array
        Mean preferred phase at each frequency (radians).
    n_spikes_used : array
        Number of spikes used at each frequency.
    """
    if freqs is None:
        freqs = np.arange(5, 101, 2)

    dt_ms = np.mean(np.diff(lfp_time_ms))

    ppc_values = np.full(len(freqs), np.nan)
    mean_phase = np.full(len(freqs), np.nan)
    n_spikes_used = np.zeros(len(freqs), dtype=int)

    # Convert spike times to sample indices
    spike_indices = np.searchsorted(lfp_time_ms, spike_times_ms)
    # Remove out-of-bounds
    valid = (spike_indices >= 0) & (spike_indices < len(lfp_signal))
    spike_indices = spike_indices[valid]

    if len(spike_indices) < min_spikes:
        return freqs, ppc_values, mean_phase, n_spikes_used

    for i, fc in enumerate(freqs):
        # Bandpass filter design
        f_low = max(fc - bandwidth / 2, 1.0)
        f_high = fc + bandwidth / 2

        if f_high >= fs / 2:
            continue

        try:
            b, a = butter(3, [f_low / (fs / 2), f_high / (fs / 2)], btype='band')
            lfp_filt = filtfilt(b, a, lfp_signal)
        except Exception:
            continue

        # Hilbert transform for instantaneous phase
        analytic = hilbert(lfp_filt)
        inst_phase = np.angle(analytic)

        # Get phases at spike times
        phases_at_spikes = inst_phase[spike_indices]

        n_spikes_used[i] = len(phases_at_spikes)

        if len(phases_at_spikes) >= min_spikes:
            ppc_values[i] = compute_ppc(phases_at_spikes)
            mean_phase[i] = circmean(phases_at_spikes, high=np.pi, low=-np.pi)

    return freqs, ppc_values, mean_phase, n_spikes_used


# =====================================================================
# Multi-trial PPC (average across trials)
# =====================================================================

def compute_ppc_multi_trial(all_trials_spike_times, all_trials_lfp,
                            lfp_time_ms, freqs=None, bandwidth=4.0,
                            fs=10000, min_spikes=20):
    """
    Compute PPC averaged across trials.

    The standard approach: pool all spike phases across trials, then
    compute PPC on the pooled set. Alternatively, compute per-trial
    and average. Here we pool (more robust with few spikes per trial).

    Parameters
    ----------
    all_trials_spike_times : list of arrays
        Spike times (ms) for each trial.
    all_trials_lfp : list of arrays
        LFP signals for each trial.
    lfp_time_ms : array
        Common time vector.

    Returns
    -------
    freqs, ppc_values, mean_phase, total_spikes
    """
    if freqs is None:
        freqs = np.arange(5, 101, 2)

    # Collect phases across all trials for each frequency
    all_phases = {i: [] for i in range(len(freqs))}

    dt_ms = np.mean(np.diff(lfp_time_ms))

    for trial_idx in range(len(all_trials_spike_times)):
        spk = all_trials_spike_times[trial_idx]
        lfp = all_trials_lfp[trial_idx]

        if len(spk) == 0:
            continue

        spike_indices = np.searchsorted(lfp_time_ms, spk)
        valid = (spike_indices >= 0) & (spike_indices < len(lfp))
        spike_indices = spike_indices[valid]

        if len(spike_indices) == 0:
            continue

        for i, fc in enumerate(freqs):
            f_low = max(fc - bandwidth / 2, 1.0)
            f_high = fc + bandwidth / 2

            if f_high >= fs / 2:
                continue

            try:
                b, a = butter(3, [f_low / (fs / 2), f_high / (fs / 2)], btype='band')
                lfp_filt = filtfilt(b, a, lfp)
                analytic = hilbert(lfp_filt)
                inst_phase = np.angle(analytic)
                all_phases[i].extend(inst_phase[spike_indices])
            except Exception:
                continue

    ppc_values = np.full(len(freqs), np.nan)
    mean_phase_vals = np.full(len(freqs), np.nan)
    total_spikes = np.zeros(len(freqs), dtype=int)

    for i in range(len(freqs)):
        phases = np.array(all_phases[i])
        total_spikes[i] = len(phases)
        if len(phases) >= min_spikes:
            ppc_values[i] = compute_ppc(phases)
            mean_phase_vals[i] = circmean(phases, high=np.pi, low=-np.pi)

    return freqs, ppc_values, mean_phase_vals, total_spikes


# =====================================================================
# High-level analysis functions
# =====================================================================

LAYER_ORDER = ['L23', 'L4AB', 'L4C', 'L5', 'L6']
CELL_TYPES = ['E', 'PV', 'SOM', 'VIP']
CELL_COLORS = {'E': '#2ca02c', 'PV': '#d62728', 'SOM': '#1f77b4', 'VIP': '#ff7f0e'}


LAYER_TO_LOCAL_CHANNELS = {
    'L6':   [3, 4],
    'L5':   [4, 5],
    'L4C':  [6, 7],
    'L4AB': [8, 9],
    'L23':  [10, 11, 12, 13],
}


def extract_spike_times_from_trial(trial_data, layer, cell_type,
                                   time_window_ms=None):
    """
    Extract spike times for a given layer/cell_type from a trial.

    Parameters
    ----------
    trial_data : dict
        Must contain 'spike_data' key with nested dict structure:
        spike_data[layer][f'{cell_type}_spikes'] = {'times_ms': ..., 'indices': ...}
    layer : str
        e.g., 'L23', 'L4C'
    cell_type : str
        e.g., 'E', 'PV', 'SOM', 'VIP'
    time_window_ms : tuple or None
        (t_start, t_end) in ms to restrict spike times.

    Returns
    -------
    spike_times : array
        Spike times in ms.
    """
    spike_data = trial_data['spike_data']

    # Try common naming conventions
    possible_keys = [
        f'{cell_type}_spikes',
        f'spikes_{cell_type}',
        f'{cell_type}',
    ]

    spk_dict = None
    for key in possible_keys:
        if key in spike_data.get(layer, {}):
            spk_dict = spike_data[layer][key]
            break

    if spk_dict is None:
        # Try to find it
        available = list(spike_data.get(layer, {}).keys())
        matching = [k for k in available if cell_type.lower() in k.lower() and 'spike' in k.lower()]
        if matching:
            spk_dict = spike_data[layer][matching[0]]
        else:
            return np.array([])

    # Handle both dict and structured array cases
    if isinstance(spk_dict, dict):
        times = np.array(spk_dict['times_ms'])
    elif hasattr(spk_dict, 'item'):
        spk_dict = spk_dict.item()
        times = np.array(spk_dict['times_ms'])
    else:
        times = np.array(spk_dict)

    if time_window_ms is not None:
        mask = (times >= time_window_ms[0]) & (times < time_window_ms[1])
        times = times[mask]

    return times


def compute_ppc_all_layers_local_channel(all_trials, time_window_ms=None,
                                          freqs=None, bandwidth=4.0,
                                          fs=10000, min_spikes=30,
                                          use_bipolar=False):
    """
    Compute PPC for each cell type in each layer, using the local LFP channel.

    For each layer, picks the middle channel from LAYER_TO_LOCAL_CHANNELS
    and computes multi-trial PPC.

    Parameters
    ----------
    all_trials : list of dicts
        Each must have 'spike_data', 'lfp_matrix' (or 'bipolar_lfp'), 'time'.
    time_window_ms : tuple or None
        Restrict analysis to this time window.
    use_bipolar : bool
        If True, use bipolar LFP instead of monopolar.

    Returns
    -------
    results : dict
        results[layer][cell_type] = {
            'freqs': ..., 'ppc': ..., 'phase': ..., 'n_spikes': ...
        }
    """
    if freqs is None:
        freqs = np.arange(5, 101, 2)

    lfp_key = 'bipolar_lfp' if use_bipolar else 'lfp_matrix'

    results = {}

    for layer in LAYER_ORDER:
        results[layer] = {}

        # Pick middle local channel
        local_chs = LAYER_TO_LOCAL_CHANNELS[layer]
        ref_ch = local_chs[len(local_chs) // 2]

        print(f"  {layer}: using LFP channel {ref_ch}")

        for ct in CELL_TYPES:
            print(f"    {ct}...", end=" ")

            # Collect spike times and LFP across trials
            trial_spikes = []
            trial_lfps = []

            for trial in all_trials:
                if 'spike_data' not in trial:
                    continue

                spk = extract_spike_times_from_trial(
                    trial, layer, ct, time_window_ms=time_window_ms
                )

                lfp = trial[lfp_key]
                if ref_ch >= lfp.shape[0]:
                    ref_ch_actual = lfp.shape[0] - 1
                else:
                    ref_ch_actual = ref_ch

                trial_spikes.append(spk)
                trial_lfps.append(lfp[ref_ch_actual])

            lfp_time = all_trials[0]['time']
            if time_window_ms is not None:
                t_mask = (lfp_time >= time_window_ms[0]) & (lfp_time < time_window_ms[1])
                lfp_time_win = lfp_time[t_mask]
                trial_lfps_win = [lfp[t_mask] for lfp in trial_lfps]
            else:
                lfp_time_win = lfp_time
                trial_lfps_win = trial_lfps

            f, ppc, phase, n_spk = compute_ppc_multi_trial(
                trial_spikes, trial_lfps_win, lfp_time_win,
                freqs=freqs, bandwidth=bandwidth, fs=fs,
                min_spikes=min_spikes
            )

            total = np.nansum(n_spk)
            print(f"({total} spikes total)")

            results[layer][ct] = {
                'freqs': f,
                'ppc': ppc,
                'phase': phase,
                'n_spikes': n_spk,
                'ref_channel': ref_ch,
            }

    return results


def compute_ppc_cross_channel(all_trials, layer, cell_type,
                               time_window_ms=None, freqs=None,
                               bandwidth=4.0, fs=10000, min_spikes=30,
                               use_bipolar=False):
    """
    Compute PPC for one cell type in one layer against ALL LFP channels.
    This produces a laminar profile of phase-locking.

    Returns
    -------
    results : dict with keys 'freqs', 'ppc_matrix' (n_channels x n_freqs),
              'phase_matrix', 'n_spikes_matrix'
    """
    if freqs is None:
        freqs = np.arange(5, 101, 2)

    lfp_key = 'bipolar_lfp' if use_bipolar else 'lfp_matrix'
    n_channels = all_trials[0][lfp_key].shape[0]

    ppc_matrix = np.full((n_channels, len(freqs)), np.nan)
    phase_matrix = np.full((n_channels, len(freqs)), np.nan)
    nspk_matrix = np.zeros((n_channels, len(freqs)), dtype=int)

    lfp_time = all_trials[0]['time']

    for ch in range(n_channels):
        print(f"  Channel {ch}/{n_channels-1}...", end=" ")

        trial_spikes = []
        trial_lfps = []

        for trial in all_trials:
            if 'spike_data' not in trial:
                continue

            spk = extract_spike_times_from_trial(
                trial, layer, cell_type, time_window_ms=time_window_ms
            )
            trial_spikes.append(spk)
            trial_lfps.append(trial[lfp_key][ch])

        if time_window_ms is not None:
            t_mask = (lfp_time >= time_window_ms[0]) & (lfp_time < time_window_ms[1])
            lfp_time_win = lfp_time[t_mask]
            trial_lfps_win = [lfp[t_mask] for lfp in trial_lfps]
        else:
            lfp_time_win = lfp_time
            trial_lfps_win = trial_lfps

        f, ppc, phase, n_spk = compute_ppc_multi_trial(
            trial_spikes, trial_lfps_win, lfp_time_win,
            freqs=freqs, bandwidth=bandwidth, fs=fs,
            min_spikes=min_spikes
        )

        ppc_matrix[ch] = ppc
        phase_matrix[ch] = phase
        nspk_matrix[ch] = n_spk
        print("done")

    return {
        'freqs': freqs,
        'ppc_matrix': ppc_matrix,
        'phase_matrix': phase_matrix,
        'n_spikes_matrix': nspk_matrix,
        'layer': layer,
        'cell_type': cell_type,
    }


# =====================================================================
# Plotting functions
# =====================================================================

def plot_ppc_all_layers(results, title_suffix=""):
    """
    Plot PPC spectra for all cell types in all layers.
    One subplot per layer, all cell types overlaid.
    """
    fig, axes = plt.subplots(1, len(LAYER_ORDER), figsize=(22, 4), sharey=True)

    for i, layer in enumerate(LAYER_ORDER):
        ax = axes[i]
        for ct in CELL_TYPES:
            if ct not in results[layer]:
                continue
            r = results[layer][ct]
            freqs = r['freqs']
            ppc = r['ppc']

            ax.plot(freqs, ppc, color=CELL_COLORS[ct], label=ct, linewidth=1.5)

        ax.set_title(f'{layer} (ch{results[layer]["E"]["ref_channel"]})')
        ax.set_xlabel('Frequency (Hz)')
        if i == 0:
            ax.set_ylabel('PPC')
        ax.legend(fontsize=8)
        ax.set_xlim([freqs[0], freqs[-1]])
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle(f'Spike-LFP Phase Locking (PPC) — Local Channel{title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.show()
    return fig


def plot_ppc_phase_polar(results, freq_band=(40, 100), title_suffix=""):
    """
    Polar plot of preferred phase and PPC strength for each cell type,
    averaged over a frequency band, for each layer.
    """
    fig, axes = plt.subplots(1, len(LAYER_ORDER), figsize=(22, 4),
                             subplot_kw={'projection': 'polar'})

    for i, layer in enumerate(LAYER_ORDER):
        ax = axes[i]
        for ct in CELL_TYPES:
            if ct not in results[layer]:
                continue
            r = results[layer][ct]
            freqs = r['freqs']
            ppc = r['ppc']
            phase = r['phase']

            mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
            if not np.any(mask) or np.all(np.isnan(ppc[mask])):
                continue

            mean_ppc = np.nanmean(ppc[mask])
            mean_phase = circmean(phase[mask][~np.isnan(phase[mask])],
                                  high=np.pi, low=-np.pi)

            ax.plot([mean_phase, mean_phase], [0, max(mean_ppc, 0)],
                    color=CELL_COLORS[ct], linewidth=2, label=ct)
            ax.scatter([mean_phase], [max(mean_ppc, 0)],
                       color=CELL_COLORS[ct], s=50, zorder=5)

        ax.set_title(f'{layer}', fontsize=10, pad=15)
        ax.legend(fontsize=7, loc='upper right')

    plt.suptitle(f'Preferred Phase & PPC ({freq_band[0]}-{freq_band[1]} Hz){title_suffix}',
                 fontsize=14)
    plt.tight_layout()
    plt.show()
    return fig


def plot_ppc_cross_channel_heatmap(cross_results, electrode_positions=None,
                                    freq_range=(5, 100)):
    """
    Heatmap of PPC (channels x frequency) for one cell type in one layer.
    Shows the laminar profile of phase-locking.
    """
    freqs = cross_results['freqs']
    ppc = cross_results['ppc_matrix'].copy()
    layer = cross_results['layer']
    ct = cross_results['cell_type']

    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_plot = freqs[freq_mask]
    ppc_plot = ppc[:, freq_mask]

    # Flip so superficial is on top
    ppc_plot = np.flipud(ppc_plot)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={'width_ratios': [3, 1]})

    # Heatmap
    ax = axes[0]
    n_ch = ppc_plot.shape[0]
    extent = [freqs_plot[0], freqs_plot[-1], -0.5, n_ch - 0.5]

    vmax = np.nanpercentile(ppc_plot, 95)
    vmin = 0

    im = ax.imshow(ppc_plot, aspect='auto', cmap='hot',
                    extent=extent, origin='upper',
                    vmin=vmin, vmax=vmax)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Channel (deep → superficial)')
    ax.set_title(f'PPC Laminar Profile: {layer} {ct}')
    plt.colorbar(im, ax=ax, label='PPC')

    # Marginal: PPC averaged over gamma band
    ax2 = axes[1]
    gamma_mask = (freqs_plot >= 25) & (freqs_plot <= 45)
    if np.any(gamma_mask):
        mean_gamma_ppc = np.nanmean(ppc_plot[:, gamma_mask], axis=1)
        channels = np.arange(n_ch)
        ax2.barh(channels, mean_gamma_ppc, color=CELL_COLORS.get(ct, 'gray'), alpha=0.7)
        ax2.set_xlabel('Mean PPC (25-45 Hz)')
        ax2.set_ylabel('Channel (deep → superficial)')
        ax2.set_ylim(-0.5, n_ch - 0.5)

    plt.suptitle(f'Cross-channel PPC: {layer} {ct} cells', fontsize=14)
    plt.tight_layout()
    plt.show()
    return fig


def plot_ppc_summary_matrix(results, freq_band=(40, 100)):
    """
    Summary heatmap: layers x cell types, showing mean PPC in a frequency band.
    Quick overview of which populations lock most strongly to gamma.
    """
    mat = np.full((len(LAYER_ORDER), len(CELL_TYPES)), np.nan)

    for i, layer in enumerate(LAYER_ORDER):
        for j, ct in enumerate(CELL_TYPES):
            if ct not in results[layer]:
                continue
            r = results[layer][ct]
            mask = (r['freqs'] >= freq_band[0]) & (r['freqs'] <= freq_band[1])
            if np.any(mask):
                mat[i, j] = np.nanmean(r['ppc'][mask])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(CELL_TYPES)))
    ax.set_xticklabels(CELL_TYPES)
    ax.set_yticks(range(len(LAYER_ORDER)))
    ax.set_yticklabels(LAYER_ORDER)
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Layer')
    ax.set_title(f'Mean PPC ({freq_band[0]}-{freq_band[1]} Hz)')

    # Annotate values
    for i in range(len(LAYER_ORDER)):
        for j in range(len(CELL_TYPES)):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9,
                        color='white' if val > np.nanmax(mat) * 0.6 else 'black')

    plt.colorbar(im, ax=ax, label='PPC')
    plt.tight_layout()
    plt.show()
    return fig


base_path = "results/trials_23_03"
n_trials =5
all_trials = []

for trial_idx in range(n_trials):
    fname = f"{base_path}/trial_{trial_idx:03d}.npz"
    data = np.load(fname, allow_pickle=True)

    trial_data = {
        'seed':       data["seed"],
        'time':       data["time_array_ms"],
        'bipolar_lfp': data["bipolar_matrix"],
        'lfp_matrix': data["lfp_matrix"],
        'csd':        data["csd"],
        'rate_data':  data["rate_data"].item() if data["rate_data"].size == 1 else data["rate_data"],
        'baseline_ms':   float(data["baseline_ms"]),
        'stim_onset_ms': float(data["stim_onset_ms"]),
        'channel_labels':     data["channel_labels"],
        'channel_depths':     data["channel_depths"],
        'electrode_positions': data["electrode_positions"],
        'spike_data': data["spike_data"].item() if "spike_data" in data else None,
    }
    all_trials.append(trial_data)

# 3. Check spike data is available
if all_trials[0]['spike_data'] is None:
    print("WARNING: No spike data found! Re-run simulations with spike saving enabled.")

# 4. Compute PPC for all layers, local channel reference
#    Restrict to post-stimulus window
stim_onset = all_trials[0]['stim_onset_ms']
time_window = (stim_onset + 500, stim_onset + 1500)  # skip initial transient

print("Computing PPC (local channel reference)...")
results = compute_ppc_all_layers_local_channel(
    all_trials,
    time_window_ms=time_window,
    freqs=np.arange(5, 101, 1),
    bandwidth=4.0,
    fs=10000,
    min_spikes=30,
    use_bipolar=False,  # use monopolar LFP
)

# 5. Plot PPC spectra
plot_ppc_all_layers(results, title_suffix=" (post-stimulus)")

# 6. Plot preferred phases (polar)
plot_ppc_phase_polar(results, freq_band=(40, 100))

# 7. Summary matrix
plot_ppc_summary_matrix(results, freq_band=(40, 100))

# 8. Cross-channel laminar profile for a specific population
print("Computing cross-channel PPC for L4C PV cells...")
cross_ppc = compute_ppc_cross_channel(
    all_trials,
    layer='L4C',
    cell_type='PV',
    time_window_ms=time_window,
    freqs=np.arange(5, 101, 2),
    use_bipolar=True,
)
plot_ppc_cross_channel_heatmap(cross_ppc)

# 9. Compare baseline vs stimulus
time_window_pre = (stim_onset - 800, stim_onset)
time_window_post = (stim_onset + 200, stim_onset + 1200)

results_pre = compute_ppc_all_layers_local_channel(
    all_trials, time_window_ms=time_window_pre)
results_post = compute_ppc_all_layers_local_channel(
    all_trials, time_window_ms=time_window_post)

plot_ppc_all_layers(results_pre, title_suffix=" (baseline)")
plot_ppc_all_layers(results_post, title_suffix=" (post-stimulus)")


