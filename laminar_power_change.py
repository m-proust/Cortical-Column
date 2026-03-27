import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import zscore
import seaborn as sns
from matplotlib.gridspec import GridSpec

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Paired")

base_path = "results/trials_27_03_4"



n_trials = 4
all_trials = []

for trial_idx in range(n_trials):
    fname = f"{base_path}/trial_{trial_idx:03d}.npz"
    # fname = f"{base_path}/trial_{trial_idx:03d}_rate_0.5x_processed.npz"
    data = np.load(fname, allow_pickle=True)

    trial_data = {
        'seed' : data["seed"],
        'time': data["time_array_ms"],
        'bipolar_lfp': data["bipolar_matrix"],
        'lfp_matrix': data["lfp_matrix"],
        'csd': data["csd"],
        'rate_data': data["rate_data"].item() if data["rate_data"].size == 1 else data["rate_data"],
        # 'spike_data': data["spike_data"].item() if data["spike_data"].size == 1 else data["spike_data"],
        'baseline_ms': float(data["baseline_ms"]),
        'stim_onset_ms': float(data["stim_onset_ms"]),
        'channel_labels': data["channel_labels"],
        'channel_depths': data["channel_depths"],
        'electrode_positions': data["electrode_positions"]
    }

    # Load Mazzoni per-layer LFP if available
    if "mazzoni_lfp_matrix" in data:
        trial_data['mazzoni_lfp_matrix'] = data["mazzoni_lfp_matrix"]
        trial_data['mazzoni_time_ms'] = data["mazzoni_time_ms"]
        trial_data['mazzoni_layer_names'] = data["mazzoni_layer_names"]

    all_trials.append(trial_data)

print(f"Loaded {len(all_trials)} trials")
print(f"Time range: {all_trials[0]['time'][0]:.1f} to {all_trials[0]['time'][-1]:.1f} ms")
print(f"Stimulus onset: {all_trials[0]['stim_onset_ms']:.1f} ms")
print(f"Number of bipolar channels: {all_trials[0]['bipolar_lfp'].shape[0]}")
print(f"Sampling rate: ~{1000 / np.mean(np.diff(all_trials[0]['time'])):.0f} Hz")

from scipy.signal import detrend
from scipy.signal.windows import dpss
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral'
})

def pmtm(x, NW=2, nfft=None, fs=1.0):
    """
    Multitaper power spectral density estimate using DPSS windows.

    Parameters:
    -----------
    x : array_like
        Input signal
    NW : float
        Time-bandwidth product (typically 2-4)
    nfft : int
        FFT length (default: next power of 2 >= len(x))
    fs : float
        Sampling frequency

    Returns:
    --------
    f : ndarray
        Frequency vector
    psd : ndarray
        Power spectral density estimate
    """
    N = len(x)

    if nfft is None:
        nfft = 2 ** int(np.ceil(np.log2(N)))

    # Number of tapers
    K = int(2 * NW - 1)

    # Generate DPSS tapers
    tapers, ratios = dpss(N, NW, K, return_ratios=True)

    # Compute tapered FFTs
    psd_mt = np.zeros(nfft)

    for k in range(K):
        # Apply taper and compute FFT
        x_tapered = x * tapers[k]
        X = np.fft.fft(x_tapered, n=nfft)

        # Accumulate weighted power (weight by eigenvalue ratio)
        psd_mt += ratios[k] * np.abs(X) ** 2

    # Normalize
    psd_mt /= np.sum(ratios)
    psd_mt /= fs  # Convert to density

    # Frequency vector (positive frequencies only)
    f = np.fft.fftfreq(nfft, d=1/fs)
    pos_mask = f >= 0

    return f[pos_mask], psd_mt[pos_mask]

def plot_laminar_spectral_profile(all_trials, pre_window_ms=1000, post_window_ms=1000,
                                 post_start_ms=500,
                                 freq_range=(1, 100), log_freq=True, remove_mean=True,
                                 do_detrend=True, fmax=100):

    from scipy.signal import detrend, welch
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np
    import matplotlib.pyplot as plt

    fs = 10000  # sampling rate
    n_channels = all_trials[0]['bipolar_lfp'].shape[0]
    n_trials = len(all_trials)

    channel_depths = all_trials[0]['channel_depths']

    if len(channel_depths) > n_channels:
        bipolar_depths = (channel_depths[:-1] + channel_depths[1:]) / 2
    else:
        bipolar_depths = channel_depths[:n_channels]

    # ---------- PSD computation (Welch, matching main.py) ----------
    all_psd_pre, all_psd_post = [], []
    f = None

    for ch in range(n_channels):
        pre_trials, post_trials = [], []

        for trial in all_trials:
            lfp = trial['bipolar_lfp'][ch]
            time = trial['time']
            stim = trial['stim_onset_ms']

            pre_mask = (time >= stim - pre_window_ms) & (time < stim)
            post_mask = (time >= stim + post_start_ms) & \
                        (time < stim + post_start_ms + post_window_ms)

            pre = lfp[pre_mask].copy()
            post = lfp[post_mask].copy()

            if len(pre) == 0 or len(post) == 0:
                continue
            if np.any(np.isnan(pre)) or np.any(np.isnan(post)):
                continue

            if do_detrend:
                pre = detrend(pre)
                post = detrend(post)
            elif remove_mean:
                pre -= np.mean(pre)
                post -= np.mean(post)

            nperseg = min(len(pre), 100 * min(1024, len(pre) // 4))
            f, psd_pre = welch(pre, fs=fs, nperseg=nperseg, window='hann')
            _, psd_post = welch(post, fs=fs, nperseg=nperseg, window='hann')

            pre_trials.append(psd_pre)
            post_trials.append(psd_post)

        if len(pre_trials) == 0:
            print(f"  WARNING: channel {ch} — all trials skipped (NaN / empty LFP)")
            all_psd_pre.append(None)
            all_psd_post.append(None)
        else:
            all_psd_pre.append(np.mean(pre_trials, axis=0))
            all_psd_post.append(np.mean(post_trials, axis=0))

    if f is None:
        raise RuntimeError("No valid trials in any channel — check your LFP data for NaN/zero signals")

    # Replace None channels with NaN arrays
    n_freq = len(f)
    for i in range(n_channels):
        if all_psd_pre[i] is None:
            all_psd_pre[i] = np.full(n_freq, np.nan)
            all_psd_post[i] = np.full(n_freq, np.nan)

    psd_pre = np.array(all_psd_pre)
    psd_post = np.array(all_psd_post)

    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    f_plot = f[freq_mask]

    psd_pre = psd_pre[:, freq_mask]
    psd_post = psd_post[:, freq_mask]

    psd_pre_db = 10 * np.log10(psd_pre + 1e-10)
    psd_post_db = 10 * np.log10(psd_post + 1e-10)

    pct_change = (psd_post - psd_pre) / psd_pre * 100

    psd_pre_db = np.flipud(psd_pre_db)
    psd_post_db = np.flipud(psd_post_db)
    pct_change = np.flipud(pct_change)

    # ---------- Plot ----------
    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.15], wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    extent = [f_plot[0], f_plot[-1], -0.5, n_channels - 0.5]

    vmin_db = np.percentile([psd_pre_db, psd_post_db], 5)
    vmax_db = np.percentile([psd_pre_db, psd_post_db], 95)

    # ---- Baseline ----
    im1 = axes[0].imshow(psd_pre_db, aspect='auto', cmap='viridis',
                         extent=extent, origin='upper',
                         vmin=vmin_db, vmax=vmax_db)
    axes[0].set_title('Baseline')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Channel (deep → superficial)')
    plt.colorbar(im1, ax=axes[0], label='Power (dB)')

    # ---- Post ----
    im2 = axes[1].imshow(psd_post_db, aspect='auto', cmap='viridis',
                         extent=extent, origin='upper',
                         vmin=vmin_db, vmax=vmax_db)
    axes[1].set_title('Post-stimulus')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Channel (deep → superficial)')
    plt.colorbar(im2, ax=axes[1], label='Power (dB)')

    # ---- Percent Change (FIXED SCALE) ----
    vmin_pct = -100
    vmax_pct = 1000
    norm = TwoSlopeNorm(vmin=vmin_pct, vcenter=0, vmax=vmax_pct)


    im3 = axes[2].imshow(pct_change, aspect='auto', cmap='RdBu_r',
                         extent=extent, origin='upper', norm=norm)
    axes[2].set_title('Stimulus-induced change (%)')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Channel (deep → superficial)')

    cbar = plt.colorbar(im3, ax=axes[2], label='% Change')
    cbar.set_ticks([-100, -50, 0, 100, 250, 400, 500])

    if log_freq:
        for ax in axes:
            ax.set_xscale('log')

    plt.suptitle('Laminar Spectral Profile (Welch)', fontsize=16)
    plt.show()

    return f_plot, bipolar_depths, psd_pre_db, psd_post_db, pct_change


def plot_mazzoni_laminar_spectral_profile(all_trials, pre_window_ms=1000, post_window_ms=1000,
                                          post_start_ms=500,
                                          freq_range=(1, 100), log_freq=True, remove_mean=True,
                                          do_detrend=True, fmax=100):
    """
    Laminar spectral profile using the Mazzoni per-layer LFP proxy.
    Each row = one cortical layer (instead of bipolar electrode channel).
    """
    from scipy.signal import detrend, welch
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np
    import matplotlib.pyplot as plt

    fs = 10000

    # Get layer names from first trial
    layer_names = list(all_trials[0]['mazzoni_layer_names'])
    n_layers = len(layer_names)
    n_trials = len(all_trials)

    # ---------- PSD computation (Welch) ----------
    all_psd_pre, all_psd_post = [], []

    f = None

    for li in range(n_layers):
        pre_trials, post_trials = [], []

        for trial in all_trials:
            lfp = trial['mazzoni_lfp_matrix'][li]
            time = trial['mazzoni_time_ms']
            stim = trial['stim_onset_ms']

            pre_mask = (time >= stim - pre_window_ms) & (time < stim)
            post_mask = (time >= stim + post_start_ms) & \
                        (time < stim + post_start_ms + post_window_ms)

            pre = lfp[pre_mask].copy()
            post = lfp[post_mask].copy()

            if len(pre) == 0 or len(post) == 0:
                continue
            if np.any(np.isnan(pre)) or np.any(np.isnan(post)):
                continue

            if do_detrend:
                pre = detrend(pre)
                post = detrend(post)
            elif remove_mean:
                pre -= np.mean(pre)
                post -= np.mean(post)

            nperseg = min(len(pre), 100 * min(1024, len(pre) // 4))
            f, psd_pre = welch(pre, fs=fs, nperseg=nperseg, window='hann')
            _, psd_post = welch(post, fs=fs, nperseg=nperseg, window='hann')

            pre_trials.append(psd_pre)
            post_trials.append(psd_post)

        if len(pre_trials) == 0:
            print(f"  WARNING: layer {li} ({layer_names[li]}) — all trials skipped (NaN / empty LFP)")
            all_psd_pre.append(None)
            all_psd_post.append(None)
        else:
            all_psd_pre.append(np.mean(pre_trials, axis=0))
            all_psd_post.append(np.mean(post_trials, axis=0))

    if f is None:
        raise RuntimeError("No valid trials in any layer — check your LFP data for NaN/zero signals")

    # Replace None layers with NaN arrays
    n_freq = len(f)
    for i in range(n_layers):
        if all_psd_pre[i] is None:
            all_psd_pre[i] = np.full(n_freq, np.nan)
            all_psd_post[i] = np.full(n_freq, np.nan)

    psd_pre = np.array(all_psd_pre)
    psd_post = np.array(all_psd_post)

    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    f_plot = f[freq_mask]

    psd_pre = psd_pre[:, freq_mask]
    psd_post = psd_post[:, freq_mask]

    psd_pre_db = 10 * np.log10(psd_pre + 1e-10)
    psd_post_db = 10 * np.log10(psd_post + 1e-10)

    pct_change = (psd_post - psd_pre) / psd_pre * 100

    # Flip so deep layers are at bottom
    psd_pre_db = np.flipud(psd_pre_db)
    psd_post_db = np.flipud(psd_post_db)
    pct_change = np.flipud(pct_change)
    layer_labels_display = list(reversed(layer_names))

    # ---------- Plot ----------
    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.15], wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    extent = [f_plot[0], f_plot[-1], -0.5, n_layers - 0.5]

    vmin_db = np.percentile([psd_pre_db, psd_post_db], 5)
    vmax_db = np.percentile([psd_pre_db, psd_post_db], 95)

    yticks = np.arange(n_layers)

    # ---- Baseline ----
    im1 = axes[0].imshow(psd_pre_db, aspect='auto', cmap='viridis',
                         extent=extent, origin='upper',
                         vmin=vmin_db, vmax=vmax_db)
    axes[0].set_title('Baseline')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Layer')
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(layer_labels_display)
    plt.colorbar(im1, ax=axes[0], label='Power (dB)')

    # ---- Post ----
    im2 = axes[1].imshow(psd_post_db, aspect='auto', cmap='viridis',
                         extent=extent, origin='upper',
                         vmin=vmin_db, vmax=vmax_db)
    axes[1].set_title('Post-stimulus')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Layer')
    axes[1].set_yticks(yticks)
    axes[1].set_yticklabels(layer_labels_display)
    plt.colorbar(im2, ax=axes[1], label='Power (dB)')

    # ---- Percent Change ----
    vmin_pct = -100
    vmax_pct = 500
    norm = TwoSlopeNorm(vmin=vmin_pct, vcenter=0, vmax=vmax_pct)

    im3 = axes[2].imshow(pct_change, aspect='auto', cmap='RdBu_r',
                         extent=extent, origin='upper', norm=norm)
    axes[2].set_title('Stimulus-induced change (%)')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Layer')
    axes[2].set_yticks(yticks)
    axes[2].set_yticklabels(layer_labels_display)

    cbar = plt.colorbar(im3, ax=axes[2], label='% Change')
    cbar.set_ticks([-100, -50, 0, 100, 250, 400, 500])

    if log_freq:
        for ax in axes:
            ax.set_xscale('log')

    plt.suptitle('Laminar Spectral Profile — Mazzoni per-layer LFP', fontsize=16)
    plt.show()

    return f_plot, layer_names, psd_pre_db, psd_post_db, pct_change


# Run the analysis — kernel bipolar LFP
f_plot, depths, psd_pre, psd_post, psd_change = plot_laminar_spectral_profile(
    all_trials,
    pre_window_ms=300,
    post_window_ms=300,
    post_start_ms=300,
    freq_range=(1, 120),
    log_freq=False,
    remove_mean=True,
    do_detrend=True,
)

# Run the analysis — Mazzoni per-layer LFP (only if data available)
if 'mazzoni_lfp_matrix' in all_trials[0]:
    f_plot_m, layers_m, psd_pre_m, psd_post_m, psd_change_m = plot_mazzoni_laminar_spectral_profile(
        all_trials,
        pre_window_ms=300,
        post_window_ms=300,
        post_start_ms=500,
        freq_range=(1, 120),
        log_freq=False,
        remove_mean=True,
        do_detrend=True,
    )
else:
    print("No Mazzoni per-layer LFP data found in trials — skipping Mazzoni plot.")
