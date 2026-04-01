import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
from scipy.signal.windows import dpss
from scipy.signal import detrend, welch
import seaborn as sns


plt.rcParams.update({
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral'
})
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Paired")

base_path = "results/trials_01_04" # change here the path to your saved trials



n_trials = 30
all_trials = []

for trial_idx in range(n_trials):
    fname = f"{base_path}/trial_{trial_idx:03d}.npz"
    data = np.load(fname, allow_pickle=True)

    trial_data = {
        'seed' : data["seed"],
        'time': data["time_array_ms"],
        'bipolar_lfp': data["bipolar_matrix"],
        'lfp_matrix': data["lfp_matrix"],
        'rate_data': data["rate_data"].item() if data["rate_data"].size == 1 else data["rate_data"],
        # 'spike_data': data["spike_data"].item() if data["spike_data"].size == 1 else data["spike_data"],
        'baseline_ms': float(data["baseline_ms"]),
        'stim_onset_ms': float(data["stim_onset_ms"]),
        'channel_labels': data["channel_labels"],
        'channel_depths': data["channel_depths"],
        'electrode_positions': data["electrode_positions"]
    }

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



def pmtm(x, NW=2, nfft=None, fs=1.0):
   
    N = len(x)

    if nfft is None:
        nfft = 2 ** int(np.ceil(np.log2(N)))

    K = int(2 * NW - 1)

    tapers, ratios = dpss(N, NW, K, return_ratios=True)

    psd_mt = np.zeros(nfft)

    for k in range(K):
        x_tapered = x * tapers[k]
        X = np.fft.fft(x_tapered, n=nfft)
        psd_mt += ratios[k] * np.abs(X) ** 2


    psd_mt /= np.sum(ratios)
    psd_mt /= fs 

    f = np.fft.fftfreq(nfft, d=1/fs)
    pos_mask = f >= 0

    return f[pos_mask], psd_mt[pos_mask]

def plot_laminar_spectral_profile(all_trials, pre_window_ms=1000, post_window_ms=1000,
                                 post_start_ms=500,
                                 freq_range=(1, 100), log_freq=True, remove_mean=True,
                                 do_detrend=True, fmax=100):



    fs = 10000 
    n_channels = all_trials[0]['bipolar_lfp'].shape[0]
    n_trials = len(all_trials)

    channel_depths = all_trials[0]['channel_depths']

    if len(channel_depths) > n_channels:
        bipolar_depths = (channel_depths[:-1] + channel_depths[1:]) / 2
    else:
        bipolar_depths = channel_depths[:n_channels]

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
            print(f"  WARNING: channel {ch} skipped")
            all_psd_pre.append(None)
            all_psd_post.append(None)
        else:
            all_psd_pre.append(np.mean(pre_trials, axis=0))
            all_psd_post.append(np.mean(post_trials, axis=0))

    if f is None:
        raise RuntimeError("No valid trials in any channel")

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

    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.15], wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    extent = [f_plot[0], f_plot[-1], -0.5, n_channels - 0.5]

    vmin_db = np.percentile([psd_pre_db, psd_post_db], 5)
    vmax_db = np.percentile([psd_pre_db, psd_post_db], 95)

    im1 = axes[0].imshow(psd_pre_db, aspect='auto', cmap='viridis',
                         extent=extent, origin='upper',
                         vmin=vmin_db, vmax=vmax_db)
    axes[0].set_title('Baseline')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Channel (deep → superficial)')
    plt.colorbar(im1, ax=axes[0], label='Power (dB)')

    im2 = axes[1].imshow(psd_post_db, aspect='auto', cmap='viridis',
                         extent=extent, origin='upper',
                         vmin=vmin_db, vmax=vmax_db)
    axes[1].set_title('Post-stimulus')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Channel (deep → superficial)')
    plt.colorbar(im2, ax=axes[1], label='Power (dB)')

    vmin_pct = -100
    vmax_pct = 700
    norm = TwoSlopeNorm(vmin=vmin_pct, vcenter=0, vmax=vmax_pct)


    im3 = axes[2].imshow(pct_change, aspect='auto', cmap='RdBu_r',
                         extent=extent, origin='upper', norm=norm)
    axes[2].set_title('Stimulus-induced change (%)')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Channel')

    cbar = plt.colorbar(im3, ax=axes[2], label='% Change')
    cbar.set_ticks([-100, -50, 0, 100, 250, 400, 500, 600, 700])

    if log_freq:
        for ax in axes:
            ax.set_xscale('log')

    plt.suptitle('Laminar Spectral Profile', fontsize=16)
    plt.show()

    return f_plot, bipolar_depths, psd_pre_db, psd_post_db, pct_change



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
