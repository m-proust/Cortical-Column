import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import detrend, welch
import seaborn as sns
import os

plt.rcParams.update({
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral'
})
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Paired")

base_path = "results/trials3_07_04"
output_dir = "results/trials3_07_04/individual_plots"
os.makedirs(output_dir, exist_ok=True)

n_trials = len([f for f in os.listdir(base_path) if f.startswith("trial_") and f.endswith(".npz")])
print(f"Found {n_trials} trials")


def load_trial(trial_idx):
    fname = f"{base_path}/trial_{trial_idx:03d}.npz"
    data = np.load(fname, allow_pickle=True)
    trial_data = {
        'time': data["time_array_ms"],
        'bipolar_lfp': data["bipolar_matrix"],
        'lfp_matrix': data["lfp_matrix"],
        'rate_data': data["rate_data"].item() if data["rate_data"].size == 1 else data["rate_data"],
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
    if "stim_rates" in data:
        trial_data['stim_rates'] = data["stim_rates"].item()
    return trial_data


def plot_laminar_spectral_profile_single(trial_data, trial_idx,
                                         pre_window_ms=300, post_window_ms=300,
                                         post_start_ms=200,
                                         freq_range=(0, 120), log_freq=False,
                                         remove_mean=True, do_detrend=True,
                                         lfp_key='bipolar_lfp'):

    fs = 10000
    n_channels = trial_data[lfp_key].shape[0]
    channel_depths = trial_data['channel_depths']

    if lfp_key == 'bipolar_lfp' and len(channel_depths) > n_channels:
        bipolar_depths = (channel_depths[:-1] + channel_depths[1:]) / 2
    else:
        bipolar_depths = channel_depths[:n_channels]

    psd_pre_list, psd_post_list = [], []
    f = None

    for ch in range(n_channels):
        lfp = trial_data[lfp_key][ch]
        time = trial_data['time']
        stim = trial_data['stim_onset_ms']

        pre_mask = (time >= stim - pre_window_ms) & (time < stim)
        post_mask = (time >= stim + post_start_ms) & \
                    (time < stim + post_start_ms + post_window_ms)

        pre = lfp[pre_mask].copy()
        post = lfp[post_mask].copy()

        if len(pre) == 0 or len(post) == 0:
            psd_pre_list.append(None)
            psd_post_list.append(None)
            continue
        if np.any(np.isnan(pre)) or np.any(np.isnan(post)):
            psd_pre_list.append(None)
            psd_post_list.append(None)
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

        psd_pre_list.append(psd_pre)
        psd_post_list.append(psd_post)

    if f is None:
        print(f"  Trial {trial_idx}: no valid data, skipping")
        return

    n_freq = len(f)
    for i in range(n_channels):
        if psd_pre_list[i] is None:
            psd_pre_list[i] = np.full(n_freq, np.nan)
            psd_post_list[i] = np.full(n_freq, np.nan)

    psd_pre_arr = np.array(psd_pre_list)
    psd_post_arr = np.array(psd_post_list)

    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    f_plot = f[freq_mask]

    psd_pre_arr = psd_pre_arr[:, freq_mask]
    psd_post_arr = psd_post_arr[:, freq_mask]

    psd_pre_db = 10 * np.log10(psd_pre_arr + 1e-10)
    psd_post_db = 10 * np.log10(psd_post_arr + 1e-10)
    pct_change = (psd_post_arr - psd_pre_arr) / psd_pre_arr * 100

    psd_pre_db = np.flipud(psd_pre_db)
    psd_post_db = np.flipud(psd_post_db)
    pct_change = np.flipud(pct_change)

    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.15], wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    extent = [f_plot[0], f_plot[-1], -0.5, n_channels - 0.5]

    vmin_db = np.nanpercentile([psd_pre_db, psd_post_db], 5)
    vmax_db = np.nanpercentile([psd_pre_db, psd_post_db], 95)

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
    vmax_pct = 500
    norm = TwoSlopeNorm(vmin=vmin_pct, vcenter=0, vmax=vmax_pct)

    im3 = axes[2].imshow(pct_change, aspect='auto', cmap='RdBu_r',
                         extent=extent, origin='upper', norm=norm)
    axes[2].set_title('Stimulus-induced change (%)')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Channel')

    cbar = plt.colorbar(im3, ax=axes[2], label='% Change')
    cbar.set_ticks([-100, -50, 0, 100, 250])

    if log_freq:
        for ax in axes:
            ax.set_xscale('log')

    suffix = ' (Bipolar)' if lfp_key == 'bipolar_lfp' else ' (LFP)'
    title = f'Trial {trial_idx:03d} — Laminar Spectral Profile{suffix}'
    if 'stim_rates' in trial_data:
        sr = trial_data['stim_rates']
        rate_str = (f"L4C E={sr['L4C_E']:.2f} Hz, PV={sr['L4C_PV']:.2f} Hz  |  "
                    f"L6 E={sr['L6_E']:.2f} Hz, PV={sr['L6_PV']:.2f} Hz")
        title += f'\n{rate_str}'
    plt.suptitle(title, fontsize=16)

    return fig


# Run for all trials, both bipolar and LFP
for trial_idx in range(n_trials):
    print(f"Processing trial {trial_idx:03d}/{n_trials-1}...")
    trial_data = load_trial(trial_idx)

    fig = plot_laminar_spectral_profile_single(trial_data, trial_idx, lfp_key='bipolar_lfp')
    if fig is not None:
        fig.savefig(f"{output_dir}/trial_{trial_idx:03d}_bipolar.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

print(f"\nDone! All plots saved to {output_dir}/")
