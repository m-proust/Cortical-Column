
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import signal
from scipy.signal import detrend
from scipy.signal.windows import dpss
import seaborn as sns


plt.rcParams.update({
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
})
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Paired')


def _load_trial(fname):
    data = np.load(fname, allow_pickle=True)
    out = {
        'time': data['time_array_ms'],
        'bipolar_lfp': data['bipolar_matrix'],
        'lfp_matrix': data['lfp_matrix'],
        'rate_data': (data['rate_data'].item()
                      if data['rate_data'].size == 1
                      else data['rate_data']),
        'spike_data': (data['spike_data'].item()
                       if data['spike_data'].size == 1
                       else data['spike_data']),
        'baseline_ms': float(data['baseline_ms']),
        'stim_onset_ms': float(data['stim_onset_ms']),
        'channel_labels': data['channel_labels'],
        'channel_depths': data['channel_depths'],
        'electrode_positions': data['electrode_positions'],
        'silenced': bool(data['silenced']),
        'condition_name': str(data['condition_name']),
        'trial_index': int(data['trial_index']) if 'trial_index' in data.files else 0,
    }
    if 'lfp_current_matrix' in data.files:
        out['lfp_current_matrix'] = data['lfp_current_matrix']
        out['time_current_ms'] = data['time_current_ms']
    return out


def list_conditions(base_path):
    """Return sorted list of condition names found under base_path."""
    sub = sorted(d for d in os.listdir(base_path)
                 if os.path.isdir(os.path.join(base_path, d)))
    return [d for d in sub if d != 'config_snapshot']


def load_condition(base_path, condition_name):
    """
    Load all trials for a condition, split into control / silenced lists
    sorted by trial_index so they are paired.
    """
    cond_dir = os.path.join(base_path, condition_name)
    ctrl_files = sorted(glob.glob(
        os.path.join(cond_dir, f"{condition_name}_trial*_control.npz")))
    sil_files = sorted(glob.glob(
        os.path.join(cond_dir, f"{condition_name}_trial*_silenced.npz")))

    control_trials = [_load_trial(f) for f in ctrl_files]
    silenced_trials = [_load_trial(f) for f in sil_files]

    return control_trials, silenced_trials


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------

def multitaper_psd(data, fs, NW=2, nfft=None):
    data_demeaned = data - np.mean(data)
    if nfft is None:
        nfft = 2 ** int(np.ceil(np.log2(len(data_demeaned))))
    K = int(2 * NW - 1)
    tapers = dpss(len(data_demeaned), NW, K)
    psds = []
    for taper in tapers:
        freqs, psd_single = signal.periodogram(
            data_demeaned * taper, fs=fs, nfft=nfft, scaling='density',
        )
        psds.append(psd_single)
    return freqs, np.mean(psds, axis=0)


def _time_vector_for_key(trial, lfp_key):
    if lfp_key in ('lfp_current_matrix', 'bipolar_lfp_current'):
        if 'time_current_ms' not in trial:
            raise KeyError("trial has no 'time_current_ms'")
        return np.asarray(trial['time_current_ms'])
    return np.asarray(trial['time'])


def _pre_post_psd_per_channel(trials,
                              lfp_key,
                              pre_window_ms,
                              post_window_ms,
                              post_start_ms,
                              do_detrend,
                              remove_mean):
    """
    Within-trial pre vs post PSD, averaged across trials per channel.

    For each silenced trial, the spectrum is computed twice:
      pre:  [stim - pre_window_ms,        stim)
      post: [stim + post_start_ms,        stim + post_start_ms + post_window_ms)
    where `stim` is the silencing-onset time (`stim_onset_ms` in the npz).

    Returns (freqs, psd_pre[n_ch, n_freq], psd_post[n_ch, n_freq]).
    """
    t0 = _time_vector_for_key(trials[0], lfp_key)
    fs = 1000.0 / float(np.mean(np.diff(t0)))
    n_channels = trials[0][lfp_key].shape[0]

    f = None
    per_ch_pre, per_ch_post = [], []
    for ch in range(n_channels):
        pre_psds, post_psds = [], []
        for trial in trials:
            lfp = trial[lfp_key][ch]
            time = _time_vector_for_key(trial, lfp_key)
            stim = trial['stim_onset_ms']

            pre_mask = (time >= stim - pre_window_ms) & (time < stim)
            post_mask = ((time >= stim + post_start_ms) &
                         (time < stim + post_start_ms + post_window_ms))

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

            nfft = 2 ** int(np.ceil(np.log2(min(len(pre), len(post)))))
            f, psd_pre = multitaper_psd(pre, fs=fs, NW=2, nfft=nfft)
            _, psd_post = multitaper_psd(post, fs=fs, NW=2, nfft=nfft)

            pre_psds.append(psd_pre)
            post_psds.append(psd_post)

        if pre_psds:
            per_ch_pre.append(np.mean(pre_psds, axis=0))
            per_ch_post.append(np.mean(post_psds, axis=0))
        else:
            per_ch_pre.append(None)
            per_ch_post.append(None)

    if f is None:
        raise RuntimeError('zero valid trials in any channel')

    n_freq = len(f)
    per_ch_pre = [p if p is not None else np.full(n_freq, np.nan)
                  for p in per_ch_pre]
    per_ch_post = [p if p is not None else np.full(n_freq, np.nan)
                   for p in per_ch_post]
    return f, np.array(per_ch_pre), np.array(per_ch_post)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_silence_spectral_profile(silenced_trials,
                                  condition_name,
                                  pre_window_ms=500,
                                  post_window_ms=500,
                                  post_start_ms=200,
                                  freq_range=(1, 100),
                                  log_freq=False,
                                  remove_mean=True,
                                  do_detrend=True,
                                  lfp_key='bipolar_lfp',
                                  title_suffix=None,
                                  save_path=None):
    if not silenced_trials:
        print(f"  {condition_name}: no silenced trials, skip")
        return None

    f, psd_pre, psd_post = _pre_post_psd_per_channel(
        silenced_trials, lfp_key,
        pre_window_ms=pre_window_ms,
        post_window_ms=post_window_ms,
        post_start_ms=post_start_ms,
        do_detrend=do_detrend,
        remove_mean=remove_mean,
    )

    n_channels = psd_pre.shape[0]
    channel_depths = silenced_trials[0]['channel_depths']
    if lfp_key == 'bipolar_lfp' and len(channel_depths) > n_channels:
        depths = (channel_depths[:-1] + channel_depths[1:]) / 2
    else:
        depths = channel_depths[:n_channels]

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
    axes[0].set_title('Pre-silencing')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Laminar depth')
    plt.colorbar(im1, ax=axes[0], label='Power (dB)')

    im2 = axes[1].imshow(psd_post_db, aspect='auto', cmap='viridis',
                         extent=extent, origin='upper',
                         vmin=vmin_db, vmax=vmax_db)
    axes[1].set_title('Post-silencing')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Laminar depth')
    plt.colorbar(im2, ax=axes[1], label='Power (dB)')

    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
    im3 = axes[2].imshow(pct_change, aspect='auto', cmap='RdBu_r',
                         extent=extent, origin='upper', norm=norm)
    axes[2].set_title('Silencing-induced change (%)')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Laminar depth')
    cbar = plt.colorbar(im3, ax=axes[2], label='% Change')
    cbar.set_ticks([-100, -50, 0, 100])

    if log_freq:
        for ax in axes:
            ax.set_xscale('log')

    if title_suffix is None:
        title_suffix = {
            'bipolar_lfp': ' (Bipolar)',
            'lfp_matrix': ' (Kernel method)',
            'lfp_current_matrix': ' (Synaptic-current method)',
        }.get(lfp_key, f' ({lfp_key})')

    plt.suptitle(f'{condition_name} — Laminar Spectral Profile{title_suffix}',
                 fontsize=16)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'  saved {save_path}')
        plt.close(fig)
    else:
        plt.show()

    return f_plot, depths, psd_pre_db, psd_post_db, pct_change


# ---------------------------------------------------------------------------
# Raster
# ---------------------------------------------------------------------------

LAYERS_ORDER = ["L23", "L4AB", "L4C", "L5", "L6"]
POP_COLORS = {
    'E':   '#2b6cb0',
    'PV':  '#c53030',
    'SOM': '#2f855a',
    'VIP': '#b7791f',
}


def plot_silence_raster(trial,
                        condition_name,
                        window_ms=(None, None),
                        save_path=None):
    """
    Raster of all spikes from one silenced trial. Rows are stacked by
    layer (top to bottom L23..L6), within each layer pops E, PV, SOM, VIP
    are stacked. A vertical line marks silencing onset.
    """
    spike_data = trial['spike_data']
    stim = trial['stim_onset_ms']

    t_lo = window_ms[0] if window_ms[0] is not None else 0.0
    t_hi = window_ms[1] if window_ms[1] is not None else (
        trial['baseline_ms'] + trial.get('post_ms', 0)
        if trial.get('post_ms', 0) else float(trial['time'][-1])
    )

    fig, ax = plt.subplots(figsize=(14, 9))
    y_offset = 0
    yticks, ytick_labels = [], []

    for layer in LAYERS_ORDER:
        if layer not in spike_data:
            continue
        layer_spikes = spike_data[layer]
        for pop in ['E', 'PV', 'SOM', 'VIP']:
            key = f'{pop}_spikes'
            if key not in layer_spikes:
                continue
            sp = layer_spikes[key]
            t = np.asarray(sp['times_ms'])
            i = np.asarray(sp['spike_indices'])
            mask = (t >= t_lo) & (t <= t_hi)
            t = t[mask]
            i = i[mask]
            if len(t):
                n_neurons = int(i.max()) + 1 if len(i) else 1
            else:
                n_neurons = 1
            ax.scatter(t, i + y_offset, s=1.5,
                       color=POP_COLORS.get(pop, 'k'),
                       alpha=0.7, linewidths=0)
            yticks.append(y_offset + n_neurons / 2)
            ytick_labels.append(f'{layer} {pop}')
            y_offset += n_neurons + 5

    ax.axvline(stim, color='k', linestyle='--', linewidth=1,
               label='silencing onset')
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(0, y_offset)
    ax.invert_yaxis()
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=8)
    ax.set_xlabel('Time (ms)')
    ax.set_title(f'{condition_name} — raster')
    ax.legend(loc='upper right', fontsize=8)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'  saved {save_path}')
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Time-frequency
# ---------------------------------------------------------------------------

def plot_silence_spectrogram(trial,
                             condition_name,
                             lfp_key='bipolar_lfp',
                             freq_range=(1, 120),
                             nperseg_ms=200,
                             noverlap_frac=0.9,
                             window_ms=(None, None),
                             save_path=None):
    """
    Per-channel spectrogram (depth = rows of subplots) for one trial.
    Power in dB; vertical line marks silencing onset.
    """
    lfp = trial[lfp_key]
    time = _time_vector_for_key(trial, lfp_key)
    fs = 1000.0 / float(np.mean(np.diff(time)))
    stim = trial['stim_onset_ms']

    t_lo = window_ms[0] if window_ms[0] is not None else float(time[0])
    t_hi = window_ms[1] if window_ms[1] is not None else float(time[-1])
    tmask = (time >= t_lo) & (time <= t_hi)

    n_channels = lfp.shape[0]
    channel_depths = trial['channel_depths']
    if lfp_key == 'bipolar_lfp' and len(channel_depths) > n_channels:
        depth_labels = [
            f'{(channel_depths[i] + channel_depths[i + 1]) / 2:.0f}'
            for i in range(n_channels)
        ]
    else:
        depth_labels = [f'{channel_depths[i]:.0f}' for i in range(n_channels)]

    nperseg = int(nperseg_ms * 1e-3 * fs)
    noverlap = int(nperseg * noverlap_frac)

    fig, axes = plt.subplots(n_channels, 1,
                             figsize=(12, 1.5 * n_channels),
                             sharex=True)
    if n_channels == 1:
        axes = [axes]

    # Compute once per channel, share color scale across channels.
    Sxx_all = []
    f_arr = t_arr = None
    for ch in range(n_channels):
        x = lfp[ch][tmask]
        x = x - np.mean(x)
        f_arr, t_arr, Sxx = signal.spectrogram(
            x, fs=fs, nperseg=nperseg, noverlap=noverlap,
            scaling='density',
        )
        Sxx_all.append(Sxx)

    Sxx_db_all = [10 * np.log10(s + 1e-12) for s in Sxx_all]
    fmask = (f_arr >= freq_range[0]) & (f_arr <= freq_range[1])
    f_plot = f_arr[fmask]
    flat = np.concatenate([s[fmask].ravel() for s in Sxx_db_all])
    vmin, vmax = np.percentile(flat, [5, 95])

    t_abs = t_arr * 1000.0 + t_lo  # spectrogram time is relative to start

    for ch in range(n_channels):
        ax = axes[ch]
        im = ax.pcolormesh(
            t_abs, f_plot, Sxx_db_all[ch][fmask],
            cmap='viridis', vmin=vmin, vmax=vmax, shading='auto',
        )
        ax.axvline(stim, color='w', linestyle='--', linewidth=1)
        ax.set_ylabel(f'd={depth_labels[ch]}\nHz', fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel('Time (ms)')
    fig.suptitle(f'{condition_name} — bipolar spectrogram', fontsize=14)
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Power (dB)')

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'  saved {save_path}')
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    base_path = 'results/trials_18_05'
    fig_dir = os.path.join(base_path, 'figures_laminar')
    raster_dir = os.path.join(base_path, 'figures_raster')
    tf_dir = os.path.join(base_path, 'figures_spectrogram')

    common = dict(
        pre_window_ms=500,
        post_window_ms=500,
        post_start_ms=200,
        freq_range=(0, 120),
        log_freq=False,
        remove_mean=True,
        do_detrend=True,
    )

    conditions = list_conditions(base_path)
    print(f'Found {len(conditions)} conditions in {base_path}')

    for cond in conditions:
        print(f'\n=== {cond} ===')
        _, sil = load_condition(base_path, cond)
        if not sil:
            print(f'  skip (no silenced trials)')
            continue
        print(f'  silenced trials: {len(sil)}')

        # Bipolar PSD only.
        save_path = os.path.join(fig_dir, f'{cond}_bipolar.png')
        plot_silence_spectral_profile(
            sil, cond,
            lfp_key='bipolar_lfp',
            save_path=save_path,
            **common,
        )

        # Raster + spectrogram from a single representative trial (idx 0).
        # Window: 500 ms before silencing through 1500 ms after, so the
        # transition is centered.
        rep = sil[0]
        stim = rep['stim_onset_ms']
        win = (stim - 500.0, stim + 1500.0)

        plot_silence_raster(
            rep, cond,
            window_ms=win,
            save_path=os.path.join(raster_dir, f'{cond}_raster.png'),
        )

        plot_silence_spectrogram(
            rep, cond,
            lfp_key='bipolar_lfp',
            freq_range=(1, 120),
            nperseg_ms=200,
            noverlap_frac=0.9,
            window_ms=win,
            save_path=os.path.join(tf_dir, f'{cond}_spectrogram.png'),
        )
