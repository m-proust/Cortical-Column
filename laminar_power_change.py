import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import signal
from scipy.signal import detrend
from scipy.signal.windows import dpss
from scipy.ndimage import gaussian_filter1d
import seaborn as sns


plt.rcParams.update({
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
})
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Paired')


SMOOTH_HZ = 1.5      # gaussian sigma along freq axis (Hz). 0 disables.
SMOOTH_DEPTH = 0.6   # gaussian sigma along depth axis (channels). 0 disables.
NW_TAPER = 3         # multitaper time-bandwidth (was 2 -> smoother now).


def _smooth_psd(psd, freqs, sigma_hz=SMOOTH_HZ, sigma_depth=SMOOTH_DEPTH):
    """Light 2D smoothing of a (channels x freq) PSD image.

    Smooths along freq (in Hz) and a tiny bit along depth (in channels)
    to kill horizontal striping while preserving spectral peaks.
    """
    if psd is None:
        return psd
    out = psd.astype(float, copy=True)
    if sigma_hz and sigma_hz > 0:
        df = float(np.median(np.diff(freqs)))
        if df > 0:
            sigma_bins = sigma_hz / df
            if sigma_bins >= 0.5:
                out = gaussian_filter1d(out, sigma=sigma_bins,
                                        axis=-1, mode='nearest')
    if sigma_depth and sigma_depth > 0:
        out = gaussian_filter1d(out, sigma=sigma_depth,
                                axis=0, mode='nearest')
    return out


def load_trials(base_path, n_trials):
    all_trials = []
    for trial_idx in range(n_trials):
        fname = f"{base_path}/trial_{trial_idx:03d}.npz"
        data = np.load(fname, allow_pickle=True)

        trial_data = {
            'time': data['time_array_ms'],
            'bipolar_lfp': data['bipolar_matrix'],
            'lfp_matrix': data['lfp_matrix'],
            'rate_data': (data['rate_data'].item()
                          if data['rate_data'].size == 1
                          else data['rate_data']),
            'baseline_ms': float(data['baseline_ms']),
            'stim_onset_ms': float(data['stim_onset_ms']),
            'channel_labels': data['channel_labels'],
            'channel_depths': data['channel_depths'],
            'electrode_positions': data['electrode_positions'],
        }

     
        if 'lfp_current_matrix' in data.files:
            trial_data['lfp_current_matrix'] = data['lfp_current_matrix']
            trial_data['time_current_ms'] = data['time_current_ms']

        if 'mazzoni_lfp_matrix' in data.files:
            trial_data['mazzoni_lfp_matrix'] = data['mazzoni_lfp_matrix']
            trial_data['mazzoni_time_ms'] = data['mazzoni_time_ms']
            trial_data['mazzoni_layer_names'] = data['mazzoni_layer_names']

        all_trials.append(trial_data)
    return all_trials

def multitaper_psd(data, fs, NW=NW_TAPER, nfft=None):
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
            raise KeyError(
                "trial has no 'time_current_ms'"
            )
        return np.asarray(trial['time_current_ms'])
    return np.asarray(trial['time'])

def plot_laminar_spectral_profile(all_trials,
                                  pre_window_ms=1000,
                                  post_window_ms=1000,
                                  post_start_ms=500,
                                  freq_range=(1, 100),
                                  log_freq=True,
                                  remove_mean=True,
                                  do_detrend=True,
                                  lfp_key='bipolar_lfp',
                                  title_suffix=None,
                                  save_path=None,
                                  show=True,
                                  smooth=True):

    for i, tr in enumerate(all_trials):
        if lfp_key not in tr:
            raise KeyError(
                f"Trial {i} has no key '{lfp_key}'.  the available keys are  "
                f"{[k for k in tr.keys() if 'lfp' in k or 'matrix' in k]}"
            )

    t0 = _time_vector_for_key(all_trials[0], lfp_key)
    fs = 1000.0 / float(np.mean(np.diff(t0)))

    n_channels = all_trials[0][lfp_key].shape[0]
    channel_depths = all_trials[0]['channel_depths']

    if lfp_key == 'bipolar_lfp' and len(channel_depths) > n_channels:
        depths = (channel_depths[:-1] + channel_depths[1:]) / 2
    else:
        depths = channel_depths[:n_channels]

    all_psd_pre, all_psd_post = [], []
    f = None

    for ch in range(n_channels):
        pre_trials, post_trials = [], []
        for trial in all_trials:
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
            f, psd_pre = multitaper_psd(pre, fs=fs, NW=NW_TAPER, nfft=nfft)
            _, psd_post = multitaper_psd(post, fs=fs, NW=NW_TAPER, nfft=nfft)

            pre_trials.append(psd_pre)
            post_trials.append(psd_post)

        if len(pre_trials) == 0:
            print(f"  channel {ch} skipped (no valid trials)")
            all_psd_pre.append(None)
            all_psd_post.append(None)
        else:
            all_psd_pre.append(np.mean(pre_trials, axis=0))
            all_psd_post.append(np.mean(post_trials, axis=0))

    if f is None:
        raise RuntimeError('zero valid trials in any channel')

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

    if smooth:
        psd_pre = _smooth_psd(psd_pre, f_plot)
        psd_post = _smooth_psd(psd_post, f_plot)

    psd_pre_db = 10 * np.log10(psd_pre + 1e-10)
    psd_post_db = 10 * np.log10(psd_post + 1e-10)
    pct_change = (psd_post - psd_pre) / psd_pre * 100
    if smooth:
        pct_change = _smooth_psd(pct_change, f_plot,
                                 sigma_hz=SMOOTH_HZ, sigma_depth=SMOOTH_DEPTH)

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
    axes[0].set_ylabel('Laminar depth')
    plt.colorbar(im1, ax=axes[0], label='Power (dB)')

    im2 = axes[1].imshow(psd_post_db, aspect='auto', cmap='viridis',
                         extent=extent, origin='upper',
                         vmin=vmin_db, vmax=vmax_db)
    axes[1].set_title('Post-stimulus')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Laminar depth')
    plt.colorbar(im2, ax=axes[1], label='Power (dB)')

    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
    im3 = axes[2].imshow(pct_change, aspect='auto', cmap='RdBu_r',
                         extent=extent, origin='upper', norm=norm)
    axes[2].set_title('Stimulus-induced change (%)')
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

    plt.suptitle('Laminar Spectral Profile' + title_suffix, fontsize=16)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches='tight')
        print(f"  saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return f_plot, depths, psd_pre_db, psd_post_db, pct_change


def _count_trials(sweep_dir):
    return len(sorted(glob.glob(os.path.join(sweep_dir, "trial_*.npz"))))


def _process_sweep(sweep_dir, fig_root, common):
    name = os.path.basename(sweep_dir.rstrip('/'))
    n_trials = _count_trials(sweep_dir)
    if n_trials == 0:
        print(f"[{name}] no trials, skip")
        return

    print(f"\n=== {name}  ({n_trials} trials) ===")
    all_trials = load_trials(sweep_dir, n_trials)
    out_dir = os.path.join(fig_root, name)

    plot_laminar_spectral_profile(
        all_trials, lfp_key='bipolar_lfp',
        save_path=os.path.join(out_dir, f"{name}_bipolar.png"),
        show=False, smooth=True, **common)

    plot_laminar_spectral_profile(
        all_trials, lfp_key='lfp_matrix',
        save_path=os.path.join(out_dir, f"{name}_kernel.png"),
        show=False, smooth=True, **common)

    if 'lfp_current_matrix' in all_trials[0]:
        plot_laminar_spectral_profile(
            all_trials, lfp_key='lfp_current_matrix',
            save_path=os.path.join(out_dir, f"{name}_current.png"),
            show=False, smooth=True, **common)
        for tr in all_trials:
            tr['bipolar_lfp_current'] = np.diff(tr['lfp_current_matrix'], axis=0)
        plot_laminar_spectral_profile(
            all_trials, lfp_key='bipolar_lfp_current',
            title_suffix=' (Bipolar synaptic current)',
            save_path=os.path.join(out_dir, f"{name}_bipolar_current.png"),
            show=False, smooth=True, **common)


if __name__ == '__main__':
    SWEEP_ROOT = 'results/trials2_pop_sweep'
    FIG_ROOT = os.path.join(SWEEP_ROOT, 'figures_laminar')

    common = dict(
        pre_window_ms=500,
        post_window_ms=500,
        post_start_ms=200,
        freq_range=(0, 120),
        log_freq=False,
        remove_mean=True,
        do_detrend=True,
    )

    sweep_dirs = sorted(
        d for d in glob.glob(os.path.join(SWEEP_ROOT, '*'))
        if os.path.isdir(d) and os.path.basename(d) != 'config_snapshot'
        and not os.path.basename(d).startswith('figures')
    )
    print(f"Found {len(sweep_dirs)} sweep dirs under {SWEEP_ROOT}")
    for sd in sweep_dirs:
        _process_sweep(sd, FIG_ROOT, common)
    print(f"\nAll figures saved under {FIG_ROOT}")