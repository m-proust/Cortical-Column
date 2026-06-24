"""Stimulus-induced laminar power change vs depth, from a folder of saved trials.
Produces raw and smoothed heatmaps for both bipolar LFP and bipolar current.

Run:
    path/to/your/venv/bin/python analysis/spectral/laminar_power_change.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import signal
from scipy.signal import detrend
from scipy.signal.windows import dpss
from scipy.ndimage import zoom, gaussian_filter
import seaborn as sns


plt.rcParams.update({
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
})
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Paired')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})


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
                                  per_trial_pct=False,
                                  robust_db=True,
                                  title_suffix=None):
    """Laminar pre vs post-stimulus power change.

    Change panel: robust_db -> mean per-trial 10*log10(post/pre) (symmetric,
    outlier-resistant); per_trial_pct -> mean per-trial %; else % of trial-mean PSD.
    """
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

    all_psd_pre, all_psd_post, all_pct, all_db = [], [], [], []
    f = None

    for ch in range(n_channels):
        pre_trials, post_trials, pct_trials, db_trials = [], [], [], []
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
            f, psd_pre = multitaper_psd(pre, fs=fs, NW=2, nfft=nfft)
            _, psd_post = multitaper_psd(post, fs=fs, NW=2, nfft=nfft)

            pre_trials.append(psd_pre)
            post_trials.append(psd_post)
            pct_trials.append((psd_post - psd_pre) / (psd_pre + 1e-20) * 100)
            db_trials.append(10 * np.log10((psd_post + 1e-20) /
                                           (psd_pre + 1e-20)))

        if len(pre_trials) == 0:
            print(f"  channel {ch} skipped (no valid trials)")
            all_psd_pre.append(None)
            all_psd_post.append(None)
            all_pct.append(None)
            all_db.append(None)
        else:
            all_psd_pre.append(np.mean(pre_trials, axis=0))
            all_psd_post.append(np.mean(post_trials, axis=0))
            all_pct.append(np.mean(pct_trials, axis=0))
            all_db.append(np.mean(db_trials, axis=0))

    if f is None:
        raise RuntimeError('zero valid trials in any channel')

    n_freq = len(f)
    for i in range(n_channels):
        if all_psd_pre[i] is None:
            all_psd_pre[i] = np.full(n_freq, np.nan)
            all_psd_post[i] = np.full(n_freq, np.nan)
            all_pct[i] = np.full(n_freq, np.nan)
            all_db[i] = np.full(n_freq, np.nan)

    psd_pre = np.array(all_psd_pre)
    psd_post = np.array(all_psd_post)
    pct_per_trial = np.array(all_pct)
    db_per_trial = np.array(all_db)

    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    f_plot = f[freq_mask]
    psd_pre = psd_pre[:, freq_mask]
    psd_post = psd_post[:, freq_mask]
    pct_per_trial = pct_per_trial[:, freq_mask]
    db_per_trial = db_per_trial[:, freq_mask]

    psd_pre_db = 10 * np.log10(psd_pre + 1e-10)
    psd_post_db = 10 * np.log10(psd_post + 1e-10)
    if robust_db:
        pct_change = db_per_trial
        change_unit = 'dB'
        change_method = 'mean of per-trial dB'
    elif per_trial_pct:
        pct_change = pct_per_trial
        change_unit = '%'
        change_method = 'mean of per-trial %'
    else:
        pct_change = (psd_post - psd_pre) / psd_pre * 100
        change_unit = '%'
        change_method = '% of trial-mean PSD'

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

    if change_unit == 'dB':
        vlim = max(float(np.nanpercentile(np.abs(pct_change), 98)), 1.0)
        norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    else:
        norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=500)
    im3 = axes[2].imshow(pct_change, aspect='auto', cmap='RdBu_r',
                         extent=extent, origin='upper', norm=norm)
    axes[2].set_title(f'Stimulus-induced change ({change_unit})\n{change_method}')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Laminar depth')
    cbar = plt.colorbar(im3, ax=axes[2], label=f'Change ({change_unit})')
    if change_unit == '%':
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
    plt.show()

    plot_laminar_pct_change_3d(f_plot, depths, pct_change,
                               title_suffix=title_suffix)

    return f_plot, depths, psd_pre_db, psd_post_db, pct_change


def plot_laminar_pct_change_3d(freqs, depths, pct_change,
                               upsample_depth=8, smooth_freq=4,
                               clip_percentile=99,
                               title_suffix=''):
    """3D surface of % power change (freq x depth)."""
    pct = np.asarray(pct_change)  # row 0 = top of cortex (already flipud'd)
    depth_axis = np.linspace(depths.max(), depths.min(), pct.shape[0])

    if upsample_depth and upsample_depth > 1 and pct.shape[0] > 1:
        pct_up = zoom(pct, (upsample_depth, 1), order=3)
        d_up = np.linspace(depth_axis[0], depth_axis[-1], pct_up.shape[0])
    else:
        pct_up = pct
        d_up = depth_axis

    if smooth_freq and smooth_freq > 1:
        kernel = np.ones(smooth_freq) / smooth_freq
        pct_up = np.apply_along_axis(
            lambda v: np.convolve(v, kernel, mode='same'), 1, pct_up)

    neg = pct_up[pct_up < 0]
    vmin = np.percentile(neg, 100 - clip_percentile) if neg.size else -1.0
    vmin = min(vmin, -1.0)
    pos = pct_up[pct_up > 0]
    vmax = np.percentile(pos, clip_percentile) if pos.size else 1.0
    vmax = max(vmax, 1.0)

    fig = plt.figure(figsize=(10, 7), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    F, D = np.meshgrid(freqs, d_up)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    surf = ax.plot_surface(F, D, pct_up, cmap='RdBu_r', norm=norm,
                           edgecolor='none', alpha=0.95, antialiased=True,
                           rcount=80, ccount=80)
    z_floor = float(np.nanmin(pct_up))
    z_ceil = float(np.nanmax(pct_up))
    ax.set_zlim(z_floor, z_ceil)
    ax.contour(F, D, pct_up, zdir='z', offset=z_floor,
               cmap='RdBu_r', norm=norm, levels=12)

    ax.set_xlabel('Frequency (Hz)', fontsize=9)
    ax.set_ylabel('Laminar depth', fontsize=9)
    ax.set_zlabel('% change', fontsize=9)
    ax.set_title(f'3D stimulus-induced % change  [{vmin:+.0f}, {vmax:+.0f}]%'
                 + title_suffix, fontsize=12)
    ax.view_init(elev=25, azim=-60)
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, label='% change')
    fig.tight_layout()
    plt.show()
    return fig


def plot_smooth_power_change_2d(all_trials, common, lfp_key='bipolar_lfp',
                                title='Stimulus-induced laminar power change',
                                upsample_depth=12, smooth_sigma=(0.8, 1.2),
                                clip_percentile=98,
                                outpath='figures/power_change_2d.png'):
    """Smoothed 2D heatmap of laminar % power change. smooth_sigma = (depth, freq)."""
    _orig_show = plt.show  # suppress the pop-up figures of the inner pipeline
    plt.show = lambda *a, **k: None
    try:
        f_plot, depths, _pre_db, _post_db, pct_change = (
            plot_laminar_spectral_profile(all_trials, lfp_key=lfp_key, **common))
    finally:
        plt.show = _orig_show
        plt.close('all')

    pct = np.asarray(pct_change)  # row 0 = most superficial
    depth_axis = np.linspace(depths.max(), depths.min(), pct.shape[0])
    if upsample_depth > 1 and pct.shape[0] > 1:
        pct_up = zoom(pct, (upsample_depth, 1), order=3)
        d_up = np.linspace(depth_axis[0], depth_axis[-1], pct_up.shape[0])
    else:
        pct_up, d_up = pct, depth_axis
    pct_up = gaussian_filter(pct_up, sigma=smooth_sigma, mode='nearest')

    neg, pos = pct_up[pct_up < 0], pct_up[pct_up > 0]
    vmin = min(np.percentile(neg, 100 - clip_percentile) if neg.size else -1.0, -1.0)
    vmax = max(np.percentile(pos, clip_percentile) if pos.size else 1.0, 1.0)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9, 6.5), facecolor='white')
    im = ax.imshow(pct_up, aspect='auto', cmap='RdBu_r', norm=norm,
                   extent=[f_plot[0], f_plot[-1], d_up[-1], d_up[0]],
                   origin='upper', interpolation='bilinear')
    ax.set_xticks(np.arange(10, f_plot[-1] + 1, 20))
    ax.set_xlabel('Frequency (Hz)', fontsize=13)
    ax.set_ylabel('Cortical depth (μm)', fontsize=13)
    ax.set_title(title, fontsize=15, pad=12)
    ax.tick_params(labelsize=11)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label('Power change (%)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f'saved {outpath}  (range [{vmin:+.0f}, {vmax:+.0f}]%)')


if __name__ == '__main__':
    import os
    BASE_PATH = 'saved_trials/2026-06-23_feedback'
    N_TRIALS = 20
    COMMON = dict(
        pre_window_ms=500,
        post_window_ms=500,
        post_start_ms=200,
        freq_range=(0, 120),
        log_freq=False,
        remove_mean=True,
        do_detrend=True,
        robust_db=True,
        per_trial_pct=True,
    )
    os.makedirs('figures', exist_ok=True)

    all_trials = load_trials(BASE_PATH, N_TRIALS)
    print(f'Loaded {len(all_trials)} trials')

    has_current = 'lfp_current_matrix' in all_trials[0]
    if has_current:
        for trial in all_trials:
            trial['bipolar_lfp_current'] = np.diff(
                trial['lfp_current_matrix'], axis=0)

    plot_laminar_spectral_profile(all_trials, lfp_key='bipolar_lfp', **COMMON)
    plot_smooth_power_change_2d(
        all_trials, COMMON, lfp_key='bipolar_lfp',
        title='Stimulus-induced laminar power change (bipolar LFP)',
        outpath='figures/bipolar_power_change_2d.png')

    if has_current:
        plot_laminar_spectral_profile(
            all_trials, lfp_key='bipolar_lfp_current',
            title_suffix=' (Bipolar synaptic current)', **COMMON)
        plot_smooth_power_change_2d(
            all_trials, COMMON, lfp_key='bipolar_lfp_current',
            title='Stimulus-induced laminar power change (bipolar synaptic current)',
            outpath='figures/bipolar_current_power_change_2d.png')
    else:
        print('No lfp_current_matrix in trials -- skipping synaptic-current figures')

    plt.show()