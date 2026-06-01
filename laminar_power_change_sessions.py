import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import signal
from scipy.signal import detrend
from scipy.signal.windows import dpss
from scipy.ndimage import zoom
import seaborn as sns


plt.rcParams.update({
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
})
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Paired')


SESSION_RE = re.compile(r'session_(\d+)$')
TRIAL_RE = re.compile(r'trial_(\d+)\.npz$')


def discover_sessions(base_path):
    """Return {session_id: [trial_path, ...]} for every session_XX dir found."""
    sessions = {}
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"{base_path} does not exist")

    for entry in sorted(os.listdir(base_path)):
        full = os.path.join(base_path, entry)
        if not os.path.isdir(full):
            continue
        m = SESSION_RE.match(entry)
        if not m:
            continue
        sid = int(m.group(1))
        trial_files = sorted(glob.glob(os.path.join(full, "trial_*.npz")))
        if trial_files:
            sessions[sid] = trial_files
    if not sessions:
        raise FileNotFoundError(
            f"No session_XX/trial_YYY.npz files found under {base_path}"
        )
    return sessions


def _load_trial_file(fname):
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
    return trial_data


def load_sessions(base_path):
    """Return {session_id: [trial_dict, ...]} loaded from disk."""
    paths_by_session = discover_sessions(base_path)
    sessions = {}
    for sid, paths in paths_by_session.items():
        sessions[sid] = [_load_trial_file(p) for p in paths]
    return sessions


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


def compute_trial_psds(trials, lfp_key,
                       pre_window_ms, post_window_ms, post_start_ms,
                       remove_mean=True, do_detrend=True):
    """For one trial list, return (f, psd_pre[ch,trial,freq], psd_post[...]).

    Trials with missing/NaN windows are dropped; the returned arrays already
    have the dropped trials removed (so shape[1] may be < len(trials)).
    """
    if not trials:
        raise ValueError("empty trial list")

    t0 = _time_vector_for_key(trials[0], lfp_key)
    fs = 1000.0 / float(np.mean(np.diff(t0)))
    n_channels = trials[0][lfp_key].shape[0]

    psd_pre_chans = [[] for _ in range(n_channels)]
    psd_post_chans = [[] for _ in range(n_channels)]
    f_ref = None

    for trial in trials:
        time = _time_vector_for_key(trial, lfp_key)
        stim = trial['stim_onset_ms']
        pre_mask = (time >= stim - pre_window_ms) & (time < stim)
        post_mask = ((time >= stim + post_start_ms) &
                     (time < stim + post_start_ms + post_window_ms))

        lfp_mat = trial[lfp_key]
        nfft_target = 2 ** int(np.ceil(np.log2(
            min(int(pre_mask.sum()), int(post_mask.sum()))
        )))

        bad_trial = False
        per_chan_pre = []
        per_chan_post = []
        for ch in range(n_channels):
            pre = lfp_mat[ch][pre_mask].copy()
            post = lfp_mat[ch][post_mask].copy()
            if (len(pre) == 0 or len(post) == 0 or
                    np.any(np.isnan(pre)) or np.any(np.isnan(post))):
                bad_trial = True
                break
            if do_detrend:
                pre = detrend(pre)
                post = detrend(post)
            elif remove_mean:
                pre -= np.mean(pre)
                post -= np.mean(post)
            f, psd_pre = multitaper_psd(pre, fs=fs, NW=2, nfft=nfft_target)
            _, psd_post = multitaper_psd(post, fs=fs, NW=2, nfft=nfft_target)
            per_chan_pre.append(psd_pre)
            per_chan_post.append(psd_post)
            f_ref = f

        if bad_trial:
            continue
        for ch in range(n_channels):
            psd_pre_chans[ch].append(per_chan_pre[ch])
            psd_post_chans[ch].append(per_chan_post[ch])

    if f_ref is None:
        raise RuntimeError("no valid trials")

    psd_pre = np.array([np.stack(c, axis=0) for c in psd_pre_chans])
    psd_post = np.array([np.stack(c, axis=0) for c in psd_post_chans])
    return f_ref, fs, psd_pre, psd_post


def _make_depth_axis(trial, lfp_key, n_channels):
    channel_depths = trial['channel_depths']
    if lfp_key == 'bipolar_lfp' and len(channel_depths) > n_channels:
        return (channel_depths[:-1] + channel_depths[1:]) / 2
    return channel_depths[:n_channels]


def _plot_panels(f_plot, psd_pre_db, psd_post_db, pct_change,
                 depths, n_channels, log_freq, title):
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

    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=500)
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

    plt.suptitle(title, fontsize=16)
    plt.show()


def plot_laminar_spectral_profile_sessions(sessions,
                                           pre_window_ms=1000,
                                           post_window_ms=1000,
                                           post_start_ms=500,
                                           freq_range=(1, 100),
                                           log_freq=True,
                                           remove_mean=True,
                                           do_detrend=True,
                                           lfp_key='bipolar_lfp',
                                           title_suffix=None):
    """Plot laminar spectral profiles:

    1. One full 3-panel figure for the grand average across all trials and
       sessions (the clean overview).
    2. A grid of % change heatmaps, one per session, so each individual
       'electrode penetration' can be inspected.
    """
    sids = sorted(sessions.keys())
    print(f"Found {len(sids)} sessions: "
          + ", ".join(f"s{sid}({len(sessions[sid])}t)" for sid in sids))

    per_session = {}
    for sid in sids:
        trials = sessions[sid]
        try:
            f, fs, psd_pre, psd_post = compute_trial_psds(
                trials, lfp_key,
                pre_window_ms, post_window_ms, post_start_ms,
                remove_mean=remove_mean, do_detrend=do_detrend,
            )
        except Exception as e:
            print(f"  session {sid} skipped: {e}")
            continue
        per_session[sid] = {
            'f': f, 'fs': fs,
            'psd_pre': psd_pre,
            'psd_post': psd_post,
            'depths': _make_depth_axis(trials[0], lfp_key,
                                       trials[0][lfp_key].shape[0]),
        }

    if not per_session:
        raise RuntimeError("No usable sessions")

    f = per_session[sids[0]]['f']
    n_channels = per_session[sids[0]]['psd_pre'].shape[0]
    depths = per_session[sids[0]]['depths']

    all_pre = np.concatenate([s['psd_pre'] for s in per_session.values()],
                             axis=1)
    all_post = np.concatenate([s['psd_post'] for s in per_session.values()],
                              axis=1)
    psd_pre_grand = all_pre.mean(axis=1)
    psd_post_grand = all_post.mean(axis=1)

    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    f_plot = f[freq_mask]

    if title_suffix is None:
        title_suffix = {
            'bipolar_lfp': ' (Bipolar)',
            'lfp_matrix': ' (Kernel method)',
            'lfp_current_matrix': ' (Synaptic-current method)',
        }.get(lfp_key, f' ({lfp_key})')

    pre_clip = psd_pre_grand[:, freq_mask]
    post_clip = psd_post_grand[:, freq_mask]
    pre_db_grand = np.flipud(10 * np.log10(pre_clip + 1e-10))
    post_db_grand = np.flipud(10 * np.log10(post_clip + 1e-10))
    pct_grand = np.flipud((post_clip - pre_clip) / pre_clip * 100)

    grand_title = (f"Laminar Spectral Profile — Grand average "
                   f"({all_pre.shape[1]} trials, {len(per_session)} sessions)"
                   f"{title_suffix}")
    _plot_panels(f_plot, pre_db_grand, post_db_grand, pct_grand,
                 depths, n_channels, log_freq, grand_title)
    plot_laminar_pct_change_3d(f_plot, depths, pct_grand,
                               title_suffix=f' — grand{title_suffix}')

    per_session_pct = {}
    for sid, s in per_session.items():
        s_pre = s['psd_pre'].mean(axis=1)[:, freq_mask]
        s_post = s['psd_post'].mean(axis=1)[:, freq_mask]
        per_session_pct[sid] = np.flipud((s_post - s_pre) / s_pre * 100)

    _plot_per_session_grid(f_plot, depths, per_session_pct,
                           per_session, n_channels, log_freq,
                           title=f"Per-session % change{title_suffix}")

    return {
        'f': f_plot,
        'depths': depths,
        'grand': {
            'psd_pre_db': pre_db_grand,
            'psd_post_db': post_db_grand,
            'pct_change': pct_grand,
        },
        'per_session_pct': per_session_pct,
    }


def _plot_per_session_grid(f_plot, depths, per_session_pct, per_session,
                           n_channels, log_freq, title):
    sids = sorted(per_session_pct.keys())
    n = len(sids)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2 * ncols, 2.8 * nrows),
                             squeeze=False)
    extent = [f_plot[0], f_plot[-1], -0.5, n_channels - 0.5]
    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=500)

    last_im = None
    for idx, sid in enumerate(sids):
        ax = axes[idx // ncols, idx % ncols]
        n_trials_sid = per_session[sid]['psd_pre'].shape[1]
        last_im = ax.imshow(per_session_pct[sid], aspect='auto',
                            cmap='RdBu_r', extent=extent,
                            origin='upper', norm=norm)
        ax.set_title(f's{sid} (n={n_trials_sid})', fontsize=9)
        if idx % ncols == 0:
            ax.set_ylabel('depth')
        if idx // ncols == nrows - 1:
            ax.set_xlabel('Hz')
        if log_freq:
            ax.set_xscale('log')

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis('off')

    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
    fig.colorbar(last_im, cax=cbar_ax, label='% change')
    plt.show()


def plot_laminar_pct_change_3d(freqs, depths, pct_change,
                               upsample_depth=8, smooth_freq=4,
                               clip_percentile=99,
                               title_suffix=''):
    pct = np.asarray(pct_change)
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

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    F, D = np.meshgrid(freqs, d_up)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    surf = ax.plot_surface(F, D, pct_up, cmap='RdBu_r', norm=norm,
                           edgecolor='none', alpha=0.95, antialiased=True,
                           rcount=80, ccount=80)
    ax.contour(F, D, pct_up, zdir='z',
               offset=np.nanmin(pct_up) - 20,
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


if __name__ == '__main__':
    base_path = 'results/trials4_19_05'

    sessions = load_sessions(base_path)
    n_total = sum(len(v) for v in sessions.values())
    first_trial = next(iter(sessions.values()))[0]

    print(f"Loaded {len(sessions)} sessions, {n_total} trials total")
    print(f"Time range: {first_trial['time'][0]:.1f} to "
          f"{first_trial['time'][-1]:.1f} ms")
    print(f"Stimulus onset: {first_trial['stim_onset_ms']:.1f} ms")
    print(f"Number of bipolar channels: {first_trial['bipolar_lfp'].shape[0]}")
    print(f"Sampling rate (kernel): "
          f"~{1000 / np.mean(np.diff(first_trial['time'])):.0f} Hz")
    if 'time_current_ms' in first_trial:
        print(f"Sampling rate (current): "
              f"~{1000 / np.mean(np.diff(first_trial['time_current_ms'])):.0f} Hz")

    common = dict(
        pre_window_ms=500,
        post_window_ms=500,
        post_start_ms=500,
        freq_range=(0, 120),
        log_freq=False,
        remove_mean=True,
        do_detrend=True,
    )

    plot_laminar_spectral_profile_sessions(sessions, lfp_key='bipolar_lfp',
                                           **common)
    plot_laminar_spectral_profile_sessions(sessions, lfp_key='lfp_matrix',
                                           **common)
    if 'lfp_current_matrix' in first_trial:
        plot_laminar_spectral_profile_sessions(sessions,
                                               lfp_key='lfp_current_matrix',
                                               **common)
        for trials in sessions.values():
            for trial in trials:
                lfp_cur = trial['lfp_current_matrix']
                trial['bipolar_lfp_current'] = np.diff(lfp_cur, axis=0)
        plot_laminar_spectral_profile_sessions(
            sessions, lfp_key='bipolar_lfp_current',
            title_suffix=' (Bipolar synaptic current)', **common)
    else:
        print('\nNo lfp_current_matrix found in trials')
