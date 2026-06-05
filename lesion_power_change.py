"""Compare lesion vs control baseline PSD across the laminar profile.

For each lesion subfolder under results/lesions_<date>/, computes the
trial-averaged PSD on the last 2000 ms of each trial (no stimulus, baseline
only), then plots 3 panels: control PSD, lesion PSD, and % change vs control.
Also produces a 3D surface of the % change. Figures are saved under
figures/lesions_<date>/<lesion>/.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import detrend
from scipy.ndimage import zoom

from laminar_power_change import (
    load_trials,
    multitaper_psd,
    _time_vector_for_key,
)


def _bipolar_current_from_trials(trials):
    """Add 'bipolar_lfp_current' = diff of lfp_current_matrix across channels."""
    for tr in trials:
        if 'lfp_current_matrix' in tr and 'bipolar_lfp_current' not in tr:
            tr['bipolar_lfp_current'] = np.diff(tr['lfp_current_matrix'], axis=0)
    return trials


def compute_baseline_psd(all_trials, lfp_key, window_ms=2000, do_detrend=True):
    """Trial-averaged PSD per channel on the last `window_ms` of each trial.

    Returns (f, psd[n_channels, n_freq], depths).
    """
    t0 = _time_vector_for_key(all_trials[0], lfp_key)
    fs = 1000.0 / float(np.mean(np.diff(t0)))

    n_channels = all_trials[0][lfp_key].shape[0]
    channel_depths = all_trials[0]['channel_depths']
    if lfp_key in ('bipolar_lfp', 'bipolar_lfp_current') and len(channel_depths) > n_channels:
        depths = (channel_depths[:-1] + channel_depths[1:]) / 2
    else:
        depths = channel_depths[:n_channels]

    psd_per_channel = []
    f = None
    for ch in range(n_channels):
        psds = []
        for trial in all_trials:
            lfp = trial[lfp_key][ch]
            time = _time_vector_for_key(trial, lfp_key)
            t_end = float(time[-1])
            mask = (time >= t_end - window_ms) & (time <= t_end)
            seg = lfp[mask].copy()
            if len(seg) == 0 or np.any(np.isnan(seg)):
                continue
            if do_detrend:
                seg = detrend(seg)
            nfft = 2 ** int(np.ceil(np.log2(len(seg))))
            f_, psd = multitaper_psd(seg, fs=fs, NW=2, nfft=nfft)
            f = f_
            psds.append(psd)
        if not psds:
            psd_per_channel.append(None)
        else:
            psd_per_channel.append(np.mean(psds, axis=0))

    if f is None:
        raise RuntimeError(f'No valid trials for lfp_key={lfp_key!r}')
    n_freq = len(f)
    psd_arr = np.array([
        p if p is not None else np.full(n_freq, np.nan)
        for p in psd_per_channel
    ])
    return f, psd_arr, np.asarray(depths)


def plot_lesion_vs_control(f, depths, psd_ctrl, psd_les,
                            lesion_name, lfp_key, out_path,
                            freq_range=(0, 120), log_freq=False):
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    f_plot = f[freq_mask]
    p_ctrl = psd_ctrl[:, freq_mask]
    p_les = psd_les[:, freq_mask]

    p_ctrl_db = 10 * np.log10(p_ctrl + 1e-10)
    p_les_db = 10 * np.log10(p_les + 1e-10)
    pct_change = (p_les - p_ctrl) / p_ctrl * 100

    p_ctrl_db = np.flipud(p_ctrl_db)
    p_les_db = np.flipud(p_les_db)
    pct_change = np.flipud(pct_change)

    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    n_channels = p_ctrl_db.shape[0]
    extent = [f_plot[0], f_plot[-1], -0.5, n_channels - 0.5]

    vmin_db = np.nanpercentile([p_ctrl_db, p_les_db], 5)
    vmax_db = np.nanpercentile([p_ctrl_db, p_les_db], 95)

    im1 = axes[0].imshow(p_ctrl_db, aspect='auto', cmap='viridis',
                         extent=extent, origin='upper',
                         vmin=vmin_db, vmax=vmax_db)
    axes[0].set_title('Control')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Laminar depth')
    plt.colorbar(im1, ax=axes[0], label='Power (dB)')

    im2 = axes[1].imshow(p_les_db, aspect='auto', cmap='viridis',
                         extent=extent, origin='upper',
                         vmin=vmin_db, vmax=vmax_db)
    axes[1].set_title(f'Lesion {lesion_name}')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Laminar depth')
    plt.colorbar(im2, ax=axes[1], label='Power (dB)')

    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
    im3 = axes[2].imshow(pct_change, aspect='auto', cmap='RdBu_r',
                         extent=extent, origin='upper', norm=norm)
    axes[2].set_title('Lesion-induced change (%)')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Laminar depth')
    cbar = plt.colorbar(im3, ax=axes[2], label='% change')
    cbar.set_ticks([-100, -50, 0, 50, 100])

    if log_freq:
        for ax in axes:
            ax.set_xscale('log')

    suffix = {
        'bipolar_lfp': ' (Bipolar kernel LFP)',
        'bipolar_lfp_current': ' (Bipolar synaptic current)',
    }.get(lfp_key, f' ({lfp_key})')
    plt.suptitle(f'Lesion {lesion_name} vs control{suffix}', fontsize=15)
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return f_plot, pct_change


def plot_pct_change_3d(f_plot, depths, pct_change, lesion_name, lfp_key,
                       out_path, upsample_depth=8, smooth_freq=4,
                       clip_percentile=99):
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

    fig = plt.figure(figsize=(10, 7), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    F, D = np.meshgrid(f_plot, d_up)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    surf = ax.plot_surface(F, D, pct_up, cmap='RdBu_r', norm=norm,
                           edgecolor='none', alpha=0.95, antialiased=True,
                           rcount=80, ccount=80)
    z_floor = float(np.nanmin(pct_up))
    ax.set_zlim(z_floor, float(np.nanmax(pct_up)))
    ax.contour(F, D, pct_up, zdir='z', offset=z_floor,
               cmap='RdBu_r', norm=norm, levels=12)
    ax.set_xlabel('Frequency (Hz)', fontsize=9)
    ax.set_ylabel('Laminar depth', fontsize=9)
    ax.set_zlabel('% change', fontsize=9)
    suffix = {
        'bipolar_lfp': ' (Bipolar kernel LFP)',
        'bipolar_lfp_current': ' (Bipolar synaptic current)',
    }.get(lfp_key, f' ({lfp_key})')
    ax.set_title(f'Lesion {lesion_name} {suffix}\n3D % change  '
                 f'[{vmin:+.0f}, {vmax:+.0f}]%', fontsize=11)
    ax.view_init(elev=25, azim=-60)
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, label='% change')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def count_trials(folder):
    return sum(1 for f in os.listdir(folder) if f.startswith('trial_') and f.endswith('.npz'))


def main(lesion_root, fig_root, window_ms=2000,
         lfp_keys=('bipolar_lfp', 'bipolar_lfp_current')):
    control_dir = os.path.join(lesion_root, 'control')
    if not os.path.isdir(control_dir):
        raise FileNotFoundError(f'No control folder at {control_dir}')

    print(f'Loading control trials from {control_dir}')
    n_ctrl = count_trials(control_dir)
    control_trials = load_trials(control_dir, n_ctrl)
    control_trials = _bipolar_current_from_trials(control_trials)
    print(f'  loaded {n_ctrl} control trials')

    control_psd = {}
    control_depths = {}
    control_f = {}
    for key in lfp_keys:
        if key not in control_trials[0] and key != 'bipolar_lfp_current':
            print(f'  control missing {key!r}, skipping that signal')
            continue
        if key == 'bipolar_lfp_current' and 'bipolar_lfp_current' not in control_trials[0]:
            print('  control has no lfp_current_matrix, skipping bipolar_lfp_current')
            continue
        f, psd, depths = compute_baseline_psd(control_trials, key, window_ms=window_ms)
        control_f[key] = f
        control_psd[key] = psd
        control_depths[key] = depths
        print(f'  control PSD ready for {key}: shape {psd.shape}')

    lesion_names = sorted(
        d for d in os.listdir(lesion_root)
        if os.path.isdir(os.path.join(lesion_root, d)) and d != 'control'
    )
    print(f'Found {len(lesion_names)} lesion folders')

    os.makedirs(fig_root, exist_ok=True)

    for lname in lesion_names:
        lesion_dir = os.path.join(lesion_root, lname)
        n = count_trials(lesion_dir)
        if n == 0:
            print(f'[{lname}] no trials, skipping')
            continue
        print(f'[{lname}] loading {n} trials')
        try:
            lesion_trials = load_trials(lesion_dir, n)
        except Exception as exc:
            print(f'[{lname}] failed to load: {exc}')
            continue
        lesion_trials = _bipolar_current_from_trials(lesion_trials)

        out_dir = os.path.join(fig_root, lname)
        os.makedirs(out_dir, exist_ok=True)

        for key in lfp_keys:
            if key not in control_psd:
                continue
            if key not in lesion_trials[0] and not (
                key == 'bipolar_lfp_current' and 'bipolar_lfp_current' in lesion_trials[0]
            ):
                print(f'[{lname}] missing {key!r}, skipping')
                continue
            try:
                f_l, psd_l, _ = compute_baseline_psd(
                    lesion_trials, key, window_ms=window_ms)
            except Exception as exc:
                print(f'[{lname}] PSD failed for {key}: {exc}')
                continue
            if not np.array_equal(f_l, control_f[key]):
                print(f'[{lname}] freq grid mismatch for {key}, skipping')
                continue

            heatmap_path = os.path.join(out_dir, f'{key}_heatmaps.png')
            f_plot, pct = plot_lesion_vs_control(
                f_l, control_depths[key], control_psd[key], psd_l,
                lesion_name=lname, lfp_key=key, out_path=heatmap_path)
            print(f'  saved {heatmap_path}')

            surf_path = os.path.join(out_dir, f'{key}_3d.png')
            plot_pct_change_3d(
                f_plot, control_depths[key], pct,
                lesion_name=lname, lfp_key=key, out_path=surf_path)
            print(f'  saved {surf_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lesion-root', default='results/lesions_2026-06-02')
    parser.add_argument('--fig-root', default='figures/lesions_2026-06-02')
    parser.add_argument('--window-ms', type=int, default=2000)
    args = parser.parse_args()
    main(args.lesion_root, args.fig_root, window_ms=args.window_ms)
