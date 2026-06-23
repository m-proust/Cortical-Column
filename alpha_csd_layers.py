"""alpha trough realigned laminar CSD, computes the Mitzdorf 1D CSD,
realigns it on the alpha trough of a reference channel.

to run:
    path/to/venv/python alpha_csd_layers.py --trial_dir path/to/trials
"""
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import butter, sosfiltfilt, hilbert, welch
from scipy.ndimage import gaussian_filter


ALPHA_BAND = (8.0, 13.0)
WARMUP_MS = 500.0
CUTOUT_MS = 200.0
HIGHPASS_HZ = 1.0

LAYER_Z_RANGES = {
    'L23':  (0.45, 1.10),
    'L4AB': (0.14, 0.45),
    'L4C':  (-0.14, 0.14),
    'L5':   (-0.34, -0.14),
    'L6':   (-0.62, -0.34),
}
LAYER_ORDER = ['L23', 'L4AB', 'L4C', 'L5', 'L6']


def load_trial(fpath):
    d = np.load(fpath, allow_pickle=True)
    return {k: d[k] for k in d.files}


def select_lfp_and_depths(trial):
    if 'lfp_matrix' in trial:
        lfp = np.asarray(trial['lfp_matrix'], dtype=float)
        time_ms = np.asarray(trial['time_array_ms'], dtype=float)
    elif 'unipolar_matrix' in trial:
        lfp = np.asarray(trial['unipolar_matrix'], dtype=float)
        time_ms = np.asarray(trial['time_array_ms'], dtype=float)
    else:
        lfp = np.cumsum(np.asarray(trial['bipolar_matrix'], dtype=float),
                        axis=0)
        time_ms = np.asarray(trial['time_array_ms'], dtype=float)

    if 'electrode_positions' in trial:
        depths = np.asarray(trial['electrode_positions'], dtype=float)[:, 2]
    else:
        depths = np.asarray(trial['channel_depths'], dtype=float)
    return lfp, depths, time_ms


def bandpass(sig, fs, band, order=4):
    nyq = fs / 2.0
    sos = butter(order, [band[0] / nyq, band[1] / nyq],
                 btype='bandpass', output='sos')
    return sosfiltfilt(sos, sig)


def preprocess_lfp(lfp, fs):
    sos = butter(2, HIGHPASS_HZ, btype='highpass', fs=fs, output='sos')
    return sosfiltfilt(sos, lfp, axis=1)


def hamming_smooth_channels(lfp_matrix):
    w = np.array([0.23, 0.54, 0.23])
    w = w / w.sum()
    out = np.empty_like(lfp_matrix)
    n_ch = lfp_matrix.shape[0]
    out[0, :] = (w[1] + w[0]) * lfp_matrix[0, :] + w[2] * lfp_matrix[1, :]
    out[-1, :] = (w[1] + w[2]) * lfp_matrix[-1, :] + w[0] * lfp_matrix[-2, :]
    for i in range(1, n_ch - 1):
        out[i, :] = (w[0] * lfp_matrix[i - 1, :]
                     + w[1] * lfp_matrix[i, :]
                     + w[2] * lfp_matrix[i + 1, :])
    return out


def compute_csd(lfp_matrix, channel_depths_mm):
    z = np.asarray(channel_depths_mm, dtype=float)
    if abs(lfp_matrix.shape[0] - len(z)) > 1:
        raise ValueError(
            f"channel mismatch: lfp has {lfp_matrix.shape[0]} rows, "
            f"depths has {len(z)}"
        )
    n_ch = min(lfp_matrix.shape[0], len(z))
    lfp_use = hamming_smooth_channels(lfp_matrix[:n_ch, :])

    csd = np.full((n_ch, lfp_matrix.shape[1]), np.nan, dtype=float)
    for i in range(1, n_ch - 1):
        dz_up = z[i + 1] - z[i]
        dz_dn = z[i] - z[i - 1]
        dz_eff = 0.5 * (dz_up + dz_dn)
        csd[i, :] = -(lfp_use[i + 1, :] - 2 * lfp_use[i, :]
                      + lfp_use[i - 1, :]) / (dz_eff ** 2)
    return csd


def find_alpha_reference_channel(lfp_matrix, channel_depths, fs, alpha_band,
                                 prefer_depth=None):
    if prefer_depth is not None:
        return int(np.argmin(np.abs(channel_depths - prefer_depth)))
    n_ch = lfp_matrix.shape[0]
    pow_per_ch = np.zeros(n_ch)
    nperseg = int(min(2.0 * fs, lfp_matrix.shape[1]))
    for ci in range(n_ch):
        f, pxx = welch(lfp_matrix[ci, :], fs=fs, nperseg=nperseg)
        band = (f >= alpha_band[0]) & (f <= alpha_band[1])
        pow_per_ch[ci] = np.trapezoid(pxx[band], f[band])
    return int(np.argmax(pow_per_ch))


def detect_alpha_troughs_phase(signal_1d, fs, alpha_band, edge_pad_samples):
    filt = bandpass(signal_1d, fs, alpha_band)
    analytic = hilbert(filt)
    phase = np.angle(analytic)
    envelope = np.abs(analytic)

    troughs = []
    for k in range(1, len(phase)):
        if phase[k - 1] > np.pi / 2 and phase[k] < -np.pi / 2:
            d_prev = np.pi - phase[k - 1]
            d_curr = phase[k] + np.pi
            troughs.append(k - 1 if d_prev < d_curr else k)
    troughs = np.asarray(troughs, dtype=int)

    n = len(signal_1d)
    keep = (troughs >= edge_pad_samples) & (troughs < n - edge_pad_samples)
    troughs = troughs[keep]
    amps = envelope[troughs] if len(troughs) else np.array([])
    return troughs, amps


def alpha_realigned_csd(trial_files, alpha_band, window_kind, ref_depth_mm):
    accum = None
    counts = None
    ref_accum = None
    ref_counts = None
    ref_depth_used = None
    depths_ref = None
    cutout_t = None
    n_troughs = 0

    for fpath in trial_files:
        trial = load_trial(fpath)
        lfp, depths, time_ms = select_lfp_and_depths(trial)
        dt = float(time_ms[1] - time_ms[0])
        fs = 1000.0 / dt
        stim = float(trial.get('stim_onset_ms', 2000))

        if window_kind == 'baseline':
            t_lo, t_hi = WARMUP_MS, stim
        elif window_kind == 'stimulus':
            t_lo, t_hi = stim, float(time_ms[-1])
        else:
            raise ValueError(f"unknown window_kind {window_kind}")

        mask = (time_ms >= t_lo) & (time_ms < t_hi)
        if not np.any(mask):
            continue
        idx_in_window = np.where(mask)[0]
        lfp = preprocess_lfp(lfp, fs)

        csd = compute_csd(lfp, depths)
        n_channels, n_t = csd.shape

        ref_idx = find_alpha_reference_channel(lfp, depths, fs, alpha_band,
                                               prefer_depth=ref_depth_mm)
        ref_signal = lfp[ref_idx, :]
        ref_signal_alpha = bandpass(ref_signal, fs, alpha_band)
        if ref_depth_used is None:
            ref_depth_used = float(depths[ref_idx])

        cutout_samples = int(round(CUTOUT_MS / dt))
        ref_in_win = ref_signal[idx_in_window]
        rel_troughs, _ = detect_alpha_troughs_phase(
            ref_in_win, fs, alpha_band, edge_pad_samples=cutout_samples)
        abs_troughs = idx_in_window[rel_troughs]
        keep_cut = ((abs_troughs - cutout_samples >= 0)
                    & (abs_troughs + cutout_samples + 1 <= n_t))
        abs_troughs = abs_troughs[keep_cut]

        if len(abs_troughs) == 0:
            continue

        if accum is None:
            n_cutout = 2 * cutout_samples + 1
            accum = np.zeros((n_channels, n_cutout))
            counts = np.zeros((n_channels, n_cutout))
            ref_accum = np.zeros(n_cutout)
            ref_counts = np.zeros(n_cutout)
            depths_ref = depths.copy()
            cutout_t = (np.arange(n_cutout) - cutout_samples) * dt

        for tr in abs_troughs:
            seg = csd[:, tr - cutout_samples: tr + cutout_samples + 1]
            ref_seg = ref_signal_alpha[
                tr - cutout_samples: tr + cutout_samples + 1]
            valid = ~np.isnan(seg)
            ref_valid = ~np.isnan(ref_seg)
            accum += np.where(valid, seg, 0.0)
            counts += valid.astype(float)
            ref_accum += np.where(ref_valid, ref_seg, 0.0)
            ref_counts += ref_valid.astype(float)
            n_troughs += 1

    if accum is None:
        raise RuntimeError("no alpha troughs were found in any trial")

    csd_avg = np.divide(accum, counts, out=np.full_like(accum, np.nan),
                        where=counts > 0)
    ref_avg = np.divide(ref_accum, ref_counts,
                        out=np.full_like(ref_accum, np.nan),
                        where=ref_counts > 0)

    return {
        'csd_avg': csd_avg,
        'depths': depths_ref,
        'cutout_t_ms': cutout_t,
        'n_troughs': n_troughs,
        'ref_alpha_avg': ref_avg,
        'ref_depth_used': ref_depth_used,
    }


def trial_average_evoked_csd(trial_files, pre_ms=50.0, post_ms=200.0):
    accum = None
    counts = None
    depths_ref = None
    t_ref = None
    n_used = 0

    for fpath in trial_files:
        trial = load_trial(fpath)
        lfp, depths, time_ms = select_lfp_and_depths(trial)
        dt = float(time_ms[1] - time_ms[0])
        fs = 1000.0 / dt
        stim = float(trial.get('stim_onset_ms', 2000))

        i_stim = int(np.argmin(np.abs(time_ms - stim)))
        i_lo = i_stim - int(round(pre_ms / dt))
        i_hi = i_stim + int(round(post_ms / dt))
        if i_lo < 0 or i_hi > len(time_ms):
            continue

        lfp = preprocess_lfp(lfp, fs)
        csd = compute_csd(lfp, depths)
        seg = csd[:, i_lo:i_hi]

        if accum is None:
            accum = np.zeros_like(seg)
            counts = np.zeros_like(seg)
            depths_ref = depths.copy()
            t_ref = (np.arange(seg.shape[1]) - int(round(pre_ms / dt))) * dt
        valid = ~np.isnan(seg)
        accum += np.where(valid, seg, 0.0)
        counts += valid.astype(float)
        n_used += 1

    if accum is None:
        raise RuntimeError("no trials had a usable stimulus-aligned window")

    csd_avg = np.divide(accum, counts, out=np.full_like(accum, np.nan),
                        where=counts > 0)
    return csd_avg, depths_ref, t_ref, n_used


def plot_csd_heatmap(ax, csd, depths, t, title, smooth_sigma=(0.5, 1.5),
                     vmax=None, show_layers=True, mark_zero_depth=True):
    csd_plot = csd.copy()
    finite = np.isfinite(csd_plot)
    csd_filled = np.where(finite, csd_plot, 0.0)
    if smooth_sigma is not None:
        csd_filled = gaussian_filter(csd_filled, sigma=smooth_sigma)
    csd_filled = np.where(finite, csd_filled, np.nan)

    if vmax is None:
        vmax = np.nanpercentile(np.abs(csd_filled), 98)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    extent = [t[0], t[-1], depths[0], depths[-1]]
    im = ax.imshow(
        csd_filled, aspect='auto', origin='lower', extent=extent,
        cmap='RdBu_r', norm=norm, interpolation='bilinear',
    )

    if show_layers:
        for layer, (lo, hi) in LAYER_Z_RANGES.items():
            ax.axhline(hi, color='k', lw=0.4, alpha=0.25)
            ax.text(t[-1] + (t[-1] - t[0]) * 0.015, (lo + hi) / 2, layer,
                    va='center', ha='left', fontsize=8, color='#333333')
    if mark_zero_depth:
        ax.axhline(0, color='k', lw=1.0, alpha=0.5)

    ax.set_xlabel('time (ms)', fontsize=10)
    ax.set_ylabel('depth (mm)\n← deep    superficial →', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    return im


def _plot_ref_alpha(ax, cutout_t, ref_avg, ref_depth_used):
    if ref_avg is None or np.all(np.isnan(ref_avg)):
        ax.set_axis_off()
        return
    ax.plot(cutout_t, ref_avg, color='k', lw=1.2)
    ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax.set_xlim(cutout_t[0], cutout_t[-1])
    ax.set_xlabel('time from alpha trough (ms)')
    label = (f'alpha LFP @ {ref_depth_used:+.2f} mm'
             if ref_depth_used is not None else 'alpha LFP (ref)')
    ax.set_ylabel(label, fontsize=9)
    ax.tick_params(labelsize=8)


def plot_alpha_csd_summary(result_b, save_path, result_s=None):
    csd_b = result_b['csd_avg']
    depths = result_b['depths']
    cutout_t = result_b['cutout_t_ms']

    ref_b = result_b.get('ref_alpha_avg')
    ref_depth_b = result_b.get('ref_depth_used')

    def _attach_aligned_pair(ax_csd, ax_ref, attach_cbar):
        div_csd = make_axes_locatable(ax_csd)
        div_ref = make_axes_locatable(ax_ref)
        cax = div_csd.append_axes('right', size='3%', pad=0.08)
        spacer = div_ref.append_axes('right', size='3%', pad=0.08)
        spacer.set_axis_off()
        ax_ref.sharex(ax_csd)
        if not attach_cbar:
            cax.set_axis_off()
            return None
        return cax

    if result_s is not None:
        csd_s = result_s['csd_avg']
        ref_s = result_s.get('ref_alpha_avg')
        fig, axes = plt.subplots(2, 2, figsize=(13, 6.5),
                                 gridspec_kw={'wspace': 0.35,
                                              'hspace': 0.05,
                                              'height_ratios': [3, 1]})
        vmax_b = np.nanpercentile(np.abs(csd_b), 98)
        vmax_s = np.nanpercentile(np.abs(csd_s), 98)
        vmax = max(vmax_b, vmax_s)
        im = plot_csd_heatmap(axes[0, 0], csd_b, depths, cutout_t,
                              title='Baseline alpha aligned CSD ',
                              vmax=vmax)
        plot_csd_heatmap(axes[0, 1], csd_s, depths, cutout_t,
                         title='Stimulus alpha aligned CSD ',
                         vmax=vmax)
        for ax in axes[0, :]:
            ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)
        _plot_ref_alpha(axes[1, 0], cutout_t, ref_b, ref_depth_b)
        _plot_ref_alpha(axes[1, 1], cutout_t, ref_s,
                        result_s.get('ref_depth_used'))
        _attach_aligned_pair(axes[0, 0], axes[1, 0], attach_cbar=False)
        cax = _attach_aligned_pair(axes[0, 1], axes[1, 1], attach_cbar=True)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label('CSD (a.u.)\n← sink     source →', fontsize=10)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5),
                                 gridspec_kw={'hspace': 0.05,
                                              'height_ratios': [3, 1]})
        ax_csd, ax_ref = axes
        im = plot_csd_heatmap(ax_csd, csd_b, depths, cutout_t,
                              title='Baseline alpha aligned CSD ')
        ax_csd.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
        ax_csd.set_xlabel('')
        ax_csd.tick_params(labelbottom=False)
        _plot_ref_alpha(ax_ref, cutout_t, ref_b, ref_depth_b)
        cax = _attach_aligned_pair(ax_csd, ax_ref, attach_cbar=True)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label('CSD (a.u.)\n← sink     source →', fontsize=10)

    fig.suptitle('Alpha-realigned CSD\ndata-driven reference; '
                 'centred on alpha trough', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(save_path, dpi=160, bbox_inches='tight')
    print(f"  saved {save_path}")
    plt.close(fig)


def plot_evoked_csd(csd, depths, t_ms, n_used, save_path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = plot_csd_heatmap(ax, csd, depths, t_ms,
                          title='CSD centered at stimulus')
    ax.axvline(0, color='k', lw=1.0, ls='--', alpha=0.6)
    ax.text(0, depths[-1], '  stim onset', va='top', ha='left', fontsize=9)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('sink to source', fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=160, bbox_inches='tight')
    print(f"  saved {save_path}")
    plt.close(fig)


def plot_single_channel_cutout(result, ref_depth_mm, save_path,
                               window_label='baseline'):
    csd_avg = result['csd_avg']
    depths = result['depths']
    cutout_t = result['cutout_t_ms']
    n_troughs = result['n_troughs']

    ch = int(np.argmin(np.abs(depths - ref_depth_mm)))
    trace = csd_avg[ch, :]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(cutout_t, trace, color='C0', lw=1.2)
    ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(0, color='k', lw=0.5, alpha=0.4)
    for t_mark in (-100, 100):
        ax.axvline(t_mark, color='r', lw=0.5, ls=':', alpha=0.5)
    ax.set_xlabel('time relative to alpha trough (ms)')
    ax.set_ylabel('CSD (a.u.)')
    ax.set_title(
        f'Single-channel cutout-averaged CSD ({window_label})\n'
        f'channel @ z={depths[ch]:+.2f} mm (target {ref_depth_mm:+.2f}); '
        f'n={n_troughs} troughs; red dotted = ±100 ms (10 Hz cycle)'
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    print(f"  saved {save_path}")
    plt.close(fig)


def plot_alpha_diagnostic(trial_files, alpha_band, ref_depth_mm, save_path):
    trial = load_trial(trial_files[0])
    lfp, depths, time_ms = select_lfp_and_depths(trial)
    dt = float(time_ms[1] - time_ms[0])
    fs = 1000.0 / dt
    stim = float(trial.get('stim_onset_ms', 2000))
    lfp = preprocess_lfp(lfp, fs)
    ref_idx = find_alpha_reference_channel(lfp, depths, fs, alpha_band,
                                           prefer_depth=ref_depth_mm)
    mask = (time_ms >= WARMUP_MS) & (time_ms < stim)
    raw = lfp[ref_idx, mask]
    t_in_win = time_ms[mask]
    filt = bandpass(raw, fs, alpha_band)
    troughs, _ = detect_alpha_troughs_phase(raw, fs, alpha_band,
                                            edge_pad_samples=int(0.05 * fs))

    n_ch = lfp.shape[0]
    pow_accum = np.zeros(n_ch)
    spec_accum = None
    f_ref = None
    n_trials_used = 0
    for fpath in trial_files:
        tr = load_trial(fpath)
        lfp_t, _, time_ms_t = select_lfp_and_depths(tr)
        if lfp_t.shape[0] != n_ch:
            continue
        stim_t = float(tr.get('stim_onset_ms', 2000))
        m = (time_ms_t >= WARMUP_MS) & (time_ms_t < stim_t)
        if not np.any(m):
            continue
        lfp_t = preprocess_lfp(lfp_t, fs)
        for ci in range(n_ch):
            f, pxx = welch(lfp_t[ci, m], fs=fs,
                           nperseg=int(min(2.0 * fs, m.sum())))
            band = (f >= alpha_band[0]) & (f <= alpha_band[1])
            pow_accum[ci] += np.trapezoid(pxx[band], f[band])
        f, pxx_ref = welch(lfp_t[ref_idx, m], fs=fs,
                           nperseg=int(min(2.0 * fs, m.sum())))
        if spec_accum is None:
            spec_accum = np.zeros_like(pxx_ref)
            f_ref = f
        spec_accum += pxx_ref
        n_trials_used += 1
    if n_trials_used > 0:
        pow_accum /= n_trials_used
        spec_accum /= n_trials_used

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot(t_in_win, raw - np.mean(raw), color='0.6', lw=0.6,
            label='raw (mean removed)')
    ax.plot(t_in_win, filt, color='C0', lw=1.0,
            label=f'bandpass {alpha_band[0]}-{alpha_band[1]} Hz')
    ax.plot(t_in_win[troughs], filt[troughs], 'r.', ms=6,
            label='detected troughs')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('reference channel LFP')
    ax.set_title(f'reference @ z={depths[ref_idx]:+.2f} mm — trial 0')
    ax.legend(fontsize=8, loc='upper right')

    ax = axes[1]
    ax.plot(pow_accum, depths, 'o-', color='C0')
    ax.axhline(depths[ref_idx], color='r', lw=0.8, ls='--',
               label=f'ref depth ({depths[ref_idx]:+.2f})')
    ax.set_xlabel(f'alpha power ({alpha_band[0]}-{alpha_band[1]} Hz)')
    ax.set_ylabel('depth (mm)')
    ax.set_title(f'alpha power vs depth (avg over {n_trials_used} trials)')
    for _, (_, hi) in LAYER_Z_RANGES.items():
        ax.axhline(hi, color='k', lw=0.3, alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[2]
    if spec_accum is not None:
        keep = f_ref <= 60
        ax.semilogy(f_ref[keep], spec_accum[keep], color='C0')
        for fb in alpha_band:
            ax.axvline(fb, color='r', lw=0.6, ls='--')
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title('reference channel spectrum (baseline)')

    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    print(f"  saved {save_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trial_dir', type=str, required=True)
    args = p.parse_args()

    save_dir = os.path.join(args.trial_dir, 'csd_alpha')
    os.makedirs(save_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.trial_dir, 'trial_*.npz')))
    if not files:
        raise FileNotFoundError(f"no trial_*.npz in {args.trial_dir}")
    print(f"loaded {len(files)} trials")

    ref_depth_mm = None

    plot_alpha_diagnostic(
        files, ALPHA_BAND, ref_depth_mm=ref_depth_mm,
        save_path=os.path.join(save_dir, 'alpha_diagnostic.png'),
    )

    try:
        csd_evoked, depths_ev, t_ev, n_evoked = trial_average_evoked_csd(files)
        plot_evoked_csd(csd_evoked, depths_ev, t_ev, n_evoked,
                        save_path=os.path.join(save_dir, 'evoked_csd.png'))
    except RuntimeError as e:
        print(f"  could not compute evoked CSD: {e}")

    print('computing baseline alpha-realigned CSD...')
    result_b = alpha_realigned_csd(files, ALPHA_BAND, window_kind='baseline',
                                   ref_depth_mm=ref_depth_mm)
    print(f"  baseline troughs averaged: {result_b['n_troughs']}")

    result_s = None
    try:
        print('computing stimulus alpha-realigned CSD...')
        result_s = alpha_realigned_csd(files, ALPHA_BAND,
                                       window_kind='stimulus',
                                       ref_depth_mm=ref_depth_mm)
        print(f"  stimulus troughs averaged: {result_s['n_troughs']}")
    except RuntimeError as e:
        print(f"  could not compute stimulus CSD: {e}")
        result_s = None

    ref_for_plot = (ref_depth_mm if ref_depth_mm is not None
                    else float(result_b['depths'][
                        np.nanargmax(np.nansum(np.abs(result_b['csd_avg']),
                                               axis=1))]))
    plot_single_channel_cutout(
        result_b, ref_depth_mm=ref_for_plot,
        save_path=os.path.join(save_dir, 'single_channel_cutout_baseline.png'),
        window_label='baseline',
    )
    if result_s is not None:
        plot_single_channel_cutout(
            result_s, ref_depth_mm=ref_for_plot,
            save_path=os.path.join(save_dir,
                                   'single_channel_cutout_stimulus.png'),
            window_label='stimulus',
        )

    plot_alpha_csd_summary(
        result_b,
        save_path=os.path.join(save_dir, 'alpha_realigned_csd.png'),
        result_s=result_s,
    )
    plot_alpha_csd_summary(
        result_b,
        save_path=os.path.join(save_dir,
                               'alpha_realigned_csd_baseline_only.png'),
        result_s=None,
    )


if __name__ == '__main__':
    main()
