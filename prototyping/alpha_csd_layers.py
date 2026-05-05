import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks, welch
from scipy.ndimage import gaussian_filter


# Same layer conventions as the PLV script so depth labels match across plots.
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


LFP_SOURCES = ('kernel', 'current')


def select_lfp_and_depths(trial, lfp_source='kernel'):
    """Return (lfp, depths, time_ms) for the chosen LFP source.

    `lfp_source='kernel'`  (default) — kernel-method unipolar LFP saved as
        `lfp_matrix`, paired with `time_array_ms`.
    `lfp_source='current'` — synaptic-current-method LFP saved as
        `lfp_current_matrix`, paired with `time_current_ms`.

    The trials pipeline saves `channel_depths` as the *bipolar* midpoint axis
    (length n_electrodes - 1), so it does NOT pair with the unipolar LFP
    matrices (length n_electrodes). For CSD we need unipolar depths, which we
    pull from `electrode_positions[:, 2]` when available.
    """
    if lfp_source not in LFP_SOURCES:
        raise ValueError(f"lfp_source must be one of {LFP_SOURCES}, "
                         f"got {lfp_source!r}")

    if lfp_source == 'current':
        if 'lfp_current_matrix' not in trial:
            raise KeyError("trial does not contain 'lfp_current_matrix' — "
                           "re-run trials.py with the synaptic-current LFP "
                           "branch enabled, or use lfp_source='kernel'")
        lfp = np.asarray(trial['lfp_current_matrix'], dtype=float)
        time_ms = np.asarray(trial['time_current_ms'], dtype=float)
    elif 'lfp_matrix' in trial:
        lfp = np.asarray(trial['lfp_matrix'], dtype=float)
        time_ms = np.asarray(trial['time_array_ms'], dtype=float)
    elif 'unipolar_matrix' in trial:
        lfp = np.asarray(trial['unipolar_matrix'], dtype=float)
        time_ms = np.asarray(trial['time_array_ms'], dtype=float)
    else:
        # Fall back: integrate the bipolar back to unipolar by cumsum.
        lfp = np.cumsum(np.asarray(trial['bipolar_matrix'], dtype=float),
                        axis=0)
        time_ms = np.asarray(trial['time_array_ms'], dtype=float)

    if 'electrode_positions' in trial:
        ep = np.asarray(trial['electrode_positions'], dtype=float)
        depths = ep[:, 2]
    else:
        depths = np.asarray(trial['channel_depths'], dtype=float)
    return lfp, depths, time_ms


def bandpass(sig, fs, band, order=4):
    nyq = fs / 2.0
    sos = butter(order, [band[0] / nyq, band[1] / nyq],
                 btype='bandpass', output='sos')
    return sosfiltfilt(sos, sig)


def preprocess_lfp(lfp, fs, highpass_hz=1.0):
    """High-pass the LFP to match acquisition-hardware filtering.

    Real LFP is hardware-filtered around ~1 Hz; without this step the
    depth-stationary DC/very-low-frequency component dominates the CSD
    second derivative and washes out the laminar pattern. Pass
    `highpass_hz=0` (or None) to skip.
    """
    if not highpass_hz:
        return lfp
    sos = butter(2, highpass_hz, btype='highpass', fs=fs, output='sos')
    return sosfiltfilt(sos, lfp, axis=1)


def hamming_smooth_channels(lfp_matrix):
    """3-point Hamming spatial smooth across channels.

    This is Mitzdorf's original recommendation (Mitzdorf 1985) and the
    standard preprocessing step before second-differencing for CSD; see
    e.g. Ulbert et al. 2001, Schroeder et al. 1998.

    Weights = [0.23, 0.54, 0.23] (Hamming, normalised to sum to 1).
    Boundary channels are smoothed with a 2-point reflective kernel.
    """
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


def compute_csd(lfp_matrix, channel_depths_mm, spatial_smooth=True):
    """Mitzdorf-style 1D CSD from unipolar LFP.

    csd[i,t] = -(lfp[i+1,t] - 2*lfp[i,t] + lfp[i-1,t]) / dz**2

    Sign convention: sinks negative (current entering the cell), sources
    positive — matches Bollimunta 2011 etc. (blue = sink in RdBu_r).

    With `spatial_smooth=True`, applies Mitzdorf's 3-point Hamming smooth
    across channels before differentiation.

    Top and bottom channels are NaN (second derivative undefined at edges).
    """
    z = np.asarray(channel_depths_mm, dtype=float)
    if abs(lfp_matrix.shape[0] - len(z)) > 1:
        raise ValueError(
            f"channel mismatch: lfp has {lfp_matrix.shape[0]} rows, "
            f"depths has {len(z)}"
        )
    n_ch = min(lfp_matrix.shape[0], len(z))

    if spatial_smooth:
        lfp_use = hamming_smooth_channels(lfp_matrix[:n_ch, :])
    else:
        lfp_use = lfp_matrix[:n_ch, :]

    csd = np.full((n_ch, lfp_matrix.shape[1]), np.nan, dtype=float)
    for i in range(1, n_ch - 1):
        dz_up = z[i + 1] - z[i]
        dz_dn = z[i] - z[i - 1]
        # symmetric stencil; if spacings are equal this reduces to /dz**2
        dz_eff = 0.5 * (dz_up + dz_dn)
        csd[i, :] = -(lfp_use[i + 1, :] - 2 * lfp_use[i, :]
                      + lfp_use[i - 1, :]) / (dz_eff ** 2)
    return csd


def find_alpha_reference_channel(lfp_matrix, channel_depths, fs, alpha_band,
                                 prefer_depth=None):
    """Pick the channel for alpha trough detection.

    If `prefer_depth` is given, pick the closest channel to that depth.
    Otherwise pick the channel with maximum alpha-band power across the whole
    trial.
    """
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
    """Realign on Hilbert phase = +-pi (the alpha trough).

    Bandpass to alpha, take Hilbert analytic signal, find samples where the
    phase wraps through +-pi (i.e. consecutive samples whose phase straddles
    pi). This is the convention used in Bollimunta 2011, Haegens 2015, etc.

    Returns (trough_indices, envelope_amplitude_at_troughs).
    """
    filt = bandpass(signal_1d, fs, alpha_band)
    analytic = hilbert(filt)
    phase = np.angle(analytic)          # in (-pi, pi]
    envelope = np.abs(analytic)

    # A trough corresponds to phase = +-pi. The phase wraps from +pi to -pi
    # exactly at the trough, so we look for sign changes from + to -.
    # Only count downward-going wraps (from positive near +pi to negative
    # near -pi); upward-going zero-crossings near 0 are peaks of the
    # filtered signal, not what we want.
    troughs = []
    for k in range(1, len(phase)):
        if phase[k - 1] > np.pi / 2 and phase[k] < -np.pi / 2:
            # interpolate to find the closer of the two samples to +-pi
            d_prev = np.pi - phase[k - 1]
            d_curr = phase[k] + np.pi
            troughs.append(k - 1 if d_prev < d_curr else k)
    troughs = np.asarray(troughs, dtype=int)

    n = len(signal_1d)
    keep = (troughs >= edge_pad_samples) & (troughs < n - edge_pad_samples)
    troughs = troughs[keep]
    amps = envelope[troughs] if len(troughs) else np.array([])
    return troughs, amps


def detect_alpha_troughs_peaks(signal_1d, fs, alpha_band, edge_pad_samples):
    """Fallback method: find_peaks on the negated filtered signal.

    Kept for backward compatibility / sanity checking. In clean signals this
    gives essentially identical results to the Hilbert-phase method.
    """
    filt = bandpass(signal_1d, fs, alpha_band)
    troughs, _ = find_peaks(-filt, distance=int(0.06 * fs))
    n = len(signal_1d)
    keep = (troughs >= edge_pad_samples) & (troughs < n - edge_pad_samples)
    troughs = troughs[keep]
    # envelope amplitude at troughs (for optional amplitude weighting)
    analytic = hilbert(filt)
    envelope = np.abs(analytic)
    amps = envelope[troughs] if len(troughs) else np.array([])
    return troughs, amps


def alpha_realigned_csd(trial_files, alpha_band, window_kind, baseline_start_ms,
                        cutout_ms=200.0, ref_depth_mm=-0.20,
                        method='phase', amplitude_weighted=False,
                        n_surrogates=0, surrogate_seed=0,
                        spatial_smooth=True, highpass_hz=1.0,
                        lfp_source='kernel'):
    """Compute alpha-realigned CSD averaged across cycles and trials.

    method            'phase' (Hilbert-based, default) or 'peaks' (find_peaks)
    amplitude_weighted    if True, weight each cutout by the envelope amplitude
                          at the trough
    n_surrogates      if > 0, also compute phase-shuffled surrogates by
                      picking n_troughs random uniformly-distributed trigger
                      times in each window. Returns surrogate mean and std
                      across surrogates for thresholding.

    Returns:
        result dict with keys:
            csd_avg         (n_channels, n_cutout_samples)
            depths          (n_channels,)
            cutout_t_ms     (n_cutout_samples,)
            n_troughs       int
            surrogate_mean  (n_channels, n_cutout_samples) or None
            surrogate_std   (n_channels, n_cutout_samples) or None
    """
    if method not in ('phase', 'peaks'):
        raise ValueError(f"method must be 'phase' or 'peaks', got {method!r}")
    detect = (detect_alpha_troughs_phase if method == 'phase'
              else detect_alpha_troughs_peaks)

    accum = None
    counts = None         # samples added per (channel, time) cell
    ref_accum = None      # alpha-bandpassed reference signal cutouts (1D)
    ref_counts = None
    ref_depth_used = None
    weight_sum = 0.0      # for amplitude weighting normalisation
    depths_ref = None
    cutout_t = None
    n_troughs = 0

    # surrogates: accumulate a separate set of cutouts at random trigger times,
    # repeated n_surrogates times to estimate the null distribution
    surrogate_runs = []  # list of (n_channels, n_cutout) arrays
    rng = np.random.default_rng(surrogate_seed)

    for fpath in trial_files:
        trial = load_trial(fpath)
        lfp, depths, time_ms = select_lfp_and_depths(trial,
                                                     lfp_source=lfp_source)
        dt = float(time_ms[1] - time_ms[0])
        fs = 1000.0 / dt
        stim = float(trial.get('stim_onset_ms', 2000))

        if window_kind == 'baseline':
            t_lo, t_hi = baseline_start_ms, stim
        elif window_kind == 'stimulus':
            t_lo, t_hi = stim, float(time_ms[-1])
        else:
            raise ValueError(f"unknown window_kind {window_kind}")

        mask = (time_ms >= t_lo) & (time_ms < t_hi)
        if not np.any(mask):
            continue
        idx_in_window = np.where(mask)[0]
        lfp = preprocess_lfp(lfp, fs, highpass_hz=highpass_hz)

        csd = compute_csd(lfp, depths, spatial_smooth=spatial_smooth)
        n_channels, n_t = csd.shape

        ref_idx = find_alpha_reference_channel(lfp, depths, fs, alpha_band,
                                               prefer_depth=ref_depth_mm)
        ref_signal = lfp[ref_idx, :]
        ref_signal_alpha = bandpass(ref_signal, fs, alpha_band)
        if ref_depth_used is None:
            ref_depth_used = float(depths[ref_idx])

        cutout_samples = int(round(cutout_ms / dt))
        ref_in_win = ref_signal[idx_in_window]
        rel_troughs, amps = detect(ref_in_win, fs, alpha_band,
                                   edge_pad_samples=cutout_samples)
        abs_troughs = idx_in_window[rel_troughs]
        # also enforce the cutout fits inside the full trial
        keep_cut = ((abs_troughs - cutout_samples >= 0)
                    & (abs_troughs + cutout_samples + 1 <= n_t))
        abs_troughs = abs_troughs[keep_cut]
        amps = amps[keep_cut] if len(amps) else amps

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

        # average raw CSD cutouts (no per-cutout demeaning — standard method)
        for k, tr in enumerate(abs_troughs):
            seg = csd[:, tr - cutout_samples: tr + cutout_samples + 1]
            ref_seg = ref_signal_alpha[
                tr - cutout_samples: tr + cutout_samples + 1]
            valid = ~np.isnan(seg)
            ref_valid = ~np.isnan(ref_seg)
            if amplitude_weighted:
                w = float(amps[k]) if len(amps) else 1.0
                accum += np.where(valid, seg * w, 0.0)
                counts += valid.astype(float) * w
                ref_accum += np.where(ref_valid, ref_seg * w, 0.0)
                ref_counts += ref_valid.astype(float) * w
                weight_sum += w
            else:
                accum += np.where(valid, seg, 0.0)
                counts += valid.astype(float)
                ref_accum += np.where(ref_valid, ref_seg, 0.0)
                ref_counts += ref_valid.astype(float)
            n_troughs += 1

        # surrogates: same number of triggers but at uniformly random times
        # within the same window. Repeat n_surrogates times.
        if n_surrogates > 0 and len(abs_troughs) > 0:
            n_tr = len(abs_troughs)
            valid_lo = max(idx_in_window[0], cutout_samples)
            valid_hi = min(idx_in_window[-1], n_t - cutout_samples - 1)
            if valid_hi > valid_lo:
                # initialise per-trial surrogate accumulator on first contact
                while len(surrogate_runs) < n_surrogates:
                    surrogate_runs.append({
                        'accum': np.zeros((n_channels, accum.shape[1])),
                        'counts': np.zeros((n_channels, accum.shape[1])),
                    })
                for s in range(n_surrogates):
                    rand_trigs = rng.integers(valid_lo, valid_hi + 1, size=n_tr)
                    for tr in rand_trigs:
                        seg = csd[:, tr - cutout_samples: tr + cutout_samples + 1]
                        valid = ~np.isnan(seg)
                        surrogate_runs[s]['accum'] += np.where(valid, seg, 0.0)
                        surrogate_runs[s]['counts'] += valid.astype(float)

    if accum is None:
        raise RuntimeError("no alpha troughs were found in any trial")

    csd_avg = np.divide(accum, counts, out=np.full_like(accum, np.nan),
                        where=counts > 0)
    ref_avg = np.divide(ref_accum, ref_counts,
                        out=np.full_like(ref_accum, np.nan),
                        where=ref_counts > 0)

    sur_mean = sur_std = None
    if surrogate_runs:
        sur_means = []
        for run in surrogate_runs:
            sm = np.divide(run['accum'], run['counts'],
                           out=np.full_like(run['accum'], np.nan),
                           where=run['counts'] > 0)
            sur_means.append(sm)
        sur_stack = np.stack(sur_means, axis=0)  # (n_surr, n_ch, n_cutout)
        sur_mean = np.nanmean(sur_stack, axis=0)
        sur_std = np.nanstd(sur_stack, axis=0)

    return {
        'csd_avg': csd_avg,
        'depths': depths_ref,
        'cutout_t_ms': cutout_t,
        'n_troughs': n_troughs,
        'surrogate_mean': sur_mean,
        'surrogate_std': sur_std,
        'ref_alpha_avg': ref_avg,
        'ref_depth_used': ref_depth_used,
    }


def trial_average_evoked_csd(trial_files, pre_ms=50.0, post_ms=200.0,
                             spatial_smooth=True, highpass_hz=1.0,
                             lfp_source='kernel'):
    """Trial-averaged stimulus-evoked CSD, the laminar-positioning landmark.

    Returns (csd_avg, depths, t_ms, n_used).
    """
    accum = None
    counts = None
    depths_ref = None
    t_ref = None
    n_used = 0

    for fpath in trial_files:
        trial = load_trial(fpath)
        lfp, depths, time_ms = select_lfp_and_depths(trial,
                                                     lfp_source=lfp_source)
        dt = float(time_ms[1] - time_ms[0])
        fs = 1000.0 / dt
        stim = float(trial.get('stim_onset_ms', 2000))

        i_stim = int(np.argmin(np.abs(time_ms - stim)))
        i_lo = i_stim - int(round(pre_ms / dt))
        i_hi = i_stim + int(round(post_ms / dt))
        if i_lo < 0 or i_hi > len(time_ms):
            continue

        lfp = preprocess_lfp(lfp, fs, highpass_hz=highpass_hz)
        csd = compute_csd(lfp, depths, spatial_smooth=spatial_smooth)
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
                     vmax=None, show_layers=True, mark_zero_depth=True,
                     significance_mask=None, depth_ylim=None):
    """Plot a CSD heatmap with depth on y, time on x, sinks blue / sources red.

    smooth_sigma=(depth, time) gaussian smoothing for visual clarity only.
    significance_mask  optional boolean array (same shape as csd); cells with
                       False are dimmed to indicate non-significant pixels.
    """
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

    if significance_mask is not None:
        # overlay translucent grey where NOT significant
        ns = (~significance_mask).astype(float)
        ns = np.where(np.isfinite(csd_plot), ns, 0.0)
        if smooth_sigma is not None:
            ns = gaussian_filter(ns, sigma=smooth_sigma)
        ax.imshow(ns, aspect='auto', origin='lower', extent=extent,
                  cmap='Greys', alpha=0.45, vmin=0, vmax=1,
                  interpolation='bilinear')

    if show_layers:
        for layer, (lo, hi) in LAYER_Z_RANGES.items():
            ax.axhline(hi, color='k', lw=0.4, alpha=0.25)
            ax.text(t[-1] + (t[-1] - t[0]) * 0.015, (lo + hi) / 2, layer,
                    va='center', ha='left', fontsize=8, color='#333333')
    if mark_zero_depth:
        ax.axhline(0, color='k', lw=1.0, alpha=0.5)

    if depth_ylim is not None:
        ax.set_ylim(depth_ylim)

    ax.set_xlabel('time (ms)', fontsize=10)
    ax.set_ylabel('depth (mm)\n← deep    superficial →', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    return im


def compute_significance_mask(csd, sur_mean, sur_std, z_thresh=2.0):
    """Boolean mask: True where |csd - sur_mean| > z_thresh * sur_std."""
    if sur_mean is None or sur_std is None:
        return None
    with np.errstate(invalid='ignore', divide='ignore'):
        z = (csd - sur_mean) / np.where(sur_std > 0, sur_std, np.nan)
    return np.abs(z) > z_thresh


def _plot_ref_alpha(ax, cutout_t, ref_avg, ref_depth_used):
    """Slim panel: trial-averaged alpha-bandpassed reference signal.

    Confirms the CSD heatmap above is actually trough-aligned — the trace
    should hit a clear minimum at t=0.
    """
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


def plot_alpha_csd_summary(result_b, save_path, result_s=None,
                           ref_depth_mm=-0.20, sig_z=None,
                           depth_ylim=None):
    """Two-panel plot: alpha-realigned CSD (baseline, and optionally stimulus)."""
    csd_b = result_b['csd_avg']
    depths = result_b['depths']
    cutout_t = result_b['cutout_t_ms']
    n_b = result_b['n_troughs']

    mask_b = (compute_significance_mask(csd_b, result_b['surrogate_mean'],
                                        result_b['surrogate_std'], sig_z)
              if sig_z is not None else None)

    ref_b = result_b.get('ref_alpha_avg')
    ref_depth_b = result_b.get('ref_depth_used')

    def _attach_aligned_pair(ax_csd, ax_ref, attach_cbar):
        """Append a colorbar slot to ax_csd and a matching invisible spacer
        to ax_ref so both have identical plot widths, and tie x-axes."""
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
        n_s = result_s['n_troughs']
        mask_s = (compute_significance_mask(csd_s, result_s['surrogate_mean'],
                                            result_s['surrogate_std'], sig_z)
                  if sig_z is not None else None)
        ref_s = result_s.get('ref_alpha_avg')
        fig, axes = plt.subplots(2, 2, figsize=(13, 6.5),
                                 gridspec_kw={'wspace': 0.35,
                                              'hspace': 0.05,
                                              'height_ratios': [3, 1]})
        vmax_b = np.nanpercentile(np.abs(csd_b), 98)
        vmax_s = np.nanpercentile(np.abs(csd_s), 98)
        vmax = max(vmax_b, vmax_s)
        im = plot_csd_heatmap(axes[0, 0], csd_b, depths, cutout_t,
                              title=f'Baseline alpha aligned CSD ',
                              vmax=vmax, significance_mask=mask_b,
                              depth_ylim=depth_ylim)
        plot_csd_heatmap(axes[0, 1], csd_s, depths, cutout_t,
                         title=f'Stimulus alpha aligned CSD ',
                         vmax=vmax, significance_mask=mask_s,
                         depth_ylim=depth_ylim)
        for ax in axes[0, :]:
            ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)
        _plot_ref_alpha(axes[1, 0], cutout_t, ref_b, ref_depth_b)
        _plot_ref_alpha(axes[1, 1], cutout_t, ref_s,
                        result_s.get('ref_depth_used'))
        # Spacer on the left column (no colorbar), real colorbar on the right.
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
                              title=f'Baseline alpha aligned CSD ',
                              significance_mask=mask_b,
                              depth_ylim=depth_ylim)
        ax_csd.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
        ax_csd.set_xlabel('')
        ax_csd.tick_params(labelbottom=False)
        _plot_ref_alpha(ax_ref, cutout_t, ref_b, ref_depth_b)
        cax = _attach_aligned_pair(ax_csd, ax_ref, attach_cbar=True)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label('CSD (a.u.)\n← sink     source →', fontsize=10)

    ref_label = (f'{ref_depth_mm:+.2f} mm' if ref_depth_mm is not None
                 else 'data-driven (max alpha power)')
    sub = (f'reference depth = {ref_label}; '
           f'centred on alpha trough')
    if sig_z is not None:
        sub += f'; non-significant cells dimmed (|z| < {sig_z})'
    fig.suptitle(f'Alpha-realigned CSD\n{sub}', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(save_path, dpi=160, bbox_inches='tight')
    print(f"  saved {save_path}")
    plt.close(fig)


def plot_evoked_csd(csd, depths, t_ms, n_used, save_path, depth_ylim=None):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = plot_csd_heatmap(ax, csd, depths, t_ms,
                          title=f'CSD centered at stimulus',
                          depth_ylim=depth_ylim)
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
    """Plot the cutout-averaged CSD trace at a single channel near `ref_depth_mm`.

    A clean ~10 Hz oscillation centred on t=0 means the alpha modulation
    survived averaging; a flat trace means cycles weren't phase-locked and
    the heatmap was just saturated by a few outlier cells.
    """
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
    # mark 100 ms (one alpha cycle at 10 Hz) on either side of t=0
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


def plot_alpha_diagnostic(trial_files, alpha_band, baseline_start_ms,
                          ref_depth_mm, save_path, method='phase',
                          highpass_hz=1.0, lfp_source='kernel'):
    """Three-panel sanity check that the alpha trough realignment makes sense:
      (1) reference channel raw + alpha-bandpassed + detected troughs (one trial)
      (2) alpha-band power vs depth (averaged across trials)
      (3) baseline reference signal spectrum (averaged across trials)
    """
    trial = load_trial(trial_files[0])
    lfp, depths, time_ms = select_lfp_and_depths(trial, lfp_source=lfp_source)
    dt = float(time_ms[1] - time_ms[0])
    fs = 1000.0 / dt
    stim = float(trial.get('stim_onset_ms', 2000))
    lfp = preprocess_lfp(lfp, fs, highpass_hz=highpass_hz)
    ref_idx = find_alpha_reference_channel(lfp, depths, fs, alpha_band,
                                           prefer_depth=ref_depth_mm)
    mask = (time_ms >= baseline_start_ms) & (time_ms < stim)
    raw = lfp[ref_idx, mask]
    t_in_win = time_ms[mask]
    filt = bandpass(raw, fs, alpha_band)
    detect = (detect_alpha_troughs_phase if method == 'phase'
              else detect_alpha_troughs_peaks)
    troughs, _ = detect(raw, fs, alpha_band, edge_pad_samples=int(0.05 * fs))

    n_ch = lfp.shape[0]
    pow_accum = np.zeros(n_ch)
    spec_accum = None
    f_ref = None
    n_trials_used = 0
    for fpath in trial_files:
        tr = load_trial(fpath)
        lfp_t, _, time_ms_t = select_lfp_and_depths(tr, lfp_source=lfp_source)
        if lfp_t.shape[0] != n_ch:
            continue
        stim_t = float(tr.get('stim_onset_ms', 2000))
        m = (time_ms_t >= baseline_start_ms) & (time_ms_t < stim_t)
        if not np.any(m):
            continue
        lfp_t = preprocess_lfp(lfp_t, fs, highpass_hz=highpass_hz)
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
            label=f'detected troughs ({method})')
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
    p.add_argument('--alpha_lo', type=float, default=7.0)
    p.add_argument('--alpha_hi', type=float, default=14.0)
    p.add_argument('--warmup_ms', type=float, default=500.0,
                   help='start of baseline window (skip warmup)')
    p.add_argument('--cutout_ms', type=float, default=200.0,
                   help='+- window around each alpha trough to average '
                        '(200 ms = ~2 alpha cycles)')
    p.add_argument('--ref_depth_mm', type=float, default=None,
                   help='cortical depth of the alpha-trough reference channel; '
                        'if omitted, picks the channel with max alpha power')
    p.add_argument('--ref_channel', type=int, default=None,
                   help='electrode channel index of the alpha-trough reference '
                        'channel (0-based, into the depths array). Mutually '
                        'exclusive with --ref_depth_mm.')
    p.add_argument('--ref_layer', type=str, default=None,
                   choices=['L23', 'L4AB', 'L4C', 'L5', 'L6'],
                   help='restrict reference-channel selection to electrodes '
                        'whose depth falls within this layer (z-ranges read '
                        'from config.CONFIG); picks the max-alpha-power '
                        'channel inside the layer. Mutually exclusive with '
                        '--ref_depth_mm and --ref_channel.')
    p.add_argument('--method', type=str, default='phase',
                   choices=['phase', 'peaks'],
                   help='trough detection method (phase: Hilbert-phase '
                        'crossing, the standard; peaks: find_peaks fallback)')
    p.add_argument('--amplitude_weighted', action='store_true',
                   help='weight each cutout by alpha envelope amplitude '
                        'at the trough')
    p.add_argument('--no_spatial_smooth', action='store_true',
                   help='disable Mitzdorf 3-pt Hamming smooth before CSD')
    p.add_argument('--highpass_hz', type=float, default=1.0,
                   help='high-pass cutoff (Hz) applied to LFP before CSD; '
                        'matches typical acquisition-hardware filtering. '
                        'Set 0 to disable. Note: cutoffs <<1 Hz need long '
                        'trials to avoid sosfiltfilt edge ringing.')
    p.add_argument('--lfp_source', type=str, default='both',
                   choices=['kernel', 'current', 'both'],
                   help="which LFP to use: 'kernel' (lfp_matrix), "
                        "'current' (lfp_current_matrix, synaptic-current "
                        "method), or 'both' to run the full pipeline once "
                        "per source into separate subdirectories.")
    p.add_argument('--n_surrogates', type=int, default=0,
                   help='number of phase-shuffled surrogate runs for '
                        'significance testing (0 = off; 200 is typical)')
    p.add_argument('--sig_z', type=float, default=2.0,
                   help='|z| threshold for surrogate significance overlay')
    p.add_argument('--depth_lo', type=float, default=None,
                   help='clip CSD heatmaps to this minimum depth (mm)')
    p.add_argument('--depth_hi', type=float, default=None,
                   help='clip CSD heatmaps to this maximum depth (mm)')
    p.add_argument('--save_dir', type=str, default=None)
    p.add_argument('--no_stim', action='store_true',
                   help='skip the stimulus-window analysis')
    args = p.parse_args()

    n_ref_opts = sum(x is not None for x in
                     (args.ref_channel, args.ref_depth_mm, args.ref_layer))
    if n_ref_opts > 1:
        p.error('--ref_channel, --ref_depth_mm and --ref_layer are mutually '
                'exclusive')

    if args.ref_layer is not None:
        from config.config import CONFIG
        layer_cfg = CONFIG['layers'].get(args.ref_layer)
        if layer_cfg is None:
            p.error(f'layer {args.ref_layer!r} not found in CONFIG[\'layers\']')
        layer_z_min, layer_z_max = layer_cfg['coordinates']['z']
    else:
        layer_z_min = layer_z_max = None

    base_save_dir = args.save_dir or os.path.join(args.trial_dir, 'csd_alpha')

    files = sorted(glob.glob(os.path.join(args.trial_dir, 'trial_*.npz')))
    if not files:
        raise FileNotFoundError(f"no trial_*.npz in {args.trial_dir}")
    print(f"loaded {len(files)} trials")

    alpha_band = (args.alpha_lo, args.alpha_hi)
    spatial_smooth = not args.no_spatial_smooth

    if args.lfp_source == 'both':
        sources = ['kernel', 'current']
    else:
        sources = [args.lfp_source]

    depth_ylim = (None if args.depth_lo is None and args.depth_hi is None
                  else (args.depth_lo, args.depth_hi))

    for lfp_source in sources:
        save_dir = (os.path.join(base_save_dir, lfp_source)
                    if len(sources) > 1 else base_save_dir)
        os.makedirs(save_dir, exist_ok=True)
        print(f"=== lfp_source={lfp_source} -> {save_dir} ===")

        if args.ref_channel is not None:
            sample_trial = np.load(files[0], allow_pickle=True)
            _, sample_depths, _ = select_lfp_and_depths(
                sample_trial, lfp_source=lfp_source)
            n_ch = len(sample_depths)
            if not -n_ch <= args.ref_channel < n_ch:
                raise IndexError(
                    f"--ref_channel {args.ref_channel} out of range for "
                    f"{lfp_source} LFP with {n_ch} channels")
            ref_depth_mm = float(sample_depths[args.ref_channel])
            print(f"  --ref_channel={args.ref_channel} -> "
                  f"depth {ref_depth_mm:+.3f} mm ({lfp_source})")
        elif args.ref_layer is not None:
            sample_trial = np.load(files[0], allow_pickle=True)
            sample_lfp, sample_depths, sample_time_ms = select_lfp_and_depths(
                sample_trial, lfp_source=lfp_source)
            in_layer = np.where((sample_depths >= layer_z_min)
                                & (sample_depths <= layer_z_max))[0]
            if in_layer.size == 0:
                raise ValueError(
                    f"no electrodes within {args.ref_layer} z-range "
                    f"[{layer_z_min:+.2f}, {layer_z_max:+.2f}] mm")
            if in_layer.size == 1:
                best = int(in_layer[0])
                print(f"  --ref_layer={args.ref_layer}: only one electrode "
                      f"in layer -> ch {best} @ {sample_depths[best]:+.3f} mm")
            else:
                fs = 1000.0 / float(sample_time_ms[1] - sample_time_ms[0])
                nperseg = int(min(2.0 * fs, sample_lfp.shape[1]))
                pow_in_layer = np.zeros(in_layer.size)
                for k, ci in enumerate(in_layer):
                    f, pxx = welch(sample_lfp[ci, :], fs=fs, nperseg=nperseg)
                    band = (f >= alpha_band[0]) & (f <= alpha_band[1])
                    pow_in_layer[k] = np.trapezoid(pxx[band], f[band])
                best = int(in_layer[int(np.argmax(pow_in_layer))])
                print(f"  --ref_layer={args.ref_layer}: max-alpha ch among "
                      f"{in_layer.tolist()} -> ch {best} @ "
                      f"{sample_depths[best]:+.3f} mm")
            ref_depth_mm = float(sample_depths[best])
        else:
            ref_depth_mm = args.ref_depth_mm

        plot_alpha_diagnostic(
            files, alpha_band, baseline_start_ms=args.warmup_ms,
            ref_depth_mm=ref_depth_mm, method=args.method,
            save_path=os.path.join(save_dir, 'alpha_diagnostic.png'),
            highpass_hz=args.highpass_hz, lfp_source=lfp_source,
        )

        try:
            csd_evoked, depths_ev, t_ev, n_evoked = trial_average_evoked_csd(
                files, spatial_smooth=spatial_smooth,
                highpass_hz=args.highpass_hz, lfp_source=lfp_source)
            plot_evoked_csd(csd_evoked, depths_ev, t_ev, n_evoked,
                            save_path=os.path.join(save_dir, 'evoked_csd.png'),
                            depth_ylim=depth_ylim)
        except RuntimeError as e:
            print(f"  could not compute evoked CSD: {e}")

        print('computing baseline alpha-realigned CSD...')
        result_b = alpha_realigned_csd(
            files, alpha_band, window_kind='baseline',
            baseline_start_ms=args.warmup_ms,
            cutout_ms=args.cutout_ms, ref_depth_mm=ref_depth_mm,
            method=args.method, amplitude_weighted=args.amplitude_weighted,
            n_surrogates=args.n_surrogates, spatial_smooth=spatial_smooth,
            highpass_hz=args.highpass_hz, lfp_source=lfp_source,
        )
        print(f"  baseline troughs averaged: {result_b['n_troughs']}")

        result_s = None
        if not args.no_stim:
            try:
                print('computing stimulus alpha-realigned CSD...')
                result_s = alpha_realigned_csd(
                    files, alpha_band, window_kind='stimulus',
                    baseline_start_ms=args.warmup_ms,
                    cutout_ms=args.cutout_ms, ref_depth_mm=ref_depth_mm,
                    method=args.method,
                    amplitude_weighted=args.amplitude_weighted,
                    n_surrogates=args.n_surrogates,
                    spatial_smooth=spatial_smooth,
                    highpass_hz=args.highpass_hz, lfp_source=lfp_source,
                )
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
            save_path=os.path.join(save_dir,
                                   'single_channel_cutout_baseline.png'),
            window_label='baseline',
        )
        if result_s is not None:
            plot_single_channel_cutout(
                result_s, ref_depth_mm=ref_for_plot,
                save_path=os.path.join(save_dir,
                                       'single_channel_cutout_stimulus.png'),
                window_label='stimulus',
            )

        sig_z = args.sig_z if args.n_surrogates > 0 else None
        plot_alpha_csd_summary(
            result_b,
            save_path=os.path.join(save_dir, 'alpha_realigned_csd.png'),
            result_s=result_s, ref_depth_mm=ref_depth_mm, sig_z=sig_z,
            depth_ylim=depth_ylim,
        )
        plot_alpha_csd_summary(
            result_b,
            save_path=os.path.join(save_dir,
                                   'alpha_realigned_csd_baseline_only.png'),
            result_s=None, ref_depth_mm=ref_depth_mm, sig_z=sig_z,
            depth_ylim=depth_ylim,
        )


if __name__ == '__main__':
    main()