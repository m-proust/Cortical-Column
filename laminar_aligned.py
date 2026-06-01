"""Laminar profile of LFP and CSD aligned to alpha and gamma troughs.

Replicates van Kerkoerle et al. 2014 (PNAS) Fig. 3 panels A,B (alpha-aligned
LFP and CSD) and E,F (gamma-aligned LFP and CSD). MUA panels (C,D / G,H)
are deliberately omitted.

Output layout::

    <save_dir>/
        L23/
            kernel/
                alpha_csd.png
                alpha_lfp.png
                gamma_csd.png
                gamma_lfp.png
            current/
                ...
        L4AB/
            ...
        ...

For each reference layer, the max-alpha-power channel inside that layer is
picked as the trough-detection reference (via the existing
`find_alpha_reference_channel` helper, fed the appropriate band).
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import welch, hilbert
from scipy.ndimage import gaussian_filter

from alpha_csd_layers import (
    LAYER_Z_RANGES,
    LAYER_ORDER,
    load_trial,
    select_lfp_and_depths,
    bandpass,
    preprocess_lfp,
    compute_csd,
    find_alpha_reference_channel,
    detect_alpha_troughs_phase,
    detect_alpha_troughs_peaks,
    plot_csd_heatmap,
)


ALPHA_BAND = (7.0, 14.0)
GAMMA_BAND = (60.0, 80.0)


def realigned_signal(trial_files, band, signal_kind, window_kind,
                     baseline_start_ms, cutout_ms, ref_layer,
                     method='phase', spatial_smooth=True, highpass_hz=1.0,
                     lfp_source='kernel', amplitude_weighted=False,
                     envelope_pct=None):
    """Trough-realigned average of either the LFP or the CSD.

    Picks the max-band-power channel inside `ref_layer` as the trough
    reference, detects troughs in that channel within the requested
    window, and averages cutouts of `signal_kind` ('lfp' or 'csd') across
    troughs and trials.

    amplitude_weighted: if True, each cutout is weighted by the band-pass
        envelope at the trough — strong-cycle troughs dominate the average,
        noisy weak-envelope troughs contribute less.
    envelope_pct: if not None (0..100), only keep troughs whose envelope
        is above this percentile (computed per-trial across the window).
        E.g. envelope_pct=75 keeps the top 25% strongest cycles.

    Returns dict with keys: avg, depths, cutout_t_ms, n_troughs,
    ref_depth_used, ref_band_avg.
    """
    if signal_kind not in ('lfp', 'csd'):
        raise ValueError(f"signal_kind must be 'lfp' or 'csd', got "
                         f"{signal_kind!r}")
    detect = (detect_alpha_troughs_phase if method == 'phase'
              else detect_alpha_troughs_peaks)
    layer_z_min, layer_z_max = LAYER_Z_RANGES[ref_layer]

    accum = None
    counts = None
    ref_accum = None
    ref_counts = None
    depths_ref = None
    cutout_t = None
    ref_depth_used = None
    n_troughs = 0

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

        # pick reference channel within the requested layer
        in_layer = np.where((depths >= layer_z_min)
                            & (depths <= layer_z_max))[0]
        if in_layer.size == 0:
            continue
        if in_layer.size == 1:
            ref_idx = int(in_layer[0])
        else:
            nperseg = int(min(2.0 * fs, lfp.shape[1]))
            pow_in_layer = np.zeros(in_layer.size)
            for k, ci in enumerate(in_layer):
                f, pxx = welch(lfp[ci, :], fs=fs, nperseg=nperseg)
                bmask = (f >= band[0]) & (f <= band[1])
                pow_in_layer[k] = np.trapezoid(pxx[bmask], f[bmask])
            ref_idx = int(in_layer[int(np.argmax(pow_in_layer))])

        if ref_depth_used is None:
            ref_depth_used = float(depths[ref_idx])

        # signal to be realigned
        if signal_kind == 'csd':
            signal_2d = compute_csd(lfp, depths, spatial_smooth=spatial_smooth)
        else:
            signal_2d = lfp
        n_channels, n_t = signal_2d.shape

        ref_signal = lfp[ref_idx, :]
        ref_signal_band = bandpass(ref_signal, fs, band)
        envelope_full = np.abs(hilbert(ref_signal_band))

        cutout_samples = int(round(cutout_ms / dt))
        ref_in_win = ref_signal[idx_in_window]
        rel_troughs, _ = detect(ref_in_win, fs, band,
                                edge_pad_samples=cutout_samples)
        abs_troughs = idx_in_window[rel_troughs]
        keep_cut = ((abs_troughs - cutout_samples >= 0)
                    & (abs_troughs + cutout_samples + 1 <= n_t))
        abs_troughs = abs_troughs[keep_cut]

        if envelope_pct is not None and len(abs_troughs) > 0:
            env_in_win = envelope_full[idx_in_window]
            thresh = np.percentile(env_in_win, envelope_pct)
            trough_env = envelope_full[abs_troughs]
            abs_troughs = abs_troughs[trough_env >= thresh]

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
            seg = signal_2d[:, tr - cutout_samples: tr + cutout_samples + 1]
            ref_seg = ref_signal_band[
                tr - cutout_samples: tr + cutout_samples + 1]
            valid = ~np.isnan(seg)
            ref_valid = ~np.isnan(ref_seg)
            w = float(envelope_full[tr]) if amplitude_weighted else 1.0
            accum += np.where(valid, seg * w, 0.0)
            counts += valid.astype(float) * w
            ref_accum += np.where(ref_valid, ref_seg * w, 0.0)
            ref_counts += ref_valid.astype(float) * w
            n_troughs += 1

    if accum is None:
        raise RuntimeError(f"no {ref_layer} {band} troughs found in any trial")

    avg = np.divide(accum, counts, out=np.full_like(accum, np.nan),
                    where=counts > 0)
    ref_avg = np.divide(ref_accum, ref_counts,
                        out=np.full_like(ref_accum, np.nan),
                        where=ref_counts > 0)

    return {
        'avg': avg,
        'depths': depths_ref,
        'cutout_t_ms': cutout_t,
        'n_troughs': n_troughs,
        'ref_depth_used': ref_depth_used,
        'ref_band_avg': ref_avg,
    }


def plot_lfp_heatmap(ax, lfp, depths, t, title, smooth_sigma=(0.5, 1.5),
                     vmax=None, show_layers=True, mark_zero_depth=True,
                     depth_ylim=None):
    """Heatmap of trough-aligned LFP. Negative blue, positive red — same
    convention as van Kerkoerle Fig. 3A/E."""
    lfp_plot = lfp.copy()
    finite = np.isfinite(lfp_plot)
    lfp_filled = np.where(finite, lfp_plot, 0.0)
    if smooth_sigma is not None:
        lfp_filled = gaussian_filter(lfp_filled, sigma=smooth_sigma)
    lfp_filled = np.where(finite, lfp_filled, np.nan)

    if vmax is None:
        vmax = np.nanpercentile(np.abs(lfp_filled), 98)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    extent = [t[0], t[-1], depths[0], depths[-1]]
    im = ax.imshow(
        lfp_filled, aspect='auto', origin='lower', extent=extent,
        cmap='RdBu_r', norm=norm, interpolation='bilinear',
    )

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


def _plot_ref_trace(ax, cutout_t, ref_avg, ref_depth_used, band):
    if ref_avg is None or np.all(np.isnan(ref_avg)):
        ax.set_axis_off()
        return
    ax.plot(cutout_t, ref_avg, color='k', lw=1.2)
    ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax.set_xlim(cutout_t[0], cutout_t[-1])
    ax.set_xlabel(f'time from trough (ms)')
    label = (f'{band[0]:.0f}-{band[1]:.0f} Hz @ {ref_depth_used:+.2f} mm'
             if ref_depth_used is not None
             else f'{band[0]:.0f}-{band[1]:.0f} Hz (ref)')
    ax.set_ylabel(label, fontsize=9)
    ax.tick_params(labelsize=8)


def plot_aligned_pair(result_b, result_s, signal_kind, band, ref_layer,
                      save_path, depth_ylim=None, time_xlim=None):
    """Two-column figure: baseline (left) and stimulus (right) trough-aligned
    average of `signal_kind`, with a slim reference-trace panel below each."""
    sig_b = result_b['avg']
    depths = result_b['depths']
    cutout_t = result_b['cutout_t_ms']
    ref_b = result_b.get('ref_band_avg')
    ref_depth_b = result_b.get('ref_depth_used')

    plotter = plot_csd_heatmap if signal_kind == 'csd' else plot_lfp_heatmap
    cbar_label = ('CSD (a.u.)\n← sink     source →' if signal_kind == 'csd'
                  else 'LFP (mV)\n← negative   positive →')

    def _attach_aligned_pair(ax_top, ax_bot, attach_cbar):
        div_top = make_axes_locatable(ax_top)
        div_bot = make_axes_locatable(ax_bot)
        cax = div_top.append_axes('right', size='3%', pad=0.08)
        spacer = div_bot.append_axes('right', size='3%', pad=0.08)
        spacer.set_axis_off()
        ax_bot.sharex(ax_top)
        if not attach_cbar:
            cax.set_axis_off()
            return None
        return cax

    band_label = f'{band[0]:.0f}-{band[1]:.0f} Hz'

    if result_s is not None:
        sig_s = result_s['avg']
        ref_s = result_s.get('ref_band_avg')
        ref_depth_s = result_s.get('ref_depth_used')

        fig, axes = plt.subplots(2, 2, figsize=(13, 6.5),
                                 gridspec_kw={'wspace': 0.35,
                                              'hspace': 0.05,
                                              'height_ratios': [3, 1]})
        vmax_b = np.nanpercentile(np.abs(sig_b), 98)
        vmax_s = np.nanpercentile(np.abs(sig_s), 98)
        vmax = max(vmax_b, vmax_s)

        kw = {} if signal_kind == 'csd' else {}
        im = plotter(axes[0, 0], sig_b, depths, cutout_t,
                     title=f'Baseline {band_label} aligned '
                           f'{signal_kind.upper()}',
                     vmax=vmax, depth_ylim=depth_ylim, **kw)
        plotter(axes[0, 1], sig_s, depths, cutout_t,
                title=f'Stimulus {band_label} aligned {signal_kind.upper()}',
                vmax=vmax, depth_ylim=depth_ylim, **kw)
        for ax in axes[0, :]:
            ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)
            if time_xlim is not None:
                ax.set_xlim(time_xlim)
        _plot_ref_trace(axes[1, 0], cutout_t, ref_b, ref_depth_b, band)
        _plot_ref_trace(axes[1, 1], cutout_t, ref_s, ref_depth_s, band)
        if time_xlim is not None:
            for ax in axes[1, :]:
                ax.set_xlim(time_xlim)
        _attach_aligned_pair(axes[0, 0], axes[1, 0], attach_cbar=False)
        cax = _attach_aligned_pair(axes[0, 1], axes[1, 1], attach_cbar=True)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(cbar_label, fontsize=10)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5),
                                 gridspec_kw={'hspace': 0.05,
                                              'height_ratios': [3, 1]})
        ax_top, ax_bot = axes
        im = plotter(ax_top, sig_b, depths, cutout_t,
                     title=f'Baseline {band_label} aligned '
                           f'{signal_kind.upper()}',
                     depth_ylim=depth_ylim)
        ax_top.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
        ax_top.set_xlabel('')
        ax_top.tick_params(labelbottom=False)
        if time_xlim is not None:
            ax_top.set_xlim(time_xlim)
        _plot_ref_trace(ax_bot, cutout_t, ref_b, ref_depth_b, band)
        if time_xlim is not None:
            ax_bot.set_xlim(time_xlim)
        cax = _attach_aligned_pair(ax_top, ax_bot, attach_cbar=True)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(cbar_label, fontsize=10)

    sub = (f'reference layer = {ref_layer} '
           f'(max {band_label} power channel); centred on trough')
    fig.suptitle(f'{band_label} trough-aligned {signal_kind.upper()}\n{sub}',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(save_path, dpi=160, bbox_inches='tight')
    print(f"  saved {save_path}")
    plt.close(fig)


def run_one(trial_files, ref_layer, lfp_source, save_dir, args):
    """Compute and save the four plots (alpha/gamma × LFP/CSD) for one
    (ref_layer, lfp_source) pair."""
    os.makedirs(save_dir, exist_ok=True)
    spatial_smooth = not args.no_spatial_smooth
    depth_ylim = (None if args.depth_lo is None and args.depth_hi is None
                  else (args.depth_lo, args.depth_hi))

    bands = [('alpha', ALPHA_BAND), ('gamma', GAMMA_BAND)]
    signals = ['csd', 'lfp']
    band_xlim = {'alpha': None, 'gamma': (-20.0, 20.0)}

    for band_name, band in bands:
        amp_weighted = (args.gamma_amp_weighted
                        if band_name == 'gamma' else False)
        env_pct = (args.gamma_envelope_pct
                   if band_name == 'gamma' else None)
        for signal_kind in signals:
            try:
                result_b = realigned_signal(
                    trial_files, band, signal_kind,
                    window_kind='baseline',
                    baseline_start_ms=args.warmup_ms,
                    cutout_ms=args.cutout_ms, ref_layer=ref_layer,
                    method=args.method, spatial_smooth=spatial_smooth,
                    highpass_hz=args.highpass_hz, lfp_source=lfp_source,
                    amplitude_weighted=amp_weighted,
                    envelope_pct=env_pct,
                )
            except RuntimeError as e:
                print(f"  baseline {band_name}/{signal_kind}: {e}")
                continue

            result_s = None
            if not args.no_stim:
                try:
                    result_s = realigned_signal(
                        trial_files, band, signal_kind,
                        window_kind='stimulus',
                        baseline_start_ms=args.warmup_ms,
                        cutout_ms=args.cutout_ms, ref_layer=ref_layer,
                        method=args.method, spatial_smooth=spatial_smooth,
                        highpass_hz=args.highpass_hz, lfp_source=lfp_source,
                        amplitude_weighted=amp_weighted,
                        envelope_pct=env_pct,
                    )
                except RuntimeError as e:
                    print(f"  stimulus {band_name}/{signal_kind}: {e}")
                    result_s = None

            fname = f'{band_name}_{signal_kind}.png'
            print(f"  [{ref_layer}/{lfp_source}] {band_name}/{signal_kind}: "
                  f"{result_b['n_troughs']} baseline troughs"
                  + (f", {result_s['n_troughs']} stim troughs"
                     if result_s is not None else ""))
            plot_aligned_pair(
                result_b, result_s, signal_kind=signal_kind, band=band,
                ref_layer=ref_layer,
                save_path=os.path.join(save_dir, fname),
                depth_ylim=depth_ylim,
                time_xlim=band_xlim.get(band_name),
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trial_dir', type=str, required=True)
    p.add_argument('--warmup_ms', type=float, default=500.0)
    p.add_argument('--cutout_ms', type=float, default=200.0)
    p.add_argument('--method', type=str, default='phase',
                   choices=['phase', 'peaks'])
    p.add_argument('--no_spatial_smooth', action='store_true')
    p.add_argument('--highpass_hz', type=float, default=1.0)
    p.add_argument('--depth_lo', type=float, default=-0.62)
    p.add_argument('--depth_hi', type=float, default=1.1)
    p.add_argument('--save_dir', type=str, default=None)
    p.add_argument('--no_stim', action='store_true')
    p.add_argument('--layers', type=str, nargs='+', default=None,
                   help='subset of layers to process; default = all '
                        f'{LAYER_ORDER}')
    p.add_argument('--lfp_sources', type=str, nargs='+',
                   default=['kernel', 'current'],
                   choices=['kernel', 'current'])
    p.add_argument('--gamma_amp_weighted', action='store_true',
                   help='weight each gamma cutout by the band-pass envelope '
                        'at the trough — strong cycles dominate the average. '
                        'Helps reveal travelling-wave structure when gamma '
                        'is a weak shoulder on a stronger rhythm.')
    p.add_argument('--gamma_envelope_pct', type=float, default=None,
                   help='only keep gamma troughs whose envelope is above '
                        'this percentile (per-trial, in-window). E.g. 75 '
                        'keeps the top 25%% strongest cycles. Drops noisy '
                        'low-amplitude triggers.')
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.trial_dir, 'trial_*.npz')))
    if not files:
        raise FileNotFoundError(f"no trial_*.npz in {args.trial_dir}")
    print(f"loaded {len(files)} trials")

    base_save_dir = (args.save_dir
                     or os.path.join(args.trial_dir, 'laminar_aligned'))
    os.makedirs(base_save_dir, exist_ok=True)

    layers = args.layers or LAYER_ORDER
    for ref_layer in layers:
        if ref_layer not in LAYER_Z_RANGES:
            print(f"skipping unknown layer {ref_layer!r}")
            continue
        for lfp_source in args.lfp_sources:
            save_dir = os.path.join(base_save_dir, ref_layer, lfp_source)
            print(f"=== ref_layer={ref_layer}  lfp_source={lfp_source} "
                  f"-> {save_dir} ===")
            try:
                run_one(files, ref_layer, lfp_source, save_dir, args)
            except Exception as e:
                print(f"  failed for {ref_layer}/{lfp_source}: {e}")


if __name__ == '__main__':
    main()
