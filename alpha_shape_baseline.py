"""
Alpha shape in the baseline period.

For each trial in results/trials_26_04, take the bipolar LFP during the
baseline window (0 .. baseline_ms), band-pass it in the alpha band
(default 7-13 Hz), detect peaks and troughs, and characterise their
variability:
  - peak amplitude vs trough amplitude variability (CV, std)
  - inter-peak interval vs inter-trough interval variability
Aggregated per cortical layer (L23, L4AB, L4C, L5, L6).

Optionally restrict the analysis to the trials with the highest alpha
power (per channel, ranked by mean baseline 7-13 Hz power).

Outputs PNGs into results/trials_26_04/alpha_shape_baseline/.
"""

from __future__ import annotations

import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks, welch, decimate


LAYER_Z_RANGES = {
    'L23':  (0.45, 1.10),
    'L4AB': (0.14, 0.45),
    'L4C':  (-0.14, 0.14),
    'L5':   (-0.34, -0.14),
    'L6':   (-0.62, -0.34),
}
LAYER_ORDER = ['L23', 'L4AB', 'L4C', 'L5', 'L6']
LAYER_COLORS = {
    'L23':  '#1f77b4',
    'L4AB': '#2ca02c',
    'L4C':  '#9467bd',
    'L5':   '#d62728',
    'L6':   '#8c564b',
}


def bandpass(x, fs, lo=5.0, hi=20.0, order=4):
    sos = butter(order, [lo, hi], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, x, axis=-1)


def assign_layer(depth_mm: float) -> str | None:
    for layer, (lo, hi) in LAYER_Z_RANGES.items():
        if lo <= depth_mm <= hi:
            return layer
    return None


def detect_peaks_troughs(sig, fs, fmin=5.0):
    # min separation = 1/(2*f_max_alpha+a bit), but to be safe use 1/(fmax+5)
    # Actually we want at least one cycle worth between same-type extrema.
    # Cycle period at the slowest passband edge (5 Hz) = 200 ms.
    # Distance between successive peaks (or successive troughs) >= ~50 ms
    # to avoid double-counting jitter; use 1/(2*hi) as a safe floor.
    min_dist_samples = max(1, int(fs * (1.0 / 30.0)))  # ~33 ms
    peaks, _ = find_peaks(sig, distance=min_dist_samples)
    troughs, _ = find_peaks(-sig, distance=min_dist_samples)
    return peaks, troughs


def cv(x):
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.nan
    m = np.mean(np.abs(x))
    if m == 0:
        return np.nan
    return float(np.std(x) / m)


def summarise_channel(sig_filt, fs):
    peaks, troughs = detect_peaks_troughs(sig_filt, fs)
    if peaks.size < 3 or troughs.size < 3:
        return None
    peak_amps = sig_filt[peaks]
    trough_amps = sig_filt[troughs]
    peak_intervals_ms = np.diff(peaks) * 1000.0 / fs
    trough_intervals_ms = np.diff(troughs) * 1000.0 / fs
    return {
        'peaks': peaks,
        'troughs': troughs,
        'peak_amps': peak_amps,
        'trough_amps': trough_amps,
        'peak_intervals_ms': peak_intervals_ms,
        'trough_intervals_ms': trough_intervals_ms,
        'peak_amp_std': float(np.std(peak_amps)),
        'trough_amp_std': float(np.std(trough_amps)),
        'peak_amp_cv': cv(peak_amps),
        'trough_amp_cv': cv(trough_amps),
        'peak_int_cv': cv(peak_intervals_ms),
        'trough_int_cv': cv(trough_intervals_ms),
    }


def load_trial(path, source='bipolar'):
    """source:
       'bipolar'         - kernel-method bipolar LFP (precomputed)
       'current'         - synaptic-current monopolar LFP (per-channel demeaned)
       'current_bipolar' - bipolar derivation of the current-method LFP
                           (computed on the fly: ch[i+1] - ch[i])
    """
    d = np.load(path, allow_pickle=True)
    if source == 'bipolar':
        sig = d['bipolar_matrix']
        t_ms = d['time_array_ms']
        depths = d['channel_depths']
    elif source in ('current', 'current_bipolar'):
        mono = np.asarray(d['lfp_current_matrix'], dtype=np.float64)
        t_ms = np.asarray(d['time_current_ms'], dtype=np.float64)
        z = np.asarray(d['electrode_positions'][:, 2], dtype=np.float64)
        if source == 'current':
            sig = mono - mono.mean(axis=-1, keepdims=True)
            depths = z
        else:
            # bipolar = ch[i+1] - ch[i]; depth = midpoint
            sig = np.diff(mono, axis=0)
            depths = 0.5 * (z[:-1] + z[1:])
    else:
        raise ValueError(f'unknown source {source!r}')
    baseline_ms = float(d['baseline_ms'])
    return sig, t_ms, depths, baseline_ms


def alpha_power(sig, fs, lo, hi):
    """Mean PSD in [lo, hi] via Welch (1-s segments)."""
    nperseg = min(sig.size, int(fs * 1.0))
    f, p = welch(sig, fs=fs, nperseg=nperseg)
    band = (f >= lo) & (f <= hi)
    if not band.any():
        return np.nan
    return float(np.trapz(p[band], f[band]))


def run(trial_dir: Path, out_dir: Path, lo=5.0, hi=20.0,
        top_frac=1.0, source='bipolar'):
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_files = sorted(glob(str(trial_dir / 'trial_*.npz')))
    if not trial_files:
        raise FileNotFoundError(f'no trial_*.npz files in {trial_dir}')

    # Read first to set up axes
    bp0, t_ms0, depths0, baseline_ms = load_trial(trial_files[0], source=source)
    n_chan = bp0.shape[0]
    fs_raw = 1000.0 / (t_ms0[1] - t_ms0[0])     # samples / s
    n_trials = len(trial_files)

    # Decimate to ~1 kHz to keep filter design well-conditioned for 7-13 Hz
    decim = max(1, int(round(fs_raw / 1000.0)))
    fs = fs_raw / decim
    print(f'source = {source}, fs_raw = {fs_raw:.1f} Hz, '
          f'decimate by {decim} -> fs = {fs:.1f} Hz')
    print(f'n_channels = {n_chan}, baseline = {baseline_ms:.0f} ms, '
          f'n_trials = {n_trials}, band = {lo:.1f}-{hi:.1f} Hz, '
          f'top_frac = {top_frac:.2f}')

    # baseline mask (in raw-sample space)
    mask = t_ms0 < baseline_ms
    t_base_ms = t_ms0[mask][::decim]

    # Map each channel to a layer
    channel_layers = [assign_layer(float(z)) for z in depths0]
    print('channel -> layer:')
    for i, (z, layer) in enumerate(zip(depths0, channel_layers)):
        print(f'  ch{i:2d}  z={z:+.3f} mm  layer={layer}')

    # ---- Pass 1: load + filter every trial, score 7-13 Hz power ------------
    filt_cache = []           # list of (n_chan, n_samples) filtered arrays
    power_table = np.full((n_trials, n_chan), np.nan)   # alpha power per (trial, ch)
    for ti, f in enumerate(trial_files):
        sig_full, t_ms, _, _ = load_trial(f, source=source)
        if sig_full.shape[1] != t_ms.size:
            raise ValueError(f'shape mismatch in {f}')
        baseline = sig_full[:, mask]
        if decim > 1:
            baseline = decimate(baseline, decim, axis=-1, ftype='iir',
                                zero_phase=True)
        filt = bandpass(baseline, fs, lo=lo, hi=hi, order=4)
        filt_cache.append(filt)
        for c in range(n_chan):
            # Score the *unfiltered* (but decimated) baseline so the band
            # ranking isn't contaminated by the filter's edge response.
            power_table[ti, c] = alpha_power(baseline[c], fs, lo, hi)

    # ---- Trial selection per channel (top_frac by alpha power) ------------
    n_keep = max(1, int(np.ceil(top_frac * n_trials)))
    keep_mask = np.zeros_like(power_table, dtype=bool)
    for c in range(n_chan):
        order = np.argsort(power_table[:, c])[::-1]   # high power first
        keep_mask[order[:n_keep], c] = True
    print(f'keeping top {n_keep}/{n_trials} trials per channel '
          f'(by {lo:.1f}-{hi:.1f} Hz power)')

    # ---- Pass 2: extract peaks/troughs from kept (trial, channel) only ----
    per_chan = [[] for _ in range(n_chan)]
    example_traces = [None] * n_chan
    example_extrema = [None] * n_chan
    example_trial_idx = [None] * n_chan
    # Pick ONE random trial to show across all channels (same trial everywhere
    # so laminar phase relationships are readable).
    rng = np.random.default_rng()
    example_ti = int(rng.integers(0, n_trials))
    print(f'example trace trial (random, shared across channels) = {example_ti}')
    for c in range(n_chan):
        kept_trials = np.where(keep_mask[:, c])[0]
        example_trial_idx[c] = example_ti
        for ti in kept_trials:
            sig = filt_cache[ti][c]
            stats = summarise_channel(sig, fs)
            if stats is None:
                continue
            per_chan[c].append(stats)
        # Example trace is from the shared random trial, regardless of whether
        # it was in the top-power kept set for this channel.
        sig_ex = filt_cache[example_ti][c]
        stats_ex = summarise_channel(sig_ex, fs)
        example_traces[c] = sig_ex
        if stats_ex is not None:
            example_extrema[c] = (stats_ex['peaks'], stats_ex['troughs'])

    # Per-channel aggregated metrics
    chan_metrics = {
        'peak_amp_cv':    np.full(n_chan, np.nan),
        'trough_amp_cv':  np.full(n_chan, np.nan),
        'peak_amp_std':   np.full(n_chan, np.nan),
        'trough_amp_std': np.full(n_chan, np.nan),
        'peak_int_cv':    np.full(n_chan, np.nan),
        'trough_int_cv':  np.full(n_chan, np.nan),
    }
    for c in range(n_chan):
        if not per_chan[c]:
            continue
        for key in chan_metrics:
            vals = [s[key] for s in per_chan[c] if np.isfinite(s[key])]
            if vals:
                chan_metrics[key][c] = float(np.mean(vals))

    # Pool by layer
    layer_pool = {layer: {'peak_amps': [], 'trough_amps': [],
                          'peak_intervals_ms': [], 'trough_intervals_ms': []}
                  for layer in LAYER_ORDER}
    for c, layer in enumerate(channel_layers):
        if layer is None:
            continue
        for s in per_chan[c]:
            for k in layer_pool[layer]:
                layer_pool[layer][k].append(s[k])
    for layer in LAYER_ORDER:
        for k in layer_pool[layer]:
            layer_pool[layer][k] = (np.concatenate(layer_pool[layer][k])
                                    if layer_pool[layer][k] else np.array([]))

    # ---- Plot 1: example filtered LFP trace for EVERY channel, grouped by layer ----
    # Build, per layer, the list of (channel_index) sorted by depth (deep -> superficial within layer).
    layer_to_chans = {layer: [] for layer in LAYER_ORDER}
    for c, layer in enumerate(channel_layers):
        if layer is not None:
            layer_to_chans[layer].append(c)
    # sort by depth descending so superficial-most is on top within a layer
    for layer in LAYER_ORDER:
        layer_to_chans[layer].sort(key=lambda i: -depths0[i])

    n_rows = sum(max(1, len(layer_to_chans[layer])) for layer in LAYER_ORDER)
    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(11, max(2.0, 1.4 * n_rows)),
                             sharex=True)
    if n_rows == 1:
        axes = [axes]
    row = 0
    legend_done = False
    for layer in LAYER_ORDER:
        chans = layer_to_chans[layer]
        if not chans:
            ax = axes[row]
            ax.set_visible(False)
            row += 1
            continue
        for c in chans:
            ax = axes[row]
            row += 1
            if example_traces[c] is None:
                ax.text(0.5, 0.5, f'{layer} ch{c} z={depths0[c]:+.2f} (no data)',
                        transform=ax.transAxes, ha='center', va='center',
                        color='grey')
                ax.set_yticks([])
                continue
            sig = example_traces[c]
            ax.plot(t_base_ms, sig, color=LAYER_COLORS[layer], lw=0.9)
            ti = example_trial_idx[c]
            ax.set_ylabel(f'{layer}\nch{c} z={depths0[c]:+.2f}\ntrial {ti}',
                          fontsize=8)
            ax.axhline(0, color='grey', lw=0.5, alpha=0.5)
    axes[-1].set_xlabel('time (ms)')
    xlo, xhi = 1000.0, 2000.0
    xticks = np.arange(xlo, xhi + 1, 100.0)
    for ax in axes:
        ax.set_xlim(xlo, xhi)
        ax.set_xticks(xticks)
        ax.grid(axis='x', color='grey', lw=0.4, alpha=0.4)
    fig.suptitle(f'Baseline LFP ({source}) filtered {lo:.1f}-{hi:.1f} Hz '
                 f'— every electrode (highest-{lo:.0f}-{hi:.0f}Hz-power trial per channel)')
    fig.tight_layout()
    p1 = out_dir / 'example_filtered_traces.png'
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # ---- Plot 2: per-channel CV of peak vs trough amplitudes (across depth) -
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    order = np.argsort(depths0)
    z_sorted = np.array(depths0)[order]
    pk_cv = chan_metrics['peak_amp_cv'][order]
    tr_cv = chan_metrics['trough_amp_cv'][order]
    pk_int = chan_metrics['peak_int_cv'][order]
    tr_int = chan_metrics['trough_int_cv'][order]
    layer_color_per_chan = [LAYER_COLORS.get(channel_layers[i], 'grey')
                            for i in order]

    ax = axes[0]
    ax.plot(pk_cv, z_sorted, '-o', color='red', label='peak amp CV')
    ax.plot(tr_cv, z_sorted, '-o', color='black', label='trough amp CV')
    ax.set_xlabel('CV of extremum amplitude')
    ax.set_ylabel('depth z (mm)')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(pk_int, z_sorted, '-o', color='red',
            label='peak-to-peak interval CV')
    ax.plot(tr_int, z_sorted, '-o', color='black',
            label='trough-to-trough interval CV')
    ax.set_xlabel('CV of inter-extremum interval')
    ax.legend()
    ax.grid(alpha=0.3)

    # shade layer ranges
    for axx in axes:
        for layer, (zlo, zhi) in LAYER_Z_RANGES.items():
            axx.axhspan(zlo, zhi, color=LAYER_COLORS[layer], alpha=0.08)
            axx.text(axx.get_xlim()[1], 0.5 * (zlo + zhi), layer,
                     va='center', ha='left', fontsize=8,
                     color=LAYER_COLORS[layer])
    fig.suptitle(f'Cycle-to-cycle variability of {lo:.1f}-{hi:.1f} Hz '
                 f'baseline LFP ({source}, '
                 f'top {n_keep}/{n_trials} trials per channel)')
    fig.tight_layout()
    p2 = out_dir / 'cv_vs_depth.png'
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    # ---- Plot 3: per-electrode histogram of peak vs trough amplitudes ------
    # one column per channel, grouped left-to-right by layer (in LAYER_ORDER)
    chan_order = []
    for layer in LAYER_ORDER:
        chan_order.extend(layer_to_chans[layer])
    n_cols = max(1, len(chan_order))
    fig, axes = plt.subplots(2, n_cols,
                             figsize=(2.2 * n_cols, 6),
                             squeeze=False)
    for j, c in enumerate(chan_order):
        layer = channel_layers[c]
        # pool peaks/troughs/intervals across kept trials for THIS channel
        pk = (np.concatenate([s['peak_amps'] for s in per_chan[c]])
              if per_chan[c] else np.array([]))
        tr = (np.concatenate([s['trough_amps'] for s in per_chan[c]])
              if per_chan[c] else np.array([]))
        pi = (np.concatenate([s['peak_intervals_ms'] for s in per_chan[c]])
              if per_chan[c] else np.array([]))
        ti = (np.concatenate([s['trough_intervals_ms'] for s in per_chan[c]])
              if per_chan[c] else np.array([]))

        ax = axes[0, j]
        if pk.size:
            ax.hist(pk, bins=30, color='red', alpha=0.6, label='peaks')
        if tr.size:
            ax.hist(tr, bins=30, color='black', alpha=0.5, label='troughs')
        ax.set_title(f'{layer}\nch{c} z={depths0[c]:+.2f}',
                     color=LAYER_COLORS.get(layer, 'grey'),
                     fontsize=9)
        ax.axvline(0, color='grey', lw=0.5)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_ylabel('count\n(amplitude)')
            ax.legend(fontsize=7)

        ax = axes[1, j]
        if pi.size:
            ax.hist(pi, bins=30, color='red', alpha=0.6, label='peak-peak')
        if ti.size:
            ax.hist(ti, bins=30, color='black', alpha=0.5, label='trough-trough')
        ax.set_xlabel('interval (ms)', fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_ylabel('count\n(interval)')
            ax.legend(fontsize=7)
    fig.suptitle(f'Per-electrode distribution of peak vs trough amplitudes '
                 f'and inter-extremum intervals ({source}, '
                 f'{lo:.1f}-{hi:.1f} Hz, '
                 f'top {n_keep}/{n_trials} trials per channel)')
    fig.tight_layout()
    p3 = out_dir / 'channel_histograms.png'
    fig.savefig(p3, dpi=160)
    plt.close(fig)

    # ---- Plot 4: alpha-power per (trial, channel), with selection ---------
    fig, ax = plt.subplots(figsize=(10, 5))
    order = np.argsort(depths0)
    z_sorted = np.array(depths0)[order]
    pt_sorted = power_table[:, order]
    im = ax.imshow(pt_sorted.T, aspect='auto', origin='lower',
                   extent=[-0.5, n_trials - 0.5,
                           z_sorted[0], z_sorted[-1]],
                   cmap='magma')
    plt.colorbar(im, ax=ax, label=f'{lo:.0f}-{hi:.0f} Hz power')
    # mark kept trials with a small dot
    keep_sorted = keep_mask[:, order]
    rows, cols = np.where(keep_sorted.T)
    if rows.size:
        ax.plot(cols, z_sorted[rows], 'w.', ms=3, alpha=0.8)
    ax.set_xlabel('trial')
    ax.set_ylabel('depth z (mm)')
    ax.set_title(f'Baseline {lo:.0f}-{hi:.0f} Hz power per (trial, channel) '
                 f'[{source}] — white dots = kept (top {n_keep}/{n_trials})')
    fig.tight_layout()
    p4 = out_dir / 'alpha_power_selection.png'
    fig.savefig(p4, dpi=160)
    plt.close(fig)

    # ---- Summary table ----------------------------------------------------
    def _f(x, fmt='{:8.4f}'):
        return fmt.format(x) if np.isfinite(x) else '   nan  '

    lines = ['== per layer ==',
             'layer  n_peaks  n_troughs  peak_amp_std  trough_amp_std  '
             'peak_amp_CV  trough_amp_CV  peakInt_CV  troughInt_CV  '
             'mean_peakInt_ms  mean_troughInt_ms']
    for layer in LAYER_ORDER:
        pk = layer_pool[layer]['peak_amps']
        tr = layer_pool[layer]['trough_amps']
        pi = layer_pool[layer]['peak_intervals_ms']
        ti = layer_pool[layer]['trough_intervals_ms']

        peak_std = float(np.std(pk)) if pk.size else np.nan
        trough_std = float(np.std(tr)) if tr.size else np.nan
        peak_amp_cv = cv(pk)
        trough_amp_cv = cv(tr)
        peak_int_cv = cv(pi)
        trough_int_cv = cv(ti)
        mean_pi = float(np.mean(pi)) if pi.size else np.nan
        mean_ti = float(np.mean(ti)) if ti.size else np.nan
        lines.append(
            f'{layer:5s}  {pk.size:7d}  {tr.size:9d}  '
            f'{_f(peak_std)}  {_f(trough_std)}  '
            f'{_f(peak_amp_cv)}  {_f(trough_amp_cv)}  '
            f'{_f(peak_int_cv)}  {_f(trough_int_cv)}  '
            f'{_f(mean_pi)}  {_f(mean_ti)}'
        )

    lines.append('')
    lines.append('== per electrode ==')
    lines.append('layer  ch  z(mm)   n_peaks  n_troughs  peak_amp_std  '
                 'trough_amp_std  peak_amp_CV  trough_amp_CV  peakInt_CV  '
                 'troughInt_CV  mean_peakInt_ms  mean_troughInt_ms')
    for layer in LAYER_ORDER:
        for c in layer_to_chans[layer]:
            pk = (np.concatenate([s['peak_amps'] for s in per_chan[c]])
                  if per_chan[c] else np.array([]))
            tr = (np.concatenate([s['trough_amps'] for s in per_chan[c]])
                  if per_chan[c] else np.array([]))
            pi = (np.concatenate([s['peak_intervals_ms'] for s in per_chan[c]])
                  if per_chan[c] else np.array([]))
            ti = (np.concatenate([s['trough_intervals_ms'] for s in per_chan[c]])
                  if per_chan[c] else np.array([]))
            peak_std = float(np.std(pk)) if pk.size else np.nan
            trough_std = float(np.std(tr)) if tr.size else np.nan
            mean_pi = float(np.mean(pi)) if pi.size else np.nan
            mean_ti = float(np.mean(ti)) if ti.size else np.nan
            lines.append(
                f'{layer:5s}  {c:2d}  {depths0[c]:+.3f}  '
                f'{pk.size:7d}  {tr.size:9d}  '
                f'{_f(peak_std)}  {_f(trough_std)}  '
                f'{_f(cv(pk))}  {_f(cv(tr))}  '
                f'{_f(cv(pi))}  {_f(cv(ti))}  '
                f'{_f(mean_pi)}  {_f(mean_ti)}'
            )
    summary = '\n'.join(lines)
    print('\n' + summary)
    (out_dir / 'summary.txt').write_text(summary + '\n')

    print(f'\nSaved:\n  {p1}\n  {p2}\n  {p3}\n  {p4}\n  '
          f'{out_dir / "summary.txt"}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trial_dir', type=str,
                   default='results/trials_26_04')
    p.add_argument('--out_dir', type=str, default=None,
                   help='default: <trial_dir>/alpha_shape_baseline_<lo>_<hi>'
                        '_top<int(top_frac*100)>')
    p.add_argument('--lo', type=float, default=5.0)
    p.add_argument('--hi', type=float, default=20.0)
    p.add_argument('--top_frac', type=float, default=0.5,
                   help='fraction of highest-alpha-power trials kept per '
                        'channel (default 0.5)')
    p.add_argument('--source', type=str, default='bipolar',
                   choices=['bipolar', 'current', 'current_bipolar'],
                   help='which LFP signal to analyse '
                        '(bipolar = kernel-method bipolar; '
                        'current = synaptic-current monopolar; '
                        'current_bipolar = bipolar derivation of current LFP)')
    args = p.parse_args()

    trial_dir = Path(args.trial_dir)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        tag = (f'{args.source}_{args.lo:.0f}_{args.hi:.0f}_'
               f'top{int(args.top_frac*100):03d}')
        out_dir = trial_dir / f'alpha_shape_baseline_{tag}'
    run(trial_dir, out_dir, lo=args.lo, hi=args.hi,
        top_frac=args.top_frac, source=args.source)


if __name__ == '__main__':
    main()
