"""Per-trial, per-electrode dominant-frequency catalog for the no-lesion control.

For each control trial and each bipolar channel we:
    1. compute the multitaper PSD on the last WINDOW_MS of the trial (no
       stimulus, baseline only) using the same pipeline as the lesion analysis
    2. detect peaks in log10(PSD) using scipy.signal.find_peaks with a minimum
       prominence and minimum separation in log-frequency
    3. record the *dominant* peak (highest log-power) and every other peak

Outputs (under figures/lesions_<date>/_control_peaks/):
    control_peaks_all.csv            one row per peak  (trial, channel, depth,
                                                        freq, log10_power,
                                                        prominence, is_dominant)
    control_dominant_per_channel.csv per-channel summary of dominant peak
                                     (mode, median, IQR, dominant band)
    control_peak_histogram.png       depth x frequency 2D histogram of all
                                     peaks (where rhythms live across layers)
    control_dominant_histogram.png   same but only dominant peaks
    control_per_trial_dominant.png   per-channel plot of dominant freq across
                                     trials (scatter / strip)
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from laminar_power_change import load_trials
from lesion_power_change import _bipolar_current_from_trials, count_trials
from lesion_significance2 import per_trial_psd, WINDOW_MS, FREQ_RANGE, BANDS


LFP_KEY = 'bipolar_lfp'
MIN_PROM_LOG10 = 0.05      # minimum prominence in log10(PSD) units
MIN_SEP_OCTAVES = 0.2      # minimum spacing between detected peaks
DOMINANT_MIN_FREQ = 2.0    # ignore the lowest-freq bump for "dominant" choice
                           # (raw bipolar LFP has a DC-adjacent rise)


def band_of(freq):
    for name, (lo, hi) in BANDS.items():
        if lo <= freq < hi:
            return name
    return 'out'


def detect_peaks_in_psd(freqs, psd, min_prominence=MIN_PROM_LOG10,
                        min_sep_octaves=MIN_SEP_OCTAVES):
    """Detect peaks in log10(psd). Returns list of dicts."""
    y = np.log10(np.maximum(psd, 1e-20))
    idx, props = find_peaks(y, prominence=min_prominence)
    if idx.size == 0:
        return []
    # enforce min separation by keeping highest-prominence peaks first
    lf = np.log2(np.maximum(freqs[idx], 1e-3))
    order = np.argsort(-props['prominences'])
    kept = []
    for j in order:
        fi = lf[j]
        if all(abs(fi - lf[k]) >= min_sep_octaves for k in kept):
            kept.append(j)
    out = []
    for j in kept:
        out.append({
            'freq': float(freqs[idx[j]]),
            'log10_power': float(y[idx[j]]),
            'prominence': float(props['prominences'][j]),
        })
    out.sort(key=lambda p: p['freq'])
    return out


def main(lesion_root, fig_root, window_ms=WINDOW_MS, lfp_key=LFP_KEY,
         min_prominence=MIN_PROM_LOG10):
    ctrl_dir = os.path.join(lesion_root, 'control')
    n_ctrl = count_trials(ctrl_dir)
    print(f'[control] loading {n_ctrl} trials')
    ctrl_trials = load_trials(ctrl_dir, n_ctrl)
    if lfp_key == 'bipolar_lfp_current':
        ctrl_trials = _bipolar_current_from_trials(ctrl_trials)

    psd_all, freqs, depths, labels = per_trial_psd(
        ctrl_trials, lfp_key, window_ms=window_ms)
    n_trial, n_ch, n_f = psd_all.shape
    print(f'  PSD shape {psd_all.shape}; freqs {freqs[0]:.1f}–{freqs[-1]:.1f} Hz')

    out_dir = os.path.join(fig_root, '_control_peaks')
    os.makedirs(out_dir, exist_ok=True)

    rows = []  # all peaks
    dominant_rows = []  # dominant peak per (trial, channel)

    for tr in range(n_trial):
        for ch in range(n_ch):
            peaks = detect_peaks_in_psd(freqs, psd_all[tr, ch],
                                        min_prominence=min_prominence)
            if not peaks:
                continue
            # choose dominant: highest log10_power among peaks above the
            # DOMINANT_MIN_FREQ cutoff; fall back to highest power overall.
            candidates = [p for p in peaks if p['freq'] >= DOMINANT_MIN_FREQ]
            if not candidates:
                candidates = peaks
            dom = max(candidates, key=lambda p: p['log10_power'])
            for p in peaks:
                rows.append({
                    'trial': tr,
                    'channel': str(labels[ch]),
                    'depth': float(depths[ch]),
                    'freq': p['freq'],
                    'log10_power': p['log10_power'],
                    'prominence': p['prominence'],
                    'band': band_of(p['freq']),
                    'is_dominant': (p is dom),
                })
            dominant_rows.append({
                'trial': tr,
                'channel': str(labels[ch]),
                'depth': float(depths[ch]),
                'freq': dom['freq'],
                'log10_power': dom['log10_power'],
                'prominence': dom['prominence'],
                'band': band_of(dom['freq']),
            })

    # ---- CSV: every peak ----
    all_csv = os.path.join(out_dir, 'control_peaks_all.csv')
    with open(all_csv, 'w') as fh:
        fh.write('trial,channel,depth,freq,log10_power,prominence,band,is_dominant\n')
        for r in rows:
            fh.write(f'{r["trial"]},{r["channel"]},{r["depth"]:+.3f},'
                     f'{r["freq"]:.3f},{r["log10_power"]:.4f},'
                     f'{r["prominence"]:.4f},{r["band"]},'
                     f'{int(r["is_dominant"])}\n')
    print(f'saved {all_csv}  ({len(rows)} peaks)')

    # ---- per-channel summary of dominant frequency ----
    summary_path = os.path.join(out_dir, 'control_dominant_per_channel.csv')
    summary_lines = [
        'channel,depth,n_trials_with_peak,mode_freq,mode_count,'
        'median_freq,q25_freq,q75_freq,dominant_band,'
        'frac_delta,frac_theta,frac_alpha,frac_beta,frac_low_gamma,frac_high_gamma'
    ]
    per_ch_dom = {}
    for ch in range(n_ch):
        per_ch_dom[ch] = [r for r in dominant_rows
                          if r['channel'] == str(labels[ch])]
    for ch in range(n_ch):
        rows_ch = per_ch_dom[ch]
        if not rows_ch:
            continue
        freqs_ch = np.array([r['freq'] for r in rows_ch])
        band_counts = {b: 0 for b in BANDS}
        for r in rows_ch:
            if r['band'] in band_counts:
                band_counts[r['band']] += 1
        dom_band = max(band_counts, key=band_counts.get)
        # mode: round to nearest 0.5 Hz to group near-identical peaks
        bins = np.round(freqs_ch * 2) / 2
        vals, counts = np.unique(bins, return_counts=True)
        mode_idx = int(np.argmax(counts))
        mode_freq = float(vals[mode_idx])
        mode_count = int(counts[mode_idx])
        median_f = float(np.median(freqs_ch))
        q25 = float(np.percentile(freqs_ch, 25))
        q75 = float(np.percentile(freqs_ch, 75))
        n = len(rows_ch)
        fracs = {b: band_counts[b] / n for b in BANDS}
        summary_lines.append(
            f'{labels[ch]},{depths[ch]:+.3f},{n},{mode_freq:.2f},{mode_count},'
            f'{median_f:.2f},{q25:.2f},{q75:.2f},{dom_band},'
            f'{fracs["delta"]:.2f},{fracs["theta"]:.2f},{fracs["alpha"]:.2f},'
            f'{fracs["beta"]:.2f},{fracs["low_gamma"]:.2f},{fracs["high_gamma"]:.2f}'
        )
    with open(summary_path, 'w') as fh:
        fh.write('\n'.join(summary_lines) + '\n')
    print(f'saved {summary_path}')

    # ---- depth x frequency histogram (all peaks) ----
    f_edges = np.logspace(np.log10(max(freqs[0], 0.5)),
                          np.log10(freqs[-1]), 50)
    d_centers = depths
    d_edges = np.concatenate([
        [d_centers[0] - (d_centers[1] - d_centers[0]) / 2],
        (d_centers[:-1] + d_centers[1:]) / 2,
        [d_centers[-1] + (d_centers[-1] - d_centers[-2]) / 2],
    ])

    def _hist2d(records, title, out_png):
        if not records:
            return
        f_arr = np.array([r['freq'] for r in records])
        d_arr = np.array([r['depth'] for r in records])
        H, _, _ = np.histogram2d(d_arr, f_arr, bins=[d_edges, f_edges])
        # flip depth so superficial is at top
        order = np.argsort(-d_centers)
        H = H[order]
        depth_lbl = d_centers[order]

        fig, ax = plt.subplots(figsize=(11, 6))
        im = ax.imshow(H, aspect='auto', origin='upper', cmap='magma',
                       extent=[f_edges[0], f_edges[-1],
                               len(depth_lbl) - 0.5, -0.5])
        ax.set_xscale('log')
        ax.set_yticks(range(len(depth_lbl)))
        ax.set_yticklabels([f'{labels[order[i]]}  z={depth_lbl[i]:+.2f}'
                            for i in range(len(depth_lbl))], fontsize=8)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_title(title, fontsize=11)
        # band boundaries
        for name, (lo, hi) in BANDS.items():
            ax.axvline(lo, color='cyan', alpha=0.4, lw=0.6, linestyle='--')
            ax.axvline(hi, color='cyan', alpha=0.4, lw=0.6, linestyle='--')
            ax.text(np.sqrt(lo * hi), -0.6, name,
                    ha='center', va='bottom', fontsize=7, color='cyan',
                    clip_on=False)
        plt.colorbar(im, ax=ax, label='peak count (trials)')
        fig.tight_layout()
        fig.savefig(out_png, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f'saved {out_png}')

    _hist2d(rows,
            'CONTROL: peak frequencies per electrode (all detected peaks)',
            os.path.join(out_dir, 'control_peak_histogram.png'))
    _hist2d(dominant_rows,
            'CONTROL: dominant peak frequency per electrode '
            f'(>={DOMINANT_MIN_FREQ:.1f} Hz, highest log-power per trial)',
            os.path.join(out_dir, 'control_dominant_histogram.png'))

    # ---- per-trial dominant freq strip plot ----
    fig, ax = plt.subplots(figsize=(11, 6))
    order = np.argsort(-depths)
    for yi, ch in enumerate(order):
        ch_label = str(labels[ch])
        ch_rows = [r for r in dominant_rows if r['channel'] == ch_label]
        if not ch_rows:
            continue
        xs = [r['freq'] for r in ch_rows]
        ax.scatter(xs, [yi] * len(xs), s=20, alpha=0.7,
                   color='C0', edgecolors='k', linewidths=0.3)
        ax.scatter([np.median(xs)], [yi], marker='|', s=160, color='C3',
                   linewidths=1.6)
    ax.set_xscale('log')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f'{labels[order[i]]}  z={depths[order[i]]:+.2f}'
                        for i in range(len(order))], fontsize=8)
    ax.set_xlabel('Dominant peak frequency (Hz)')
    ax.set_title('CONTROL: per-trial dominant peak per electrode  '
                 '(blue dots = trials, red bar = median)', fontsize=11)
    for name, (lo, hi) in BANDS.items():
        ax.axvline(lo, color='gray', alpha=0.4, lw=0.6, linestyle='--')
        ax.axvline(hi, color='gray', alpha=0.4, lw=0.6, linestyle='--')
        ax.text(np.sqrt(lo * hi), -0.6, name, ha='center', va='bottom',
                fontsize=7, color='gray', clip_on=False)
    ax.set_xlim(max(freqs[0], 0.5), freqs[-1])
    ax.grid(True, which='both', alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'control_per_trial_dominant.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_dir}/control_per_trial_dominant.png')

    # ---- print a quick textual report ----
    print('\n=== CONTROL DOMINANT PEAK PER CHANNEL ===')
    print(f'{"channel":>12} {"depth":>7} {"mode":>7} {"median":>7} '
          f'{"IQR":>15} {"dom_band":>10} {"n":>4}')
    for ch in range(n_ch):
        rows_ch = per_ch_dom[ch]
        if not rows_ch:
            continue
        freqs_ch = np.array([r['freq'] for r in rows_ch])
        bins = np.round(freqs_ch * 2) / 2
        vals, counts = np.unique(bins, return_counts=True)
        mode_freq = float(vals[np.argmax(counts)])
        median_f = float(np.median(freqs_ch))
        q25 = float(np.percentile(freqs_ch, 25))
        q75 = float(np.percentile(freqs_ch, 75))
        band_counts = {b: 0 for b in BANDS}
        for r in rows_ch:
            if r['band'] in band_counts:
                band_counts[r['band']] += 1
        dom_band = max(band_counts, key=band_counts.get)
        print(f'{labels[ch]:>12} {depths[ch]:+7.3f} {mode_freq:>7.2f} '
              f'{median_f:>7.2f} {q25:>5.1f}-{q75:<7.1f} '
              f'{dom_band:>10} {len(rows_ch):>4d}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lesion-root', default='results/lesions_2026-06-02')
    parser.add_argument('--fig-root', default='figures/lesions_2026-06-02')
    parser.add_argument('--window-ms', type=int, default=WINDOW_MS)
    parser.add_argument('--lfp-key', default=LFP_KEY)
    parser.add_argument('--min-prominence', type=float, default=MIN_PROM_LOG10,
                        help='Minimum peak prominence in log10(PSD) units.')
    args = parser.parse_args()
    main(args.lesion_root, args.fig_root, window_ms=args.window_ms,
         lfp_key=args.lfp_key, min_prominence=args.min_prominence)
