"""Per-lesion frequency-resolved significance test vs control baseline.

For each lesion folder under results/lesions_<date>/:
  - load all trials, compute per-trial multitaper PSD on the last
    `WINDOW_MS` of each trial, on bipolar_lfp_current
  - per channel and per freq bin, test lesion vs control trials with a
    permutation test (n_perm shuffles of trial labels); FDR-correct across
    freq bins per channel
  - signed log2 fold-change masked by significance

Outputs:
  figures/lesions_<date>/_significance/
    <lesion>_sig_heatmap.png        # channel x freq, signed log2FC, sig-masked
    <lesion>_sig_summary.txt        # peak freqs, direction, layer
  figures/lesions_<date>/_significance/
    grand_summary_log2fc.png        # lesion x freq (averaged across mid layers)
    grand_summary_sig_counts.png    # lesion x band, count of sig channels
    grand_summary.csv               # machine-readable table
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import detrend

from laminar_power_change import load_trials, multitaper_psd, _time_vector_for_key
from lesion_power_change import _bipolar_current_from_trials, count_trials


WINDOW_MS = 2000
FREQ_RANGE = (1, 100)
N_PERM = 1000
ALPHA_FDR = 0.05
LFP_KEY = 'bipolar_lfp_current'
RNG_SEED = 0

BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'low_gamma':  (30, 50),
    'high_gamma': (50, 100),
}


def _draw_band_lines(ax, freqs, label=True):
    """Overlay dashed vertical lines at the edges of each frequency band."""
    edges = sorted({e for band in BANDS.values() for e in band})
    f_lo, f_hi = float(freqs[0]), float(freqs[-1])
    for e in edges:
        if f_lo <= e <= f_hi:
            ax.axvline(e, color='k', alpha=0.45, lw=0.7, linestyle='--')
    if label:
        y_top = ax.get_ylim()[1]
        for name, (b_lo, b_hi) in BANDS.items():
            lo = max(b_lo, f_lo)
            hi = min(b_hi, f_hi)
            if hi <= lo:
                continue
            ax.text((lo + hi) / 2, y_top, name,
                    ha='center', va='bottom', fontsize=7, color='k',
                    clip_on=False)


def per_trial_psd(trials, lfp_key, window_ms=WINDOW_MS):
    """Return psd_all[n_trial, n_channel, n_freq], freqs, depths, channel_labels."""
    t0 = _time_vector_for_key(trials[0], lfp_key)
    fs = 1000.0 / float(np.mean(np.diff(t0)))
    n_channels = trials[0][lfp_key].shape[0]
    depths = trials[0]['channel_depths']
    if lfp_key in ('bipolar_lfp', 'bipolar_lfp_current') and len(depths) > n_channels:
        depths = (depths[:-1] + depths[1:]) / 2
    else:
        depths = depths[:n_channels]
    labels = trials[0]['channel_labels'][:n_channels]

    psd_list = []
    freqs_ref = None
    for trial in trials:
        time = _time_vector_for_key(trial, lfp_key)
        t_end = float(time[-1])
        mask = (time >= t_end - window_ms) & (time <= t_end)
        trial_psd = []
        valid = True
        for ch in range(n_channels):
            seg = trial[lfp_key][ch][mask].copy()
            if len(seg) == 0 or np.any(np.isnan(seg)):
                valid = False
                break
            seg = detrend(seg)
            nfft = 2 ** int(np.ceil(np.log2(len(seg))))
            f, psd = multitaper_psd(seg, fs=fs, NW=2, nfft=nfft)
            if freqs_ref is None:
                freqs_ref = f
            trial_psd.append(psd)
        if valid:
            psd_list.append(np.stack(trial_psd, axis=0))

    if not psd_list:
        raise RuntimeError(f'no valid trials for {lfp_key}')

    psd_all = np.stack(psd_list, axis=0)
    mask_f = (freqs_ref >= FREQ_RANGE[0]) & (freqs_ref <= FREQ_RANGE[1])
    return psd_all[:, :, mask_f], freqs_ref[mask_f], np.asarray(depths), np.asarray(labels)


def perm_test_log2fc(psd_ctrl, psd_les, n_perm=N_PERM, rng=None):
    """Permutation test on log-power means, per channel x freq.

    psd_ctrl : (n_ctrl, n_ch, n_f)
    psd_les  : (n_les,  n_ch, n_f)
    Returns:
        log2fc : (n_ch, n_f)  signed log2 fold-change (lesion / control)
        pvals  : (n_ch, n_f)  two-sided permutation p-values
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    log_ctrl = np.log(psd_ctrl + 1e-20)
    log_les = np.log(psd_les + 1e-20)
    mean_ctrl = log_ctrl.mean(0)
    mean_les = log_les.mean(0)
    log2fc = (mean_les - mean_ctrl) / np.log(2.0)
    obs = mean_les - mean_ctrl  # natural log diff; sign matches log2fc

    pooled = np.concatenate([log_ctrl, log_les], axis=0)
    n_ctrl, n_les = log_ctrl.shape[0], log_les.shape[0]
    n_total = n_ctrl + n_les

    abs_obs = np.abs(obs)
    count = np.zeros_like(obs)
    for _ in range(n_perm):
        idx = rng.permutation(n_total)
        a = pooled[idx[:n_ctrl]].mean(0)
        b = pooled[idx[n_ctrl:]].mean(0)
        count += (np.abs(b - a) >= abs_obs).astype(np.float64)
    pvals = (count + 1.0) / (n_perm + 1.0)
    return log2fc, pvals


def fdr_bh(pvals, alpha=ALPHA_FDR):
    """Benjamini-Hochberg FDR per row (per channel)."""
    p = np.asarray(pvals)
    out = np.zeros_like(p, dtype=bool)
    for i in range(p.shape[0]):
        row = p[i]
        n = row.size
        order = np.argsort(row)
        ranked = row[order]
        thresh = alpha * (np.arange(1, n + 1) / n)
        passed = ranked <= thresh
        if passed.any():
            k = np.max(np.where(passed)[0])
            cutoff = ranked[k]
            out[i] = row <= cutoff
    return out


def plot_sig_heatmap(freqs, depths, labels, log2fc, sig_mask,
                     lesion_name, out_path, vlim=None):
    """Channel x freq heatmap of signed log2FC, with non-sig grayed out."""
    masked = np.where(sig_mask, log2fc, np.nan)
    n_ch = log2fc.shape[0]

    if vlim is None:
        finite = log2fc[np.isfinite(log2fc)]
        if finite.size:
            vlim = float(np.nanpercentile(np.abs(finite), 98))
            vlim = max(vlim, 0.1)
        else:
            vlim = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5),
                             gridspec_kw={'width_ratios': [1, 1]})
    extent = [freqs[0], freqs[-1], n_ch - 0.5, -0.5]
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)

    im0 = axes[0].imshow(log2fc, aspect='auto', cmap='RdBu_r',
                         extent=extent, norm=norm, interpolation='nearest')
    axes[0].set_title('log2 fold-change (lesion / control) — all bins')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('bipolar channel')
    axes[0].set_yticks(range(n_ch))
    axes[0].set_yticklabels([f'{lab}  z={z:+.2f}' for lab, z in zip(labels, depths)],
                            fontsize=7)
    _draw_band_lines(axes[0], freqs)
    plt.colorbar(im0, ax=axes[0], label='log2 FC')

    im1 = axes[1].imshow(masked, aspect='auto', cmap='RdBu_r',
                         extent=extent, norm=norm, interpolation='nearest')
    axes[1].set_facecolor('#dddddd')
    axes[1].set_title(f'log2 FC, FDR<{ALPHA_FDR} only')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_yticks(range(n_ch))
    axes[1].set_yticklabels([f'{lab}  z={z:+.2f}' for lab, z in zip(labels, depths)],
                            fontsize=7)
    _draw_band_lines(axes[1], freqs)
    plt.colorbar(im1, ax=axes[1], label='log2 FC')

    fig.suptitle(f'Lesion {lesion_name}  — significance (n_perm={N_PERM})',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def summarize_lesion(freqs, depths, labels, log2fc, sig_mask):
    """Per-band summary stats for the lesion."""
    rows = []
    n_ch = log2fc.shape[0]
    for band, (f_lo, f_hi) in BANDS.items():
        m = (freqs >= f_lo) & (freqs <= f_hi)
        if not np.any(m):
            continue
        for ch in range(n_ch):
            lfc = log2fc[ch, m]
            sig = sig_mask[ch, m]
            n_sig = int(sig.sum())
            n_tot = int(m.sum())
            # only summarize bands where >25% of bins are significant
            if n_sig / max(n_tot, 1) < 0.25:
                continue
            # signed direction
            sig_vals = lfc[sig]
            direction = 'up' if np.median(sig_vals) > 0 else 'down'
            rows.append({
                'band': band,
                'channel': str(labels[ch]),
                'depth': float(depths[ch]),
                'frac_sig': n_sig / n_tot,
                'median_log2fc': float(np.median(sig_vals)),
                'max_abs_log2fc': float(np.max(np.abs(sig_vals))),
                'direction': direction,
            })
    return rows


def write_summary_text(rows, lesion_name, out_path):
    with open(out_path, 'w') as fh:
        fh.write(f'# Lesion {lesion_name}\n')
        fh.write(f'# Significant band/channel combinations (>=25% of bins '
                 f'pass FDR<{ALPHA_FDR})\n\n')
        if not rows:
            fh.write('  (no band/channel combination reached the threshold — '
                     'this lesion looks like noise)\n')
            return
        fh.write(f'{"band":10} {"channel":12} {"depth":>7} '
                 f'{"frac_sig":>9} {"med_log2FC":>11} {"max_abs":>9} '
                 f'{"dir":>5}\n')
        for r in rows:
            fh.write(f'{r["band"]:10} {r["channel"]:12} {r["depth"]:+7.3f} '
                     f'{r["frac_sig"]:9.2f} {r["median_log2fc"]:+11.3f} '
                     f'{r["max_abs_log2fc"]:9.3f} {r["direction"]:>5}\n')


def run(lesion_root, fig_root, window_ms=WINDOW_MS, n_perm=N_PERM,
        lfp_key=LFP_KEY):
    control_dir = os.path.join(lesion_root, 'control')
    if not os.path.isdir(control_dir):
        raise FileNotFoundError(f'no control folder in {lesion_root}')

    n_ctrl = count_trials(control_dir)
    print(f'[control] loading {n_ctrl} trials')
    ctrl_trials = load_trials(control_dir, n_ctrl)
    ctrl_trials = _bipolar_current_from_trials(ctrl_trials)
    psd_ctrl, freqs, depths, labels = per_trial_psd(
        ctrl_trials, lfp_key, window_ms=window_ms)
    print(f'  ctrl PSD shape {psd_ctrl.shape}, freqs {freqs[0]:.1f}–{freqs[-1]:.1f} Hz')

    out_dir = os.path.join(fig_root, '_significance')
    os.makedirs(out_dir, exist_ok=True)

    lesion_names = sorted(
        d for d in os.listdir(lesion_root)
        if os.path.isdir(os.path.join(lesion_root, d)) and d != 'control'
    )

    rng = np.random.default_rng(RNG_SEED)

    grand_log2fc = []        # one row per lesion: log2FC averaged across mid layers
    grand_sig_count = []     # one row per lesion: sig channels per band (up - down)
    grand_names = []
    per_lesion_rows = {}

    for lname in lesion_names:
        ldir = os.path.join(lesion_root, lname)
        n = count_trials(ldir)
        if n == 0:
            print(f'[{lname}] no trials — skip')
            continue
        print(f'[{lname}] loading {n} trials')
        try:
            les_trials = load_trials(ldir, n)
        except Exception as exc:
            print(f'[{lname}] load failed: {exc}')
            continue
        les_trials = _bipolar_current_from_trials(les_trials)
        try:
            psd_les, freqs_l, _, _ = per_trial_psd(
                les_trials, lfp_key, window_ms=window_ms)
        except Exception as exc:
            print(f'[{lname}] PSD failed: {exc}')
            continue
        if not np.allclose(freqs_l, freqs):
            print(f'[{lname}] freq grid mismatch — skip')
            continue

        log2fc, pvals = perm_test_log2fc(psd_ctrl, psd_les,
                                         n_perm=n_perm, rng=rng)
        sig_mask = fdr_bh(pvals, alpha=ALPHA_FDR)

        n_sig = int(sig_mask.sum())
        n_total = sig_mask.size
        print(f'  {n_sig}/{n_total} bins significant ({100*n_sig/n_total:.1f}%)')

        plot_sig_heatmap(
            freqs, depths, labels, log2fc, sig_mask,
            lname, os.path.join(out_dir, f'{lname}_sig_heatmap.png'))

        rows = summarize_lesion(freqs, depths, labels, log2fc, sig_mask)
        write_summary_text(
            rows, lname, os.path.join(out_dir, f'{lname}_summary.txt'))
        per_lesion_rows[lname] = rows

        # mid-layer = channels whose depth is between -0.3 and +0.7 (rough L4-L5)
        mid_mask = (depths >= -0.3) & (depths <= 0.7)
        if not mid_mask.any():
            mid_mask = np.ones_like(depths, dtype=bool)
        mid_lfc = np.where(sig_mask, log2fc, 0.0)[mid_mask].mean(0)
        grand_log2fc.append(mid_lfc)

        band_score = []
        for band, (f_lo, f_hi) in BANDS.items():
            m = (freqs >= f_lo) & (freqs <= f_hi)
            up = int(((log2fc > 0) & sig_mask)[:, m].any(axis=1).sum())
            dn = int(((log2fc < 0) & sig_mask)[:, m].any(axis=1).sum())
            band_score.append(up - dn)  # net signed channel-count
        grand_sig_count.append(band_score)
        grand_names.append(lname)

    if not grand_names:
        print('no lesions processed')
        return

    grand_log2fc = np.stack(grand_log2fc, axis=0)
    grand_sig_count = np.array(grand_sig_count, dtype=int)

    # ---------- grand heatmap (lesion x freq) ----------
    vlim = float(np.nanpercentile(np.abs(grand_log2fc), 98))
    vlim = max(vlim, 0.1)
    fig, ax = plt.subplots(figsize=(13, max(4, 0.35 * len(grand_names) + 2)))
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    extent = [freqs[0], freqs[-1], len(grand_names) - 0.5, -0.5]
    im = ax.imshow(grand_log2fc, aspect='auto', cmap='RdBu_r',
                   extent=extent, norm=norm, interpolation='nearest')
    ax.set_yticks(range(len(grand_names)))
    ax.set_yticklabels(grand_names, fontsize=9)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title('Sig-masked log2 FC, averaged across mid-layer (z∈[-0.3,+0.7]) channels')
    _draw_band_lines(ax, freqs)
    plt.colorbar(im, ax=ax, label='log2 FC')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'grand_summary_log2fc.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_dir}/grand_summary_log2fc.png')

    # ---------- per-band sig channel-count ----------
    band_names = list(BANDS.keys())
    vlim2 = max(1, int(np.max(np.abs(grand_sig_count))))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(grand_names) + 2)))
    norm2 = TwoSlopeNorm(vmin=-vlim2, vcenter=0, vmax=vlim2)
    im = ax.imshow(grand_sig_count, aspect='auto', cmap='RdBu_r', norm=norm2,
                   interpolation='nearest')
    ax.set_yticks(range(len(grand_names)))
    ax.set_yticklabels(grand_names, fontsize=9)
    ax.set_xticks(range(len(band_names)))
    ax.set_xticklabels(band_names, rotation=30, fontsize=9)
    ax.set_title('Sig channels per band  (up − down)')
    for i in range(grand_sig_count.shape[0]):
        for j in range(grand_sig_count.shape[1]):
            v = grand_sig_count[i, j]
            if v != 0:
                ax.text(j, i, f'{v:+d}', ha='center', va='center',
                        fontsize=8,
                        color='white' if abs(v) > vlim2 * 0.6 else 'black')
    plt.colorbar(im, ax=ax, label='channels (up − down)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'grand_summary_sig_counts.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_dir}/grand_summary_sig_counts.png')

    # ---------- machine-readable CSV ----------
    csv_path = os.path.join(out_dir, 'grand_summary.csv')
    with open(csv_path, 'w') as fh:
        fh.write('lesion,band,channel,depth,frac_sig,median_log2fc,'
                 'max_abs_log2fc,direction\n')
        for lname in grand_names:
            for r in per_lesion_rows.get(lname, []):
                fh.write(f'{lname},{r["band"]},{r["channel"]},'
                         f'{r["depth"]:+.3f},{r["frac_sig"]:.3f},'
                         f'{r["median_log2fc"]:+.4f},'
                         f'{r["max_abs_log2fc"]:.4f},{r["direction"]}\n')
    print(f'saved {csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lesion-root', default='results/lesions_2026-06-02')
    parser.add_argument('--fig-root', default='figures/lesions_2026-06-02')
    parser.add_argument('--window-ms', type=int, default=WINDOW_MS)
    parser.add_argument('--n-perm', type=int, default=N_PERM)
    parser.add_argument('--lfp-key', default=LFP_KEY)
    args = parser.parse_args()
    run(args.lesion_root, args.fig_root,
        window_ms=args.window_ms, n_perm=args.n_perm, lfp_key=args.lfp_key)
