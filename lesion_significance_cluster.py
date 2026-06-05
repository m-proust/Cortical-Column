"""Paired cluster-based permutation test (Maris & Oostenveld 2007) for lesions.

This is the field-standard test for spectral/spatiotemporal data where bin-wise
FDR is too conservative because neighbouring frequency bins are correlated
(here, via the multitaper / Welch bandwidth). It controls the family-wise error
rate at level ALPHA across the whole (channel x frequency) map by working in
terms of contiguous clusters of supra-threshold bins.

Design assumption: trial i is matched between control and each lesion (same
network/baseline/stim seeds — see trials.py). We exploit this with a paired
sign-flip test on per-trial log2 fold-changes:

    d_i(ch, f) = log2( P_les[i, ch, f] / P_ctrl[i, ch, f] )

Procedure (per lesion):
  1. Per-bin observed paired t-statistic:  t_obs(ch,f) = mean_i(d_i) / SEM_i(d_i)
  2. Threshold |t_obs| > T_CLUSTER (cluster-forming threshold, default
     t-critical at p=0.01 two-sided for n_trial-1 df).
  3. Find connected components ("clusters") in the (channel x freq) binary
     map. Connectivity is along the frequency axis only by default (each
     channel is its own row); set CLUSTER_CONN_CHANNELS=True to also connect
     neighbouring channels (spatial clustering across depth).
  4. Cluster mass = sum of t_obs over each cluster (signed).
  5. Sign-flip null: for each of N_PERM permutations, flip each trial's sign
     independently, recompute t-map, threshold the same way, find clusters,
     keep the largest |mass|. Build the null distribution of max-|cluster
     mass|.
  6. Two-sided p for each observed cluster:
        p = (1 + #{null_max >= |obs_mass|}) / (N_PERM + 1)
  7. Clusters with p < ALPHA are reported as significant; the union of their
     bins is the significance mask used for plotting / summary.

Outputs (under figures/lesions_<date>/_significance_cluster/):
    <lesion>_cluster_heatmap.png   channel x freq, signed log2FC, sig-masked
    <lesion>_cluster_summary.txt   one line per surviving cluster + per-band
                                   tally of sig channels for the human reader
    grand_summary_log2fc.png       lesion x freq, mid-layer mean log2FC
    grand_summary_cluster_counts.png  lesion x band, signed #sig channels
    grand_summary_clusters.csv     machine-readable cluster table
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats as sp_stats
from scipy.ndimage import label as nd_label, zoom as nd_zoom
from scipy.signal import find_peaks, medfilt

from laminar_power_change import load_trials
from lesion_power_change import _bipolar_current_from_trials, count_trials
from lesion_significance2 import (
    per_trial_psd, _band_overlay,
    WINDOW_MS, FREQ_RANGE, BANDS,
)


N_PERM = 1000
ALPHA = 0.05
CLUSTER_P_FORMING = 0.01     # cluster-forming threshold (two-sided p)
CLUSTER_CONN_CHANNELS = False  # if True, clusters span adjacent channels too
LFP_KEY = 'bipolar_lfp'
RNG_SEED = 0


def paired_t_map(d):
    """Per-bin paired t-statistic across trials.

    d : (n_trial, n_ch, n_f) per-trial log2 fold-changes
    Returns t : (n_ch, n_f).
    """
    n = d.shape[0]
    mean = d.mean(axis=0)
    sd = d.std(axis=0, ddof=1)
    sem = sd / np.sqrt(n)
    sem = np.where(sem > 0, sem, np.inf)
    return mean / sem


def _connectivity_structure(connect_channels):
    """3x3 structuring element for scipy.ndimage.label.

    Axis 0 = channels, axis 1 = frequency. Frequency always connected.
    Channels optionally connected (4-neighbourhood).
    """
    if connect_channels:
        # 4-connectivity (no diagonals): up/down/left/right
        s = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]], dtype=int)
    else:
        # only horizontal neighbours (along freq) within a row
        s = np.array([[0, 0, 0],
                      [1, 1, 1],
                      [0, 0, 0]], dtype=int)
    return s


def _find_clusters_signed(t_map, t_thresh, structure):
    """Find positive and negative clusters separately.

    Positive clusters: connected components of (t_map > +t_thresh).
    Negative clusters: connected components of (t_map < -t_thresh).
    Returns list of dicts with keys: mask (bool ndarray same shape as t_map),
    sign (+1/-1), mass (signed sum of t inside cluster).
    """
    clusters = []
    for sign, sup in ((+1, t_map > +t_thresh),
                      (-1, t_map < -t_thresh)):
        if not sup.any():
            continue
        lbl, n = nd_label(sup, structure=structure)
        for k in range(1, n + 1):
            m = (lbl == k)
            mass = float(t_map[m].sum())  # already signed
            clusters.append({'mask': m, 'sign': sign, 'mass': mass})
    return clusters


def _max_abs_cluster_mass(t_map, t_thresh, structure):
    """Helper for the null: just the largest |signed mass|."""
    best = 0.0
    for sign, sup in ((+1, t_map > +t_thresh),
                      (-1, t_map < -t_thresh)):
        if not sup.any():
            continue
        lbl, n = nd_label(sup, structure=structure)
        if n == 0:
            continue
        # vectorised sum-per-label
        flat_lbl = lbl.ravel()
        flat_t = t_map.ravel()
        # bincount weighted by t, indexed by label
        sums = np.bincount(flat_lbl, weights=flat_t, minlength=n + 1)[1:]
        m = float(np.max(np.abs(sums)))
        if m > best:
            best = m
    return best


def cluster_permutation_test(d, n_perm=N_PERM, p_forming=CLUSTER_P_FORMING,
                              connect_channels=CLUSTER_CONN_CHANNELS,
                              rng=None):
    """Paired sign-flip cluster permutation test.

    d : (n_trial, n_ch, n_f) per-trial log2 fold-changes
    Returns:
        t_obs : (n_ch, n_f) observed paired t-statistic
        clusters : list of dicts (mask, sign, mass, p)  -- all observed
                   clusters above the forming threshold, with cluster p-value
        sig_mask : (n_ch, n_f) bool, union of clusters with p < ALPHA
        null_max : (n_perm,) array of max |cluster mass| under H0
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    n_trial = d.shape[0]
    df = n_trial - 1
    # two-sided t critical for cluster-forming
    t_thresh = float(sp_stats.t.ppf(1 - p_forming / 2, df))
    structure = _connectivity_structure(connect_channels)

    t_obs = paired_t_map(d)
    obs_clusters = _find_clusters_signed(t_obs, t_thresh, structure)

    null_max = np.empty(n_perm, dtype=np.float64)
    for p in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=(n_trial, 1, 1))
        d_perm = d * signs
        t_perm = paired_t_map(d_perm)
        null_max[p] = _max_abs_cluster_mass(t_perm, t_thresh, structure)

    # cluster p-values: fraction of null max >= |obs mass|
    sig_mask = np.zeros_like(t_obs, dtype=bool)
    for c in obs_clusters:
        p_val = (1.0 + float((null_max >= abs(c['mass'])).sum())) / (n_perm + 1.0)
        c['p'] = p_val
        if p_val < ALPHA:
            sig_mask |= c['mask']

    return t_obs, obs_clusters, sig_mask, null_max


def plot_cluster_heatmap(freqs, depths, labels, log2fc, t_obs, sig_mask,
                          lesion_name, out_path):
    """Two-panel: left = log2FC all bins, right = log2FC sig-clusters only."""
    masked = np.where(sig_mask, log2fc, np.nan)
    n_ch = log2fc.shape[0]
    finite = log2fc[np.isfinite(log2fc)]
    vlim = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
    vlim = max(vlim, 0.1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5),
                             gridspec_kw={'width_ratios': [1, 1]})
    extent = [freqs[0], freqs[-1], n_ch - 0.5, -0.5]
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)

    im0 = axes[0].imshow(log2fc, aspect='auto', cmap='RdBu_r',
                         extent=extent, norm=norm, interpolation='nearest')
    axes[0].set_title('log2 FC (lesion / control) -- all bins')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('bipolar channel')
    axes[0].set_yticks(range(n_ch))
    axes[0].set_yticklabels([f'{lab}  z={z:+.2f}' for lab, z in zip(labels, depths)],
                            fontsize=7)
    _band_overlay(axes[0], freqs[0], freqs[-1], label_bands=True)
    plt.colorbar(im0, ax=axes[0], label='log2 FC')

    im1 = axes[1].imshow(masked, aspect='auto', cmap='RdBu_r',
                         extent=extent, norm=norm, interpolation='nearest')
    axes[1].set_facecolor('#dddddd')
    axes[1].set_title(f'log2 FC, cluster p<{ALPHA} only')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_yticks(range(n_ch))
    axes[1].set_yticklabels([f'{lab}  z={z:+.2f}' for lab, z in zip(labels, depths)],
                            fontsize=7)
    _band_overlay(axes[1], freqs[0], freqs[-1], label_bands=True)
    plt.colorbar(im1, ax=axes[1], label='log2 FC')

    fig.suptitle(
        f'Lesion {lesion_name}  -- cluster permutation (n_perm={N_PERM}, '
        f'forming p<{CLUSTER_P_FORMING}, '
        f'channels {"connected" if CLUSTER_CONN_CHANNELS else "independent"})',
        fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_cluster_3d(freqs, depths, labels, log2fc, sig_mask,
                    lesion_name, out_path,
                    upsample_depth=8, smooth_freq=2, only_sig=True,
                    peaks=None):
    """3D surface of signed log2FC: height = log2FC, colour = red/blue by sign.

    Designed to make narrow peaks INSIDE a broad cluster visible: a band-wide
    decrease shows as a wide negative plateau, and a focal sub-band that drops
    even further sticks down as a deeper trough; same for narrow excess peaks
    on a positive plateau. Non-significant bins are set to NaN (only_sig=True)
    so the surface only shows what survived the cluster test.
    """
    n_ch = log2fc.shape[0]
    z = log2fc.copy().astype(float)
    if only_sig:
        z[~sig_mask] = np.nan

    if upsample_depth and upsample_depth > 1 and n_ch > 1:
        # cubic spline along depth for a smooth surface; keep NaN where the
        # entire interpolated value would be NaN.
        z_filled = np.where(np.isfinite(z), z, 0.0)
        valid = np.isfinite(z).astype(float)
        z_up = nd_zoom(z_filled, (upsample_depth, 1), order=3)
        v_up = nd_zoom(valid, (upsample_depth, 1), order=1)
        z_up = np.where(v_up > 0.5, z_up, np.nan)
        d_up = np.linspace(depths[0], depths[-1], z_up.shape[0])
    else:
        z_up = z
        d_up = np.asarray(depths)

    if smooth_freq and smooth_freq > 1:
        kernel = np.ones(smooth_freq) / smooth_freq
        z_up = np.apply_along_axis(
            lambda v: np.where(
                np.isfinite(v),
                np.convolve(np.where(np.isfinite(v), v, 0.0), kernel, mode='same'),
                np.nan),
            1, z_up)

    finite = z_up[np.isfinite(z_up)]
    if finite.size:
        vlim = float(np.nanpercentile(np.abs(finite), 99))
        vlim = max(vlim, 0.5)
    else:
        vlim = 1.0
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)

    F, D = np.meshgrid(freqs, d_up)

    fig = plt.figure(figsize=(13, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    surf = ax.plot_surface(F, D, z_up, cmap='RdBu_r', norm=norm,
                           edgecolor='none', alpha=0.95, antialiased=True,
                           rcount=120, ccount=120)

    # zero plane reference (helps eye gauge "above"/"below" baseline)
    ax.plot_surface(F, D, np.zeros_like(F), color='lightgray',
                    alpha=0.18, edgecolor='none', shade=False)

    # band boundaries projected to the floor
    z_floor = float(np.nanmin(z_up)) if finite.size else -1.0
    z_floor = min(z_floor, -0.5)
    edges = sorted({e for band in BANDS.values() for e in band})
    f_lo, f_hi = float(freqs[0]), float(freqs[-1])
    for e in edges:
        if f_lo <= e <= f_hi:
            ax.plot([e, e], [d_up[0], d_up[-1]], [z_floor, z_floor],
                    color='k', alpha=0.3, lw=0.6, linestyle='--')

    # band labels on the floor
    for name, (b_lo, b_hi) in BANDS.items():
        lo = max(b_lo, f_lo)
        hi = min(b_hi, f_hi)
        if hi <= lo:
            continue
        ax.text((lo + hi) / 2, d_up[-1], z_floor, name,
                ha='center', va='top', fontsize=7, color='k')

    ax.set_xlabel('Frequency (Hz)', fontsize=9)
    ax.set_ylabel('Laminar depth (mm)', fontsize=9)
    ax.set_zlabel('log2 FC (lesion / control)', fontsize=9)
    ax.set_title(
        f'Lesion {lesion_name} -- 3D log2FC (cluster-significant bins only)\n'
        f'red = power up, blue = power down. '
        f'Height shows magnitude; sub-band peaks/troughs visible inside plateaus.',
        fontsize=11)
    if peaks:
        for p in peaks:
            color = 'crimson' if p['sign'] > 0 else 'royalblue'
            ax.scatter([p['freq']], [depths[p['ch']]], [p['log2fc']],
                       s=55, c=color, edgecolors='k', linewidths=0.7,
                       depthshade=False, zorder=10)
            ax.text(p['freq'], depths[p['ch']], p['log2fc'],
                    f"  {p['freq']:.0f}Hz",
                    fontsize=7, color='k', zorder=11)
    ax.view_init(elev=28, azim=-62)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, label='log2 FC')
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def cluster_band_for_freqs(f_lo, f_hi):
    for name, (b_lo, b_hi) in BANDS.items():
        if f_lo >= b_lo and f_hi <= b_hi:
            return name
    # cluster spans multiple bands
    return 'multi'


def _log_baseline(freqs, y, baseline_octaves=0.8):
    """Smooth baseline = running median in log-frequency.

    Width = `baseline_octaves` octaves on each side of every bin. Wide enough
    to track broadband plateaus, narrow enough to leave focal peaks alone.
    Operates in linear log2FC space along a logarithmically-spaced freq axis.
    """
    lf = np.log2(np.maximum(freqs, 1e-3))
    out = np.empty_like(y)
    for i, fi in enumerate(lf):
        lo = fi - baseline_octaves
        hi = fi + baseline_octaves
        m = (lf >= lo) & (lf <= hi)
        out[i] = np.median(y[m])
    return out


def find_residual_peaks(freqs, log2fc, sig_mask,
                        baseline_octaves=0.8,
                        min_prominence=0.4,
                        min_separation_octaves=0.15):
    """Detect narrow peaks/troughs of log2FC that stick out of a smooth baseline.

    Per channel:
        1. baseline = running median of log2FC over `baseline_octaves`
        2. residual = log2FC - baseline
        3. find local maxima in +residual (peaks) and -residual (troughs)
           with prominence >= `min_prominence`
        4. keep only peaks whose freq bin is inside `sig_mask` (cluster-sig)
        5. enforce minimum separation in log2(freq)

    Returns list of dicts with keys:
        ch, freq, log2fc, baseline, residual, sign, prominence.
    """
    peaks_out = []
    n_ch, n_f = log2fc.shape
    lf = np.log2(np.maximum(freqs, 1e-3))

    for ch in range(n_ch):
        y = log2fc[ch].copy()
        if not sig_mask[ch].any():
            continue
        base = _log_baseline(freqs, y, baseline_octaves=baseline_octaves)
        resid = y - base

        for sign, signal in ((+1, resid), (-1, -resid)):
            idx, props = find_peaks(signal, prominence=min_prominence)
            if idx.size == 0:
                continue
            # keep only sig-mask bins
            keep = sig_mask[ch, idx]
            idx = idx[keep]
            prom = props['prominences'][keep]
            if idx.size == 0:
                continue
            # sort by prominence desc, then enforce log-freq separation
            order = np.argsort(-prom)
            chosen = []
            for j in order:
                fi = lf[idx[j]]
                if all(abs(fi - lf[idx[k]]) >= min_separation_octaves
                       for k in chosen):
                    chosen.append(j)
            for j in chosen:
                k = int(idx[j])
                peaks_out.append({
                    'ch': ch,
                    'freq': float(freqs[k]),
                    'log2fc': float(y[k]),
                    'baseline': float(base[k]),
                    'residual': float(resid[k]),
                    'sign': sign,
                    'prominence': float(prom[j]),
                })
    return peaks_out


def write_peaks_summary(peaks, depths, labels, lesion_name, out_path):
    """Write a per-lesion table of detected residual peaks."""
    with open(out_path, 'w') as fh:
        fh.write(f'# Lesion {lesion_name} -- residual peaks/troughs\n')
        fh.write(f'# Local extrema of (log2FC - smooth baseline) that lie '
                 f'inside cluster-significant bins.\n')
        fh.write(f'# Sign +1 = focal power increase on top of baseline.\n')
        fh.write(f'# Sign -1 = focal power decrease on top of baseline.\n\n')
        if not peaks:
            fh.write('  (no residual peaks detected above prominence threshold)\n')
            return
        fh.write(f'{"sign":>4} {"channel":>12} {"depth":>7} '
                 f'{"freq(Hz)":>9} {"log2FC":>8} {"baseline":>9} '
                 f'{"residual":>9} {"prominence":>11}\n')
        # sort: by sign, then by prominence desc
        peaks_sorted = sorted(peaks,
                              key=lambda p: (-p['sign'], -p['prominence']))
        for p in peaks_sorted:
            sign_str = '+' if p['sign'] > 0 else '-'
            fh.write(f'{sign_str:>4} {str(labels[p["ch"]]):>12} '
                     f'{depths[p["ch"]]:+7.3f} '
                     f'{p["freq"]:9.2f} {p["log2fc"]:+8.3f} '
                     f'{p["baseline"]:+9.3f} {p["residual"]:+9.3f} '
                     f'{p["prominence"]:11.3f}\n')


def plot_peaks_per_channel(freqs, depths, labels, log2fc, sig_mask, peaks,
                           lesion_name, out_path, baseline_octaves=0.8):
    """Per-channel diagnostic: log2FC, baseline, residual, marked peaks."""
    n_ch = log2fc.shape[0]
    # only plot channels with at least one peak or any significant bin
    rows = [ch for ch in range(n_ch)
            if sig_mask[ch].any() or any(p['ch'] == ch for p in peaks)]
    if not rows:
        return
    ncols = 3
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.0 * nrows),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax_i, ch in enumerate(rows):
        ax = axes[ax_i]
        y = log2fc[ch]
        base = _log_baseline(freqs, y, baseline_octaves=baseline_octaves)
        ax.plot(freqs, y, color='k', lw=1.2, label='log2FC')
        ax.plot(freqs, base, color='gray', lw=1.0, linestyle='--',
                label='baseline')
        ax.axhline(0, color='k', lw=0.4, alpha=0.4)
        # shade significant bins
        sig = sig_mask[ch]
        if sig.any():
            ax.fill_between(freqs, y, base, where=sig & (y > base),
                            color='C3', alpha=0.25)
            ax.fill_between(freqs, y, base, where=sig & (y < base),
                            color='C0', alpha=0.25)
        # mark peaks
        for p in peaks:
            if p['ch'] != ch:
                continue
            color = 'C3' if p['sign'] > 0 else 'C0'
            ax.plot(p['freq'], p['log2fc'], marker='v' if p['sign'] < 0 else '^',
                    color=color, ms=9, mec='k', mew=0.6)
            ax.annotate(f'{p["freq"]:.1f} Hz\nres {p["residual"]:+.2f}',
                        xy=(p['freq'], p['log2fc']),
                        xytext=(4, 8 if p['sign'] > 0 else -22),
                        textcoords='offset points',
                        fontsize=7, color=color)
        ax.set_xscale('log')
        ax.set_xlim(freqs[0], freqs[-1])
        ax.set_title(f'{labels[ch]}  z={depths[ch]:+.2f}', fontsize=9)
        ax.set_ylabel('log2 FC', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, which='both', alpha=0.25)
        if ax_i == 0:
            ax.legend(fontsize=7, loc='best')
    # hide unused axes
    for j in range(len(rows), len(axes)):
        axes[j].set_visible(False)
    for ax in axes[-ncols:]:
        ax.set_xlabel('Frequency (Hz)', fontsize=8)
    fig.suptitle(f'Lesion {lesion_name} -- residual peaks/troughs '
                 f'(red = focal up, blue = focal down)', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def write_cluster_summary(clusters, freqs, depths, labels, log2fc, sig_mask,
                          lesion_name, out_path):
    """Per-cluster table + per-band/channel tally (uses cluster sig mask)."""
    sig_clusters = [c for c in clusters if c.get('p', 1.0) < ALPHA]
    with open(out_path, 'w') as fh:
        fh.write(f'# Lesion {lesion_name}\n')
        fh.write(f'# Cluster permutation test, n_perm={N_PERM}, '
                 f'forming p<{CLUSTER_P_FORMING}, alpha={ALPHA}\n')
        fh.write(f'# Channels connectivity: '
                 f'{"on (4-conn)" if CLUSTER_CONN_CHANNELS else "off (per-channel)"}\n\n')

        if not clusters:
            fh.write('  (no suprathreshold bins -- this lesion looks like noise)\n')
            return
        if not sig_clusters:
            fh.write('  (suprathreshold bins existed but no cluster survived '
                     f'permutation at alpha={ALPHA})\n\n')

        fh.write('## Surviving clusters\n')
        fh.write(f'{"id":>3} {"sign":>4} {"channels":>22} '
                 f'{"f_range (Hz)":>14} {"n_bins":>6} '
                 f'{"mass":>10} {"p":>8} {"med_log2FC":>11}\n')
        for k, c in enumerate(sig_clusters, start=1):
            m = c['mask']
            ch_idx = np.where(m.any(axis=1))[0]
            f_idx = np.where(m.any(axis=0))[0]
            ch_lo, ch_hi = int(ch_idx.min()), int(ch_idx.max())
            f_lo, f_hi = float(freqs[f_idx.min()]), float(freqs[f_idx.max()])
            n_bins = int(m.sum())
            med = float(np.median(log2fc[m]))
            ch_str = (f'{labels[ch_lo]}' if ch_lo == ch_hi
                      else f'{labels[ch_lo]}..{labels[ch_hi]}')
            sign_str = '+' if c['sign'] > 0 else '-'
            fh.write(f'{k:>3} {sign_str:>4} {ch_str:>22} '
                     f'{f_lo:6.1f}-{f_hi:6.1f} {n_bins:>6d} '
                     f'{c["mass"]:>+10.1f} {c["p"]:>8.4f} {med:>+11.3f}\n')

        # Per-band, per-channel tally (descriptive)
        fh.write('\n## Sig channels per band (>=25% bins inside cluster)\n')
        fh.write(f'{"band":10} {"channel":12} {"depth":>7} '
                 f'{"frac_sig":>9} {"med_log2FC":>11} {"dir":>5}\n')
        for band, (b_lo, b_hi) in BANDS.items():
            band_mask_f = (freqs >= b_lo) & (freqs <= b_hi)
            n_in_band = int(band_mask_f.sum())
            if n_in_band == 0:
                continue
            for ch in range(log2fc.shape[0]):
                sig_in_band = sig_mask[ch, band_mask_f]
                n_sig = int(sig_in_band.sum())
                if n_sig / n_in_band < 0.25:
                    continue
                lfc = log2fc[ch, band_mask_f][sig_in_band]
                if lfc.size == 0:
                    continue
                med = float(np.median(lfc))
                fh.write(f'{band:10} {str(labels[ch]):12} {depths[ch]:+7.3f} '
                         f'{n_sig / n_in_band:9.2f} {med:+11.3f} '
                         f'{"up" if med > 0 else "down":>5}\n')


def run(lesion_root, fig_root, window_ms=WINDOW_MS, n_perm=N_PERM,
        lfp_key=LFP_KEY, connect_channels=CLUSTER_CONN_CHANNELS,
        plot_3d=False, plot_3d_only_sig=True,
        find_peaks_flag=False,
        peak_baseline_octaves=0.8, peak_min_prominence=0.4):
    ctrl_dir = os.path.join(lesion_root, 'control')
    n_ctrl = count_trials(ctrl_dir)
    print(f'[control] loading {n_ctrl} trials')
    ctrl_trials = load_trials(ctrl_dir, n_ctrl)
    if lfp_key == 'bipolar_lfp_current':
        ctrl_trials = _bipolar_current_from_trials(ctrl_trials)
    psd_ctrl, freqs, depths, labels = per_trial_psd(
        ctrl_trials, lfp_key, window_ms=window_ms)
    print(f'  ctrl PSD shape {psd_ctrl.shape}')

    out_dir = os.path.join(fig_root, '_significance_cluster')
    os.makedirs(out_dir, exist_ok=True)

    lesion_names = sorted(
        d for d in os.listdir(lesion_root)
        if os.path.isdir(os.path.join(lesion_root, d)) and d != 'control'
    )

    rng = np.random.default_rng(RNG_SEED)
    grand_log2fc = []
    grand_sig_count = []
    grand_names = []
    cluster_rows = []  # for CSV

    for lname in lesion_names:
        ldir = os.path.join(lesion_root, lname)
        n = count_trials(ldir)
        if n == 0:
            continue
        print(f'[{lname}] loading {n} trials')
        try:
            les_trials = load_trials(ldir, n)
            if lfp_key == 'bipolar_lfp_current':
                les_trials = _bipolar_current_from_trials(les_trials)
            psd_les, freqs_l, _, _ = per_trial_psd(
                les_trials, lfp_key, window_ms=window_ms)
        except Exception as exc:
            print(f'[{lname}] failed: {exc}')
            continue
        if not np.allclose(freqs_l, freqs):
            print(f'[{lname}] freq grid mismatch -- skip')
            continue

        n_pair = min(psd_ctrl.shape[0], psd_les.shape[0])
        d = np.log2((psd_les[:n_pair] + 1e-20) /
                    (psd_ctrl[:n_pair] + 1e-20))
        log2fc = d.mean(axis=0)

        t_obs, clusters, sig_mask, null_max = cluster_permutation_test(
            d, n_perm=n_perm, connect_channels=connect_channels, rng=rng)

        n_sig_cl = sum(1 for c in clusters if c.get('p', 1.0) < ALPHA)
        n_sig_bins = int(sig_mask.sum())
        print(f'  {len(clusters)} suprathreshold clusters; '
              f'{n_sig_cl} survive (alpha={ALPHA}); '
              f'{n_sig_bins}/{sig_mask.size} bins '
              f'({100*n_sig_bins/sig_mask.size:.1f}%)')

        plot_cluster_heatmap(
            freqs, depths, labels, log2fc, t_obs, sig_mask, lname,
            os.path.join(out_dir, f'{lname}_cluster_heatmap.png'))

        peaks = None
        if find_peaks_flag and sig_mask.any():
            peaks = find_residual_peaks(
                freqs, log2fc, sig_mask,
                baseline_octaves=peak_baseline_octaves,
                min_prominence=peak_min_prominence)
            write_peaks_summary(
                peaks, depths, labels, lname,
                os.path.join(out_dir, f'{lname}_peaks_summary.txt'))
            plot_peaks_per_channel(
                freqs, depths, labels, log2fc, sig_mask, peaks, lname,
                os.path.join(out_dir, f'{lname}_peaks_per_channel.png'),
                baseline_octaves=peak_baseline_octaves)
            print(f'  found {len(peaks)} residual peaks/troughs')

        if plot_3d and sig_mask.any():
            plot_cluster_3d(
                freqs, depths, labels, log2fc, sig_mask, lname,
                os.path.join(out_dir, f'{lname}_cluster_3d.png'),
                only_sig=plot_3d_only_sig, peaks=peaks)
        write_cluster_summary(
            clusters, freqs, depths, labels, log2fc, sig_mask,
            lname, os.path.join(out_dir, f'{lname}_cluster_summary.txt'))

        for k, c in enumerate(
            [c for c in clusters if c.get('p', 1.0) < ALPHA], start=1
        ):
            m = c['mask']
            ch_idx = np.where(m.any(axis=1))[0]
            f_idx = np.where(m.any(axis=0))[0]
            cluster_rows.append({
                'lesion': lname,
                'cluster_id': k,
                'sign': c['sign'],
                'ch_lo': str(labels[int(ch_idx.min())]),
                'ch_hi': str(labels[int(ch_idx.max())]),
                'depth_lo': float(depths[int(ch_idx.min())]),
                'depth_hi': float(depths[int(ch_idx.max())]),
                'f_lo': float(freqs[f_idx.min()]),
                'f_hi': float(freqs[f_idx.max()]),
                'n_bins': int(m.sum()),
                'mass': float(c['mass']),
                'p_cluster': float(c['p']),
                'median_log2fc': float(np.median(log2fc[m])),
                'band_hint': cluster_band_for_freqs(
                    float(freqs[f_idx.min()]), float(freqs[f_idx.max()])),
            })

        mid_mask = (depths >= -0.3) & (depths <= 0.7)
        if not mid_mask.any():
            mid_mask = np.ones_like(depths, dtype=bool)
        mid_lfc = np.where(sig_mask, log2fc, 0.0)[mid_mask].mean(0)
        grand_log2fc.append(mid_lfc)

        band_score = []
        for band, (b_lo, b_hi) in BANDS.items():
            m = (freqs >= b_lo) & (freqs <= b_hi)
            up = int(((log2fc > 0) & sig_mask)[:, m].any(axis=1).sum())
            dn = int(((log2fc < 0) & sig_mask)[:, m].any(axis=1).sum())
            band_score.append(up - dn)
        grand_sig_count.append(band_score)
        grand_names.append(lname)

    if not grand_names:
        print('no lesions processed')
        return

    grand_log2fc = np.stack(grand_log2fc, axis=0)
    grand_sig_count = np.array(grand_sig_count, dtype=int)

    # ---- grand log2FC heatmap ----
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
    ax.set_title('CLUSTER test: sig-masked log2 FC, '
                 'mid-layer (z in [-0.3, +0.7]) mean')
    _band_overlay(ax, freqs[0], freqs[-1], label_bands=True)
    plt.colorbar(im, ax=ax, label='log2 FC')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'grand_summary_log2fc.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)

    # ---- per-band sig channel-count ----
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
    ax.set_title('CLUSTER test: sig channels per band (up - down)')
    for i in range(grand_sig_count.shape[0]):
        for j in range(grand_sig_count.shape[1]):
            v = grand_sig_count[i, j]
            if v != 0:
                ax.text(j, i, f'{v:+d}', ha='center', va='center',
                        fontsize=8,
                        color='white' if abs(v) > vlim2 * 0.6 else 'black')
    plt.colorbar(im, ax=ax, label='channels (up - down)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'grand_summary_cluster_counts.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)

    # ---- CSV of all surviving clusters ----
    csv_path = os.path.join(out_dir, 'grand_summary_clusters.csv')
    with open(csv_path, 'w') as fh:
        fh.write('lesion,cluster_id,sign,ch_lo,ch_hi,depth_lo,depth_hi,'
                 'f_lo,f_hi,n_bins,mass,p_cluster,median_log2fc,band_hint\n')
        for r in cluster_rows:
            fh.write(f'{r["lesion"]},{r["cluster_id"]},{r["sign"]:+d},'
                     f'{r["ch_lo"]},{r["ch_hi"]},'
                     f'{r["depth_lo"]:+.3f},{r["depth_hi"]:+.3f},'
                     f'{r["f_lo"]:.2f},{r["f_hi"]:.2f},'
                     f'{r["n_bins"]},{r["mass"]:+.3f},{r["p_cluster"]:.4f},'
                     f'{r["median_log2fc"]:+.4f},{r["band_hint"]}\n')
    print(f'saved {csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lesion-root', default='results/lesions_2026-06-03')
    parser.add_argument('--fig-root', default='figures/lesions_2026-06-03')
    parser.add_argument('--window-ms', type=int, default=WINDOW_MS)
    parser.add_argument('--n-perm', type=int, default=N_PERM)
    parser.add_argument('--lfp-key', default=LFP_KEY)
    parser.add_argument('--connect-channels', action='store_true',
                        help='Enable 2D clustering (also connect adjacent '
                             'channels along depth). Default: per-channel '
                             '1D clustering along frequency only.')
    parser.add_argument('--plot-3d', action='store_true',
                        help='Also save a 3D surface plot per lesion '
                             '(depth x frequency x signed log2FC).')
    parser.add_argument('--plot-3d-all-bins', action='store_true',
                        help='In the 3D plot, show all bins (default shows '
                             'only cluster-significant bins).')
    parser.add_argument('--find-peaks', action='store_true',
                        help='Detect narrow residual peaks/troughs of log2FC '
                             'that stick out of a smooth baseline, save a '
                             'per-channel diagnostic and a peaks table, and '
                             'overlay the peaks on the 3D plot.')
    parser.add_argument('--peak-baseline-octaves', type=float, default=0.8,
                        help='Half-width (in octaves) of the running-median '
                             'baseline used for peak detection. Larger = '
                             'baseline tracks only very broad shifts.')
    parser.add_argument('--peak-min-prominence', type=float, default=0.4,
                        help='Minimum prominence of a residual peak/trough '
                             '(in log2 units of FC) to be reported.')
    args = parser.parse_args()
    run(args.lesion_root, args.fig_root,
        window_ms=args.window_ms, n_perm=args.n_perm,
        lfp_key=args.lfp_key, connect_channels=args.connect_channels,
        plot_3d=args.plot_3d, plot_3d_only_sig=not args.plot_3d_all_bins,
        find_peaks_flag=args.find_peaks,
        peak_baseline_octaves=args.peak_baseline_octaves,
        peak_min_prominence=args.peak_min_prominence)
