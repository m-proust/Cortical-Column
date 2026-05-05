"""
Plot the inter-layer connectivity strength sweep produced by
interlayer_p_sweep.py.

For each p run, compute the bipolar LFP PSD on every channel after the
transient, then produce:
    - one figure with subplots per channel showing (p x freq) heatmaps
    - one figure showing band power (delta/theta/alpha/beta/gamma) vs p
      per channel
"""

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import detrend
from scipy.signal.windows import dpss
from scipy.ndimage import gaussian_filter1d, zoom
from scipy import signal as _sig


# ---------------------------------------------------------------------------
# USER-EDITABLE
# ---------------------------------------------------------------------------
SWEEP_DIR     = "results/interlayer_p_sweep"
TRANSIENT_MS  = 500
ANALYSIS_MS   = 2000
FREQ_RANGE    = (1, 100)
LFP_KEY       = "bipolar_matrix"
SAVE_FIGS     = True
FIG_DIR       = os.path.join(SWEEP_DIR, "figures")
PCT_REF_P     = 0.0          # reference p for the % change plot

# Smoothing controls (kill horizontal striping on heatmaps)
NW            = 4            # multitaper time-bandwidth (was 2). Higher = smoother.
SMOOTH_HZ     = 1.5          # gaussian smoothing of PSD along freq axis (Hz). 0 disables.

BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 80),
}
# ---------------------------------------------------------------------------


def multitaper_psd(x, fs, NW=NW, nfft=None):
    x = x - np.mean(x)
    if nfft is None:
        nfft = 2 ** int(np.ceil(np.log2(len(x))))
    K = int(2 * NW - 1)
    tapers = dpss(len(x), NW, K)
    psds = []
    for taper in tapers:
        f, p = _sig.periodogram(x * taper, fs=fs, nfft=nfft, scaling='density')
        psds.append(p)
    return f, np.mean(psds, axis=0)


def smooth_along_freq(psd, freqs, sigma_hz):
    if sigma_hz is None or sigma_hz <= 0:
        return psd
    df = float(np.median(np.diff(freqs)))
    if df <= 0:
        return psd
    sigma_bins = sigma_hz / df
    if sigma_bins < 0.5:
        return psd
    return gaussian_filter1d(psd, sigma=sigma_bins, axis=-1, mode='nearest')


def _parse_p(fname):
    m = re.search(r"p_([0-9.]+)\.npz", os.path.basename(fname))
    return float(m.group(1)) if m else None


def load_runs(sweep_dir):
    files = sorted(glob.glob(os.path.join(sweep_dir, "p_*.npz")))
    runs = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if LFP_KEY not in d.files:
            print(f"  skip {f}: missing {LFP_KEY}")
            continue
        runs.append({
            'p':          float(d['p']) if 'p' in d.files else _parse_p(f),
            'lfp':        d[LFP_KEY],
            'time_ms':    np.asarray(d['time_array_ms']),
            'transient_ms': float(d['transient_ms'])
                if 'transient_ms' in d.files else TRANSIENT_MS,
            'channel_labels': d['channel_labels'],
            'channel_depths': d['channel_depths'],
        })
    runs.sort(key=lambda r: r['p'])
    return runs


def compute_p_freq_matrix(runs):
    fs = 1000.0 / float(np.mean(np.diff(runs[0]['time_ms'])))
    n_channels = runs[0]['lfp'].shape[0]
    p_vals = np.array([r['p'] for r in runs])

    freqs_ref = None
    freq_mask = None
    psd_stack = None

    for p_idx, r in enumerate(runs):
        time = r['time_ms']
        t0 = r['transient_ms']
        mask = (time >= t0) & (time < t0 + ANALYSIS_MS)
        for ch in range(n_channels):
            seg = r['lfp'][ch][mask].copy()
            if len(seg) == 0 or np.any(np.isnan(seg)):
                continue
            seg = detrend(seg)
            nfft = 2 ** int(np.ceil(np.log2(len(seg))))
            f, psd = multitaper_psd(seg, fs=fs, NW=NW, nfft=nfft)
            if freqs_ref is None:
                freq_mask = (f >= FREQ_RANGE[0]) & (f <= FREQ_RANGE[1])
                freqs_ref = f[freq_mask]
                psd_stack = np.full(
                    (n_channels, len(runs), freqs_ref.size), np.nan)
            psd_stack[ch, p_idx, :] = psd[freq_mask]

    if SMOOTH_HZ and SMOOTH_HZ > 0 and freqs_ref is not None:
        psd_stack = smooth_along_freq(psd_stack, freqs_ref, SMOOTH_HZ)

    return p_vals, freqs_ref, psd_stack


def plot_heatmaps(p_vals, freqs, psd_stack, channel_labels, channel_depths):
    n_channels = psd_stack.shape[0]
    bip_depths = ((channel_depths[:-1] + channel_depths[1:]) / 2
                  if len(channel_depths) > n_channels
                  else channel_depths[:n_channels])

    psd_db = 10 * np.log10(psd_stack + 1e-20)

    ncols = 5
    nrows = int(np.ceil(n_channels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()

    vmin = np.nanpercentile(psd_db, 5)
    vmax = np.nanpercentile(psd_db, 95)
    extent = [p_vals[0], p_vals[-1], freqs[0], freqs[-1]]

    last_im = None
    for ch in range(n_channels):
        ax = axes[ch]
        im = ax.imshow(psd_db[ch].T, aspect='auto', origin='lower',
                       extent=extent, cmap='viridis', vmin=vmin, vmax=vmax)
        last_im = im
        lbl = channel_labels[ch] if ch < len(channel_labels) else f"ch{ch}"
        ax.set_title(f"{lbl}  z={bip_depths[ch]:.2f}", fontsize=9)
        if ch % ncols == 0:
            ax.set_ylabel('Freq (Hz)')
        if ch // ncols == nrows - 1:
            ax.set_xlabel('inter-layer scaling p')

    for k in range(n_channels, len(axes)):
        axes[k].axis('off')

    fig.suptitle('Inter-layer p sweep - PSD per channel', fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.95, 0.96])
    cbar_ax = fig.add_axes([0.96, 0.12, 0.012, 0.76])
    fig.colorbar(last_im, cax=cbar_ax, label='Power (dB)')
    if SAVE_FIGS:
        os.makedirs(FIG_DIR, exist_ok=True)
        out = os.path.join(FIG_DIR, "psd_heatmaps_per_channel.png")
        fig.savefig(out, dpi=140, bbox_inches='tight')
        print(f"  saved {out}")
    return fig


def plot_pct_change_heatmaps(p_vals, freqs, psd_stack,
                             channel_labels, channel_depths,
                             ref_p=PCT_REF_P):
    n_channels = psd_stack.shape[0]
    bip_depths = ((channel_depths[:-1] + channel_depths[1:]) / 2
                  if len(channel_depths) > n_channels
                  else channel_depths[:n_channels])

    ref_idx = int(np.argmin(np.abs(p_vals - ref_p)))
    actual_ref = p_vals[ref_idx]
    ref_psd = psd_stack[:, ref_idx:ref_idx + 1, :]
    pct = (psd_stack - ref_psd) / (ref_psd + 1e-20) * 100.0

    ncols = 5
    nrows = int(np.ceil(n_channels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()

    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=500)
    extent = [p_vals[0], p_vals[-1], freqs[0], freqs[-1]]

    last_im = None
    for ch in range(n_channels):
        ax = axes[ch]
        im = ax.imshow(pct[ch].T, aspect='auto', origin='lower',
                       extent=extent, cmap='RdBu_r', norm=norm)
        last_im = im
        lbl = channel_labels[ch] if ch < len(channel_labels) else f"ch{ch}"
        ax.set_title(f"{lbl}  z={bip_depths[ch]:.2f}", fontsize=9)
        if ch % ncols == 0:
            ax.set_ylabel('Freq (Hz)')
        if ch // ncols == nrows - 1:
            ax.set_xlabel('inter-layer scaling p')

    for k in range(n_channels, len(axes)):
        axes[k].axis('off')

    fig.suptitle(f'Inter-layer p sweep - PSD %% change vs p={actual_ref:.2f}',
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.95, 0.96])
    cbar_ax = fig.add_axes([0.96, 0.12, 0.012, 0.76])
    cbar = fig.colorbar(last_im, cax=cbar_ax, label='% change')
    cbar.set_ticks([-100, -50, 0, 100, 250, 500])
    if SAVE_FIGS:
        os.makedirs(FIG_DIR, exist_ok=True)
        out = os.path.join(FIG_DIR,
                           f"psd_pct_change_per_channel_ref_p_{actual_ref:.2f}.png")
        fig.savefig(out, dpi=140, bbox_inches='tight')
        print(f"  saved {out}")
    return fig


def plot_pct_change_3d_surfaces(p_vals, freqs, psd_stack,
                                channel_labels, channel_depths,
                                ref_p=PCT_REF_P,
                                upsample_p=8, smooth_freq=4,
                                clip_percentile=99):
    """One 3D surface per channel: x=freq, y=p, z=% change vs ref_p.

    Mirrors the styling of plot_3d_surface in laminar_power_visualizations.py
    (cubic-spline upsample along the swept axis, light freq smoothing,
    data-driven asymmetric color scale centered on 0, RdBu_r diverging map).
    """
    n_channels = psd_stack.shape[0]
    bip_depths = ((channel_depths[:-1] + channel_depths[1:]) / 2
                  if len(channel_depths) > n_channels
                  else channel_depths[:n_channels])

    ref_idx = int(np.argmin(np.abs(p_vals - ref_p)))
    actual_ref = p_vals[ref_idx]
    ref_psd = psd_stack[:, ref_idx:ref_idx + 1, :]
    pct_all = (psd_stack - ref_psd) / (ref_psd + 1e-20) * 100.0  # (C, P, F)

    ncols = min(5, n_channels)
    nrows = int(np.ceil(n_channels / ncols))
    fig = plt.figure(figsize=(5.5 * ncols, 4.5 * nrows))

    for ch in range(n_channels):
        pct = pct_all[ch]   # (P, F)

        if upsample_p and upsample_p > 1 and len(p_vals) > 1:
            pct_up = zoom(pct, (upsample_p, 1), order=3)
            p_up = np.linspace(p_vals.min(), p_vals.max(), pct_up.shape[0])
        else:
            pct_up = pct
            p_up = p_vals

        if smooth_freq and smooth_freq > 1:
            kernel = np.ones(smooth_freq) / smooth_freq
            pct_up = np.apply_along_axis(
                lambda v: np.convolve(v, kernel, mode='same'), 1, pct_up)

        neg = pct_up[pct_up < 0]
        vmin = (np.percentile(neg, 100 - clip_percentile)
                if neg.size else -1.0)
        vmin = min(vmin, -1.0)
        pos = pct_up[pct_up > 0]
        vmax = (np.percentile(pos, clip_percentile) if pos.size else 1.0)
        vmax = max(vmax, 1.0)

        ax = fig.add_subplot(nrows, ncols, ch + 1, projection='3d')
        F, P = np.meshgrid(freqs, p_up)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        surf = ax.plot_surface(F, P, pct_up, cmap='RdBu_r', norm=norm,
                               edgecolor='none', alpha=0.95, antialiased=True,
                               rcount=80, ccount=80)
        ax.contour(F, P, pct_up, zdir='z',
                   offset=np.nanmin(pct_up) - 20,
                   cmap='RdBu_r', norm=norm, levels=12)

        ax.set_xlabel('Frequency (Hz)', fontsize=8)
        ax.set_ylabel('inter-layer p', fontsize=8)
        ax.set_zlabel('% change', fontsize=8)
        lbl = channel_labels[ch] if ch < len(channel_labels) else f"ch{ch}"
        ax.set_title(f"{lbl}  z={bip_depths[ch]:+.2f}  "
                     f"[{vmin:+.0f}, {vmax:+.0f}]%", fontsize=9)
        ax.view_init(elev=25, azim=-60)
        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.08, label='% change')

    fig.suptitle(
        f'Inter-layer p sweep - 3D %% power change vs p={actual_ref:.2f}',
        fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if SAVE_FIGS:
        os.makedirs(FIG_DIR, exist_ok=True)
        out = os.path.join(
            FIG_DIR,
            f"psd_pct_change_3d_per_channel_ref_p_{actual_ref:.2f}.png")
        fig.savefig(out, dpi=140, bbox_inches='tight')
        print(f"  saved {out}")
    return fig


def plot_single_channel(p_vals, freqs, psd_stack,
                        channel_labels, channel_depths,
                        ch_idx=7, ref_p=PCT_REF_P):
    """Readable 4-panel figure for a single bipolar channel.

    Default ch_idx=7 -> 'Ch8-Ch7' (z=+0.18, around the L4C/L5 border).
    Panels: PSD heatmap (dB), %-change heatmap, 3D %-change surface,
            band power curves vs p.
    """
    n_channels = psd_stack.shape[0]
    bip_depths = ((channel_depths[:-1] + channel_depths[1:]) / 2
                  if len(channel_depths) > n_channels
                  else channel_depths[:n_channels])
    ch_idx = max(0, min(n_channels - 1, int(ch_idx)))
    lbl = channel_labels[ch_idx] if ch_idx < len(channel_labels) else f"ch{ch_idx}"
    z = bip_depths[ch_idx]

    psd_ch = psd_stack[ch_idx]                      # (P, F)
    psd_db = 10 * np.log10(psd_ch + 1e-20)

    ref_idx = int(np.argmin(np.abs(p_vals - ref_p)))
    actual_ref = p_vals[ref_idx]
    pct = (psd_ch - psd_ch[ref_idx:ref_idx + 1, :]) / \
          (psd_ch[ref_idx:ref_idx + 1, :] + 1e-20) * 100.0

    band_power = {}
    for b, (f_lo, f_hi) in BANDS.items():
        m = (freqs >= f_lo) & (freqs <= f_hi)
        band_power[b] = (np.trapz(psd_ch[:, m], freqs[m], axis=1)
                         if np.any(m) else np.full(len(p_vals), np.nan))

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    extent = [p_vals[0], p_vals[-1], freqs[0], freqs[-1]]
    im0 = ax0.imshow(psd_db.T, aspect='auto', origin='lower',
                     extent=extent, cmap='viridis',
                     vmin=np.nanpercentile(psd_db, 5),
                     vmax=np.nanpercentile(psd_db, 95))
    ax0.set_title('PSD (dB)')
    ax0.set_xlabel('inter-layer scaling p')
    ax0.set_ylabel('Frequency (Hz)')
    fig.colorbar(im0, ax=ax0, label='Power (dB)')

    ax1 = fig.add_subplot(gs[0, 1])
    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=500)
    im1 = ax1.imshow(pct.T, aspect='auto', origin='lower',
                     extent=extent, cmap='RdBu_r', norm=norm)
    ax1.set_title(f'% change vs p={actual_ref:.2f}')
    ax1.set_xlabel('inter-layer scaling p')
    ax1.set_ylabel('Frequency (Hz)')
    cbar1 = fig.colorbar(im1, ax=ax1, label='% change')
    cbar1.set_ticks([-100, -50, 0, 100, 250, 500])

    ax2 = fig.add_subplot(gs[1, 0], projection='3d')
    pct_up = zoom(pct, (8, 1), order=3) if len(p_vals) > 1 else pct
    p_up = np.linspace(p_vals.min(), p_vals.max(), pct_up.shape[0]) \
        if len(p_vals) > 1 else p_vals
    kernel = np.ones(4) / 4.0
    pct_up = np.apply_along_axis(
        lambda v: np.convolve(v, kernel, mode='same'), 1, pct_up)
    F, P = np.meshgrid(freqs, p_up)
    neg = pct_up[pct_up < 0]
    pos = pct_up[pct_up > 0]
    vmin = min(np.percentile(neg, 1) if neg.size else -1.0, -1.0)
    vmax = max(np.percentile(pos, 99) if pos.size else 1.0, 1.0)
    norm3d = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    surf = ax2.plot_surface(F, P, pct_up, cmap='RdBu_r', norm=norm3d,
                            edgecolor='none', alpha=0.95, antialiased=True,
                            rcount=80, ccount=80)
    ax2.contour(F, P, pct_up, zdir='z',
                offset=np.nanmin(pct_up) - 20,
                cmap='RdBu_r', norm=norm3d, levels=12)
    ax2.set_xlabel('Freq (Hz)')
    ax2.set_ylabel('inter-layer p')
    ax2.set_zlabel('% change')
    ax2.set_title(f'3D % change vs p={actual_ref:.2f}')
    ax2.view_init(elev=25, azim=-60)
    fig.colorbar(surf, ax=ax2, shrink=0.55, pad=0.08, label='% change')

    ax3 = fig.add_subplot(gs[1, 1])
    colors = {'delta': '#e41a1c', 'theta': '#377eb8', 'alpha': '#4daf4a',
              'beta': '#984ea3', 'gamma': '#ff7f00'}
    for b in BANDS:
        ax3.plot(p_vals, band_power[b], marker='o', linewidth=1.8,
                 color=colors[b], label=b)
    ax3.set_yscale('log')
    ax3.set_xlabel('inter-layer scaling p')
    ax3.set_ylabel('Band power (log)')
    ax3.set_title('Band power vs p')
    ax3.legend(fontsize=9, ncol=2)
    ax3.grid(True, which='both', alpha=0.3)

    fig.suptitle(f'Single channel - {lbl}  (z = {z:+.2f} mm)', fontsize=15)
    if SAVE_FIGS:
        os.makedirs(FIG_DIR, exist_ok=True)
        out = os.path.join(FIG_DIR, f"single_channel_{lbl}_z{z:+.2f}.png")
        fig.savefig(out, dpi=140, bbox_inches='tight')
        print(f"  saved {out}")
    return fig


def plot_band_power_curves(p_vals, freqs, psd_stack,
                           channel_labels, channel_depths):
    n_channels = psd_stack.shape[0]
    bip_depths = ((channel_depths[:-1] + channel_depths[1:]) / 2
                  if len(channel_depths) > n_channels
                  else channel_depths[:n_channels])

    band_power = {b: np.full((n_channels, len(p_vals)), np.nan)
                  for b in BANDS}
    for b, (f_lo, f_hi) in BANDS.items():
        m = (freqs >= f_lo) & (freqs <= f_hi)
        if not np.any(m):
            continue
        band_power[b] = np.trapz(psd_stack[:, :, m], freqs[m], axis=2)

    ncols = 5
    nrows = int(np.ceil(n_channels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             sharex=True)
    axes = np.atleast_2d(axes).ravel()

    colors = {'delta': '#e41a1c', 'theta': '#377eb8', 'alpha': '#4daf4a',
              'beta': '#984ea3', 'gamma': '#ff7f00'}

    for ch in range(n_channels):
        ax = axes[ch]
        for b in BANDS:
            ax.plot(p_vals, band_power[b][ch], label=b, color=colors[b],
                    linewidth=1.4)
        ax.set_yscale('log')
        lbl = channel_labels[ch] if ch < len(channel_labels) else f"ch{ch}"
        ax.set_title(f"{lbl}  z={bip_depths[ch]:.2f}", fontsize=9)
        if ch % ncols == 0:
            ax.set_ylabel('Band power (log)')
        if ch // ncols == nrows - 1:
            ax.set_xlabel('inter-layer scaling p')
        if ch == 0:
            ax.legend(fontsize=7, ncol=2, loc='best')

    for k in range(n_channels, len(axes)):
        axes[k].axis('off')

    fig.suptitle('Inter-layer p sweep - band power vs p (per channel)',
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if SAVE_FIGS:
        os.makedirs(FIG_DIR, exist_ok=True)
        out = os.path.join(FIG_DIR, "band_power_vs_p.png")
        fig.savefig(out, dpi=140, bbox_inches='tight')
        print(f"  saved {out}")
    return fig


def main():
    runs = load_runs(SWEEP_DIR)
    if len(runs) < 2:
        raise FileNotFoundError(
            f"need at least 2 runs in {SWEEP_DIR}, got {len(runs)}")
    print(f"loaded {len(runs)} runs: p = {[r['p'] for r in runs]}")

    p_vals, freqs, psd_stack = compute_p_freq_matrix(runs)

    plot_heatmaps(p_vals, freqs, psd_stack,
                  runs[0]['channel_labels'], runs[0]['channel_depths'])
    plot_band_power_curves(p_vals, freqs, psd_stack,
                           runs[0]['channel_labels'],
                           runs[0]['channel_depths'])
    plot_pct_change_heatmaps(p_vals, freqs, psd_stack,
                             runs[0]['channel_labels'],
                             runs[0]['channel_depths'])
    plot_pct_change_3d_surfaces(p_vals, freqs, psd_stack,
                                runs[0]['channel_labels'],
                                runs[0]['channel_depths'])
    plot_single_channel(p_vals, freqs, psd_stack,
                        runs[0]['channel_labels'],
                        runs[0]['channel_depths'],
                        ch_idx=7)        # Ch8-Ch7, z=+0.18
    plt.show()


if __name__ == "__main__":
    main()
