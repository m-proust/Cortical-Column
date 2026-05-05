

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import detrend
from scipy.signal.windows import dpss
from scipy.ndimage import gaussian_filter1d
from scipy import signal as _sig


# ---------------------------------------------------------------------------
# USER-EDITABLE
# ---------------------------------------------------------------------------
SWEEP_DIR     = "results/pop_stim_sweep"
TRANSIENT_MS  = 500          # skip leading transient
ANALYSIS_MS   = 2000         # length of analysis window after transient
FREQ_RANGE    = (1, 100)
MODES         = ["power", "pct"]   # produce one figure per mode per condition
LFP_KEY       = "bipolar_matrix"
SAVE_FIGS     = True
FIG_DIR       = os.path.join(SWEEP_DIR, "figures")

# Smoothing controls (these kill the horizontal striping)
NW            = 4            # multitaper time-bandwidth (was 2). Higher = smoother PSD.
SMOOTH_HZ     = 1.5          # gaussian smoothing of PSD along freq axis (Hz). 0 disables.
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
    """Gaussian smooth a PSD (or 2D array along last axis) over frequency."""
    if sigma_hz is None or sigma_hz <= 0:
        return psd
    df = float(np.median(np.diff(freqs)))
    if df <= 0:
        return psd
    sigma_bins = sigma_hz / df
    if sigma_bins < 0.5:
        return psd
    return gaussian_filter1d(psd, sigma=sigma_bins, axis=-1, mode='nearest')


def _parse_rate(fname):
    m = re.search(r"rate_(\d+)Hz", os.path.basename(fname))
    return int(m.group(1)) if m else None


def load_condition(cond_dir):
    files = sorted(glob.glob(os.path.join(cond_dir, "*.npz")))
    trials = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if LFP_KEY not in d.files:
            print(f"  skip {f}: missing {LFP_KEY}")
            continue
        rate = (float(d['rate_hz']) if 'rate_hz' in d.files
                else _parse_rate(f))
        trials.append({
            'rate_hz': rate,
            'lfp':     d[LFP_KEY],
            'time_ms': np.asarray(d['time_array_ms']),
            'transient_ms': float(d['transient_ms'])
                if 'transient_ms' in d.files else TRANSIENT_MS,
            'channel_labels': d['channel_labels'],
            'channel_depths': d['channel_depths'],
        })
    trials.sort(key=lambda t: t['rate_hz'])
    return trials


def compute_rate_freq_matrix(trials):
    fs = 1000.0 / float(np.mean(np.diff(trials[0]['time_ms'])))
    n_channels = trials[0]['lfp'].shape[0]
    rates = np.array([t['rate_hz'] for t in trials])

    freqs_ref = None
    freq_mask = None
    psd_stack = None

    for r_idx, tr in enumerate(trials):
        time = tr['time_ms']
        t0 = tr['transient_ms']
        mask = (time >= t0) & (time < t0 + ANALYSIS_MS)
        for ch in range(n_channels):
            seg = tr['lfp'][ch][mask].copy()
            if len(seg) == 0 or np.any(np.isnan(seg)):
                continue
            seg = detrend(seg)
            nfft = 2 ** int(np.ceil(np.log2(len(seg))))
            f, psd = multitaper_psd(seg, fs=fs, NW=NW, nfft=nfft)

            if freqs_ref is None:
                freq_mask = (f >= FREQ_RANGE[0]) & (f <= FREQ_RANGE[1])
                freqs_ref = f[freq_mask]
                psd_stack = np.full(
                    (n_channels, len(trials), freqs_ref.size), np.nan)
            psd_stack[ch, r_idx, :] = psd[freq_mask]

    if SMOOTH_HZ and SMOOTH_HZ > 0 and freqs_ref is not None:
        psd_stack = smooth_along_freq(psd_stack, freqs_ref, SMOOTH_HZ)

    return rates, freqs_ref, psd_stack


def apply_mode(psd_stack, rates, mode):
    if mode == 'power':
        return (10 * np.log10(psd_stack + 1e-20),
                'Power (dB)', 'viridis', None)
    zero_idx = np.argmin(np.abs(rates - 0))
    ref = psd_stack[:, zero_idx:zero_idx + 1, :]
    if mode == 'pct':
        pct = (psd_stack - ref) / (ref + 1e-20) * 100.0
        return (pct, '% change vs 0 Hz', 'RdBu_r',
                TwoSlopeNorm(vmin=-100, vcenter=0, vmax=500))
    if mode == 'db_diff':
        diff = 10 * np.log10((psd_stack + 1e-20) / (ref + 1e-20))
        return (diff, 'dB change vs 0 Hz', 'RdBu_r',
                TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10))
    raise ValueError(f"unknown mode {mode}")


def plot_condition_one_mode(cond_name, rates, freqs, psd_stack,
                            labels, depths, mode):
    mat, cbar_label, cmap, norm = apply_mode(psd_stack, rates, mode)
    n_channels = psd_stack.shape[0]
    bip_depths = ((depths[:-1] + depths[1:]) / 2
                  if len(depths) > n_channels else depths[:n_channels])

    ncols = 5
    nrows = int(np.ceil(n_channels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()

    if mode == 'power':
        vmin = np.nanpercentile(mat, 5)
        vmax = np.nanpercentile(mat, 95)
    else:
        vmin = vmax = None

    extent = [rates[0], rates[-1], freqs[0], freqs[-1]]
    last_im = None
    for ch in range(n_channels):
        ax = axes[ch]
        z = mat[ch].T
        if mode == 'power':
            im = ax.imshow(z, aspect='auto', origin='lower',
                           extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(z, aspect='auto', origin='lower',
                           extent=extent, cmap=cmap, norm=norm)
        last_im = im
        lbl = labels[ch] if ch < len(labels) else f"ch{ch}"
        dep = bip_depths[ch] if ch < len(bip_depths) else np.nan
        ax.set_title(f"{lbl}  z={dep:.2f}", fontsize=9)
        if ch % ncols == 0:
            ax.set_ylabel('Freq (Hz)')
        if ch // ncols == nrows - 1:
            ax.set_xlabel('Stim rate (Hz)')

    for k in range(n_channels, len(axes)):
        axes[k].axis('off')

    fig.suptitle(
        f"{cond_name}  |  analysis window {TRANSIENT_MS}-"
        f"{TRANSIENT_MS + ANALYSIS_MS} ms  |  mode={mode}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.95, 0.96])
    cbar_ax = fig.add_axes([0.96, 0.12, 0.012, 0.76])
    cbar = fig.colorbar(last_im, cax=cbar_ax, label=cbar_label)
    if mode == 'pct':
        cbar.set_ticks([-100, -50, 0, 100, 250, 500])

    if SAVE_FIGS:
        os.makedirs(FIG_DIR, exist_ok=True)
        out = os.path.join(FIG_DIR, f"{cond_name}_{mode}.png")
        fig.savefig(out, dpi=140, bbox_inches='tight')
        print(f"  saved {out}")
    return fig


def plot_condition(cond_name, trials):
    rates, freqs, psd_stack = compute_rate_freq_matrix(trials)
    labels = trials[0]['channel_labels']
    depths = trials[0]['channel_depths']
    figs = []
    for mode in MODES:
        figs.append(
            plot_condition_one_mode(cond_name, rates, freqs, psd_stack,
                                    labels, depths, mode)
        )
    return figs


def main():
    cond_dirs = [
        d for d in sorted(glob.glob(os.path.join(SWEEP_DIR, "*")))
        if os.path.isdir(d) and os.path.basename(d) not in
           ("figures", "config_snapshot")
    ]
    if not cond_dirs:
        raise FileNotFoundError(f"no condition dirs in {SWEEP_DIR}")

    for cd in cond_dirs:
        cond_name = os.path.basename(cd)
        print(f"\n=== {cond_name} ===")
        trials = load_condition(cd)
        if len(trials) < 2:
            print(f"  skip: only {len(trials)} rates")
            continue
        print(f"  loaded {len(trials)} rates: "
              f"{[t['rate_hz'] for t in trials]}")
        plot_condition(cond_name, trials)

    plt.show()


if __name__ == "__main__":
    main()
