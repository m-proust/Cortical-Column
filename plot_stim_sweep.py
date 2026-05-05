"""
Plot stimulus-rate sweeps produced by stim_sweep.py.

For each condition directory under SWEEP_DIR, load all rate trials, compute the
post-stimulus PSD per bipolar channel for each rate, then produce one figure
per condition with 15 subplots (one per bipolar channel). Each subplot is a
rate (x) x frequency (y) heatmap showing how post-stim spectral content
changes with stimulus rate.

Mode:
    'power'     -> dB power of the post-stim window
    'pct'       -> % change vs the 0-Hz post-stim PSD (per channel)
    'db_diff'   -> dB difference vs the 0-Hz post-stim PSD (per channel)
"""

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import detrend
from scipy.signal.windows import dpss
from scipy import signal as _sig


# ---------------------------------------------------------------------------
# USER-EDITABLE
# ---------------------------------------------------------------------------
SWEEP_DIR = "results/stim_sweep_23_04"

POST_START_MS = 500      # skip this long after stim onset
POST_WINDOW_MS = 500    # length of post-stim window to analyse
FREQ_RANGE = (1, 100)    # Hz
MODE = "pct"             # 'power' | 'pct' | 'db_diff'
LFP_KEY = "bipolar_matrix"  # what to read from the npz
SAVE_FIGS = True
FIG_DIR = os.path.join(SWEEP_DIR, "figures")
# ---------------------------------------------------------------------------


def multitaper_psd(data, fs, NW=2, nfft=None):
    data = data - np.mean(data)
    if nfft is None:
        nfft = 2 ** int(np.ceil(np.log2(len(data))))
    K = int(2 * NW - 1)
    tapers = dpss(len(data), NW, K)
    psds = []
    for taper in tapers:
        freqs, psd = _sig.periodogram(
            data * taper, fs=fs, nfft=nfft, scaling='density'
        )
        psds.append(psd)
    return freqs, np.mean(psds, axis=0)


def _parse_rate_from_name(fname):
    # expects ..._rate_XXHz.npz
    m = re.search(r"rate_(\d+)Hz", os.path.basename(fname))
    if m is None:
        return None
    return int(m.group(1))


def load_condition(cond_dir):
    """Return list of trial dicts sorted by sweep_rate_hz."""
    files = sorted(glob.glob(os.path.join(cond_dir, "*.npz")))
    trials = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        if LFP_KEY not in data.files:
            print(f"  skipping {f}: missing {LFP_KEY}")
            continue
        rate = (float(data['sweep_rate_hz'])
                if 'sweep_rate_hz' in data.files
                else _parse_rate_from_name(f))
        trials.append({
            'file': f,
            'rate_hz': rate,
            'bipolar': data[LFP_KEY],
            'time_ms': data['time_array_ms'],
            'stim_onset_ms': float(data['stim_onset_ms']),
            'channel_labels': data['channel_labels'],
            'channel_depths': data['channel_depths'],
        })
    trials.sort(key=lambda t: t['rate_hz'])
    return trials


def compute_rate_freq_matrix(trials):
    """Return (rates, freqs, psd_matrix[n_channels, n_rates, n_freqs])."""
    fs = 1000.0 / float(np.mean(np.diff(trials[0]['time_ms'])))
    n_channels = trials[0]['bipolar'].shape[0]
    rates = np.array([t['rate_hz'] for t in trials])

    freqs_ref = None
    freq_mask = None
    psd_stack = None

    for r_idx, tr in enumerate(trials):
        stim = tr['stim_onset_ms']
        time = tr['time_ms']
        post_mask = ((time >= stim + POST_START_MS) &
                     (time < stim + POST_START_MS + POST_WINDOW_MS))

        for ch in range(n_channels):
            seg = tr['bipolar'][ch][post_mask].copy()
            if len(seg) == 0 or np.any(np.isnan(seg)):
                continue
            seg = detrend(seg)
            nfft = 2 ** int(np.ceil(np.log2(len(seg))))
            f, psd = multitaper_psd(seg, fs=fs, NW=2, nfft=nfft)

            if freqs_ref is None:
                freq_mask = (f >= FREQ_RANGE[0]) & (f <= FREQ_RANGE[1])
                freqs_ref = f[freq_mask]
                psd_stack = np.full(
                    (n_channels, len(trials), freqs_ref.size),
                    np.nan,
                )
            psd_stack[ch, r_idx, :] = psd[freq_mask]

    return rates, freqs_ref, psd_stack


def apply_mode(psd_stack, rates, mode):
    """Transform raw PSD stack into the quantity to plot."""
    if mode == 'power':
        return 10 * np.log10(psd_stack + 1e-20), 'Power (dB)', 'viridis', None
    # both 'pct' and 'db_diff' are relative to the 0-Hz trial
    zero_idx = np.argmin(np.abs(rates - 0))
    ref = psd_stack[:, zero_idx:zero_idx + 1, :]
    if mode == 'pct':
        pct = (psd_stack - ref) / (ref + 1e-20) * 100.0
        return pct, '% change vs 0 Hz', 'RdBu_r', TwoSlopeNorm(
            vmin=-100, vcenter=0, vmax=500
        )
    if mode == 'db_diff':
        diff = 10 * np.log10((psd_stack + 1e-20) / (ref + 1e-20))
        return diff, 'dB change vs 0 Hz', 'RdBu_r', TwoSlopeNorm(
            vmin=-10, vcenter=0, vmax=10
        )
    raise ValueError(f"unknown mode {mode}")


def plot_condition(cond_name, trials):
    rates, freqs, psd_stack = compute_rate_freq_matrix(trials)
    mat, cbar_label, cmap, norm = apply_mode(psd_stack, rates, MODE)
    n_channels = psd_stack.shape[0]

    labels = trials[0]['channel_labels']
    depths = trials[0]['channel_depths']
    bip_depths = (depths[:-1] + depths[1:]) / 2 if len(depths) > n_channels \
        else depths[:n_channels]

    ncols = 5
    nrows = int(np.ceil(n_channels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()

    # shared color scale across channels for 'power' mode
    if MODE == 'power':
        vmin = np.nanpercentile(mat, 5)
        vmax = np.nanpercentile(mat, 95)
    else:
        vmin = vmax = None  # norm handles it for diverging modes

    extent = [rates[0], rates[-1], freqs[0], freqs[-1]]

    last_im = None
    for ch in range(n_channels):
        ax = axes[ch]
        z = mat[ch].T  # [freqs, rates]
        if MODE == 'power':
            im = ax.imshow(z, aspect='auto', origin='lower',
                           extent=extent, cmap=cmap,
                           vmin=vmin, vmax=vmax)
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

    # hide unused axes
    for k in range(n_channels, len(axes)):
        axes[k].axis('off')

    fig.suptitle(f"{cond_name}  |  post-stim {POST_START_MS}-"
                 f"{POST_START_MS + POST_WINDOW_MS} ms  |  mode={MODE}",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.95, 0.96])

    # one shared colorbar on the right
    cbar_ax = fig.add_axes([0.96, 0.12, 0.012, 0.76])
    fig.colorbar(last_im, cax=cbar_ax, label=cbar_label)

    if SAVE_FIGS:
        os.makedirs(FIG_DIR, exist_ok=True)
        out = os.path.join(FIG_DIR, f"{cond_name}_{MODE}.png")
        fig.savefig(out, dpi=140, bbox_inches='tight')
        print(f"  saved {out}")

    return fig


def main():
    cond_dirs = [
        d for d in sorted(glob.glob(os.path.join(SWEEP_DIR, "*")))
        if os.path.isdir(d) and os.path.basename(d) not in
           ("figures", "config_snapshot")
    ]
    if not cond_dirs:
        raise FileNotFoundError(f"No condition subdirs in {SWEEP_DIR}")

    for cd in cond_dirs:
        cond_name = os.path.basename(cd)
        print(f"\n=== {cond_name} ===")
        trials = load_condition(cd)
        if len(trials) < 2:
            print(f"  skip: only {len(trials)} trials")
            continue
        print(f"  loaded {len(trials)} rates: "
              f"{[t['rate_hz'] for t in trials]}")
        plot_condition(cond_name, trials)

    plt.show()


if __name__ == "__main__":
    main()
