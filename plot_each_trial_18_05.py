"""Per-trial laminar spectral profile for results/trials_18_05.

Uses the same windows/detrend/multitaper params as laminar_power_change.py
(pre 500 ms, post 500 ms starting 500 ms after stim, NW=2). Saves one PNG
per trial into results/trials_18_05/laminar_per_trial/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from laminar_power_change import (
    load_trials,
    plot_laminar_spectral_profile,  # not used directly but imported for parity
    multitaper_psd,
)
from scipy.signal import detrend


BASE = 'results/trials_18_05'
OUT_DIR = os.path.join(BASE, 'laminar_per_trial')
N_TRIALS = 36

COMMON = dict(
    pre_window_ms=500,
    post_window_ms=500,
    post_start_ms=500,
    freq_range=(0, 120),
    do_detrend=True,
)


def compute_laminar(trial, lfp_key, pre_window_ms, post_window_ms,
                    post_start_ms, freq_range, do_detrend):
    lfp = trial[lfp_key]
    time = np.asarray(trial['time'])
    stim = float(trial['stim_onset_ms'])
    fs = 1000.0 / float(np.mean(np.diff(time)))

    n_ch = lfp.shape[0]
    depths = trial['channel_depths']
    if lfp_key == 'bipolar_lfp' and len(depths) > n_ch:
        depths = (depths[:-1] + depths[1:]) / 2
    else:
        depths = depths[:n_ch]

    pre_mask = (time >= stim - pre_window_ms) & (time < stim)
    post_mask = ((time >= stim + post_start_ms) &
                 (time < stim + post_start_ms + post_window_ms))

    psd_pre_rows, psd_post_rows = [], []
    f = None
    for ch in range(n_ch):
        pre = lfp[ch][pre_mask].copy()
        post = lfp[ch][post_mask].copy()
        if len(pre) == 0 or len(post) == 0:
            psd_pre_rows.append(None)
            psd_post_rows.append(None)
            continue
        if do_detrend:
            pre = detrend(pre)
            post = detrend(post)
        nfft = 2 ** int(np.ceil(np.log2(min(len(pre), len(post)))))
        f, ppre = multitaper_psd(pre, fs=fs, NW=2, nfft=nfft)
        _, ppost = multitaper_psd(post, fs=fs, NW=2, nfft=nfft)
        psd_pre_rows.append(ppre)
        psd_post_rows.append(ppost)

    n_freq = len(f)
    for i in range(n_ch):
        if psd_pre_rows[i] is None:
            psd_pre_rows[i] = np.full(n_freq, np.nan)
            psd_post_rows[i] = np.full(n_freq, np.nan)

    psd_pre = np.array(psd_pre_rows)
    psd_post = np.array(psd_post_rows)
    mask = (f >= freq_range[0]) & (f <= freq_range[1])
    f_plot = f[mask]
    psd_pre = psd_pre[:, mask]
    psd_post = psd_post[:, mask]

    pre_db = 10 * np.log10(psd_pre + 1e-10)
    post_db = 10 * np.log10(psd_post + 1e-10)
    pct = (psd_post - psd_pre) / psd_pre * 100.0

    pre_db = np.flipud(pre_db)
    post_db = np.flipud(post_db)
    pct = np.flipud(pct)
    return f_plot, depths, pre_db, post_db, pct


def plot_panel(axes_row, f_plot, n_ch, pre_db, post_db, pct, row_label):
    extent = [f_plot[0], f_plot[-1], -0.5, n_ch - 0.5]
    vmin = np.nanpercentile([pre_db, post_db], 5)
    vmax = np.nanpercentile([pre_db, post_db], 95)

    im0 = axes_row[0].imshow(pre_db, aspect='auto', cmap='viridis',
                             extent=extent, origin='upper',
                             vmin=vmin, vmax=vmax)
    axes_row[0].set_title(f'{row_label}: Baseline')
    axes_row[0].set_xlabel('Frequency (Hz)')
    axes_row[0].set_ylabel('Laminar depth')
    plt.colorbar(im0, ax=axes_row[0], label='Power (dB)')

    im1 = axes_row[1].imshow(post_db, aspect='auto', cmap='viridis',
                             extent=extent, origin='upper',
                             vmin=vmin, vmax=vmax)
    axes_row[1].set_title(f'{row_label}: Post-stimulus')
    axes_row[1].set_xlabel('Frequency (Hz)')
    axes_row[1].set_ylabel('Laminar depth')
    plt.colorbar(im1, ax=axes_row[1], label='Power (dB)')

    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=500)
    im2 = axes_row[2].imshow(pct, aspect='auto', cmap='RdBu_r',
                             extent=extent, origin='upper', norm=norm)
    axes_row[2].set_title(f'{row_label}: % change')
    axes_row[2].set_xlabel('Frequency (Hz)')
    axes_row[2].set_ylabel('Laminar depth')
    cbar = plt.colorbar(im2, ax=axes_row[2], label='% Change')
    cbar.set_ticks([-100, -50, 0, 100])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    trials = load_trials(BASE, N_TRIALS)
    print(f'Loaded {len(trials)} trials → saving to {OUT_DIR}/')

    for idx, trial in enumerate(trials):
        f_b, depths_b, pre_b, post_b, pct_b = compute_laminar(
            trial, lfp_key='bipolar_lfp', **COMMON)
        f_k, depths_k, pre_k, post_k, pct_k = compute_laminar(
            trial, lfp_key='lfp_matrix', **COMMON)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        plot_panel(axes[0], f_b, pre_b.shape[0], pre_b, post_b, pct_b,
                   'Bipolar')
        plot_panel(axes[1], f_k, pre_k.shape[0], pre_k, post_k, pct_k,
                   'Kernel')
        fig.suptitle(f'Trial {idx:03d}  -  laminar spectral profile',
                     fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(OUT_DIR, f'trial_{idx:03d}_laminar.png')
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f'  wrote {out}')


if __name__ == '__main__':
    main()
