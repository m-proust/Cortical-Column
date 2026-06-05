"""For each lesion, find where in frequency the largest *significant* increases
and decreases actually sit. Tells us whether 'red around 10 Hz' is really
10 Hz or actually a 5 Hz theta peak.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from laminar_power_change import load_trials
from lesion_power_change import _bipolar_current_from_trials, count_trials
from lesion_significance import (
    per_trial_psd, perm_test_log2fc, fdr_bh,
    LFP_KEY, WINDOW_MS, N_PERM, FREQ_RANGE, BANDS,
)

LESION_ROOT = 'results/lesions_2026-06-01'
OUT_DIR = 'figures/lesions_2026-06-01/_significance'
os.makedirs(OUT_DIR, exist_ok=True)

# Mid-layer = where the alpha generator lives (L4C / L5 border)
MID_Z_LOW, MID_Z_HIGH = -0.3, 0.7

LO_HZ, HI_HZ = 2.0, 80.0

BAND_COLORS = {
    'delta':      '#f2e6f7',
    'theta':      '#e6f0fa',
    'alpha':      '#e8f5e6',
    'beta':       '#fff4d6',
    'low_gamma':  '#fde0d2',
    'high_gamma': '#f9d2d2',
}


def shade_bands(ax, lo_hz, hi_hz, label=False):
    """Shade frequency bands as colored backgrounds."""
    for bname, (b_lo, b_hi) in BANDS.items():
        x0 = max(b_lo, lo_hz)
        x1 = min(b_hi, hi_hz)
        if x1 <= x0:
            continue
        ax.axvspan(x0, x1, color=BAND_COLORS[bname], alpha=0.6, lw=0, zorder=0)
        if label:
            ax.text((x0 + x1) / 2, 1.01, bname,
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=8, color='#444',
                    clip_on=False)


def main():
    ctrl_dir = os.path.join(LESION_ROOT, 'control')
    n_ctrl = count_trials(ctrl_dir)
    ctrl_trials = _bipolar_current_from_trials(
        load_trials(ctrl_dir, n_ctrl))
    psd_ctrl, freqs, depths, labels = per_trial_psd(
        ctrl_trials, LFP_KEY, window_ms=WINDOW_MS)

    mid_mask = (depths >= MID_Z_LOW) & (depths <= MID_Z_HIGH)
    print(f'mid layer = {labels[mid_mask].tolist()} '
          f'(z = {depths[mid_mask].tolist()})')

    f_zoom = (freqs >= LO_HZ) & (freqs <= HI_HZ)
    f_plot = freqs[f_zoom]

    # control mid-layer mean PSD as reference
    ctrl_mid = psd_ctrl[:, mid_mask, :].mean(axis=(0, 1))  # over trials, channels
    ctrl_ref = ctrl_mid[f_zoom]

    rng = np.random.default_rng(0)

    lesion_names = sorted(
        d for d in os.listdir(LESION_ROOT)
        if os.path.isdir(os.path.join(LESION_ROOT, d)) and d != 'control'
    )

    print(f'\n{"lesion":12s}  {"peak_up_Hz":>10s} {"log2FC_up":>10s}   '
          f'{"peak_down_Hz":>12s} {"log2FC_down":>11s}   '
          f'{"alpha_peak_Hz":>13s}')
    print('-' * 90)

    fig, axes = plt.subplots(5, 4, figsize=(22, 20), sharex=True)
    axes = axes.ravel()
    plot_idx = 0
    n_lesions_processed = 0

    rows = []
    for lname in lesion_names:
        ldir = os.path.join(LESION_ROOT, lname)
        n = count_trials(ldir)
        if n == 0:
            continue
        try:
            les_trials = _bipolar_current_from_trials(
                load_trials(ldir, n))
            psd_les, freqs_l, _, _ = per_trial_psd(
                les_trials, LFP_KEY, window_ms=WINDOW_MS)
        except Exception as exc:
            print(f'[{lname}] skipped ({exc})')
            continue
        if not np.allclose(freqs_l, freqs):
            continue

        log2fc, pvals = perm_test_log2fc(psd_ctrl, psd_les,
                                         n_perm=N_PERM, rng=rng)
        sig = fdr_bh(pvals)

        # mid-layer summary: mean log2FC across mid channels, sig-masked
        log2fc_mid = log2fc[mid_mask].mean(0)
        sig_any_mid = sig[mid_mask].any(0)
        lfc_z = log2fc_mid[f_zoom]
        sig_z = sig_any_mid[f_zoom]

        # find peak increase and peak decrease within significant bins
        sig_pos = sig_z & (lfc_z > 0)
        sig_neg = sig_z & (lfc_z < 0)
        if sig_pos.any():
            i_up = np.argmax(np.where(sig_pos, lfc_z, -np.inf))
            peak_up = (f_plot[i_up], lfc_z[i_up])
        else:
            peak_up = (np.nan, np.nan)
        if sig_neg.any():
            i_dn = np.argmin(np.where(sig_neg, lfc_z, np.inf))
            peak_dn = (f_plot[i_dn], lfc_z[i_dn])
        else:
            peak_dn = (np.nan, np.nan)

        # also: where is the dominant resting peak in the control mid-layer?
        # (the alpha generator frequency)
        ctrl_peak_idx = np.argmax(ctrl_ref)
        ctrl_peak_freq = f_plot[ctrl_peak_idx]

        rows.append((lname, peak_up, peak_dn))
        print(f'{lname:12s}  {peak_up[0]:10.2f} {peak_up[1]:+10.2f}   '
              f'{peak_dn[0]:12.2f} {peak_dn[1]:+11.2f}   '
              f'{ctrl_peak_freq:13.2f}')

        # plot mid-layer PSD + log2FC overlay for this lesion
        if plot_idx < len(axes):
            ax = axes[plot_idx]
            shade_bands(ax, LO_HZ, HI_HZ, label=(plot_idx < 4))

            les_mid = psd_les[:, mid_mask, :].mean(axis=(0, 1))[f_zoom]
            ax.semilogy(f_plot, ctrl_ref, color='k', lw=1.6,
                        label='control', zorder=5)
            ax.semilogy(f_plot, les_mid, color='#c8202a', lw=1.6,
                        label='lesion', zorder=5)

            # mark sig bins as small markers along the bottom of the panel
            ymin, ymax = ax.get_ylim()
            sig_y = ymin * 1.3
            for i in np.where(sig_z)[0]:
                ax.plot(f_plot[i], sig_y, marker='|', markersize=6,
                        color='#c8202a' if lfc_z[i] > 0 else '#1f77b4',
                        zorder=4)

            # vertical line at peak sig increase / decrease
            if np.isfinite(peak_up[0]):
                ax.axvline(peak_up[0], color='#c8202a', ls='--', lw=1,
                           alpha=0.7, zorder=3)
                ax.text(peak_up[0], ymax, f' +{peak_up[0]:.1f}Hz',
                        color='#c8202a', va='top', fontsize=8, zorder=6)
            if np.isfinite(peak_dn[0]):
                ax.axvline(peak_dn[0], color='#1f77b4', ls='--', lw=1,
                           alpha=0.7, zorder=3)
                ax.text(peak_dn[0], ymin, f' {peak_dn[0]:.1f}Hz',
                        color='#1f77b4', va='bottom', fontsize=8, zorder=6)

            ax.set_title(lname, fontsize=10)
            ax.set_xlim(LO_HZ, HI_HZ)
            ax.set_xscale('log')
            ax.grid(True, which='both', alpha=0.25, zorder=1)
            if plot_idx == 0:
                ax.legend(fontsize=8, loc='lower left')
            plot_idx += 1
            n_lesions_processed += 1

    for k in range(plot_idx, len(axes)):
        axes[k].axis('off')
    fig.suptitle(
        f'Mid-layer PSD (z in [{MID_Z_LOW:+.1f}, {MID_Z_HIGH:+.1f}]):  '
        f'control (black) vs lesion (red).  '
        f'Tick marks at the bottom = freq bins with FDR<0.05 vs control  '
        f'(red=lesion > control, blue=lesion < control).',
        fontsize=13)
    fig.supxlabel('Frequency (Hz, log scale)')
    fig.supylabel('PSD (log)')
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'mid_layer_psd_{int(LO_HZ)}_{int(HI_HZ)}Hz.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'\nsaved {out}')


if __name__ == '__main__':
    main()
