"""Paired version of lesion_significance.py.

The simulation uses the same (network_seed, baseline_seed, stim_seed) triple
for trial i in every condition. So trial i in control and trial i in lesion
are running matched noise realisations -- only the lesion differs. We exploit
that here:

    log2FC_i(ch, f) = log2( P_les[i,ch,f] / P_ctrl[i,ch,f] )

i.e. compute the change PER TRIAL, then average across trials. The
significance test becomes a one-sample sign-flip permutation on those
per-trial log ratios -- a paired test, much more powerful than the
two-sample test used in the unpaired analysis.
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from laminar_power_change import load_trials
from lesion_power_change import _bipolar_current_from_trials, count_trials
from lesion_significance2 import (
    per_trial_psd, fdr_bh, plot_sig_heatmap, summarize_lesion,
    write_summary_text, _band_overlay,
    WINDOW_MS, N_PERM, ALPHA_FDR, FREQ_RANGE, BANDS,
)

LFP_KEY = 'bipolar_lfp'


RNG_SEED = 0


def paired_log2fc_perm(psd_ctrl, psd_les, n_perm=N_PERM, rng=None):
    """Paired permutation test on log2 fold-changes.

    psd_ctrl, psd_les : (n_trial, n_ch, n_f), with row i matched across both.

    Per-trial log2FC:  d[i, ch, f] = log2(P_les[i] / P_ctrl[i])
    Observed statistic: mean across trials of d.
    Null: under H0 the lesion has no effect, so for each trial the sign of
    d_i could equally well be flipped. We sample 2^n / n_perm sign patterns
    and recompute mean(d).
    Two-sided p:  fraction of sign-flipped means whose |value| >= |obs|.
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    if psd_ctrl.shape != psd_les.shape:
        raise ValueError(f'paired test requires equal shapes; got '
                         f'{psd_ctrl.shape} vs {psd_les.shape}')

    n_trial = psd_ctrl.shape[0]
    # d has shape (n_trial, n_ch, n_f)
    d = np.log2((psd_les + 1e-20) / (psd_ctrl + 1e-20))
    obs = d.mean(axis=0)
    abs_obs = np.abs(obs)

    count = np.zeros_like(obs)
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=(n_trial, 1, 1))
        m = (d * signs).mean(axis=0)
        count += (np.abs(m) >= abs_obs).astype(np.float64)
    pvals = (count + 1.0) / (n_perm + 1.0)
    return obs, pvals, d


def plot_paired_diagnostic(freqs, depths, labels, d_per_trial,
                           log2fc, sig_mask, lesion_name, out_path,
                           mid_z=(-0.3, 0.7)):
    """One-page paired diagnostic: per-trial log2FC + group mean + sig."""
    mid_mask = (depths >= mid_z[0]) & (depths <= mid_z[1])
    if not mid_mask.any():
        mid_mask = np.ones_like(depths, dtype=bool)

    # mid-layer per-trial fold change, averaged across mid channels
    d_mid = d_per_trial[:, mid_mask, :].mean(axis=1)  # (n_trial, n_f)
    mean_mid = d_mid.mean(0)
    sem_mid = d_mid.std(0, ddof=1) / np.sqrt(d_mid.shape[0])
    sig_mid = sig_mask[mid_mask].any(0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1.3, 1]})

    ax = axes[0]
    for i in range(d_mid.shape[0]):
        ax.plot(freqs, d_mid[i], color='gray', alpha=0.35, lw=0.8)
    ax.plot(freqs, mean_mid, color='k', lw=2.0, label='mean across trials')
    ax.fill_between(freqs, mean_mid - sem_mid, mean_mid + sem_mid,
                    color='k', alpha=0.18, label='+/- SEM')
    ax.axhline(0, color='k', lw=0.5, alpha=0.5)
    # mark sig bins
    for i in np.where(sig_mid)[0]:
        ax.plot(freqs[i], ax.get_ylim()[0] * 0.95, marker='|',
                color='C3' if mean_mid[i] > 0 else 'C0', markersize=6)
    _band_overlay(ax, freqs[0], freqs[-1], label_bands=True)
    ax.set_ylabel('per-trial log2 FC')
    ax.set_title(f'Lesion {lesion_name} -- paired log2 FC per trial '
                 f'(gray) and group mean (black). '
                 f'Mid-layer (z in [{mid_z[0]:+.1f}, {mid_z[1]:+.1f}]).')
    ax.set_xscale('log')
    ax.set_xlim(freqs[0], freqs[-1])
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, which='both', alpha=0.25)

    # channel x freq heatmap of mean log2FC, sig-masked
    ax = axes[1]
    masked = np.where(sig_mask, log2fc, np.nan)
    n_ch = log2fc.shape[0]
    extent = [freqs[0], freqs[-1], n_ch - 0.5, -0.5]
    vlim = float(np.nanpercentile(np.abs(log2fc), 98))
    vlim = max(vlim, 0.1)
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    im = ax.imshow(masked, aspect='auto', cmap='RdBu_r',
                   extent=extent, norm=norm, interpolation='nearest')
    ax.set_facecolor('#dddddd')
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels([f'{lab}  z={z:+.2f}' for lab, z in zip(labels, depths)],
                       fontsize=7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title(f'channel x freq, FDR<{ALPHA_FDR} only')
    _band_overlay(ax, freqs[0], freqs[-1], label_bands=False)
    plt.colorbar(im, ax=ax, label='log2 FC')

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def run(lesion_root, fig_root, window_ms=WINDOW_MS, n_perm=N_PERM,
        lfp_key=LFP_KEY):
    ctrl_dir = os.path.join(lesion_root, 'control')
    n_ctrl = count_trials(ctrl_dir)
    print(f'[control] loading {n_ctrl} trials')
    ctrl_trials = load_trials(ctrl_dir, n_ctrl)
    if lfp_key == 'bipolar_lfp_current':
        ctrl_trials = _bipolar_current_from_trials(ctrl_trials)
    psd_ctrl, freqs, depths, labels = per_trial_psd(
        ctrl_trials, lfp_key, window_ms=window_ms)
    print(f'  ctrl PSD shape {psd_ctrl.shape}')

    out_dir = os.path.join(fig_root, '_significance_paired_may21')
    os.makedirs(out_dir, exist_ok=True)

    lesion_names = sorted(
        d for d in os.listdir(lesion_root)
        if os.path.isdir(os.path.join(lesion_root, d)) and d != 'control'
    )

    rng = np.random.default_rng(RNG_SEED)
    grand_log2fc = []
    grand_sig_count = []
    grand_names = []
    per_lesion_rows = {}

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
            print(f'[{lname}] freq grid mismatch — skip')
            continue
        # paired test requires same n_trial; truncate to min
        n_pair = min(psd_ctrl.shape[0], psd_les.shape[0])
        log2fc, pvals, d_per_trial = paired_log2fc_perm(
            psd_ctrl[:n_pair], psd_les[:n_pair], n_perm=n_perm, rng=rng)
        sig = fdr_bh(pvals, alpha=ALPHA_FDR)
        n_sig = int(sig.sum())
        n_tot = sig.size
        print(f'  paired: {n_sig}/{n_tot} bins significant '
              f'({100*n_sig/n_tot:.1f}%)')

        # reuse the unpaired plot for the channel x freq heatmap...
        plot_sig_heatmap(
            freqs, depths, labels, log2fc, sig, lname,
            os.path.join(out_dir, f'{lname}_sig_heatmap.png'))
        # ...and add the paired diagnostic
        plot_paired_diagnostic(
            freqs, depths, labels, d_per_trial, log2fc, sig, lname,
            os.path.join(out_dir, f'{lname}_paired_diagnostic.png'))
        rows = summarize_lesion(freqs, depths, labels, log2fc, sig)
        write_summary_text(rows, lname,
                           os.path.join(out_dir, f'{lname}_summary.txt'))
        per_lesion_rows[lname] = rows

        mid_mask = (depths >= -0.3) & (depths <= 0.7)
        if not mid_mask.any():
            mid_mask = np.ones_like(depths, dtype=bool)
        mid_lfc = np.where(sig, log2fc, 0.0)[mid_mask].mean(0)
        grand_log2fc.append(mid_lfc)

        band_score = []
        for band, (f_lo, f_hi) in BANDS.items():
            m = (freqs >= f_lo) & (freqs <= f_hi)
            up = int(((log2fc > 0) & sig)[:, m].any(axis=1).sum())
            dn = int(((log2fc < 0) & sig)[:, m].any(axis=1).sum())
            band_score.append(up - dn)
        grand_sig_count.append(band_score)
        grand_names.append(lname)

    if not grand_names:
        print('no lesions processed')
        return

    grand_log2fc = np.stack(grand_log2fc, axis=0)
    grand_sig_count = np.array(grand_sig_count, dtype=int)

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
    ax.set_title('PAIRED test: sig-masked log2 FC, mid-layer (z in [-0.3,+0.7]) average')
    _band_overlay(ax, freqs[0], freqs[-1], label_bands=True)
    plt.colorbar(im, ax=ax, label='log2 FC')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'grand_summary_log2fc.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_dir}/grand_summary_log2fc.png')

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
    ax.set_title('PAIRED test: sig channels per band (up - down)')
    for i in range(grand_sig_count.shape[0]):
        for j in range(grand_sig_count.shape[1]):
            v = grand_sig_count[i, j]
            if v != 0:
                ax.text(j, i, f'{v:+d}', ha='center', va='center',
                        fontsize=8,
                        color='white' if abs(v) > vlim2 * 0.6 else 'black')
    plt.colorbar(im, ax=ax, label='channels (up - down)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'grand_summary_sig_counts.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_dir}/grand_summary_sig_counts.png')


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
