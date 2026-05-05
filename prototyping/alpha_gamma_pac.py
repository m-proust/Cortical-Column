"""
Phase-amplitude coupling between alpha phase and gamma amplitude.

Implements the Tort (2010) modulation index (MI):
    1. bandpass-filter LFP into the alpha band -> instantaneous phase
    2. bandpass-filter LFP into the gamma band -> instantaneous amplitude
    3. bin the gamma amplitude by alpha phase (n_bins)
    4. normalise the binned amplitude into a probability distribution
    5. MI = (log(n_bins) - H(P)) / log(n_bins)
       where H is Shannon entropy

A significance test is computed by phase-shuffling: circularly shift the
alpha phase relative to the gamma amplitude n_perm times, recompute MI,
and report a z-score and p-value.

Computed per bipolar channel, separately for the baseline and post-stim
windows, averaged across trials.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert, detrend


# ---------------------------------------------------------------------------
# USER-EDITABLE
# ---------------------------------------------------------------------------
TRIAL_DIR      = "results/trials_26_04"
LFP_KEY        = "bipolar_matrix"
ALPHA_BAND     = (8, 13)
GAMMA_BAND     = (30, 80)

BASE_WINDOW_MS = 1500
POST_START_MS  = 200
POST_WINDOW_MS = 1500

N_BINS         = 18           # phase bins (Tort default)
N_PERM         = 200          # permutations for significance test
DETREND_PRE    = True

SAVE_DIR = os.path.join(TRIAL_DIR, "alpha_gamma_pac")
# ---------------------------------------------------------------------------


def make_sos(low, high, fs, order=4):
    nyq = fs / 2.0
    return butter(order, [low / nyq, high / nyq],
                  btype='bandpass', output='sos')


def bandpass(x, fs, band):
    sos = make_sos(band[0], band[1], fs)
    return sosfiltfilt(sos, x)


def modulation_index(phase, amp, n_bins=N_BINS):
    """Tort 2010 MI given alpha phase and gamma amplitude vectors."""
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    binned = np.zeros(n_bins)
    for k in range(n_bins):
        m = (phase >= edges[k]) & (phase < edges[k + 1])
        if np.any(m):
            binned[k] = np.mean(amp[m])
    if binned.sum() <= 0:
        return np.nan, binned
    P = binned / binned.sum()
    P = np.clip(P, 1e-12, None)
    H = -np.sum(P * np.log(P))
    Hmax = np.log(n_bins)
    mi = (Hmax - H) / Hmax
    return mi, binned


def mi_with_permutation(phase, amp, n_bins=N_BINS, n_perm=N_PERM,
                        rng=None):
    rng = np.random.default_rng() if rng is None else rng
    mi_obs, binned = modulation_index(phase, amp, n_bins)
    if not np.isfinite(mi_obs) or n_perm <= 0:
        return mi_obs, binned, np.nan, np.nan, np.nan

    n = len(amp)
    min_shift = max(int(n * 0.1), 1)
    surr = np.empty(n_perm)
    for k in range(n_perm):
        shift = int(rng.integers(min_shift, n - min_shift))
        amp_s = np.roll(amp, shift)
        mi_s, _ = modulation_index(phase, amp_s, n_bins)
        surr[k] = mi_s
    mu = np.nanmean(surr)
    sd = np.nanstd(surr)
    z = (mi_obs - mu) / sd if sd > 0 else np.nan
    p = float(np.mean(surr >= mi_obs))   # one-sided
    return mi_obs, binned, z, p, surr


def load_files(trial_dir):
    files = sorted(glob.glob(os.path.join(trial_dir, "trial_*.npz")))
    if not files:
        raise FileNotFoundError(f"no trial_*.npz under {trial_dir}")
    return files


def process(files):
    n_trials = len(files)
    first = np.load(files[0], allow_pickle=True)
    n_channels = first[LFP_KEY].shape[0]
    channel_labels = first['channel_labels']
    channel_depths = first['channel_depths']

    mi_base = np.full((n_trials, n_channels), np.nan)
    mi_post = np.full((n_trials, n_channels), np.nan)
    z_base  = np.full((n_trials, n_channels), np.nan)
    z_post  = np.full((n_trials, n_channels), np.nan)
    p_base  = np.full((n_trials, n_channels), np.nan)
    p_post  = np.full((n_trials, n_channels), np.nan)

    binned_base = np.full((n_trials, n_channels, N_BINS), np.nan)
    binned_post = np.full((n_trials, n_channels, N_BINS), np.nan)

    rng = np.random.default_rng(0)

    for ti, fpath in enumerate(files):
        d = np.load(fpath, allow_pickle=True)
        lfp = d[LFP_KEY]
        time = np.asarray(d['time_array_ms'])
        stim = float(d['stim_onset_ms'])
        fs = 1000.0 / float(np.mean(np.diff(time)))

        base_mask = (time >= stim - BASE_WINDOW_MS) & (time < stim)
        post_mask = ((time >= stim + POST_START_MS) &
                     (time < stim + POST_START_MS + POST_WINDOW_MS))

        for ch in range(n_channels):
            sig_full = lfp[ch].astype(float)
            base_seg = sig_full[base_mask]
            post_seg = sig_full[post_mask]
            if len(base_seg) < 50 or len(post_seg) < 50:
                continue
            if DETREND_PRE:
                base_seg = detrend(base_seg)
                post_seg = detrend(post_seg)

            phi_b = np.angle(hilbert(bandpass(base_seg, fs, ALPHA_BAND)))
            amp_b = np.abs(hilbert(bandpass(base_seg, fs, GAMMA_BAND)))
            phi_p = np.angle(hilbert(bandpass(post_seg, fs, ALPHA_BAND)))
            amp_p = np.abs(hilbert(bandpass(post_seg, fs, GAMMA_BAND)))

            mi_b, bin_b, zb, pb, _ = mi_with_permutation(phi_b, amp_b,
                                                         rng=rng)
            mi_p, bin_p, zp, pp, _ = mi_with_permutation(phi_p, amp_p,
                                                         rng=rng)
            mi_base[ti, ch] = mi_b
            mi_post[ti, ch] = mi_p
            z_base[ti, ch]  = zb
            z_post[ti, ch]  = zp
            p_base[ti, ch]  = pb
            p_post[ti, ch]  = pp
            if bin_b.sum() > 0:
                binned_base[ti, ch] = bin_b / bin_b.sum()
            if bin_p.sum() > 0:
                binned_post[ti, ch] = bin_p / bin_p.sum()
        print(f"  trial {ti+1}/{n_trials} processed")

    return {
        'mi_base':        mi_base,
        'mi_post':        mi_post,
        'z_base':         z_base,
        'z_post':         z_post,
        'p_base':         p_base,
        'p_post':         p_post,
        'binned_base':    binned_base,
        'binned_post':    binned_post,
        'channel_labels': channel_labels,
        'channel_depths': channel_depths,
    }


def plot_bar(out, save_path):
    n_channels = out['mi_base'].shape[1]
    depths = out['channel_depths']
    labels = out['channel_labels']
    bip_d = ((depths[:-1] + depths[1:]) / 2
             if len(depths) > n_channels else depths[:n_channels])

    mb = np.nanmean(out['mi_base'], axis=0)
    mp = np.nanmean(out['mi_post'], axis=0)
    sb = np.nanstd(out['mi_base'], axis=0, ddof=1) / \
         np.sqrt(np.sum(np.isfinite(out['mi_base']), axis=0))
    sp = np.nanstd(out['mi_post'], axis=0, ddof=1) / \
         np.sqrt(np.sum(np.isfinite(out['mi_post']), axis=0))

    # Channel-level p-value: fraction of trials whose MI was non-significant
    # is not what we want -> use the median trial p, plus average z-score.
    pb_med = np.nanmedian(out['p_base'], axis=0)
    pp_med = np.nanmedian(out['p_post'], axis=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    y = np.arange(n_channels)
    width = 0.4
    ax.barh(y - width / 2, mb, width, xerr=sb, label='baseline',
            color='steelblue', error_kw=dict(lw=0.7))
    ax.barh(y + width / 2, mp, width, xerr=sp, label='post-stim',
            color='indianred', error_kw=dict(lw=0.7))
    for i in range(n_channels):
        if np.isfinite(pb_med[i]) and pb_med[i] < 0.05:
            ax.text(mb[i], y[i] - width / 2, '*', va='center', fontsize=12)
        if np.isfinite(pp_med[i]) and pp_med[i] < 0.05:
            ax.text(mp[i], y[i] + width / 2, '*', va='center', fontsize=12)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{labels[i]} z={bip_d[i]:.2f}" for i in range(n_channels)],
        fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('Modulation Index (Tort 2010)')
    ax.set_title('Alpha-phase / Gamma-amplitude PAC per channel\n'
                 '* = median permutation p < 0.05')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    print(f"  saved {save_path}")
    return fig, mb, sb, mp, sp, pb_med, pp_med


def plot_phase_distributions(out, save_path):
    """Polar plot of mean gamma-amp distribution across phase, per channel."""
    n_channels = out['mi_base'].shape[1]
    labels = out['channel_labels']
    depths = out['channel_depths']
    bip_d = ((depths[:-1] + depths[1:]) / 2
             if len(depths) > n_channels else depths[:n_channels])

    edges = np.linspace(-np.pi, np.pi, N_BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    width = (2 * np.pi) / N_BINS

    ncols = 5
    nrows = int(np.ceil(n_channels / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.4 * ncols, 3.0 * nrows),
                             subplot_kw={'projection': 'polar'})
    axes = np.atleast_2d(axes).ravel()

    mean_b = np.nanmean(out['binned_base'], axis=0)
    mean_p = np.nanmean(out['binned_post'], axis=0)

    for ch in range(n_channels):
        ax = axes[ch]
        b = mean_b[ch]
        p = mean_p[ch]
        if np.all(np.isnan(b)) and np.all(np.isnan(p)):
            ax.set_visible(False)
            continue
        if not np.all(np.isnan(b)):
            ax.bar(centres, b, width=width, alpha=0.5, color='steelblue',
                   edgecolor='steelblue', label='base')
        if not np.all(np.isnan(p)):
            ax.bar(centres, p, width=width, alpha=0.5, color='indianred',
                   edgecolor='indianred', label='post')
        ax.set_title(f"{labels[ch]} z={bip_d[ch]:.2f}", fontsize=8)
        ax.set_xticks([0, np.pi/2, np.pi, -np.pi/2])
        ax.set_xticklabels(['0', 'π/2', '±π', '-π/2'], fontsize=6)
        ax.tick_params(labelsize=5)

    for k in range(n_channels, len(axes)):
        axes[k].axis('off')

    handles, lbls = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, lbls, loc='upper right', fontsize=9)
    fig.suptitle(
        'Gamma amplitude distribution by alpha phase\n'
        '(probability per phase bin, averaged across trials)',
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    print(f"  saved {save_path}")
    return fig


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    files = load_files(TRIAL_DIR)
    print(f"loaded {len(files)} trials from {TRIAL_DIR}")
    out = process(files)

    bar_path = os.path.join(SAVE_DIR, "pac_modulation_index.png")
    _, mb, sb, mp, sp, pb_med, pp_med = plot_bar(out, bar_path)

    polar_path = os.path.join(SAVE_DIR, "pac_phase_distribution.png")
    plot_phase_distributions(out, polar_path)

    n_channels = out['mi_base'].shape[1]
    n_trials = out['mi_base'].shape[0]
    bip_d = (
        (out['channel_depths'][:-1] + out['channel_depths'][1:]) / 2
        if len(out['channel_depths']) > n_channels
        else out['channel_depths'][:n_channels]
    )

    z_base_mean = np.nanmean(out['z_base'], axis=0)
    z_post_mean = np.nanmean(out['z_post'], axis=0)

    df = pd.DataFrame({
        'channel': out['channel_labels'][:n_channels],
        'depth_mm': bip_d,
        'MI_baseline_mean':  mb,
        'MI_baseline_sem':   sb,
        'MI_baseline_z':     z_base_mean,
        'MI_baseline_pmed':  pb_med,
        'MI_poststim_mean':  mp,
        'MI_poststim_sem':   sp,
        'MI_poststim_z':     z_post_mean,
        'MI_poststim_pmed':  pp_med,
    })
    csv_path = os.path.join(SAVE_DIR, "pac_modulation_index.csv")
    df.to_csv(csv_path, index=False)
    print(f"  saved {csv_path}")

    print("\n" + "=" * 100)
    print("PHASE-AMPLITUDE COUPLING SUMMARY (alpha phase -> gamma amplitude)")
    print(f"  trials      : {n_trials}")
    print(f"  channels    : {n_channels}")
    print(f"  alpha band  : {ALPHA_BAND[0]}-{ALPHA_BAND[1]} Hz")
    print(f"  gamma band  : {GAMMA_BAND[0]}-{GAMMA_BAND[1]} Hz")
    print(f"  baseline    : {BASE_WINDOW_MS} ms before stim onset")
    print(f"  post-stim   : {POST_START_MS}-"
          f"{POST_START_MS + POST_WINDOW_MS} ms after stim onset")
    print(f"  phase bins  : {N_BINS}    permutations: {N_PERM}")
    print(f"  LFP key     : {LFP_KEY}")
    print("=" * 100)
    print(f"\n{'ch':>3}  {'label':<14} {'depth':>7}   "
          f"{'MI_base':>9}  {'sem':>7}  {'z_base':>7}  {'p_base':>7}   "
          f"{'MI_post':>9}  {'sem':>7}  {'z_post':>7}  {'p_post':>7}")

    n_sig_b = n_sig_p = 0
    for c in range(n_channels):
        lbl = str(out['channel_labels'][c])
        zb = z_base_mean[c]
        zp = z_post_mean[c]
        flag_b = '*' if (np.isfinite(pb_med[c]) and pb_med[c] < 0.05) else ' '
        flag_p = '*' if (np.isfinite(pp_med[c]) and pp_med[c] < 0.05) else ' '
        if flag_b == '*':
            n_sig_b += 1
        if flag_p == '*':
            n_sig_p += 1
        print(f"{c:>3}  {lbl:<14} {bip_d[c]:>+7.2f}   "
              f"{mb[c]:>9.4e}  {sb[c]:>7.1e}  {zb:>+7.2f}  "
              f"{pb_med[c]:>7.3f}{flag_b}  "
              f"{mp[c]:>9.4e}  {sp[c]:>7.1e}  {zp:>+7.2f}  "
              f"{pp_med[c]:>7.3f}{flag_p}")
    print("-" * 100)
    print(f"  baseline : mean MI over channels = {np.nanmean(mb):.4e}, "
          f"mean z = {np.nanmean(z_base_mean):+.2f}  "
          f"| {n_sig_b}/{n_channels} channels with median p < 0.05")
    print(f"  post-stim: mean MI over channels = {np.nanmean(mp):.4e}, "
          f"mean z = {np.nanmean(z_post_mean):+.2f}  "
          f"| {n_sig_p}/{n_channels} channels with median p < 0.05")
    print("=" * 100 + "\n")

    plt.show()


if __name__ == "__main__":
    main()
