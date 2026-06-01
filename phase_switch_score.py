"""Quantify the alpha→gamma "switch" per phase per channel.

Two scores, computed from the same pre/post multitaper PSDs as
phase_laminar_power.py:

  1) MIN-SCORE (% units)
       suppression = -%Δ power in (0, 20) Hz   → positive when <20Hz drops
       increase    = +%Δ power in (20, 120) Hz → positive when >20Hz rises
       switch_min  = min(suppression, increase)
     A phase only scores high if BOTH effects happen. Penalises asymmetric
     responses (e.g. big gamma boost with no alpha suppression).

  2) SPECTRAL-TILT CHANGE (dimensionless)
       Fit log10(PSD) = a + b * log10(f) over (2, 120) Hz pre and post.
       tilt_change = b_post - b_pre
     Positive means spectrum got flatter / less low-frequency-dominated.
     Independent of any band edge — sanity check for the min-score.

For each (channel, phase) we report mean ± SEM across trials.
Outputs:
  - phase_switch_score.png   (heatmaps + ranking)
  - phase_switch_score.csv   (full table)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import detrend

from laminar_power_change import multitaper_psd, _time_vector_for_key
from phase_laminar_power import load_phase_trials
from trials_phase import LAYER_Z_CENTER

RESULTS_DIR = 'results/trials_phase_07_05'
PHASES = (0, 45, 90, 135, 180, 225, 270, 315)
LAYERS_TO_PROBE = ('L23', 'L4AB', 'L4C', 'L5', 'L6')

PRE_WINDOW_MS = 500
POST_WINDOW_MS = 500
POST_START_MS = 300

LOW_BAND = (10.0, 16.0)        # narrow alpha/low-beta gap (the actual blue stripe)
HIGH_BAND = (20.0, 120.0)
TILT_FIT_BAND = (2.0, 120.0)
LOW_BAND_WIDE = (0.5, 20.0)    # legacy wide band, reported for comparison


def pick_bipolar_channel(channel_depths, layer_name, n_bipolar):
    depths = np.asarray(channel_depths)
    mid = (depths[:-1] + depths[1:]) / 2
    mid = mid[:n_bipolar]
    target = LAYER_Z_CENTER[layer_name]
    return int(np.argmin(np.abs(mid - target)))


def per_trial_scores(trial, ch_idx):
    lfp = trial['bipolar_lfp'][ch_idx]
    time = _time_vector_for_key(trial, 'bipolar_lfp')
    stim = trial['stim_onset_ms']
    pre_mask = (time >= stim - PRE_WINDOW_MS) & (time < stim)
    post_mask = ((time >= stim + POST_START_MS) &
                 (time < stim + POST_START_MS + POST_WINDOW_MS))
    pre = lfp[pre_mask].copy()
    post = lfp[post_mask].copy()
    if len(pre) == 0 or len(post) == 0:
        return None
    pre = detrend(pre)
    post = detrend(post)
    fs = 1000.0 / float(np.mean(np.diff(time)))
    nfft = 2 ** int(np.ceil(np.log2(min(len(pre), len(post)))))
    f, psd_pre = multitaper_psd(pre, fs=fs, NW=2, nfft=nfft)
    _, psd_post = multitaper_psd(post, fs=fs, NW=2, nfft=nfft)

    # Band-integrated power
    def band_power(psd, lo, hi):
        m = (f >= lo) & (f <= hi)
        return float(np.trapezoid(psd[m], f[m])) if np.any(m) else np.nan

    p_lo_pre = band_power(psd_pre, *LOW_BAND)
    p_lo_post = band_power(psd_post, *LOW_BAND)
    p_hi_pre = band_power(psd_pre, *HIGH_BAND)
    p_hi_post = band_power(psd_post, *HIGH_BAND)
    p_lo_wide_pre = band_power(psd_pre, *LOW_BAND_WIDE)
    p_lo_wide_post = band_power(psd_post, *LOW_BAND_WIDE)

    if p_lo_pre <= 0 or p_hi_pre <= 0 or p_lo_wide_pre <= 0:
        return None
    pct_lo = (p_lo_post - p_lo_pre) / p_lo_pre * 100
    pct_hi = (p_hi_post - p_hi_pre) / p_hi_pre * 100
    pct_lo_wide = (p_lo_wide_post - p_lo_wide_pre) / p_lo_wide_pre * 100
    suppression = -pct_lo
    increase = pct_hi
    switch_min = min(suppression, increase)

    # Spectral tilt: slope of log10(PSD) vs log10(f)
    tilt_mask = (f >= TILT_FIT_BAND[0]) & (f <= TILT_FIT_BAND[1]) & (f > 0)
    log_f = np.log10(f[tilt_mask])
    pre_safe = psd_pre[tilt_mask].copy()
    post_safe = psd_post[tilt_mask].copy()
    pre_safe[pre_safe <= 0] = np.nan
    post_safe[post_safe <= 0] = np.nan
    good = np.isfinite(pre_safe) & np.isfinite(post_safe)
    if good.sum() < 5:
        slope_pre = slope_post = np.nan
    else:
        slope_pre = float(np.polyfit(log_f[good],
                                     np.log10(pre_safe[good]), 1)[0])
        slope_post = float(np.polyfit(log_f[good],
                                      np.log10(post_safe[good]), 1)[0])
    tilt_change = slope_post - slope_pre

    return {
        'pct_lo': pct_lo,
        'pct_hi': pct_hi,
        'pct_lo_wide': pct_lo_wide,
        'suppression': suppression,
        'suppression_wide': -pct_lo_wide,
        'increase': increase,
        'switch_min': switch_min,
        'tilt_change': tilt_change,
    }


def collect():
    """Returns dict[layer][phase] = list of per-trial score dicts."""
    res = {ln: {ph: [] for ph in PHASES} for ln in LAYERS_TO_PROBE}
    for ph in PHASES:
        trials = load_phase_trials(RESULTS_DIR, ph)
        if not trials:
            continue
        for tr in trials:
            for ln in LAYERS_TO_PROBE:
                ch = pick_bipolar_channel(tr['channel_depths'], ln,
                                          tr['bipolar_lfp'].shape[0])
                s = per_trial_scores(tr, ch)
                if s is not None:
                    res[ln][ph].append(s)
    return res


def summarize_metric(per_phase, key):
    phases = np.array(sorted(per_phase.keys()))
    means, sems, n = [], [], []
    for ph in phases:
        vals = np.array([s[key] for s in per_phase[ph]
                         if np.isfinite(s[key])])
        if len(vals) == 0:
            means.append(np.nan)
            sems.append(np.nan)
            n.append(0)
        else:
            means.append(float(np.mean(vals)))
            sems.append(float(np.std(vals) / np.sqrt(len(vals))))
            n.append(len(vals))
    return phases, np.array(means), np.array(sems), np.array(n)


def main():
    print("Loading trials & computing per-trial scores...")
    res = collect()

    # ---- Build (layer × phase) matrices for the two scores ----
    sw_mat = np.full((len(LAYERS_TO_PROBE), len(PHASES)), np.nan)
    sw_sem = np.full_like(sw_mat, np.nan)
    tilt_mat = np.full_like(sw_mat, np.nan)
    tilt_sem = np.full_like(sw_mat, np.nan)
    supp_mat = np.full_like(sw_mat, np.nan)
    supp_sem = np.full_like(sw_mat, np.nan)
    supp_wide_mat = np.full_like(sw_mat, np.nan)
    inc_mat = np.full_like(sw_mat, np.nan)
    inc_sem = np.full_like(sw_mat, np.nan)
    n_mat = np.zeros_like(sw_mat, dtype=int)

    for i, ln in enumerate(LAYERS_TO_PROBE):
        _, m_sw, s_sw, n = summarize_metric(res[ln], 'switch_min')
        sw_mat[i] = m_sw
        sw_sem[i] = s_sw
        n_mat[i] = n
        _, m_t, s_t, _ = summarize_metric(res[ln], 'tilt_change')
        tilt_mat[i] = m_t
        tilt_sem[i] = s_t
        _, m_sp, s_sp, _ = summarize_metric(res[ln], 'suppression')
        supp_mat[i] = m_sp
        supp_sem[i] = s_sp
        _, m_sw_wide, _, _ = summarize_metric(res[ln], 'suppression_wide')
        supp_wide_mat[i] = m_sw_wide
        _, m_in, s_in, _ = summarize_metric(res[ln], 'increase')
        inc_mat[i] = m_in
        inc_sem[i] = s_in

    # ---- Figure: 4 heatmaps (switch_min, tilt, suppression, increase)
    # + 2 ranked line plots ----
    fig = plt.figure(figsize=(15, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.25)

    # (a) switch_min heatmap (layer × phase)
    ax = fig.add_subplot(gs[0, 0])
    vmax = float(np.nanmax(np.abs(sw_mat)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(sw_mat, aspect='auto', cmap='RdBu_r', norm=norm)
    ax.set_xticks(range(len(PHASES)))
    ax.set_xticklabels([f'{p}°' for p in PHASES])
    ax.set_yticks(range(len(LAYERS_TO_PROBE)))
    ax.set_yticklabels(LAYERS_TO_PROBE)
    ax.set_xlabel('Phase at stim onset')
    ax.set_title(f'Switch score = min(suppression {int(LOW_BAND[0])}–'
                 f'{int(LOW_BAND[1])} Hz, '
                 f'increase {int(HIGH_BAND[0])}–{int(HIGH_BAND[1])} Hz)  [%]')
    for i in range(sw_mat.shape[0]):
        for j in range(sw_mat.shape[1]):
            if np.isfinite(sw_mat[i, j]):
                ax.text(j, i, f'{sw_mat[i, j]:.0f}',
                        ha='center', va='center', fontsize=8,
                        color='k' if abs(sw_mat[i, j]) < 0.5 * vmax
                        else 'white')
    fig.colorbar(im, ax=ax, shrink=0.85, label='%')

    # (b) tilt_change heatmap
    ax = fig.add_subplot(gs[0, 1])
    vmax_t = float(np.nanmax(np.abs(tilt_mat)))
    norm = TwoSlopeNorm(vmin=-vmax_t, vcenter=0, vmax=vmax_t)
    im = ax.imshow(tilt_mat, aspect='auto', cmap='RdBu_r', norm=norm)
    ax.set_xticks(range(len(PHASES)))
    ax.set_xticklabels([f'{p}°' for p in PHASES])
    ax.set_yticks(range(len(LAYERS_TO_PROBE)))
    ax.set_yticklabels(LAYERS_TO_PROBE)
    ax.set_xlabel('Phase at stim onset')
    ax.set_title(f'Spectral tilt change Δslope of log-PSD '
                 f'({TILT_FIT_BAND[0]:.0f}–{TILT_FIT_BAND[1]:.0f} Hz)')
    for i in range(tilt_mat.shape[0]):
        for j in range(tilt_mat.shape[1]):
            if np.isfinite(tilt_mat[i, j]):
                ax.text(j, i, f'{tilt_mat[i, j]:+.2f}',
                        ha='center', va='center', fontsize=8,
                        color='k' if abs(tilt_mat[i, j]) < 0.5 * vmax_t
                        else 'white')
    fig.colorbar(im, ax=ax, shrink=0.85, label='Δslope')

    # (c) suppression alone vs phase
    ax = fig.add_subplot(gs[1, 0])
    layer_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(LAYERS_TO_PROBE)))
    for i, ln in enumerate(LAYERS_TO_PROBE):
        ax.errorbar(PHASES, supp_mat[i], yerr=supp_sem[i], fmt='o-',
                    color=layer_colors[i], lw=1.5, capsize=3, label=ln)
    ax.axhline(0, color='k', ls=':', lw=0.7)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel('Phase at stim onset (°)')
    ax.set_ylabel(f'Suppression {int(LOW_BAND[0])}–{int(LOW_BAND[1])} Hz (%)\n'
                  '(positive = power drops)')
    ax.set_title(f'Low-band suppression ({int(LOW_BAND[0])}–'
                 f'{int(LOW_BAND[1])} Hz) vs. phase')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=5, fontsize=8, loc='best')

    # (d) increase alone vs phase
    ax = fig.add_subplot(gs[1, 1])
    for i, ln in enumerate(LAYERS_TO_PROBE):
        ax.errorbar(PHASES, inc_mat[i], yerr=inc_sem[i], fmt='o-',
                    color=layer_colors[i], lw=1.5, capsize=3, label=ln)
    ax.axhline(0, color='k', ls=':', lw=0.7)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel('Phase at stim onset (°)')
    ax.set_ylabel(f'Increase >{int(HIGH_BAND[0])} Hz (%)')
    ax.set_title(f'High-band increase ({int(HIGH_BAND[0])}–'
                 f'{int(HIGH_BAND[1])} Hz) vs. phase')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=5, fontsize=8, loc='best')

    # (e) switch_min vs phase, lines per layer
    ax = fig.add_subplot(gs[2, 0])
    for i, ln in enumerate(LAYERS_TO_PROBE):
        ax.errorbar(PHASES, sw_mat[i], yerr=sw_sem[i], fmt='o-',
                    color=layer_colors[i], lw=1.5, capsize=3, label=ln)
    ax.axhline(0, color='k', ls=':', lw=0.7)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel('Phase at stim onset (°)')
    ax.set_ylabel('Switch score (%)')
    ax.set_title('Switch score vs. phase (mean ± SEM across trials)')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=5, fontsize=8, loc='best')

    # (f) tilt_change vs phase, lines per layer
    ax = fig.add_subplot(gs[2, 1])
    for i, ln in enumerate(LAYERS_TO_PROBE):
        ax.errorbar(PHASES, tilt_mat[i], yerr=tilt_sem[i], fmt='o-',
                    color=layer_colors[i], lw=1.5, capsize=3, label=ln)
    ax.axhline(0, color='k', ls=':', lw=0.7)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel('Phase at stim onset (°)')
    ax.set_ylabel('Δslope (post − pre)')
    ax.set_title('Spectral tilt change vs. phase (mean ± SEM)')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=5, fontsize=8, loc='best')

    fig.suptitle('Alpha→gamma switch strength per phase '
                 f'(pre 500 ms vs. post {POST_START_MS}–'
                 f'{POST_START_MS + POST_WINDOW_MS} ms)', y=0.995)
    out_path = os.path.join(RESULTS_DIR, 'phase_switch_score.png')
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"saved {out_path}")

    # ---- CSV ----
    out_csv = os.path.join(RESULTS_DIR, 'phase_switch_score.csv')
    with open(out_csv, 'w') as fh:
        fh.write(f"layer,phase_deg,n_trials,"
                 f"suppression_{int(LOW_BAND[0])}_{int(LOW_BAND[1])}Hz_pct,"
                 f"suppression_wide_below20Hz_pct,"
                 f"increase_above{int(HIGH_BAND[0])}Hz_pct,"
                 f"switch_min_pct,switch_min_sem,"
                 f"tilt_change,tilt_change_sem\n")
        for i, ln in enumerate(LAYERS_TO_PROBE):
            for j, ph in enumerate(PHASES):
                fh.write(f"{ln},{ph},{n_mat[i, j]},"
                         f"{supp_mat[i, j]:.3f},{supp_wide_mat[i, j]:.3f},"
                         f"{inc_mat[i, j]:.3f},"
                         f"{sw_mat[i, j]:.3f},{sw_sem[i, j]:.3f},"
                         f"{tilt_mat[i, j]:.4f},{tilt_sem[i, j]:.4f}\n")
    print(f"saved {out_csv}")

    # ---- Console: best/worst phases per layer ----
    print("\n=== Best phase per layer (highest switch_min) ===")
    print(f"{'layer':>5} | {'best φ':>6} | {'score':>8} | "
          f"{'worst φ':>7} | {'score':>8} | {'range':>8}")
    for i, ln in enumerate(LAYERS_TO_PROBE):
        row = sw_mat[i]
        if np.all(np.isnan(row)):
            continue
        j_best = int(np.nanargmax(row))
        j_worst = int(np.nanargmin(row))
        print(f"{ln:>5} | {PHASES[j_best]:>6}° | {row[j_best]:>+8.2f} | "
              f"{PHASES[j_worst]:>7}° | {row[j_worst]:>+8.2f} | "
              f"{row[j_best] - row[j_worst]:>+8.2f}")

    print("\n=== Best phase per layer (highest Δtilt) ===")
    print(f"{'layer':>5} | {'best φ':>6} | {'Δslope':>8} | "
          f"{'worst φ':>7} | {'Δslope':>8} | {'range':>8}")
    for i, ln in enumerate(LAYERS_TO_PROBE):
        row = tilt_mat[i]
        if np.all(np.isnan(row)):
            continue
        j_best = int(np.nanargmax(row))
        j_worst = int(np.nanargmin(row))
        print(f"{ln:>5} | {PHASES[j_best]:>6}° | {row[j_best]:>+8.3f} | "
              f"{PHASES[j_worst]:>7}° | {row[j_worst]:>+8.3f} | "
              f"{row[j_best] - row[j_worst]:>+8.3f}")

    print("\n=== Suppression vs increase decomposition per layer (mean across "
          "phases) ===")
    print(f"{'layer':>5} | {'-Δ narrow':>10} | {'-Δ wide<20':>11} | "
          f"{'+Δ>20':>8} | switch dominated by?")
    for i, ln in enumerate(LAYERS_TO_PROBE):
        sp = np.nanmean(supp_mat[i])
        sp_w = np.nanmean(supp_wide_mat[i])
        inc = np.nanmean(inc_mat[i])
        which = 'increase' if inc > sp else 'suppression'
        if abs(inc - sp) < 5:
            which = 'balanced'
        print(f"{ln:>5} | {sp:>+10.2f} | {sp_w:>+11.2f} | "
              f"{inc:>+8.2f} | {which}")

    print(f"\n=== Phases where {int(LOW_BAND[0])}–{int(LOW_BAND[1])} Hz "
          "suppression > 0 (real suppression) ===")
    for i, ln in enumerate(LAYERS_TO_PROBE):
        good_phases = [(PHASES[j], supp_mat[i, j])
                       for j in range(len(PHASES))
                       if np.isfinite(supp_mat[i, j]) and supp_mat[i, j] > 0]
        if good_phases:
            ss = ', '.join(f'{p}°({v:+.1f}%)' for p, v in good_phases)
            print(f"  {ln}: {ss}")
        else:
            print(f"  {ln}: (none)")


if __name__ == '__main__':
    main()
