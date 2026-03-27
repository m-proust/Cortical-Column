"""
L6 Pre-stimulus E & PV Time Course
====================================
Focused script to identify *when* L6 E and PV firing rates diverge
between good (L4 alpha suppressed) and bad trials.

Plots:
  1. Full pre-stimulus time course (−500 ms → 0) for L6 E and PV,
     good vs bad, with SEM shading.
  2. Running divergence index (|mean_good − mean_bad| / pooled SD)
     to pinpoint the onset of separation.
  3. Point-by-point Mann-Whitney U p-value over time (with α = 0.05 line).
  4. Same three plots repeated for L5 (for comparison).
  5. Scatter: mean rate in early (−500→−250 ms) vs late (−250→0 ms)
     pre-window — does the divergence grow or stay constant?
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sosfiltfilt, butter
from scipy.stats import sem, mannwhitneyu
from scipy.signal import detrend
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
    'font.size': 11,
    'axes.titlesize': 13,
    'figure.dpi': 100,
})

# ============================================================
# CONFIG — match your main analysis script
# ============================================================
base_path      = "results/trials_27_03_3"
n_trials       = 38
fs             = 10000

alpha_band     = (7, 14)
pre_window_ms  = 500
post_window_ms = 500
post_start_ms  = 200
alpha_threshold = -10   # % change threshold for good/bad

L6_channels  = [0, 1, 2]
L5_channels  = [3, 4]
L4C_channels = [6]
L4AB_channels = [7, 8]
L4_channels  = [6, 7, 8]

# Time grid for interpolation (pre-stimulus only)
N_POINTS = 300   # resolution of the common time grid
T_COMMON = np.linspace(-pre_window_ms, 0, N_POINTS)

# Smoothing for the time courses (Gaussian-like via low-pass)
SMOOTH_MS  = 20   # ms half-width for smoothing
SMOOTH_HZ  = 1000 / SMOOTH_MS  # convert to Hz cutoff


# ============================================================
# HELPERS
# ============================================================
def band_power(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    filt = sosfiltfilt(sos, x)
    return np.mean(filt ** 2)


def smooth(x, fs, cutoff_hz=50, order=2):
    """Low-pass smooth a rate trace."""
    if len(x) < 10:
        return x
    sos = butter(order, min(cutoff_hz, fs / 2 - 1),
                 btype='low', fs=fs, output='sos')
    return sosfiltfilt(sos, x)


# ============================================================
# DATA LOADING
# ============================================================
def load_all_trials(base_path, n_trials):
    all_trials = []
    for idx in range(n_trials):
        fname = f"{base_path}/trial_{idx:03d}.npz"
        try:
            data = np.load(fname, allow_pickle=True)
        except FileNotFoundError:
            continue
        trial = {
            'trial_idx':      idx,
            'time':           data['time_array_ms'],
            'bipolar_lfp':    data['bipolar_matrix'],
            'rate_data':      data['rate_data'].item() if data['rate_data'].size == 1
                              else data['rate_data'],
            'stim_onset_ms':  float(data['stim_onset_ms']),
        }
        all_trials.append(trial)
    print(f"Loaded {len(all_trials)} trials")
    return all_trials


def classify_trials(all_trials):
    """Classify by L4 combined alpha suppression (same as main script)."""
    alpha_changes = []
    for trial in all_trials:
        bp   = trial['bipolar_lfp']
        t    = trial['time']
        stim = trial['stim_onset_ms']
        n_ch = bp.shape[0]

        pre_mask  = (t >= stim - pre_window_ms) & (t < stim)
        post_mask = (t >= stim + post_start_ms) & \
                    (t < stim + post_start_ms + post_window_ms)

        pre_a, post_a = [], []
        for ch in L4_channels:
            if ch < n_ch:
                pre_a.append(band_power(detrend(bp[ch][pre_mask]),  *alpha_band, fs))
                post_a.append(band_power(detrend(bp[ch][post_mask]), *alpha_band, fs))

        mp  = np.mean(pre_a)  if pre_a  else np.nan
        mpo = np.mean(post_a) if post_a else np.nan
        alpha_changes.append((mpo - mp) / (mp + 1e-30) * 100)

    alpha_changes = np.array(alpha_changes)
    good_mask = alpha_changes < alpha_threshold
    print(f"Good: {good_mask.sum()},  Bad: {(~good_mask).sum()}")
    return alpha_changes, good_mask


# ============================================================
# EXTRACT PRE-STIMULUS RATE TRACES
# ============================================================
def extract_prestim_traces(all_trials, layer, cell_type, smooth_hz=SMOOTH_HZ):
    """
    For each trial, extract the firing rate in the pre-stimulus window
    and interpolate onto T_COMMON (ms relative to stim).

    Returns array of shape (n_trials, N_POINTS). NaN where data missing.
    """
    mon_key = f"{cell_type}_rate"
    traces  = []

    for trial in all_trials:
        rd   = trial['rate_data']
        stim = trial['stim_onset_ms']

        if layer not in rd or mon_key not in rd[layer]:
            traces.append(np.full(N_POINTS, np.nan))
            continue

        t_ms = rd[layer][mon_key]['t_ms']
        r_hz = rd[layer][mon_key]['rate_hz']

        # Restrict to pre-window (with small buffer for smoothing)
        buf  = 50  # ms buffer before window
        mask = (t_ms >= stim - pre_window_ms - buf) & (t_ms <= stim + buf)
        if np.sum(mask) < 5:
            traces.append(np.full(N_POINTS, np.nan))
            continue

        t_rel    = t_ms[mask] - stim        # relative time (ms)
        rate_win = r_hz[mask]

        # Smooth
        rate_smooth = smooth(rate_win, fs=1000 / np.median(np.diff(t_ms)),
                             cutoff_hz=smooth_hz)

        # Interpolate onto common grid
        interp = np.interp(T_COMMON, t_rel, rate_smooth,
                           left=np.nan, right=np.nan)
        traces.append(interp)

    return np.array(traces)   # (n_trials, N_POINTS)


# ============================================================
# DIVERGENCE & STATISTICS OVER TIME
# ============================================================
def compute_divergence(traces, good_mask):
    """
    Point-by-point:
      - Cohen's d  (effect size of good vs bad)
      - Mann-Whitney U p-value
    Returns arrays of shape (N_POINTS,).
    """
    d_arr = np.full(N_POINTS, np.nan)
    p_arr = np.full(N_POINTS, np.nan)

    for t in range(N_POINTS):
        col = traces[:, t]
        g   = col[good_mask  & ~np.isnan(col)]
        b   = col[~good_mask & ~np.isnan(col)]

        if len(g) < 2 or len(b) < 2:
            continue

        pooled_sd = np.sqrt((np.var(g) + np.var(b)) / 2 + 1e-30)
        d_arr[t]  = (np.mean(b) - np.mean(g)) / pooled_sd   # + means bad > good
        _, p_arr[t] = mannwhitneyu(g, b)

    return d_arr, p_arr


# ============================================================
# EARLY vs LATE SCATTER
# ============================================================
def early_late_scatter(traces, good_mask, ax, layer, cell_type):
    """
    Scatter: mean rate in first half (−500→−250) vs second half (−250→0)
    of the pre-window, coloured by good/bad.
    """
    early_mask_t = T_COMMON <= -250
    late_mask_t  = T_COMMON >  -250

    early = np.nanmean(traces[:, early_mask_t], axis=1)
    late  = np.nanmean(traces[:, late_mask_t],  axis=1)

    valid = ~(np.isnan(early) | np.isnan(late))
    colors = ['green' if g else 'red' for g in good_mask[valid]]
    ax.scatter(early[valid], late[valid], c=colors,
               edgecolors='k', s=40, alpha=0.8)

    # Identity line
    lo = min(np.nanmin(early[valid]), np.nanmin(late[valid]))
    hi = max(np.nanmax(early[valid]), np.nanmax(late[valid]))
    ax.plot([lo, hi], [lo, hi], 'gray', ls='--', lw=1, alpha=0.5)

    ax.set_xlabel('Early pre-stim rate −500→−250 ms (Hz)')
    ax.set_ylabel('Late pre-stim rate −250→0 ms (Hz)')
    ax.set_title(f'{layer} {cell_type}: early vs late\n'
                 '(above diagonal = rate rising toward stim)')

    # Annotate means per group
    for mask, color, label in [(good_mask[valid], 'green', 'Good'),
                                (~good_mask[valid], 'red', 'Bad')]:
        if mask.sum() > 0:
            me = np.mean(early[valid][mask])
            ml = np.mean(late[valid][mask])
            ax.scatter([me], [ml], c=color, s=120, marker='D',
                       edgecolors='k', zorder=5)
            ax.annotate(label, (me, ml), textcoords='offset points',
                        xytext=(6, 4), fontsize=9, color=color)


# ============================================================
# MAIN PLOT FUNCTION
# ============================================================
def plot_layer_timecourse(all_trials, good_mask, layer, cell_types=('E', 'PV')):
    """
    For one layer, produce the 3-panel × n_cell_types figure:
      Row 0: mean ± SEM time course
      Row 1: Cohen's d over time
      Row 2: −log10(p) over time  (dashed line at p = 0.05)
    Plus an extra column for early/late scatter per cell type.
    """
    n_ct   = len(cell_types)
    n_cols  = n_ct + 1   # extra col for early/late scatter overlay
    fig, axes = plt.subplots(3, n_cols, figsize=(6 * n_cols, 13))

    for col, ct in enumerate(cell_types):
        traces = extract_prestim_traces(all_trials, layer, ct)
        d_arr, p_arr = compute_divergence(traces, good_mask)

        # ---- Row 0: time course ----
        ax = axes[0, col]
        for mask, color, label in [(good_mask, 'green', 'Good (α↓)'),
                                    (~good_mask, 'red',   'Bad (no α↓)')]:
            sub = traces[mask]
            mn  = np.nanmean(sub, axis=0)
            se  = sem(sub, axis=0, nan_policy='omit')
            ax.plot(T_COMMON, mn, color=color, lw=2, label=label)
            ax.fill_between(T_COMMON, mn - se, mn + se,
                            color=color, alpha=0.18)

        ax.axvline(0, color='black', ls='--', lw=1, label='Stim onset')
        ax.set_xlabel('Time relative to stimulus (ms)')
        ax.set_ylabel('Firing rate (Hz)')
        ax.set_title(f'{layer} {ct} — pre-stim time course')
        ax.legend(fontsize=9)
        ax.set_xlim(-pre_window_ms, 0)

        # ---- Row 1: Cohen's d ----
        ax = axes[1, col]
        ax.plot(T_COMMON, d_arr, color='steelblue', lw=2)
        ax.axhline(0,    color='gray',   ls='--', lw=1, alpha=0.6)
        ax.axhline(0.5,  color='orange', ls=':',  lw=1, label="d = 0.5 (medium)")
        ax.axhline(-0.5, color='orange', ls=':',  lw=1)
        ax.axhline(0.8,  color='red',    ls=':',  lw=1, label="d = 0.8 (large)")
        ax.axhline(-0.8, color='red',    ls=':',  lw=1)
        ax.axvline(0, color='black', ls='--', lw=1)
        ax.set_xlabel('Time relative to stimulus (ms)')
        ax.set_ylabel("Cohen's d  (Bad − Good)")
        ax.set_title(f'{layer} {ct} — effect size over time\n'
                     '(+ = bad trials higher)')
        ax.legend(fontsize=8)
        ax.set_xlim(-pre_window_ms, 0)

        # ---- Row 2: significance ----
        ax = axes[2, col]
        neg_log_p = -np.log10(np.clip(p_arr, 1e-10, 1))
        ax.plot(T_COMMON, neg_log_p, color='purple', lw=2)
        ax.axhline(-np.log10(0.05), color='red', ls='--', lw=1.2,
                   label='p = 0.05')
        ax.axhline(-np.log10(0.01), color='darkred', ls=':', lw=1.2,
                   label='p = 0.01')
        ax.axvline(0, color='black', ls='--', lw=1)
        ax.fill_between(T_COMMON, -np.log10(0.05), neg_log_p,
                        where=neg_log_p > -np.log10(0.05),
                        color='purple', alpha=0.15, label='p < 0.05')
        ax.set_xlabel('Time relative to stimulus (ms)')
        ax.set_ylabel('−log₁₀(p)')
        ax.set_title(f'{layer} {ct} — Mann-Whitney p over time')
        ax.legend(fontsize=8)
        ax.set_xlim(-pre_window_ms, 0)
        ax.set_ylim(bottom=0)

    # ---- Extra column: early/late scatter for both cell types ----
    for row, ct in enumerate(cell_types):
        traces = extract_prestim_traces(all_trials, layer, ct)
        ax = axes[row, n_ct]
        early_late_scatter(traces, good_mask, ax, layer, ct)

    # Hide the third row of the extra column (no natural content)
    axes[2, n_ct].axis('off')
    axes[2, n_ct].text(0.5, 0.5,
                        'Green ● = good trials\nRed ● = bad trials\n'
                        '◆ = group mean',
                        ha='center', va='center', transform=axes[2, n_ct].transAxes,
                        fontsize=11, color='gray')

    plt.suptitle(f'{layer} Pre-stimulus E & PV: When Does Divergence Appear?',
                 fontsize=15, y=1.01)
    plt.tight_layout()
    fname = f'L6_prestim_{layer}_timecourse.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.show()


# ============================================================
# SUMMARY: OVERLAY L5 AND L6 ON ONE FIGURE
# ============================================================
def plot_summary_overlay(all_trials, good_mask):
    """
    Single figure overlaying L5 and L6 E and PV effect sizes,
    so you can compare whether L5 diverges earlier or later than L6.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)

    configs = [
        (0, 0, 'L6', 'E',  '#1a6ea8', '#a8341a'),
        (0, 1, 'L6', 'PV', '#1a6ea8', '#a8341a'),
        (1, 0, 'L5', 'E',  '#2a9d5c', '#9d2a2a'),
        (1, 1, 'L5', 'PV', '#2a9d5c', '#9d2a2a'),
    ]

    for r, c, layer, ct, col_tc, col_d in configs:
        traces = extract_prestim_traces(all_trials, layer, ct)
        d_arr, p_arr = compute_divergence(traces, good_mask)

        ax_top = axes[r, c]

        # Time course (thin lines, both groups)
        for mask, color, label in [(good_mask, 'green', 'Good'),
                                    (~good_mask, 'red', 'Bad')]:
            sub = traces[mask]
            mn  = np.nanmean(sub, axis=0)
            se  = sem(sub, axis=0, nan_policy='omit')
            ax_top.plot(T_COMMON, mn, color=color, lw=2, label=label)
            ax_top.fill_between(T_COMMON, mn - se, mn + se,
                                color=color, alpha=0.15)

        # Shade regions where p < 0.05
        sig = p_arr < 0.05
        for t in range(N_POINTS - 1):
            if sig[t]:
                ax_top.axvspan(T_COMMON[t], T_COMMON[t+1],
                               color='purple', alpha=0.08, lw=0)

        ax_top.axvline(0, color='black', ls='--', lw=1)
        ax_top.set_title(f'{layer} {ct}  —  shaded = p < 0.05 (MW)')
        ax_top.set_ylabel('Rate (Hz)')
        ax_top.set_xlim(-pre_window_ms, 0)
        if r == 0 and c == 0:
            ax_top.legend(fontsize=9)

    for ax in axes[1]:
        ax.set_xlabel('Time relative to stimulus (ms)')

    plt.suptitle('Summary: Pre-stimulus L5 & L6 E and PV divergence\n'
                 '(purple shading = significant good vs bad difference, p < 0.05)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('L6_prestim_summary_overlay.png', dpi=150, bbox_inches='tight')
    print("Saved L6_prestim_summary_overlay.png")
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Loading trials...")
    all_trials = load_all_trials(base_path, n_trials)

    print("\nClassifying trials...")
    alpha_changes, good_mask = classify_trials(all_trials)

    print("\nPlotting L6 E & PV pre-stimulus time courses...")
    plot_layer_timecourse(all_trials, good_mask, layer='L6',
                          cell_types=('E', 'PV'))

    print("\nPlotting L5 E & PV pre-stimulus time courses...")
    plot_layer_timecourse(all_trials, good_mask, layer='L5',
                          cell_types=('E', 'PV'))

    print("\nPlotting summary overlay (L5 & L6, E & PV)...")
    plot_summary_overlay(all_trials, good_mask)

    print("\nDone.")