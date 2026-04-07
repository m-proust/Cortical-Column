"""
analyze_trial_variability.py — What drives trial-to-trial variability?

Good trials are manually selected by visual inspection of laminar power
change plots. Bad trials = all the rest.

For each trial, extracts:
  1. Stim rates (jittered Poisson rates) if saved
  2. Alpha phase at stimulus onset (from bipolar LFP, bandpass 7-14 Hz)
  3. Per-layer alpha change (%)
  4. Mean alpha change across all layers

Then tests whether variability is driven by rate jitter or alpha phase.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfiltfilt, detrend
from scipy.stats import pearsonr, mannwhitneyu, circmean
import os

plt.rcParams.update({
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
    'font.size': 11,
    'axes.titlesize': 13,
})
plt.style.use('seaborn-v0_8-darkgrid')

# ── Config ──
base_path = "results/trials3_06_04_3"
output_dir = os.path.join(base_path, "variability_analysis")
os.makedirs(output_dir, exist_ok=True)

fs = 10000
alpha_band = (7, 14)
gamma_band = (30, 80)

pre_window_ms = 300
post_window_ms = 300
post_start_ms = 300

# ── Manually selected good trials (by visual inspection) ──
# Edit this list with the trial numbers that look good
GOOD_TRIALS = [0,3,4,5,8,10,12,14,15,16,18,20,23,27,29,31,32,35,36,38,39,40]   # e.g. [4, 7, 12, 23]

# ── Channel assignments (bipolar, 0-indexed) — same as alpha_analysis_4.py ──
LAYER_DEFS = {
    'L6':   [2, 3],
    'L5':   [4],
    'L4C':  [5, 6],
    'L4AB': [7, 8],
    'L23':  [9, 10, 11, 12, 13],
}

# Channels for phase extraction (L4/L5 depth where alpha is strongest)
phase_channels = [4, 5, 6]


# ── Helpers ──
def bandpass(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)


def band_power(x, lo, hi, fs):
    filt = bandpass(x, lo, hi, fs)
    return np.mean(filt ** 2)


def extract_alpha_phase_at_onset(bipolar, time_ms, stim_onset_ms, channels, fs):
    """Get instantaneous alpha phase at stimulus onset, averaged over channels."""
    n_ch = bipolar.shape[0]
    valid_chs = [ch for ch in channels if ch < n_ch]
    if not valid_chs:
        return np.nan
    lfp = np.mean([bipolar[ch] for ch in valid_chs], axis=0)

    # Use 500 ms window ending at onset for filtering
    pre_start = stim_onset_ms - 500
    mask = (time_ms >= pre_start) & (time_ms <= stim_onset_ms + 50)
    seg = lfp[mask]
    if len(seg) < fs * 0.1:
        return np.nan
    seg = detrend(seg)
    filtered = bandpass(seg, alpha_band[0], alpha_band[1], fs)
    analytic = hilbert(filtered)
    phase = np.angle(analytic)

    t_seg = time_ms[mask]
    onset_idx = np.searchsorted(t_seg, stim_onset_ms) - 1
    onset_idx = np.clip(onset_idx, 0, len(phase) - 1)
    return phase[onset_idx]


def compute_layer_changes(bipolar, time_ms, stim_onset_ms):
    """Per-layer alpha power change (%). Returns layer_changes dict and mean."""
    n_ch = bipolar.shape[0]
    pre_mask = (time_ms >= stim_onset_ms - pre_window_ms) & (time_ms < stim_onset_ms)
    post_mask = (time_ms >= stim_onset_ms + post_start_ms) & \
                (time_ms < stim_onset_ms + post_start_ms + post_window_ms)

    layer_changes = {}
    all_changes = []

    for layer_name, chs in LAYER_DEFS.items():
        pre_a, post_a = [], []
        for ch in chs:
            if ch < n_ch:
                pre_a.append(band_power(detrend(bipolar[ch][pre_mask]), *alpha_band, fs))
                post_a.append(band_power(detrend(bipolar[ch][post_mask]), *alpha_band, fs))

        if pre_a and post_a:
            mp = np.mean(pre_a)
            mpo = np.mean(post_a)
            chg = (mpo - mp) / (mp + 1e-30) * 100
        else:
            chg = np.nan

        layer_changes[layer_name] = chg
        if not np.isnan(chg):
            all_changes.append(chg)

    mean_change = np.mean(all_changes) if all_changes else np.nan
    return layer_changes, mean_change


# ── Load and process all trials ──
trial_files = sorted([f for f in os.listdir(base_path)
                       if f.startswith("trial_") and f.endswith(".npz")])
n_trials = len(trial_files)
print(f"Found {n_trials} trials in {base_path}")

# Storage
stim_rates_L4C_E = []
stim_rates_L4C_PV = []
stim_rates_L6_E = []
stim_rates_L6_PV = []
stim_rates_total = []
has_rates = False

alpha_phases = []
mean_alpha_changes = []
trial_indices = []
per_layer_changes = {name: [] for name in LAYER_DEFS}

for i, fname in enumerate(trial_files):
    data = np.load(os.path.join(base_path, fname), allow_pickle=True)

    time_ms = data["time_array_ms"]
    bipolar = data["bipolar_matrix"]
    stim_onset = float(data["stim_onset_ms"])

    trial_indices.append(i)

    # Stim rates
    if "stim_rates" in data:
        has_rates = True
        sr = data["stim_rates"].item()
        stim_rates_L4C_E.append(sr['L4C_E'])
        stim_rates_L4C_PV.append(sr['L4C_PV'])
        stim_rates_L6_E.append(sr['L6_E'])
        stim_rates_L6_PV.append(sr['L6_PV'])
        stim_rates_total.append(sr['L4C_E'] + sr['L4C_PV'] + sr['L6_E'] + sr['L6_PV'])

    # Alpha phase at onset
    phase = extract_alpha_phase_at_onset(bipolar, time_ms, stim_onset, phase_channels, fs)
    alpha_phases.append(phase)

    # Per-layer alpha changes
    layer_chg, mean_chg = compute_layer_changes(bipolar, time_ms, stim_onset)
    mean_alpha_changes.append(mean_chg)
    for name in LAYER_DEFS:
        per_layer_changes[name].append(layer_chg[name])

    if (i + 1) % 20 == 0:
        print(f"  Processed {i+1}/{n_trials}")

alpha_phases = np.array(alpha_phases)
mean_alpha_changes = np.array(mean_alpha_changes)
trial_indices = np.array(trial_indices)
for name in LAYER_DEFS:
    per_layer_changes[name] = np.array(per_layer_changes[name])

# ── Good/bad from manual list ──
good_mask = np.array([i in GOOD_TRIALS for i in trial_indices])
bad_mask = ~good_mask
n_good = np.sum(good_mask)
n_bad = np.sum(bad_mask)

valid = np.isfinite(alpha_phases) & np.isfinite(mean_alpha_changes)

print(f"\nGood trials (manually selected): {n_good}  {sorted(GOOD_TRIALS)}")
print(f"Bad trials: {n_bad}")

# ── Correlation analysis ──
print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

phases_valid = alpha_phases[valid]
alpha_valid = mean_alpha_changes[valid]
good_valid = good_mask[valid]

# Circular-linear correlation: phase vs mean alpha change
cos_phase = np.cos(phases_valid)
sin_phase = np.sin(phases_valid)

r_cos, p_cos = pearsonr(cos_phase, alpha_valid)
r_sin, p_sin = pearsonr(sin_phase, alpha_valid)
R2_phase = r_cos**2 + r_sin**2
print(f"\nAlpha phase at onset vs mean alpha change (%):")
print(f"  cos(phase): r={r_cos:.3f}, p={p_cos:.4f}")
print(f"  sin(phase): r={r_sin:.3f}, p={p_sin:.4f}")
print(f"  Circular-linear R² = {R2_phase:.3f}")

# Phase distribution in good vs bad trials
good_phases = alpha_phases[good_mask & valid]
bad_phases = alpha_phases[bad_mask & valid]
if len(good_phases) > 0 and len(bad_phases) > 0:
    print(f"\nPhase in good trials: circular mean = {circmean(good_phases, high=np.pi, low=-np.pi):.2f} rad")
    print(f"Phase in bad trials:  circular mean = {circmean(bad_phases, high=np.pi, low=-np.pi):.2f} rad")

rate_labels = ['L4C_E', 'L4C_PV', 'L6_E', 'L6_PV', 'Total']
rate_corrs = {}  # {label: (r, p, U_mw, p_mw)}

if has_rates:
    all_rate_arrays = {
        'L4C_E':  np.array(stim_rates_L4C_E),
        'L4C_PV': np.array(stim_rates_L4C_PV),
        'L6_E':   np.array(stim_rates_L6_E),
        'L6_PV':  np.array(stim_rates_L6_PV),
        'Total':  np.array(stim_rates_total),
    }

    print(f"\nPer-rate correlations with mean alpha change:")
    print(f"  {'Rate':<10s} {'r':>7s} {'p':>8s} {'R²':>7s}   "
          f"{'Good mean':>10s} {'Bad mean':>10s} {'MW p':>8s}")
    print(f"  {'-'*65}")

    for label in rate_labels:
        arr = all_rate_arrays[label]
        arr_valid = arr[valid]

        r, p = pearsonr(arr_valid, alpha_valid)

        good_r = arr[good_mask]
        bad_r = arr[bad_mask]
        if len(good_r) > 1 and len(bad_r) > 1:
            U, p_mw = mannwhitneyu(good_r, bad_r, alternative='two-sided')
        else:
            U, p_mw = np.nan, np.nan

        rate_corrs[label] = (r, p, U, p_mw)

        print(f"  {label:<10s} {r:>7.3f} {p:>8.4f} {r**2:>7.3f}   "
              f"{np.mean(good_r):>10.2f} {np.mean(bad_r):>10.2f} {p_mw:>8.4f}")

# ── Summary ──
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Alpha phase circular-linear R² on mean alpha change: {R2_phase:.3f}")
if has_rates:
    best_rate_label = max(rate_corrs, key=lambda k: rate_corrs[k][0]**2)
    best_r2 = rate_corrs[best_rate_label][0]**2
    total_r2 = rate_corrs['Total'][0]**2
    print(f"Best single rate R² ({best_rate_label}):                    {best_r2:.3f}")
    print(f"Total rate R²:                                       {total_r2:.3f}")
    if R2_phase > best_r2 * 1.5:
        print("  --> Alpha phase explains more variance => phase-dependent")
    elif best_r2 > R2_phase * 1.5:
        print(f"  --> Rate jitter ({best_rate_label}) explains more variance => reduce jitter or fix rates")
    else:
        print("  --> Both contribute similarly")
else:
    print("(No stim_rates saved — can only assess phase effect)")

print(f"\nMean alpha change: good={np.mean(mean_alpha_changes[good_mask]):.1f}%, "
      f"bad={np.mean(mean_alpha_changes[bad_mask]):.1f}%")
if has_rates:
    for label in rate_labels:
        arr = all_rate_arrays[label]
        print(f"  {label} range: {arr.min():.2f} - {arr.max():.2f} Hz")

# ── Interaction analysis: rate × phase ──
print("\n" + "=" * 60)
print("INTERACTION ANALYSIS (rate × phase)")
print("=" * 60)

def ols_r2(X, y):
    """OLS R² using numpy least-squares (no sklearn needed)."""
    X_b = np.column_stack([np.ones(len(y)), X])  # add intercept
    beta, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
    y_pred = X_b @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

# Build feature matrix: cos(phase), sin(phase), each rate, and rate×cos, rate×sin interactions
if has_rates:
    # Phase-only model
    X_phase = np.column_stack([cos_phase, sin_phase])
    R2_phase_only = ols_r2(X_phase, alpha_valid)

    print(f"\nModel comparisons (R² on mean alpha change):")
    print(f"  Phase only (cos + sin):                    R² = {R2_phase_only:.3f}")

    # Rate-only model (all 4 rates)
    ind_labels = ['L4C_E', 'L4C_PV', 'L6_E', 'L6_PV']
    X_rates = np.column_stack([all_rate_arrays[l][valid] for l in ind_labels])
    R2_rates_only = ols_r2(X_rates, alpha_valid)
    print(f"  Rates only (4 rates):                      R² = {R2_rates_only:.3f}")

    # Additive model: phase + rates
    X_add = np.column_stack([cos_phase, sin_phase, X_rates])
    R2_additive = ols_r2(X_add, alpha_valid)
    print(f"  Additive (phase + rates):                  R² = {R2_additive:.3f}")

    # Full interaction model: phase + rates + rate×cos + rate×sin
    interactions = []
    interaction_names = []
    for j, l in enumerate(ind_labels):
        interactions.append(X_rates[:, j] * cos_phase)
        interactions.append(X_rates[:, j] * sin_phase)
        interaction_names.extend([f'{l}×cos', f'{l}×sin'])
    X_full = np.column_stack([X_add] + interactions)
    R2_full = ols_r2(X_full, alpha_valid)
    print(f"  Full interaction (phase + rates + rate×phase): R² = {R2_full:.3f}")

    # Interaction contribution
    R2_interaction_gain = R2_full - R2_additive
    print(f"\n  Interaction gain (full - additive):         ΔR² = {R2_interaction_gain:.3f}")
    print(f"  Additive gain over phase alone:             ΔR² = {R2_additive - R2_phase_only:.3f}")
    print(f"  Additive gain over rates alone:             ΔR² = {R2_additive - R2_rates_only:.3f}")

    # Per-rate interaction: which rate×phase interaction matters most?
    print(f"\n  Per-rate interaction contributions:")
    for j, l in enumerate(ind_labels):
        X_single_int = np.column_stack([X_add,
                                         X_rates[:, j] * cos_phase,
                                         X_rates[:, j] * sin_phase])
        R2_si = ols_r2(X_single_int, alpha_valid)
        print(f"    {l}×phase: R² = {R2_si:.3f}  (ΔR² = {R2_si - R2_additive:.3f})")

    # Adjusted R² (penalize for number of predictors)
    n_obs = len(alpha_valid)
    def adj_r2(r2, n, p):
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print(f"\n  Adjusted R² (n={n_obs}):")
    print(f"    Phase only:      adj R² = {adj_r2(R2_phase_only, n_obs, 2):.3f}")
    print(f"    Rates only:      adj R² = {adj_r2(R2_rates_only, n_obs, 4):.3f}")
    print(f"    Additive:        adj R² = {adj_r2(R2_additive, n_obs, 6):.3f}")
    print(f"    Full interaction: adj R² = {adj_r2(R2_full, n_obs, 14):.3f}")

    # ── Pairwise rate interactions ──
    print("\n" + "=" * 60)
    print("RATE × RATE INTERACTIONS")
    print("=" * 60)

    from itertools import combinations
    pair_labels = list(combinations(ind_labels, 2))

    # Rate ratios and products
    print(f"\n  Pairwise rate ratios (r with mean alpha change):")
    print(f"  {'Pair':<20s} {'r(ratio)':>8s} {'p':>8s} {'r(product)':>10s} {'p':>8s}")
    print(f"  {'-'*58}")

    best_pair_r2 = 0
    best_pair_label = ""
    for l1, l2 in pair_labels:
        r1 = all_rate_arrays[l1][valid]
        r2 = all_rate_arrays[l2][valid]
        ratio = r1 / (r2 + 1e-10)
        product = r1 * r2

        r_ratio, p_ratio = pearsonr(ratio, alpha_valid)
        r_prod, p_prod = pearsonr(product, alpha_valid)

        best_r = max(abs(r_ratio), abs(r_prod))
        if best_r**2 > best_pair_r2:
            best_pair_r2 = best_r**2
            best_pair_label = f"{l1}/{l2}" if abs(r_ratio) > abs(r_prod) else f"{l1}×{l2}"

        print(f"  {l1+'/'+l2:<20s} {r_ratio:>8.3f} {p_ratio:>8.4f} {r_prod:>10.3f} {p_prod:>8.4f}")

    # Rates + all pairwise interaction products
    pair_features = []
    pair_names = []
    for l1, l2 in pair_labels:
        pair_features.append(all_rate_arrays[l1][valid] * all_rate_arrays[l2][valid])
        pair_names.append(f"{l1}×{l2}")
    X_rates_inter = np.column_stack([X_rates] + pair_features)
    R2_rates_inter = ols_r2(X_rates_inter, alpha_valid)
    n_p_ri = 4 + len(pair_labels)
    print(f"\n  Rates + pairwise products: R² = {R2_rates_inter:.3f}  "
          f"(adj R² = {adj_r2(R2_rates_inter, n_obs, n_p_ri):.3f})")
    print(f"  Gain over rates alone:     ΔR² = {R2_rates_inter - R2_rates_only:.3f}")
    print(f"  Best single pair:          {best_pair_label}  R² = {best_pair_r2:.3f}")

    # E/PV ratio per layer
    print(f"\n  E/PV ratios per layer:")
    for layer in ['L4C', 'L6']:
        e_label, pv_label = f'{layer}_E', f'{layer}_PV'
        ratio = all_rate_arrays[e_label][valid] / (all_rate_arrays[pv_label][valid] + 1e-10)
        r_ep, p_ep = pearsonr(ratio, alpha_valid)
        # Good vs bad
        ratio_all = all_rate_arrays[e_label] / (all_rate_arrays[pv_label] + 1e-10)
        good_ratio = ratio_all[good_mask]
        bad_ratio = ratio_all[bad_mask]
        if len(good_ratio) > 1 and len(bad_ratio) > 1:
            _, p_mw_ep = mannwhitneyu(good_ratio, bad_ratio, alternative='two-sided')
        else:
            p_mw_ep = np.nan
        print(f"    {layer} E/PV: r={r_ep:.3f}, p={p_ep:.4f}, "
              f"good mean={np.mean(good_ratio):.3f}, bad mean={np.mean(bad_ratio):.3f}, MW p={p_mw_ep:.4f}")

else:
    print("(No stim_rates saved — skipping interaction analysis)")

# ── Plots ──
fig = plt.figure(figsize=(20, 20))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

# --- Row 1: Phase analysis ---

# 1a. Polar plot: phase vs mean alpha change
ax1 = fig.add_subplot(gs[0, 0], projection='polar')
sc = ax1.scatter(phases_valid, np.abs(alpha_valid), c=alpha_valid, cmap='RdBu_r',
                 s=30, alpha=0.7, edgecolors='k', linewidths=0.3,
                 vmin=-80, vmax=80)
ax1.set_title(f'Phase vs alpha change\n(circ-lin R²={R2_phase:.3f})', pad=15)
plt.colorbar(sc, ax=ax1, label='Mean alpha change (%)', shrink=0.8)

# 1b. Phase histograms: good vs bad
ax2 = fig.add_subplot(gs[0, 1])
bins = np.linspace(-np.pi, np.pi, 25)
if len(good_phases) > 0:
    ax2.hist(good_phases, bins=bins, alpha=0.6, color='green', label=f'Good (n={len(good_phases)})',
             edgecolor='white', density=True)
if len(bad_phases) > 0:
    ax2.hist(bad_phases, bins=bins, alpha=0.6, color='red', label=f'Bad (n={len(bad_phases)})',
             edgecolor='white', density=True)
ax2.set_xlabel('Alpha phase at stimulus onset (rad)')
ax2.set_ylabel('Density')
ax2.set_title('Phase distribution: good vs bad trials')
ax2.legend()

# 1c. Phase distribution of ALL trials (polar histogram)
ax_polar = fig.add_subplot(gs[0, 2], projection='polar')
n_bins_polar = 24
theta_bins = np.linspace(-np.pi, np.pi, n_bins_polar + 1)
theta_centers = (theta_bins[:-1] + theta_bins[1:]) / 2
bar_width = 2 * np.pi / n_bins_polar
if len(bad_phases) > 0:
    counts_bad, _ = np.histogram(bad_phases, bins=theta_bins)
    ax_polar.bar(theta_centers, counts_bad, width=bar_width, alpha=0.5,
                 color='red', edgecolor='white', linewidth=0.5)
if len(good_phases) > 0:
    counts_good, _ = np.histogram(good_phases, bins=theta_bins)
    ax_polar.bar(theta_centers, counts_good, width=bar_width, alpha=0.7,
                 color='green', edgecolor='white', linewidth=0.5)
# Mark circular means
if len(good_phases) > 0:
    cm_good = circmean(good_phases, high=np.pi, low=-np.pi)
    ax_polar.axvline(cm_good, color='green', linestyle='--', linewidth=2, label=f'Good: {cm_good:.2f} rad')
if len(bad_phases) > 0:
    cm_bad = circmean(bad_phases, high=np.pi, low=-np.pi)
    ax_polar.axvline(cm_bad, color='red', linestyle='--', linewidth=2, label=f'Bad: {cm_bad:.2f} rad')
ax_polar.set_title('Alpha phase at stimulus onset\n(polar distribution)', pad=15)
ax_polar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)

# --- Row 2: Per-layer alpha change ---

# 2a. Per-layer alpha change: good vs bad
ax3 = fig.add_subplot(gs[1, 0])
layer_names = list(LAYER_DEFS.keys())
x_pos = np.arange(len(layer_names))
width = 0.35
good_means = [np.nanmean(per_layer_changes[name][good_mask]) for name in layer_names]
bad_means = [np.nanmean(per_layer_changes[name][bad_mask]) for name in layer_names]
good_sems = [np.nanstd(per_layer_changes[name][good_mask]) / np.sqrt(max(1, np.sum(good_mask))) for name in layer_names]
bad_sems = [np.nanstd(per_layer_changes[name][bad_mask]) / np.sqrt(max(1, np.sum(bad_mask))) for name in layer_names]
ax3.bar(x_pos - width/2, good_means, width, yerr=good_sems, color='green', alpha=0.7, label='Good', capsize=3)
ax3.bar(x_pos + width/2, bad_means, width, yerr=bad_sems, color='red', alpha=0.7, label='Bad', capsize=3)
ax3.axhline(0, color='k', linewidth=0.5)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(layer_names)
ax3.set_ylabel('Alpha power change (%)')
ax3.set_title('Per-layer alpha change: good vs bad')
ax3.legend(fontsize=9)

# 2b-c. Good vs bad boxplots for each rate + mean alpha change scatter
if has_rates:
    ax = fig.add_subplot(gs[1, 1])
    box_data_good = [all_rate_arrays[l][good_mask] for l in rate_labels]
    box_data_bad = [all_rate_arrays[l][bad_mask] for l in rate_labels]
    positions = np.arange(len(rate_labels))
    w = 0.3
    bp1 = ax.boxplot(box_data_good, positions=positions - w/2, widths=w,
                      patch_artist=True, showfliers=False)
    bp2 = ax.boxplot(box_data_bad, positions=positions + w/2, widths=w,
                      patch_artist=True, showfliers=False)
    for patch in bp1['boxes']:
        patch.set_facecolor('green')
        patch.set_alpha(0.5)
    for patch in bp2['boxes']:
        patch.set_facecolor('red')
        patch.set_alpha(0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(rate_labels, rotation=30, ha='right')
    ax.set_ylabel('Rate (Hz)')
    ax.set_title('Rate distributions: good (green) vs bad (red)')
    for k, label in enumerate(rate_labels):
        _, _, _, p_mw_k = rate_corrs[label]
        if np.isfinite(p_mw_k):
            star = '*' if p_mw_k < 0.05 else ''
            ax.text(k, ax.get_ylim()[1] * 0.95, f'p={p_mw_k:.3f}{star}',
                    ha='center', va='top', fontsize=8)

    # Mean alpha change distribution
    ax = fig.add_subplot(gs[1, 2])
    if n_good > 0:
        ax.hist(mean_alpha_changes[good_mask], bins=15, alpha=0.6, color='green',
                label=f'Good (n={n_good})', edgecolor='white', density=True)
    if n_bad > 0:
        ax.hist(mean_alpha_changes[bad_mask], bins=15, alpha=0.6, color='red',
                label=f'Bad (n={n_bad})', edgecolor='white', density=True)
    ax.set_xlabel('Mean alpha change (%)')
    ax.set_ylabel('Density')
    ax.set_title('Mean alpha change distribution')
    ax.legend()

# --- Row 3: Per-rate scatter plots ---
individual_rates = ['L4C_E', 'L4C_PV', 'L6_E']
if has_rates:
    for j, label in enumerate(individual_rates):
        ax = fig.add_subplot(gs[2, j])
        arr_valid = all_rate_arrays[label][valid]
        r, p_val = rate_corrs[label][0], rate_corrs[label][1]
        colors = ['green' if g else 'red' for g in good_mask[valid]]
        ax.scatter(arr_valid, alpha_valid, c=colors, s=30, alpha=0.6,
                   edgecolors='k', linewidths=0.3)
        ax.set_xlabel(f'{label} rate (Hz)')
        ax.set_ylabel('Mean alpha change (%)')
        ax.set_title(f'{label} (r={r:.3f}, p={p_val:.3f})')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

# --- Row 4: L6_PV scatter, Total scatter, summary text ---
if has_rates:
    for j, label in enumerate(['L6_PV', 'Total']):
        ax = fig.add_subplot(gs[3, j])
        arr_valid = all_rate_arrays[label][valid]
        r, p_val = rate_corrs[label][0], rate_corrs[label][1]
        colors = ['green' if g else 'red' for g in good_mask[valid]]
        ax.scatter(arr_valid, alpha_valid, c=colors, s=30, alpha=0.6,
                   edgecolors='k', linewidths=0.3)
        ax.set_xlabel(f'{label} rate (Hz)')
        ax.set_ylabel('Mean alpha change (%)')
        ax.set_title(f'{label} (r={r:.3f}, p={p_val:.3f})')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

# Summary text panel
ax_txt = fig.add_subplot(gs[3, 2])
ax_txt.axis('off')
summary_lines = [
    f"Good trials: {sorted(GOOD_TRIALS)}",
    f"n_good={n_good}, n_bad={n_bad}",
    "",
    f"Phase circ-lin R² = {R2_phase:.3f}",
    f"  cos(phase): r={r_cos:.3f}, p={p_cos:.4f}",
    f"  sin(phase): r={r_sin:.3f}, p={p_sin:.4f}",
]
if has_rates:
    summary_lines += [
        "",
        "Rate correlations (r, p):",
    ]
    for label in rate_labels:
        r, p_val, _, p_mw_k = rate_corrs[label]
        summary_lines.append(f"  {label}: r={r:.3f}, p={p_val:.3f}, MW p={p_mw_k:.3f}")
ax_txt.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax_txt.transAxes,
            va='top', ha='left', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Trial-to-trial variability: rate jitter vs alpha phase', fontsize=16)
fig.savefig(os.path.join(output_dir, "variability_analysis.png"),
            dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"\nPlot saved to {output_dir}/variability_analysis.png")
