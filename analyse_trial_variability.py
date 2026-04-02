"""
Analyse what differentiates "good" trials (clean alpha→gamma switch)
from "bad" ones, using the metadata saved per trial.

Computes a switch-quality score per trial, then correlates with:
  - synapse counts per projection
  - mean delays per projection
  - intrinsic param summary stats (C, gL, tauw, b, a, EL)
  - initial membrane voltage
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, detrend
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

plt.rcParams.update({'mathtext.fontset': 'stix', 'font.family': 'STIXGeneral'})
plt.style.use('seaborn-v0_8-darkgrid')

# ── Settings ────────────────────────────────────────────────────────
BASE_PATH = "results/trials_01_04_3"
FS = 10000
PRE_WINDOW_MS = 300
POST_WINDOW_MS = 300
POST_START_MS = 300  # skip transient after stim onset

ALPHA_BAND = (7, 15)
GAMMA_BAND = (30, 80)

# ── 1. Load all trials ──────────────────────────────────────────────
trial_files = sorted(
    [f for f in os.listdir(BASE_PATH) if f.startswith("trial_") and f.endswith(".npz")]
)
print(f"Found {len(trial_files)} trial files")

trials = []
for fname in trial_files:
    data = np.load(os.path.join(BASE_PATH, fname), allow_pickle=True)
    trials.append({
        'trial_id': int(data['trial_id']),
        'seed': int(data['seed']),
        'metadata': data['metadata'].item(),
        'bipolar_lfp': data['bipolar_matrix'],
        'time': data['time_array_ms'],
        'stim_onset_ms': float(data['stim_onset_ms']),
        'channel_labels': data['channel_labels'],
    })

n_trials = len(trials)
print(f"Loaded {n_trials} trials")


# ── 2. Compute switch-quality score per trial ───────────────────────
def compute_switch_score(trial, alpha_band=ALPHA_BAND, gamma_band=GAMMA_BAND):
    """
    Score = negative alpha change (dB), averaged across channels.
    More negative = better alpha suppression = better trial.
    Gamma is returned for info but not used in score since it increases in all trials.
    """
    lfp = trial['bipolar_lfp']
    time = trial['time']
    stim = trial['stim_onset_ms']
    n_ch = lfp.shape[0]

    alpha_changes = []
    gamma_changes = []

    for ch in range(n_ch):
        sig = lfp[ch]
        pre_mask = (time >= stim - PRE_WINDOW_MS) & (time < stim)
        post_mask = (time >= stim + POST_START_MS) & (time < stim + POST_START_MS + POST_WINDOW_MS)

        pre = detrend(sig[pre_mask])
        post = detrend(sig[post_mask])

        if len(pre) < 100 or len(post) < 100:
            continue

        nperseg = min(len(pre), 1024)
        f, psd_pre = welch(pre, fs=FS, nperseg=nperseg, window='hann')
        _, psd_post = welch(post, fs=FS, nperseg=nperseg, window='hann')

        alpha_mask = (f >= alpha_band[0]) & (f <= alpha_band[1])
        gamma_mask = (f >= gamma_band[0]) & (f <= gamma_band[1])

        alpha_pre = np.mean(psd_pre[alpha_mask])
        alpha_post = np.mean(psd_post[alpha_mask])
        gamma_pre = np.mean(psd_pre[gamma_mask])
        gamma_post = np.mean(psd_post[gamma_mask])

        if alpha_pre > 0 and gamma_pre > 0:
            alpha_changes.append(10 * np.log10(alpha_post / alpha_pre))
            gamma_changes.append(10 * np.log10(gamma_post / gamma_pre))

    if len(alpha_changes) == 0:
        return np.nan, np.nan, np.nan

    alpha_change = np.mean(alpha_changes)
    gamma_change = np.mean(gamma_changes)
    # Score = alpha suppression (more negative = better)
    score = alpha_change
    return score, alpha_change, gamma_change


scores = []
alpha_changes_all = []
gamma_changes_all = []

for trial in trials:
    s, a, g = compute_switch_score(trial)
    scores.append(s)
    alpha_changes_all.append(a)
    gamma_changes_all.append(g)

scores = np.array(scores)
alpha_changes_all = np.array(alpha_changes_all)
gamma_changes_all = np.array(gamma_changes_all)

valid = ~np.isnan(scores)
print(f"\nSwitch score: mean={np.nanmean(scores):.2f}, std={np.nanstd(scores):.2f}")
print(f"  10-20 Hz change (dB): mean={np.nanmean(alpha_changes_all):.2f}")
print(f"  Gamma change (dB): mean={np.nanmean(gamma_changes_all):.2f}")

# ── 3. Extract metadata features ────────────────────────────────────
# Build a feature matrix from metadata
feature_names = []
feature_matrix = []

# Use first trial to discover keys
meta0 = trials[0]['metadata']

# Synapse counts
for key in sorted(meta0['synapse_counts'].keys()):
    feature_names.append(f"nsyn_{key}")

# Mean delays
for key in sorted(meta0['synapse_mean_delay_ms'].keys()):
    feature_names.append(f"delay_{key}")

# Intrinsic params
for pop_key in sorted(meta0['intrinsic_params'].keys()):
    for param in ['C_mean', 'C_std', 'gL_mean', 'gL_std',
                   'tauw_mean', 'tauw_std', 'b_mean', 'b_std',
                   'a_mean', 'a_std', 'EL_mean', 'EL_std']:
        feature_names.append(f"intr_{pop_key}_{param}")

# Initial V
for key in sorted(meta0['initial_v_mean_mV'].keys()):
    feature_names.append(f"v0_{key}")

# Now fill matrix
for trial in trials:
    meta = trial['metadata']
    row = []
    for key in sorted(meta['synapse_counts'].keys()):
        row.append(meta['synapse_counts'][key])
    for key in sorted(meta['synapse_mean_delay_ms'].keys()):
        row.append(meta['synapse_mean_delay_ms'].get(key, 0.0))
    for pop_key in sorted(meta['intrinsic_params'].keys()):
        for param in ['C_mean', 'C_std', 'gL_mean', 'gL_std',
                       'tauw_mean', 'tauw_std', 'b_mean', 'b_std',
                       'a_mean', 'a_std', 'EL_mean', 'EL_std']:
            row.append(meta['intrinsic_params'][pop_key].get(param, 0.0))
    for key in sorted(meta['initial_v_mean_mV'].keys()):
        row.append(meta['initial_v_mean_mV'][key])
    feature_matrix.append(row)

feature_matrix = np.array(feature_matrix)
print(f"\nFeature matrix: {feature_matrix.shape[0]} trials x {feature_matrix.shape[1]} features")

# ── 4. Correlate features with switch score ──────────────────────────
print("\n" + "="*80)
print("TOP CORRELATES WITH SWITCH SCORE (|r| > 0.15)")
print("="*80)

correlations = []
for i, name in enumerate(feature_names):
    vals = feature_matrix[valid, i]
    sc = scores[valid]
    # Skip constant features
    if np.std(vals) < 1e-10:
        continue
    r, p = spearmanr(vals, sc)
    correlations.append((name, r, p))

# Sort by absolute correlation
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"\n{'Feature':<55} {'Spearman r':>10} {'p-value':>10}")
print("-"*80)
for name, r, p in correlations:
    if abs(r) > 0.15:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{name:<55} {r:>10.3f} {p:>10.4f} {sig}")

# ── 5. Plots ─────────────────────────────────────────────────────────

# 5a. Distribution of switch scores
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(scores[valid], bins=20, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('10-20 Hz change (dB)')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of low-freq change')
axes[0].axvline(0, color='black', ls='-', alpha=0.5)
axes[0].axvline(np.nanmedian(scores), color='red', ls='--', label='median')
axes[0].legend()

axes[1].hist(alpha_changes_all[valid], bins=20, edgecolor='black', alpha=0.7, color='purple')
axes[1].set_xlabel('Low-freq power change (dB)')
axes[1].set_title('10-20 Hz change')
axes[1].axvline(0, color='black', ls='-')

axes[2].hist(gamma_changes_all[valid], bins=20, edgecolor='black', alpha=0.7, color='orange')
axes[2].set_xlabel('Gamma power change (dB)')
axes[2].set_title('Gamma (30-80 Hz) change')
axes[2].axvline(0, color='black', ls='-')

plt.tight_layout()
plt.savefig('switch_score_distribution.png', dpi=150)
plt.show()

# 5b. Top correlated features scatter plots
top_n = min(12, sum(1 for _, r, _ in correlations if abs(r) > 0.15))
if top_n > 0:
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (name, r, p) in enumerate(correlations[:top_n]):
        feat_idx = feature_names.index(name)
        ax = axes[idx]
        ax.scatter(feature_matrix[valid, feat_idx], scores[valid],
                   alpha=0.5, s=15, edgecolors='none')
        ax.set_xlabel(name, fontsize=8)
        ax.set_ylabel('10-20 Hz change (dB)')
        ax.set_title(f'r={r:.3f}, p={p:.3f}', fontsize=9)

    for idx in range(top_n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Top features correlated with switch quality', fontsize=14)
    plt.tight_layout()
    plt.savefig('top_correlates_scatter.png', dpi=150)
    plt.show()

# 5c. Compare best vs worst trials: which synapse counts differ most?
n_compare = max(5, n_trials // 5)  # top/bottom 20%
rank = np.argsort(scores[valid])
worst_idx = rank[:n_compare]
best_idx = rank[-n_compare:]

# Focus on synapse counts only
syn_features = [(i, name) for i, name in enumerate(feature_names) if name.startswith('nsyn_')]

if syn_features:
    diffs = []
    for feat_i, feat_name in syn_features:
        best_vals = feature_matrix[valid][best_idx, feat_i]
        worst_vals = feature_matrix[valid][worst_idx, feat_i]
        mean_diff_pct = (np.mean(best_vals) - np.mean(worst_vals)) / np.mean(worst_vals) * 100
        diffs.append((feat_name, mean_diff_pct, np.mean(best_vals), np.mean(worst_vals)))

    diffs.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'='*80}")
    print(f"SYNAPSE COUNT: best {n_compare} vs worst {n_compare} trials")
    print(f"{'='*80}")
    print(f"{'Projection':<55} {'Best mean':>10} {'Worst mean':>10} {'Diff %':>8}")
    print("-"*80)
    for name, diff_pct, best_m, worst_m in diffs[:20]:
        if abs(diff_pct) > 0.5:
            print(f"{name:<55} {best_m:>10.1f} {worst_m:>10.1f} {diff_pct:>+8.1f}%")


# 5d. Alpha change vs gamma change colored by score
fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(alpha_changes_all[valid], gamma_changes_all[valid],
                c=scores[valid], cmap='RdYlGn', s=20, edgecolors='gray', linewidth=0.3)
ax.set_xlabel('Low-freq power change (dB)')
ax.set_ylabel('Gamma power change (dB)')
ax.set_title('Alpha vs Gamma change per trial')
ax.axhline(0, color='black', ls=':', alpha=0.5)
ax.axvline(0, color='black', ls=':', alpha=0.5)
plt.colorbar(sc, label='10-20 Hz change (dB)')
plt.tight_layout()
plt.savefig('alpha_vs_gamma_per_trial.png', dpi=150)
plt.show()

# ── 6. Per-layer alpha suppression ───────────────────────────────────
# Map bipolar channels to layers using z-coordinates

LAYER_Z_RANGES = {
    'L23':  (0.45, 1.10),
    'L4AB': (0.14, 0.45),
    'L4C':  (-0.14, 0.14),
    'L5':   (-0.34, -0.14),
    'L6':   (-0.62, -0.34),
}

ELECTRODE_Z = [-0.94, -0.79, -0.64, -0.49, -0.34, -0.19, -0.04,
               0.10, 0.26, 0.40, 0.56, 0.70, 0.86, 1.00, 1.16]

# Bipolar channel midpoints
bipolar_z = [(ELECTRODE_Z[i] + ELECTRODE_Z[i+1]) / 2 for i in range(len(ELECTRODE_Z)-1)]

def assign_layer(z):
    for layer_name, (z_lo, z_hi) in LAYER_Z_RANGES.items():
        if z_lo <= z <= z_hi:
            return layer_name
    return None

channel_to_layer = [assign_layer(z) for z in bipolar_z]


def compute_per_layer_alpha(trial, alpha_band=ALPHA_BAND):
    """Compute alpha change (dB) per layer, averaging across channels in that layer."""
    lfp = trial['bipolar_lfp']
    time = trial['time']
    stim = trial['stim_onset_ms']

    layer_alpha = {name: [] for name in LAYER_Z_RANGES}

    for ch in range(lfp.shape[0]):
        layer = channel_to_layer[ch] if ch < len(channel_to_layer) else None
        if layer is None:
            continue

        sig = lfp[ch]
        pre_mask = (time >= stim - PRE_WINDOW_MS) & (time < stim)
        post_mask = (time >= stim + POST_START_MS) & (time < stim + POST_START_MS + POST_WINDOW_MS)

        pre = detrend(sig[pre_mask])
        post = detrend(sig[post_mask])

        if len(pre) < 100 or len(post) < 100:
            continue

        nperseg = min(len(pre), 1024)
        f, psd_pre = welch(pre, fs=FS, nperseg=nperseg, window='hann')
        _, psd_post = welch(post, fs=FS, nperseg=nperseg, window='hann')

        alpha_mask = (f >= alpha_band[0]) & (f <= alpha_band[1])
        a_pre = np.mean(psd_pre[alpha_mask])
        a_post = np.mean(psd_post[alpha_mask])

        if a_pre > 0:
            layer_alpha[layer].append(10 * np.log10(a_post / a_pre))

    result = {}
    for layer_name in LAYER_Z_RANGES:
        if layer_alpha[layer_name]:
            result[layer_name] = np.mean(layer_alpha[layer_name])
        else:
            result[layer_name] = np.nan
    return result


# Compute per-layer low-freq for all trials
layer_names = list(LAYER_Z_RANGES.keys())
per_layer_alpha = {ln: [] for ln in layer_names}

for trial in trials:
    res = compute_per_layer_alpha(trial)
    for ln in layer_names:
        per_layer_alpha[ln].append(res[ln])

for ln in layer_names:
    per_layer_alpha[ln] = np.array(per_layer_alpha[ln])

# Print channel-to-layer mapping
print("\n" + "="*80)
print("BIPOLAR CHANNEL → LAYER MAPPING")
print("="*80)
for ch, (z, layer) in enumerate(zip(bipolar_z, channel_to_layer)):
    print(f"  Ch{ch}: z={z:.3f} mm → {layer if layer else 'outside'}")

# Print per-layer stats
print("\n" + "="*80)
print("PER-LAYER LOW-FREQ (1-20 Hz) CHANGE (dB)")
print("="*80)
print(f"{'Layer':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'N chans':>8}")
print("-"*50)
for ln in layer_names:
    vals = per_layer_alpha[ln]
    n_ch = sum(1 for c in channel_to_layer if c == ln)
    print(f"{ln:<10} {np.nanmean(vals):>+8.2f} {np.nanstd(vals):>8.2f} "
          f"{np.nanmin(vals):>+8.2f} {np.nanmax(vals):>+8.2f} {n_ch:>8}")

# Per-layer alpha correlation with global score
print("\n" + "="*80)
print("CORRELATION: per-layer 10-20 Hz ↔ global score")
print("="*80)
for ln in layer_names:
    vals = per_layer_alpha[ln]
    mask = valid & ~np.isnan(vals)
    if np.sum(mask) > 5:
        r, p = spearmanr(vals[mask], scores[mask])
        print(f"  {ln:<8} r={r:+.3f}  p={p:.4f}")

# ── Plot: per-layer low-freq change distributions ──────────────────────
fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']

for i, ln in enumerate(layer_names):
    vals = per_layer_alpha[ln][~np.isnan(per_layer_alpha[ln])]
    axes[i].hist(vals, bins=15, edgecolor='black', alpha=0.7, color=colors[i])
    axes[i].axvline(0, color='black', ls='-', alpha=0.5)
    axes[i].axvline(np.mean(vals), color='red', ls='--', label=f'mean={np.mean(vals):+.1f}')
    axes[i].set_xlabel('10-20 Hz change (dB)')
    axes[i].set_title(ln)
    axes[i].legend(fontsize=8)

axes[0].set_ylabel('Count')
plt.suptitle('Per-layer 10-20 Hz change across trials', fontsize=14)
plt.tight_layout()
plt.savefig('per_layer_alpha_suppression.png', dpi=150)
plt.show()

# ── Plot: per-layer scatter vs global score ─────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)

for i, ln in enumerate(layer_names):
    vals = per_layer_alpha[ln]
    mask = valid & ~np.isnan(vals)
    axes[i].scatter(vals[mask], scores[mask], alpha=0.4, s=15, color=colors[i], edgecolors='none')
    axes[i].set_xlabel(f'{ln} alpha change (dB)')
    axes[i].set_title(ln)
    # identity line
    lims = [min(vals[mask].min(), scores[mask].min()), max(vals[mask].max(), scores[mask].max())]
    axes[i].plot(lims, lims, 'k--', alpha=0.3)
    if np.sum(mask) > 5:
        r, _ = spearmanr(vals[mask], scores[mask])
        axes[i].annotate(f'r={r:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10)

axes[0].set_ylabel('Global alpha score (dB)')
plt.suptitle('Per-layer alpha vs global alpha score', fontsize=14)
plt.tight_layout()
plt.savefig('per_layer_vs_global.png', dpi=150)
plt.show()

# ── Plot: inter-layer correlations ──────────────────────────────────
corr_matrix = np.zeros((5, 5))
for i, ln1 in enumerate(layer_names):
    for j, ln2 in enumerate(layer_names):
        v1 = per_layer_alpha[ln1]
        v2 = per_layer_alpha[ln2]
        mask = ~np.isnan(v1) & ~np.isnan(v2)
        if np.sum(mask) > 5:
            corr_matrix[i, j], _ = spearmanr(v1[mask], v2[mask])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(layer_names)
ax.set_yticklabels(layer_names)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center', fontsize=10)
plt.colorbar(im, label='Spearman r')
ax.set_title('Inter-layer alpha suppression correlation')
plt.tight_layout()
plt.savefig('inter_layer_alpha_correlation.png', dpi=150)
plt.show()

print("\nDone! Check the plots and the correlation table above.")
print("Key: more negative score = stronger alpha suppression = better trial")
print("     score near 0 or positive = alpha persists during stimulus")

# ── 7. Multivariate analysis ─────────────────────────────────────────
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

print("\n" + "="*80)
print("MULTIVARIATE ANALYSIS (Lasso regression)")
print("="*80)

# Use only valid trials, remove constant features
X = feature_matrix[valid]
y = scores[valid]

# Remove constant columns
col_std = np.std(X, axis=0)
varying = col_std > 1e-10
X_var = X[:, varying]
names_var = [n for n, v in zip(feature_names, varying) if v]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_var)

# LassoCV to find best regularization + select features
lasso = LassoCV(cv=5, n_alphas=100, max_iter=10000, random_state=42)
lasso.fit(X_scaled, y)

# Cross-validated R²
cv_scores = cross_val_score(lasso, X_scaled, y, cv=5, scoring='r2')
print(f"\nLasso best alpha: {lasso.alpha_:.4f}")
print(f"Cross-validated R²: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
print(f"Training R²: {lasso.score(X_scaled, y):.3f}")

# Non-zero coefficients = selected features
nonzero = np.where(lasso.coef_ != 0)[0]
print(f"\nFeatures selected: {len(nonzero)} / {len(names_var)}")

if len(nonzero) > 0:
    coef_pairs = [(names_var[i], lasso.coef_[i]) for i in nonzero]
    coef_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'Feature':<55} {'Coeff':>10} {'Direction':<20}")
    print("-"*85)
    for name, coef in coef_pairs:
        direction = "→ less alpha suppr." if coef > 0 else "→ more alpha suppr."
        print(f"{name:<55} {coef:>+10.4f} {direction}")

    # Plot coefficients
    fig, ax = plt.subplots(figsize=(10, max(4, len(coef_pairs) * 0.35)))
    names_plot = [c[0] for c in coef_pairs]
    coefs_plot = [c[1] for c in coef_pairs]
    colors_bar = ['#e41a1c' if c > 0 else '#377eb8' for c in coefs_plot]
    ax.barh(range(len(names_plot)), coefs_plot, color=colors_bar, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(names_plot)))
    ax.set_yticklabels(names_plot, fontsize=8)
    ax.set_xlabel('Lasso coefficient (standardized)')
    ax.set_title(f'Multivariate predictors of alpha suppression (CV R²={np.mean(cv_scores):.2f})')
    ax.axvline(0, color='black', ls='-', linewidth=0.5)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('multivariate_lasso_coefficients.png', dpi=150)
    plt.show()
else:
    print("No features selected — model is fully regularized (no signal above noise).")
