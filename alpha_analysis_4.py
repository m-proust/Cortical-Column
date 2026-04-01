"""
Broad Alpha Suppression Analysis
==================================
What pre-stimulus factors predict a broad alpha decrease across
*most* cortical layers after stimulus onset?

A trial is classified as "good" when a majority of layers (>=3 out of 5)
show alpha power decrease exceeding the threshold.

Layer channel mapping (bipolar, 0-indexed):
  L6   ~ bipolar channels 0–2
  L5   ~ bipolar channels 3–4
  L4C  ~ bipolar channel  5
  L4AB ~ bipolar channels 6–7
  L23  ~ bipolar channels 8–(end)

Analyses:
1.  Per-layer pre-stim alpha power → broad alpha suppression
2.  Inter-layer pre-stim alpha coherence
3.  Phase-amplitude coupling across layers
4.  Time-resolved firing rates by cell type and layer
5.  SOM activity as alpha generator proxy
6.  Temporal precedence (alpha envelope cross-correlation)
7.  E & SOM rate correlation with alpha power across layers
8.  Summary dashboard: ranked feature predictors

Usage:
    Set base_path, n_trials, and channel assignments to match your data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert, detrend, sosfiltfilt, butter, coherence
from scipy.stats import pearsonr, mannwhitneyu, sem
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
# CONFIG — EDIT THESE
# ============================================================
base_path = "results/trials_01_04"
n_trials = 61
fs = 10000

alpha_band = (7, 14)
gamma_band = (30, 80)
beta_band  = (15, 30)

pre_window_ms  = 500
post_window_ms = 500
post_start_ms  = 200

alpha_threshold = -10  # % change per layer to count as suppressed

# ---- Channel assignments (bipolar, 0-indexed) ----
L6_channels   = [0, 1, 2]
L5_channels   = [3, 4]
L4C_channels  = [5, 6]
L4AB_channels = [7, 8]
L23_channels  = [9, 10, 11, 12, 13, 14]

# All layers used for broad classification
LAYER_DEFS = {
    'L6':   L6_channels,
    'L5':   L5_channels,
    'L4C':  L4C_channels,
    'L4AB': L4AB_channels,
    'L23':  L23_channels,
}

# A trial is "good" if at least this many layers show alpha decrease
min_layers_suppressed = 3

# Legacy aliases kept for analysis functions
L4_channels = L4C_channels + L4AB_channels
L4C_label   = 'L4C'
L4AB_label  = 'L4AB'
L4_label    = 'L4 (4C+4AB)'


# ============================================================
# HELPERS
# ============================================================
def bandpass(x, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x)


def band_power(x, lo, hi, fs):
    filt = bandpass(x, lo, hi, fs)
    return np.mean(filt ** 2)


def inst_phase_amp(x, lo, hi, fs):
    filt = bandpass(x, lo, hi, fs)
    analytic = hilbert(filt)
    return np.angle(analytic), np.abs(analytic), filt


def mean_lfp(bipolar, channels):
    """Average LFP across specified bipolar channels."""
    return np.mean([bipolar[ch] for ch in channels], axis=0)


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
            'trial_idx':         idx,
            'time':              data['time_array_ms'],
            'bipolar_lfp':       data['bipolar_matrix'],
            'lfp_matrix':        data['lfp_matrix'],
            'rate_data':         data['rate_data'].item() if data['rate_data'].size == 1
                                 else data['rate_data'],
            'stim_onset_ms':     float(data['stim_onset_ms']),
            'baseline_ms':       float(data['baseline_ms']),
            'channel_depths':    data['channel_depths'],
            'electrode_positions': data['electrode_positions'],
        }
        all_trials.append(trial)
    print(f"Loaded {len(all_trials)} trials")
    return all_trials


def classify_trials(all_trials, layer_defs, alpha_band,
                    pre_window_ms, post_window_ms, post_start_ms, fs,
                    threshold, min_layers):
    """
    Classify trials by broad alpha suppression across layers.

    A trial is "good" if at least `min_layers` out of the defined layers
    show alpha power decrease exceeding `threshold` (%).

    Returns per-trial metrics including per-layer alpha changes and an
    overall mean alpha change (average across all layers).
    """
    metrics = []
    for trial in all_trials:
        bp   = trial['bipolar_lfp']
        t    = trial['time']
        stim = trial['stim_onset_ms']
        n_ch = bp.shape[0]

        pre_mask  = (t >= stim - pre_window_ms) & (t < stim)
        post_mask = (t >= stim + post_start_ms) & (t < stim + post_start_ms + post_window_ms)

        layer_changes = {}
        n_suppressed = 0
        all_changes = []

        for layer_name, chs in layer_defs.items():
            pre_a, post_a = [], []
            for ch in chs:
                if ch < n_ch:
                    pre_a.append(band_power(detrend(bp[ch][pre_mask]),  *alpha_band, fs))
                    post_a.append(band_power(detrend(bp[ch][post_mask]), *alpha_band, fs))

            if pre_a and post_a:
                mp  = np.mean(pre_a)
                mpo = np.mean(post_a)
                chg = (mpo - mp) / (mp + 1e-30) * 100
            else:
                chg = np.nan

            layer_changes[layer_name] = chg
            if not np.isnan(chg):
                all_changes.append(chg)
                if chg < threshold:
                    n_suppressed += 1

        mean_change = np.mean(all_changes) if all_changes else np.nan

        metrics.append({
            'trial_idx':        trial['trial_idx'],
            'layer_changes':    layer_changes,
            'n_suppressed':     n_suppressed,
            'n_layers':         len([c for c in all_changes if not np.isnan(c)]),
            'alpha_change_pct': mean_change,
        })

    alpha_changes = np.array([m['alpha_change_pct'] for m in metrics])
    n_suppressed  = np.array([m['n_suppressed'] for m in metrics])
    good_mask = n_suppressed >= min_layers
    bad_mask  = ~good_mask

    # Print summary
    print(f"Good trials (α↓ in >={min_layers} layers): {np.sum(good_mask)},  "
          f"Bad trials: {np.sum(bad_mask)}")
    print(f"  Suppressed-layer distribution: "
          f"{dict(zip(*np.unique(n_suppressed, return_counts=True)))}")

    return metrics, alpha_changes, good_mask, bad_mask


# ============================================================
# ANALYSIS 1: PER-LAYER PRE-STIM ALPHA POWER
# ============================================================
def analysis_prestim_alpha(all_trials, alpha_changes, good_mask):
    """
    Does pre-stimulus alpha power in each layer predict broad
    alpha suppression (majority of layers)?
    """
    plot_layers = {
        'L6':    L6_channels,
        'L5':    L5_channels,
        'L4C':   L4C_channels,
        'L4AB':  L4AB_channels,
        'L23':   L23_channels,
    }

    n_cols = len(plot_layers)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 10))

    for col, (layer_label, chs) in enumerate(plot_layers.items()):
        pre_alpha = []
        for trial in all_trials:
            bp   = trial['bipolar_lfp']
            t    = trial['time']
            stim = trial['stim_onset_ms']
            n_ch = bp.shape[0]
            pre_mask = (t >= stim - pre_window_ms) & (t < stim)

            ch_a = []
            for ch in chs:
                if ch < n_ch:
                    ch_a.append(band_power(detrend(bp[ch][pre_mask]), *alpha_band, fs))
            pre_alpha.append(np.mean(ch_a) if ch_a else 0)

        pre_alpha = np.array(pre_alpha)

        # Row 0: scatter — pre-stim alpha vs mean alpha change
        ax = axes[0, col]
        ax.scatter(pre_alpha, alpha_changes,
                   c=['green' if g else 'red' for g in good_mask],
                   edgecolors='k', s=35, alpha=0.7)
        ax.set_xlabel(f'{layer_label} pre-stim alpha power')
        ax.set_ylabel('Mean alpha change (%)')
        ax.set_title(f'{layer_label} Alpha → Broad Suppression')
        r, p = pearsonr(pre_alpha, alpha_changes)
        ax.text(0.05, 0.95, f'r = {r:.2f}, p = {p:.4f}',
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axhline(0, color='gray', ls='--', alpha=0.3)

        # Row 1: boxplot good vs bad
        ax = axes[1, col]
        good_vals = pre_alpha[good_mask]
        bad_vals  = pre_alpha[~good_mask]
        bp_plot = ax.boxplot([good_vals, bad_vals],
                              labels=['Good\n(broad α↓)', 'Bad\n(no broad α↓)'],
                              patch_artist=True)
        bp_plot['boxes'][0].set_facecolor('green'); bp_plot['boxes'][0].set_alpha(0.4)
        bp_plot['boxes'][1].set_facecolor('red');   bp_plot['boxes'][1].set_alpha(0.4)
        ax.set_ylabel(f'{layer_label} pre-stim alpha power')
        if len(good_vals) > 1 and len(bad_vals) > 1:
            _, p_mw = mannwhitneyu(good_vals, bad_vals)
            d = np.abs(np.mean(good_vals) - np.mean(bad_vals)) / \
                np.sqrt((np.var(good_vals) + np.var(bad_vals)) / 2 + 1e-30)
            ax.text(0.5, 0.95, f'MW p={p_mw:.4f}\nd={d:.2f}',
                    transform=ax.transAxes, ha='center', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Analysis 1: Per-Layer Pre-stimulus Alpha Power\n'
                 f'Predicting Broad Alpha Suppression (>={min_layers_suppressed} layers)',
                 fontsize=15)
    plt.tight_layout()
    plt.savefig('broad_01_prestim_alpha_power.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# ANALYSIS 2: DEEP ↔ L4 ALPHA COHERENCE
# ============================================================
def analysis_alpha_coherence(all_trials, alpha_changes, good_mask):
    """
    Pre-stimulus alpha-band coherence between layer pairs.
    Does stronger inter-layer coupling predict broad alpha suppression?
    """
    pairs = [
        ('L6 ↔ L5',   L6_channels,  L5_channels),
        ('L6 ↔ L4C',  L6_channels,  L4C_channels),
        ('L5 ↔ L4C',  L5_channels,  L4C_channels),
        ('L5 ↔ L4AB', L5_channels,  L4AB_channels),
        ('L4AB ↔ L23', L4AB_channels, L23_channels),
        ('L6 ↔ L23',  L6_channels,  L23_channels),
    ]

    fig, axes = plt.subplots(2, len(pairs), figsize=(5 * len(pairs), 10))

    for col, (pair_label, deep_chs, tgt_chs) in enumerate(pairs):
        coh_alpha_pre = []
        coh_full_pre  = []
        f_c_ref = None

        for trial in all_trials:
            bp   = trial['bipolar_lfp']
            t    = trial['time']
            stim = trial['stim_onset_ms']
            n_ch = bp.shape[0]

            pre_mask = (t >= stim - pre_window_ms) & (t < stim)

            valid_deep = [ch for ch in deep_chs if ch < n_ch]
            valid_tgt  = [ch for ch in tgt_chs  if ch < n_ch]
            if not valid_deep or not valid_tgt:
                coh_alpha_pre.append(np.nan)
                continue

            deep_pre = np.mean([bp[ch][pre_mask] for ch in valid_deep], axis=0)
            tgt_pre  = np.mean([bp[ch][pre_mask] for ch in valid_tgt],  axis=0)

            nperseg = min(len(deep_pre), 2048)
            f_c, c_pre = coherence(detrend(deep_pre), detrend(tgt_pre),
                                    fs=fs, nperseg=nperseg)
            if f_c_ref is None:
                f_c_ref = f_c

            amask = (f_c >= alpha_band[0]) & (f_c <= alpha_band[1])
            coh_alpha_pre.append(np.mean(c_pre[amask]))
            coh_full_pre.append(c_pre)

        coh_alpha_pre = np.array(coh_alpha_pre)

        # Row 0: coherence spectrum (good vs bad, pre-stim)
        ax = axes[0, col]
        if coh_full_pre and f_c_ref is not None:
            good_spec = [coh_full_pre[i] for i in range(len(coh_full_pre))
                         if i < len(good_mask) and good_mask[i]]
            bad_spec  = [coh_full_pre[i] for i in range(len(coh_full_pre))
                         if i < len(good_mask) and not good_mask[i]]
            freq_mask = f_c_ref <= 100
            for spec, color, label in [(good_spec, 'green', 'Good'),
                                        (bad_spec,  'red',   'Bad')]:
                if spec:
                    mn  = np.mean(spec, axis=0)[freq_mask]
                    se  = sem(spec,  axis=0)[freq_mask]
                    ax.plot(f_c_ref[freq_mask], mn, color=color, lw=2, label=label)
                    ax.fill_between(f_c_ref[freq_mask], mn - se, mn + se,
                                    color=color, alpha=0.15)
        ax.axvspan(*alpha_band, color='purple', alpha=0.08, label='Alpha')
        ax.axvspan(*gamma_band, color='orange',  alpha=0.05, label='Gamma')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Coherence')
        ax.set_title(f'{pair_label}\nPre-stim coherence')
        ax.legend(fontsize=7)
        ax.set_xlim(1, 100)

        # Row 1: pre-stim alpha coherence vs L4 alpha change
        ax = axes[1, col]
        valid = ~np.isnan(coh_alpha_pre)
        ax.scatter(coh_alpha_pre[valid], alpha_changes[valid],
                   c=['green' if g else 'red' for g in good_mask[valid]],
                   edgecolors='k', s=35, alpha=0.7)
        ax.set_xlabel(f'Alpha coherence')
        ax.set_ylabel('Mean alpha change (%)')
        ax.set_title(f'Alpha Coh → Suppression')
        if np.sum(valid) > 5:
            r, p = pearsonr(coh_alpha_pre[valid], alpha_changes[valid])
            ax.text(0.05, 0.95, f'r={r:.2f}\np={p:.4f}',
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Analysis 2: Inter-Layer Alpha Coherence\n'
                 f'Predicting Broad Alpha Suppression', fontsize=15)
    plt.tight_layout()
    plt.savefig('broad_02_alpha_coherence.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# ANALYSIS 3: DEEP ALPHA PHASE → L4 ALPHA/GAMMA AMPLITUDE (PAC)
# ============================================================
def analysis_deep_L4_pac(all_trials, alpha_changes, good_mask):
    """
    Phase-amplitude coupling: does deep-layer alpha phase modulate
    L4 alpha/gamma amplitude? Computed separately for L4C and L4AB.
    Uses Tort et al. (2010) modulation index.
    """
    def modulation_index(phase_sig, amp_sig, n_bins=18):
        bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        mean_amp  = np.zeros(n_bins)
        for b in range(n_bins):
            mask = (phase_sig >= bin_edges[b]) & (phase_sig < bin_edges[b + 1])
            if np.sum(mask) > 0:
                mean_amp[b] = np.mean(amp_sig[mask])
        if np.sum(mean_amp) == 0:
            return 0.0, mean_amp
        p = mean_amp / np.sum(mean_amp)
        p = p[p > 0]
        kl = np.sum(p * np.log(p / (1.0 / n_bins)))
        return kl / np.log(n_bins), mean_amp

    n_bins = 18
    bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False) + np.pi / n_bins
    bin_centers_deg = np.rad2deg(bin_centers)

    deep_defs = {'L6': L6_channels, 'L5': L5_channels}
    tgt_defs  = {'L4C': L4C_channels, 'L4AB': L4AB_channels, 'L23': L23_channels}

    # Storage: [deep][target] → list of MI values
    mi_alpha = {d: {t: [] for t in tgt_defs} for d in deep_defs}
    mi_gamma = {d: {t: [] for t in tgt_defs} for d in deep_defs}
    amp_dist_alpha = {d: {t: {'good': [], 'bad': []} for t in tgt_defs} for d in deep_defs}
    amp_dist_gamma = {d: {t: {'good': [], 'bad': []} for t in tgt_defs} for d in deep_defs}

    for i, trial in enumerate(all_trials):
        bp   = trial['bipolar_lfp']
        t    = trial['time']
        stim = trial['stim_onset_ms']
        n_ch = bp.shape[0]
        pre_mask = (t >= stim - pre_window_ms) & (t < stim)

        for tgt_label, tgt_chs in tgt_defs.items():
            valid_tgt = [ch for ch in tgt_chs if ch < n_ch]
            if not valid_tgt:
                for dl in deep_defs:
                    mi_alpha[dl][tgt_label].append(np.nan)
                    mi_gamma[dl][tgt_label].append(np.nan)
                continue

            tgt_sig  = mean_lfp(bp, valid_tgt)
            tgt_pre  = detrend(tgt_sig[pre_mask])
            _, tgt_a_amp, _ = inst_phase_amp(tgt_pre, *alpha_band, fs)
            _, tgt_g_amp, _ = inst_phase_amp(tgt_pre, *gamma_band, fs)

            for deep_label, deep_chs in deep_defs.items():
                valid_d = [ch for ch in deep_chs if ch < n_ch]
                if not valid_d:
                    mi_alpha[deep_label][tgt_label].append(np.nan)
                    mi_gamma[deep_label][tgt_label].append(np.nan)
                    continue

                deep_pre = detrend(mean_lfp(bp, valid_d)[pre_mask])
                d_phase, _, _ = inst_phase_amp(deep_pre, *alpha_band, fs)

                mia, ampa = modulation_index(d_phase, tgt_a_amp)
                mig, ampg = modulation_index(d_phase, tgt_g_amp)

                mi_alpha[deep_label][tgt_label].append(mia)
                mi_gamma[deep_label][tgt_label].append(mig)

                group = 'good' if good_mask[i] else 'bad'
                amp_dist_alpha[deep_label][tgt_label][group].append(ampa)
                amp_dist_gamma[deep_label][tgt_label][group].append(ampg)

    # ---- Plot: 4 rows (L5/L6 × L4C/L4AB), 4 cols (MI alpha, MI gamma, amp alpha, amp gamma)
    n_rows = len(deep_defs) * len(tgt_defs)  # 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(26, 5 * n_rows))

    row = 0
    for deep_label in deep_defs:
        for tgt_label in tgt_defs:
            mia = np.array(mi_alpha[deep_label][tgt_label])
            mig = np.array(mi_gamma[deep_label][tgt_label])

            # Col 0: MI alpha scatter
            ax = axes[row, 0]
            valid = ~np.isnan(mia)
            ax.scatter(mia[valid], alpha_changes[valid],
                       c=['green' if g else 'red' for g in good_mask[valid]],
                       edgecolors='k', s=35, alpha=0.7)
            ax.set_xlabel(f'{deep_label} α-phase → {tgt_label} α-amp MI')
            ax.set_ylabel('Mean alpha change (%)')
            ax.set_title(f'{deep_label}→{tgt_label} Alpha PAC')
            if np.sum(valid) > 5:
                r, p = pearsonr(mia[valid], alpha_changes[valid])
                ax.text(0.05, 0.95, f'r={r:.2f}\np={p:.4f}',
                        transform=ax.transAxes, va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Col 1: MI gamma scatter
            ax = axes[row, 1]
            valid = ~np.isnan(mig)
            ax.scatter(mig[valid], alpha_changes[valid],
                       c=['green' if g else 'red' for g in good_mask[valid]],
                       edgecolors='k', s=35, alpha=0.7)
            ax.set_xlabel(f'{deep_label} α-phase → {tgt_label} γ-amp MI')
            ax.set_ylabel('Mean alpha change (%)')
            ax.set_title(f'{deep_label}→{tgt_label} Gamma PAC')
            if np.sum(valid) > 5:
                r, p = pearsonr(mig[valid], alpha_changes[valid])
                ax.text(0.05, 0.95, f'r={r:.2f}\np={p:.4f}',
                        transform=ax.transAxes, va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Col 2: alpha amplitude distribution over phase bins
            ax = axes[row, 2]
            for grp, color, lbl in [('good', 'green', 'Good'), ('bad', 'red', 'Bad')]:
                dists = amp_dist_alpha[deep_label][tgt_label][grp]
                if dists:
                    md = np.mean(dists, axis=0)
                    if np.sum(md) > 0:
                        md = md / np.sum(md)
                    se = sem(dists, axis=0) / (np.sum(np.mean(dists, axis=0)) + 1e-30)
                    ax.plot(bin_centers_deg, md, color=color, lw=2, label=lbl)
                    ax.fill_between(bin_centers_deg, md - se, md + se,
                                    color=color, alpha=0.15)
            ax.axhline(1.0 / n_bins, color='gray', ls='--', alpha=0.5, label='Uniform')
            ax.set_xlabel(f'{deep_label} alpha phase (deg)')
            ax.set_ylabel(f'{tgt_label} alpha amp (norm)')
            ax.set_title(f'{tgt_label} α-Amp by {deep_label} α-Phase')
            ax.legend(fontsize=8)

            # Col 3: gamma amplitude distribution over phase bins
            ax = axes[row, 3]
            for grp, color, lbl in [('good', 'green', 'Good'), ('bad', 'red', 'Bad')]:
                dists = amp_dist_gamma[deep_label][tgt_label][grp]
                if dists:
                    md = np.mean(dists, axis=0)
                    if np.sum(md) > 0:
                        md = md / np.sum(md)
                    se = sem(dists, axis=0) / (np.sum(np.mean(dists, axis=0)) + 1e-30)
                    ax.plot(bin_centers_deg, md, color=color, lw=2, label=lbl)
                    ax.fill_between(bin_centers_deg, md - se, md + se,
                                    color=color, alpha=0.15)
            ax.axhline(1.0 / n_bins, color='gray', ls='--', alpha=0.5, label='Uniform')
            ax.set_xlabel(f'{deep_label} alpha phase (deg)')
            ax.set_ylabel(f'{tgt_label} gamma amp (norm)')
            ax.set_title(f'{tgt_label} γ-Amp by {deep_label} α-Phase')
            ax.legend(fontsize=8)

            row += 1

    plt.suptitle(f'Analysis 3: Alpha Phase → Amplitude Coupling (PAC)\n'
                 f'Predicting Broad Alpha Suppression', fontsize=15)
    plt.tight_layout()
    plt.savefig('broad_03_pac.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# ANALYSIS 4: TIME-RESOLVED DEEP-LAYER FIRING RATES
# ============================================================
def analysis_firing_rates(all_trials, alpha_changes, good_mask):
    """
    Time-resolved firing rates for all layers and cell types,
    split by good vs bad broad alpha suppression trials.
    """
    layers     = ['L23', 'L4AB', 'L4C', 'L5', 'L6']
    cell_types = ['E', 'PV', 'SOM', 'VIP']
    win_pre    = 500
    win_post   = 700

    fig, axes = plt.subplots(len(layers), len(cell_types),
                              figsize=(22, 5 * len(layers)), sharey='row')

    for row, layer in enumerate(layers):
        for col, ct in enumerate(cell_types):
            ax = axes[row, col]
            mon_key = f"{ct}_rate"
            good_traces, bad_traces = [], []

            for i, trial in enumerate(all_trials):
                rd   = trial['rate_data']
                stim = trial['stim_onset_ms']

                if layer not in rd or mon_key not in rd[layer]:
                    continue

                t_ms   = rd[layer][mon_key]['t_ms']
                r_hz   = rd[layer][mon_key]['rate_hz']
                mask   = (t_ms >= stim - win_pre) & (t_ms < stim + win_post)
                if np.sum(mask) == 0:
                    continue

                t_rel    = t_ms[mask] - stim
                rate_win = r_hz[mask]

                if good_mask[i]:
                    good_traces.append((t_rel, rate_win))
                else:
                    bad_traces.append((t_rel, rate_win))

            t_common = np.linspace(-win_pre, win_post, 500)
            for traces, color, label in [(good_traces, 'green', 'Good'),
                                          (bad_traces,  'red',   'Bad')]:
                if not traces:
                    continue
                interped = np.array([np.interp(t_common, tr, rr) for tr, rr in traces])
                mean_r   = np.mean(interped, axis=0)
                sem_r    = sem(interped,  axis=0)
                ax.plot(t_common, mean_r, color=color, lw=2, label=label)
                ax.fill_between(t_common, mean_r - sem_r, mean_r + sem_r,
                                color=color, alpha=0.15)

            ax.axvline(0, color='black', ls='--', lw=1, label='Stim')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Rate (Hz)')
            ax.set_title(f'{layer} {ct}')
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    plt.suptitle(f'Analysis 4: Time-Resolved Firing Rates (All Layers)\n'
                 f'Good (broad α↓) vs Bad Trials', fontsize=15)
    plt.tight_layout()
    plt.savefig('broad_04_firing_rates.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# ANALYSIS 5: L5/L6 SOM RATE AS ALPHA GENERATOR PROXY
# ============================================================
def analysis_som_alpha_link(all_trials, alpha_changes, good_mask):
    """
    Test whether SOM pre-stim rate in each layer correlates with
    local alpha power, and whether SOM rate change predicts broad
    alpha suppression.
    """
    layer_defs = [
        ('L6',   L6_channels),
        ('L5',   L5_channels),
        ('L4C',  L4C_channels),
        ('L4AB', L4AB_channels),
        ('L23',  L23_channels),
    ]

    n_cols = len(layer_defs)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10))

    for col, (layer, tgt_chs) in enumerate(layer_defs):
        som_pre_rates = []
        som_post_rates = []
        local_alpha_pre = []

        for trial in all_trials:
            rd   = trial['rate_data']
            stim = trial['stim_onset_ms']
            bp   = trial['bipolar_lfp']
            t    = trial['time']
            n_ch = bp.shape[0]

            mon_key = 'SOM_rate'
            if layer not in rd or mon_key not in rd[layer]:
                som_pre_rates.append(np.nan)
                som_post_rates.append(np.nan)
                local_alpha_pre.append(np.nan)
                continue

            t_ms = rd[layer][mon_key]['t_ms']
            r_hz = rd[layer][mon_key]['rate_hz']

            pre_r_mask  = (t_ms >= stim - pre_window_ms) & (t_ms < stim)
            post_r_mask = (t_ms >= stim + post_start_ms) & \
                          (t_ms < stim + post_start_ms + post_window_ms)

            som_pre_rates.append(np.mean(r_hz[pre_r_mask])  if np.sum(pre_r_mask)  > 0 else np.nan)
            som_post_rates.append(np.mean(r_hz[post_r_mask]) if np.sum(post_r_mask) > 0 else np.nan)

            pre_lfp_mask = (t >= stim - pre_window_ms) & (t < stim)
            valid_tgt = [ch for ch in tgt_chs if ch < n_ch]
            a_pow = [band_power(detrend(bp[ch][pre_lfp_mask]), *alpha_band, fs)
                     for ch in valid_tgt]
            local_alpha_pre.append(np.mean(a_pow) if a_pow else np.nan)

        som_pre_rates  = np.array(som_pre_rates)
        som_post_rates = np.array(som_post_rates)
        local_alpha_pre = np.array(local_alpha_pre)
        som_change     = som_post_rates - som_pre_rates

        # Row 0: SOM pre-stim rate vs local alpha pre-stim power
        ax = axes[0, col]
        valid = ~(np.isnan(som_pre_rates) | np.isnan(local_alpha_pre))
        if np.sum(valid) > 5:
            ax.scatter(som_pre_rates[valid], local_alpha_pre[valid],
                       c=['green' if g else 'red' for g in good_mask[valid]],
                       edgecolors='k', s=35, alpha=0.7)
            r, p = pearsonr(som_pre_rates[valid], local_alpha_pre[valid])
            ax.text(0.05, 0.95, f'r={r:.2f}\np={p:.4f}',
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlabel(f'{layer} SOM pre-stim rate (Hz)')
        ax.set_ylabel(f'{layer} pre-stim alpha power')
        ax.set_title(f'{layer} SOM Rate → {layer} Alpha Power')

        # Row 1: SOM rate change vs broad alpha suppression
        ax = axes[1, col]
        valid = ~np.isnan(som_change)
        if np.sum(valid) > 5:
            ax.scatter(som_change[valid], alpha_changes[valid],
                       c=['green' if g else 'red' for g in good_mask[valid]],
                       edgecolors='k', s=35, alpha=0.7)
            r, p = pearsonr(som_change[valid], alpha_changes[valid])
            ax.text(0.05, 0.95, f'r={r:.2f}\np={p:.4f}',
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlabel(f'{layer} SOM ΔRate (Hz)')
        ax.set_ylabel('Mean alpha change (%)')
        ax.set_title(f'{layer} SOM ΔRate → Broad Suppression')
        ax.axhline(0, color='gray', ls='--', alpha=0.3)
        ax.axvline(0, color='gray', ls='--', alpha=0.3)

    plt.suptitle(f'Analysis 5: SOM Interneurons as Alpha Generators\n'
                 f'SOM rate vs local alpha & broad suppression', fontsize=15)
    plt.tight_layout()
    plt.savefig('broad_05_som_alpha.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# ANALYSIS 6: TEMPORAL PRECEDENCE (ALPHA LEAD/LAG)
# ============================================================
def analysis_temporal_precedence(all_trials, alpha_changes, good_mask):
    """
    Cross-correlation of alpha envelopes between deep and L4 layers.
    Does deep alpha lead L4 alpha? Does this differ for good vs bad?
    L4C and L4AB are tested separately.
    """
    pairs = [
        ('L6 → L5',    L6_channels, L5_channels),
        ('L6 → L4C',   L6_channels, L4C_channels),
        ('L5 → L4C',   L5_channels, L4C_channels),
        ('L5 → L4AB',  L5_channels, L4AB_channels),
        ('L4AB → L23', L4AB_channels, L23_channels),
        ('L6 → L23',   L6_channels, L23_channels),
    ]

    max_lag_ms      = 100
    max_lag_samples = int(max_lag_ms / 1000 * fs)

    fig, axes = plt.subplots(2, len(pairs), figsize=(6 * len(pairs), 10))

    for col, (pair_label, src_chs, tgt_chs) in enumerate(pairs):
        good_xcorrs, bad_xcorrs, peak_lags = [], [], []

        for i, trial in enumerate(all_trials):
            bp   = trial['bipolar_lfp']
            t    = trial['time']
            stim = trial['stim_onset_ms']
            n_ch = bp.shape[0]
            pre_mask = (t >= stim - pre_window_ms) & (t < stim)

            valid_src = [ch for ch in src_chs if ch < n_ch]
            valid_tgt = [ch for ch in tgt_chs if ch < n_ch]
            if not valid_src or not valid_tgt:
                peak_lags.append(np.nan)
                continue

            src_sig = mean_lfp(bp, valid_src)
            tgt_sig = mean_lfp(bp, valid_tgt)

            _, src_env, _ = inst_phase_amp(detrend(src_sig[pre_mask]), *alpha_band, fs)
            _, tgt_env, _ = inst_phase_amp(detrend(tgt_sig[pre_mask]), *alpha_band, fs)

            src_env = (src_env - np.mean(src_env)) / (np.std(src_env) + 1e-10)
            tgt_env = (tgt_env - np.mean(tgt_env)) / (np.std(tgt_env) + 1e-10)

            xcorr = np.correlate(tgt_env, src_env, mode='full') / len(src_env)
            lags  = np.arange(-len(src_env) + 1, len(src_env))
            lag_mask      = np.abs(lags) <= max_lag_samples
            xcorr_trimmed = xcorr[lag_mask]
            lags_trimmed  = lags[lag_mask] / fs * 1000  # ms

            peak_lags.append(lags_trimmed[np.argmax(xcorr_trimmed)])

            (good_xcorrs if good_mask[i] else bad_xcorrs).append(xcorr_trimmed)

        peak_lags = np.array(peak_lags)

        # Row 0: average cross-correlation
        ax = axes[0, col]
        for traces, color, label in [(good_xcorrs, 'green', 'Good'),
                                      (bad_xcorrs,  'red',   'Bad')]:
            if not traces:
                continue
            min_len    = min(len(x) for x in traces)
            traces_arr = np.array([x[:min_len] for x in traces])
            lags_plot  = np.linspace(-max_lag_ms, max_lag_ms, min_len)
            mn = np.mean(traces_arr, axis=0)
            se = sem(traces_arr,  axis=0)
            ax.plot(lags_plot, mn, color=color, lw=2, label=label)
            ax.fill_between(lags_plot, mn - se, mn + se, color=color, alpha=0.15)

        ax.axvline(0, color='black', ls='--', lw=1)
        ax.set_xlabel('Lag (ms)\n(+ = source leads)')
        ax.set_ylabel('Cross-correlation')
        ax.set_title(f'{pair_label}\nAlpha envelope cross-corr')
        ax.legend(fontsize=9)

        # Row 1: peak lag vs alpha change
        ax = axes[1, col]
        valid = ~np.isnan(peak_lags)
        if np.sum(valid) > 5:
            ax.scatter(peak_lags[valid], alpha_changes[valid],
                       c=['green' if g else 'red' for g in good_mask[valid]],
                       edgecolors='k', s=35, alpha=0.7)
            r, p = pearsonr(peak_lags[valid], alpha_changes[valid])
            ax.text(0.05, 0.95, f'r={r:.2f}\np={p:.4f}',
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlabel('Peak lag (ms, + = source leads)')
        ax.set_ylabel('Mean alpha change (%)')
        ax.set_title('Peak Lag vs Suppression')
        ax.axvline(0, color='gray', ls='--', alpha=0.5)

    plt.suptitle('Analysis 6: Temporal Precedence — Alpha Envelope Cross-Correlation\n'
                 f'Predicting Broad Alpha Suppression', fontsize=15)
    plt.tight_layout()
    plt.savefig('broad_06_temporal_precedence.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# ANALYSIS 7: L5/L6 E & SOM RATE vs L4 ALPHA POWER
# ============================================================
def analysis_E_rate_vs_alpha(all_trials, alpha_changes, good_mask):
    """
    Does E-cell / SOM rate in each layer correlate with local alpha power?
    Tested pre- and post-stimulus.
    """
    layer_defs = [
        ('L6',   L6_channels),
        ('L5',   L5_channels),
        ('L4C',  L4C_channels),
        ('L4AB', L4AB_channels),
        ('L23',  L23_channels),
    ]
    periods  = [('Pre-stim', True), ('Post-stim', False)]

    n_cols = len(periods)
    fig, axes = plt.subplots(len(layer_defs) * 2, n_cols,
                              figsize=(6 * n_cols, 6 * len(layer_defs) * 2))

    for row_block, (layer, tgt_chs) in enumerate(layer_defs):
        for per_idx, (per_label, is_pre) in enumerate(periods):
            col = per_idx

            e_rates  = []
            som_rates = []
            local_alpha = []

            for trial in all_trials:
                rd   = trial['rate_data']
                stim = trial['stim_onset_ms']
                bp   = trial['bipolar_lfp']
                t    = trial['time']
                n_ch = bp.shape[0]

                if is_pre:
                    t_mask_lfp = (t >= stim - pre_window_ms) & (t < stim)
                else:
                    t_mask_lfp = (t >= stim + post_start_ms) & \
                                 (t < stim + post_start_ms + post_window_ms)

                e_key   = 'E_rate'
                som_key = 'SOM_rate'
                if layer not in rd or e_key not in rd[layer]:
                    e_rates.append(np.nan)
                    som_rates.append(np.nan)
                    local_alpha.append(np.nan)
                    continue

                t_r = rd[layer][e_key]['t_ms']
                r_e = rd[layer][e_key]['rate_hz']
                r_s = rd[layer][som_key]['rate_hz'] if som_key in rd[layer] \
                      else np.zeros_like(r_e)

                r_mask = (t_r >= stim - pre_window_ms) & (t_r < stim) if is_pre \
                         else (t_r >= stim + post_start_ms) & \
                              (t_r < stim + post_start_ms + post_window_ms)

                e_rates.append(np.mean(r_e[r_mask])  if np.sum(r_mask) > 0 else np.nan)
                som_rates.append(np.mean(r_s[r_mask]) if np.sum(r_mask) > 0 else np.nan)

                valid_tgt = [ch for ch in tgt_chs if ch < n_ch]
                a_pow = [band_power(detrend(bp[ch][t_mask_lfp]), *alpha_band, fs)
                         for ch in valid_tgt]
                local_alpha.append(np.mean(a_pow) if a_pow else np.nan)

            e_rates   = np.array(e_rates)
            som_rates = np.array(som_rates)
            local_alpha = np.array(local_alpha)

            # E-rate subplot (even row)
            ax = axes[row_block * 2, col]
            valid = ~(np.isnan(e_rates) | np.isnan(local_alpha))
            ax.scatter(e_rates[valid], local_alpha[valid],
                       c=['green' if g else 'red' for g in good_mask[valid]],
                       edgecolors='k', s=35, alpha=0.7)
            ax.set_xlabel(f'{layer} E rate (Hz)')
            ax.set_ylabel(f'{layer} alpha power')
            ax.set_title(f'{per_label}: {layer} E → {layer} Alpha')
            if np.sum(valid) > 5:
                r, p = pearsonr(e_rates[valid], local_alpha[valid])
                ax.text(0.05, 0.95, f'r={r:.2f}\np={p:.4f}',
                        transform=ax.transAxes, va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # SOM-rate subplot (odd row)
            ax = axes[row_block * 2 + 1, col]
            valid_s = ~(np.isnan(som_rates) | np.isnan(local_alpha))
            ax.scatter(som_rates[valid_s], local_alpha[valid_s],
                       c=['green' if g else 'red' for g in good_mask[valid_s]],
                       edgecolors='k', s=35, alpha=0.7)
            ax.set_xlabel(f'{layer} SOM rate (Hz)')
            ax.set_ylabel(f'{layer} alpha power')
            ax.set_title(f'{per_label}: {layer} SOM → {layer} Alpha')
            if np.sum(valid_s) > 5:
                r, p = pearsonr(som_rates[valid_s], local_alpha[valid_s])
                ax.text(0.05, 0.95, f'r={r:.2f}\np={p:.4f}',
                        transform=ax.transAxes, va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Analysis 7: E & SOM Rates vs Local Alpha Power (All Layers)\n'
                 f'Pre- and Post-stimulus', fontsize=15)
    plt.tight_layout()
    plt.savefig('broad_07_E_SOM_rate_vs_alpha.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# ANALYSIS 8: SUMMARY DASHBOARD
# ============================================================
def analysis_summary(all_trials, alpha_changes, good_mask):
    """
    Collect features from all layers and rank them by predictive power
    for broad alpha suppression (majority of layers).
    """
    n = len(all_trials)

    all_layer_names = ['L6', 'L5', 'L4C', 'L4AB', 'L23']
    all_layer_chs   = [L6_channels, L5_channels, L4C_channels,
                       L4AB_channels, L23_channels]
    cell_types      = ['E', 'PV', 'SOM', 'VIP']

    # Pre-allocate feature arrays
    alpha_pre = {lname: np.full(n, np.nan) for lname in all_layer_names}
    rate_pre  = {lname: {ct: np.full(n, np.nan) for ct in cell_types}
                 for lname in all_layer_names}

    # Coherence between adjacent layer pairs
    coh_pairs = [
        ('L6', 'L5',   L6_channels, L5_channels),
        ('L5', 'L4C',  L5_channels, L4C_channels),
        ('L4C', 'L4AB', L4C_channels, L4AB_channels),
        ('L4AB', 'L23', L4AB_channels, L23_channels),
        ('L6', 'L4C',  L6_channels, L4C_channels),
        ('L6', 'L23',  L6_channels, L23_channels),
    ]
    coh_pre = {f'{a}↔{b}': np.full(n, np.nan) for a, b, _, _ in coh_pairs}

    for i, trial in enumerate(all_trials):
        bp   = trial['bipolar_lfp']
        t    = trial['time']
        stim = trial['stim_onset_ms']
        rd   = trial['rate_data']
        n_ch = bp.shape[0]
        pre_mask = (t >= stim - pre_window_ms) & (t < stim)

        # LFP alpha power per layer
        for lname, chs in zip(all_layer_names, all_layer_chs):
            valid_chs = [ch for ch in chs if ch < n_ch]
            if valid_chs:
                alpha_pre[lname][i] = np.mean(
                    [band_power(detrend(bp[ch][pre_mask]), *alpha_band, fs)
                     for ch in valid_chs])

        # Firing rates per layer per cell type
        for lname in all_layer_names:
            for ct in cell_types:
                mon_key = f'{ct}_rate'
                if lname in rd and mon_key in rd[lname]:
                    t_r  = rd[lname][mon_key]['t_ms']
                    r_hz = rd[lname][mon_key]['rate_hz']
                    r_mask = (t_r >= stim - pre_window_ms) & (t_r < stim)
                    if np.sum(r_mask) > 0:
                        rate_pre[lname][ct][i] = np.mean(r_hz[r_mask])

        # Coherence
        for a, b, a_chs, b_chs in coh_pairs:
            valid_a = [ch for ch in a_chs if ch < n_ch]
            valid_b = [ch for ch in b_chs if ch < n_ch]
            if valid_a and valid_b:
                a_sig = detrend(mean_lfp(bp, valid_a)[pre_mask])
                b_sig = detrend(mean_lfp(bp, valid_b)[pre_mask])
                nperseg = min(len(a_sig), 2048)
                f_c, coh = coherence(a_sig, b_sig, fs=fs, nperseg=nperseg)
                amask = (f_c >= alpha_band[0]) & (f_c <= alpha_band[1])
                coh_pre[f'{a}↔{b}'][i] = np.mean(coh[amask])

    # Build flat feature dict
    all_features = {}
    for lname in all_layer_names:
        all_features[f'{lname} alpha power'] = alpha_pre[lname]
    for lname in all_layer_names:
        for ct in cell_types:
            all_features[f'{lname} {ct} rate'] = rate_pre[lname][ct]
    for key, arr in coh_pre.items():
        all_features[f'{key} α coh'] = arr

    # Compute correlations
    results = []
    for feat_name, feat_vals in all_features.items():
        valid = ~np.isnan(feat_vals)
        if np.sum(valid) > 10:
            r, p = pearsonr(feat_vals[valid], alpha_changes[valid])
            good_v = feat_vals[valid & good_mask]
            bad_v  = feat_vals[valid & ~good_mask]
            if len(good_v) > 1 and len(bad_v) > 1:
                _, p_mw = mannwhitneyu(good_v, bad_v)
                d = np.abs(np.mean(good_v) - np.mean(bad_v)) / \
                    np.sqrt((np.var(good_v) + np.var(bad_v)) / 2 + 1e-30)
            else:
                p_mw, d = 1.0, 0.0
            results.append((feat_name, r, p, p_mw, d))

    results.sort(key=lambda x: abs(x[1]), reverse=True)

    # ---- Plots ----
    fig_h = max(8, 0.35 * len(results))
    fig, axes = plt.subplots(1, 2, figsize=(22, fig_h))

    # Left: bar chart of correlations
    ax = axes[0]
    names     = [r[0] for r in results]
    corrs     = [r[1] for r in results]
    p_vals    = [r[2] for r in results]
    bar_colors = ['darkgreen' if p < 0.01 else 'green' if p < 0.05
                  else 'orange' if p < 0.1 else 'gray' for p in p_vals]

    y_pos = range(len(names))
    ax.barh(y_pos, corrs, color=bar_colors, edgecolor='k', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Pearson r with mean alpha change (%)')
    ax.set_title('Feature Correlations with Broad Alpha Suppression\n'
                 '(dark green=p<0.01, green=p<0.05, orange=p<0.1, gray=ns)')
    ax.axvline(0, color='black', lw=1)
    for i, (name, r, p, p_mw, d) in enumerate(results):
        ax.text(r + 0.02 * np.sign(r), i, f'r={r:.2f}\np={p:.3f}',
                va='center', fontsize=8)
    ax.invert_yaxis()

    # Right: feature correlation matrix
    ax = axes[1]
    feat_names_short = [r[0][:13] for r in results] + ['α_chg']
    feat_matrix = np.column_stack([all_features[r[0]] for r in results])
    full = np.column_stack([feat_matrix, alpha_changes])
    valid_all = ~np.any(np.isnan(full), axis=1)
    corr_mat  = np.corrcoef(full[valid_all].T)

    im = ax.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(feat_names_short)))
    ax.set_yticks(range(len(feat_names_short)))
    ax.set_xticklabels(feat_names_short, rotation=60, ha='right', fontsize=7)
    ax.set_yticklabels(feat_names_short, fontsize=7)
    for ii in range(len(feat_names_short)):
        for jj in range(len(feat_names_short)):
            ax.text(jj, ii, f'{corr_mat[ii, jj]:.2f}',
                    ha='center', va='center', fontsize=5,
                    color='white' if abs(corr_mat[ii, jj]) > 0.5 else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title('Feature Correlation Matrix')

    plt.suptitle(f'Analysis 8: Summary — Predictors of Broad Alpha Suppression\n'
                 f'(>={min_layers_suppressed} layers with alpha decrease)',
                 fontsize=15)
    plt.tight_layout()
    plt.savefig('broad_08_summary.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print ranked table
    print("\n" + "=" * 72)
    print(f"RANKED FEATURES: Predicting Broad Alpha Suppression")
    print("=" * 72)
    print(f"{'Feature':<25s} {'r':>7s} {'p':>8s}    {'MW p':>8s}  {'d':>6s}")
    print("-" * 72)
    for name, r, p, p_mw, d in results:
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"{name:<25s} {r:+.3f}  {p:.4f} {sig:3s}  {p_mw:.4f}  {d:.2f}")
    print("=" * 72)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Loading trials...")
    all_trials = load_all_trials(base_path, n_trials)

    print(f"\nClassifying trials by broad alpha suppression "
          f"(>={min_layers_suppressed} of {len(LAYER_DEFS)} layers)...")
    metrics, alpha_changes, good_mask, bad_mask = classify_trials(
        all_trials, LAYER_DEFS, alpha_band,
        pre_window_ms, post_window_ms, post_start_ms, fs,
        alpha_threshold, min_layers_suppressed
    )

    print("\n" + "=" * 60)
    print("ANALYSIS 1: Per-Layer Pre-stimulus Alpha Power")
    print("=" * 60)
    analysis_prestim_alpha(all_trials, alpha_changes, good_mask)

    print("\n" + "=" * 60)
    print("ANALYSIS 2: Inter-Layer Alpha Coherence")
    print("=" * 60)
    analysis_alpha_coherence(all_trials, alpha_changes, good_mask)

    print("\n" + "=" * 60)
    print("ANALYSIS 3: Alpha Phase → Amplitude Coupling (PAC)")
    print("=" * 60)
    analysis_deep_L4_pac(all_trials, alpha_changes, good_mask)

    print("\n" + "=" * 60)
    print("ANALYSIS 4: Time-Resolved Firing Rates (All Layers)")
    print("=" * 60)
    analysis_firing_rates(all_trials, alpha_changes, good_mask)

    print("\n" + "=" * 60)
    print("ANALYSIS 5: SOM as Alpha Generator")
    print("=" * 60)
    analysis_som_alpha_link(all_trials, alpha_changes, good_mask)

    print("\n" + "=" * 60)
    print("ANALYSIS 6: Temporal Precedence")
    print("=" * 60)
    analysis_temporal_precedence(all_trials, alpha_changes, good_mask)

    print("\n" + "=" * 60)
    print("ANALYSIS 7: E & SOM Rates vs Local Alpha")
    print("=" * 60)
    analysis_E_rate_vs_alpha(all_trials, alpha_changes, good_mask)

    print("\n" + "=" * 60)
    print("ANALYSIS 8: Summary Dashboard")
    print("=" * 60)
    analysis_summary(all_trials, alpha_changes, good_mask)

    print("\nAll broad alpha suppression analyses complete!")