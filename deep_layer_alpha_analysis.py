"""
Deep-Layer Alpha Modulation Analysis
=====================================
Testing the hypothesis that L5/L6 activity modulates L2/3 alpha,
and that trial-to-trial variability in L2/3 alpha suppression
is explained by deep-layer dynamics.

Analyses:
1.  Pre-stim deep-layer alpha power → L2/3 alpha suppression
2.  Pre-stim alpha coherence between deep and superficial layers
3.  Deep-to-superficial alpha phase-amplitude coupling (PAC)
4.  L5/L6 firing rate dynamics: time-resolved good vs bad
5.  Deep-layer SOM activity as alpha generator proxy
6.  Cross-frequency: deep alpha phase → superficial gamma amplitude
7.  Granger-like temporal precedence (deep alpha leads superficial?)
8.  L5/L6 E and SOM rate correlation with L2/3 alpha power (per trial)
9.  Deep-layer alpha phase coherence (L5↔L2/3, L6↔L2/3)
10. Summary dashboard: which deep-layer feature best predicts outcome

Usage:
    Set base_path, n_trials, and channel assignments to match your data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert, detrend, sosfiltfilt, butter, coherence, csd
from scipy.signal.windows import dpss
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, sem, circmean
from matplotlib.colors import TwoSlopeNorm
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
base_path = "results/trials_27_03_2"
n_trials = 21
fs = 10000

alpha_band = (7, 14)
gamma_band = (30, 80)
beta_band = (15, 30)

pre_window_ms = 500
post_window_ms = 500
post_start_ms = 200

alpha_threshold = -20  # % change to classify good vs bad

# Channel assignments (bipolar channels, 0-indexed)
# Based on your 16-electrode layout with z from -0.94 to 1.30
# Bipolar channels = 15 channels (differences between adjacent electrodes)
# Deep channels (L6/L5 region): channels 0-4
# Granular (L4): channels 5-8
# Superficial (L2/3): channels 11-14
#
# More specific mapping for deep-layer separation:
# L6 ~ electrodes at z = -0.94 to -0.34 → bipolar channels 0-2
# L5 ~ electrodes at z = -0.34 to -0.14 → bipolar channels 3-4
# L2/3 ~ electrodes at z = 0.45 to 1.30 → bipolar channels 11-14

L6_channels = [0, 1, 2]
L5_channels = [3, 4]
L4_channels = [5, 6, 7, 8]
L23_channels = [11, 12, 13, 14]  # or [-1, -2, -3, -4]


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


def circ_r(angles):
    return np.abs(np.mean(np.exp(1j * np.array(angles))))


def ppc(phases):
    n = len(phases)
    if n < 2:
        return np.nan
    z = np.exp(1j * np.array(phases))
    S = np.sum(z)
    return (np.real(S)**2 + np.imag(S)**2 - n) / (n * (n - 1))


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
            'trial_idx': idx,
            'time': data['time_array_ms'],
            'bipolar_lfp': data['bipolar_matrix'],
            'lfp_matrix': data['lfp_matrix'],
            'rate_data': data['rate_data'].item() if data['rate_data'].size == 1 else data['rate_data'],
            'stim_onset_ms': float(data['stim_onset_ms']),
            'baseline_ms': float(data['baseline_ms']),
            'channel_depths': data['channel_depths'],
            'electrode_positions': data['electrode_positions'],
        }
        all_trials.append(trial)
    print(f"Loaded {len(all_trials)} trials")
    return all_trials


def classify_trials(all_trials, sup_chs, alpha_band, pre_window_ms,
                     post_window_ms, post_start_ms, fs, threshold):
    """Compute alpha change in superficial channels, classify good/bad."""
    metrics = []
    for trial in all_trials:
        bp = trial['bipolar_lfp']
        t = trial['time']
        stim = trial['stim_onset_ms']

        pre_mask = (t >= stim - pre_window_ms) & (t < stim)
        post_mask = (t >= stim + post_start_ms) & (t < stim + post_start_ms + post_window_ms)

        pre_a, post_a = [], []
        for ch in sup_chs:
            pre_a.append(band_power(detrend(bp[ch][pre_mask]), *alpha_band, fs))
            post_a.append(band_power(detrend(bp[ch][post_mask]), *alpha_band, fs))

        mp = np.mean(pre_a)
        mpo = np.mean(post_a)
        chg = (mpo - mp) / (mp + 1e-30) * 100

        metrics.append({
            'trial_idx': trial['trial_idx'],
            'alpha_pre': mp,
            'alpha_post': mpo,
            'alpha_change_pct': chg,
        })

    alpha_changes = np.array([m['alpha_change_pct'] for m in metrics])
    good_mask = alpha_changes < threshold
    bad_mask = ~good_mask
    print(f"Good trials (α↓): {np.sum(good_mask)}, Bad trials: {np.sum(bad_mask)}")
    return metrics, alpha_changes, good_mask, bad_mask


# ============================================================
# ANALYSIS 1: DEEP-LAYER PRE-STIM ALPHA POWER
# ============================================================
def analysis_deep_prestim_alpha(all_trials, alpha_changes, good_mask):
    """
    Does pre-stimulus alpha power in L5 and L6 (separately)
    predict L2/3 alpha suppression?
    """
    layer_defs = {
        'L6': L6_channels,
        'L5': L5_channels,
        'L5+L6': L5_channels + L6_channels,
        'L2/3': L23_channels,
    }

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for col, (layer_label, chs) in enumerate(layer_defs.items()):
        pre_alpha = []
        pre_gamma = []
        for trial in all_trials:
            bp = trial['bipolar_lfp']
            t = trial['time']
            stim = trial['stim_onset_ms']
            pre_mask = (t >= stim - pre_window_ms) & (t < stim)

            ch_a, ch_g = [], []
            n_ch = bp.shape[0]
            for ch in chs:
                if ch < n_ch:
                    sig = detrend(bp[ch][pre_mask])
                    ch_a.append(band_power(sig, *alpha_band, fs))
                    ch_g.append(band_power(sig, *gamma_band, fs))
            pre_alpha.append(np.mean(ch_a) if ch_a else 0)
            pre_gamma.append(np.mean(ch_g) if ch_g else 0)

        pre_alpha = np.array(pre_alpha)
        pre_gamma = np.array(pre_gamma)

        # Row 0: pre-stim alpha vs L2/3 alpha change
        ax = axes[0, col]
        ax.scatter(pre_alpha, alpha_changes,
                   c=['green' if g else 'red' for g in good_mask],
                   edgecolors='k', s=35, alpha=0.7)
        ax.set_xlabel(f'{layer_label} pre-stim alpha power')
        ax.set_ylabel('L2/3 alpha change (%)')
        ax.set_title(f'{layer_label} Alpha → L2/3 Suppression')
        r, p = pearsonr(pre_alpha, alpha_changes)
        ax.text(0.05, 0.95, f'r = {r:.2f}, p = {p:.4f}',
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axhline(0, color='gray', ls='--', alpha=0.3)

        # Row 1: boxplot good vs bad
        ax = axes[1, col]
        good_vals = pre_alpha[good_mask]
        bad_vals = pre_alpha[~good_mask]
        bp_plot = ax.boxplot([good_vals, bad_vals],
                              labels=['Good\n(α↓)', 'Bad\n(no α↓)'],
                              patch_artist=True)
        bp_plot['boxes'][0].set_facecolor('green'); bp_plot['boxes'][0].set_alpha(0.4)
        bp_plot['boxes'][1].set_facecolor('red'); bp_plot['boxes'][1].set_alpha(0.4)
        ax.set_ylabel(f'{layer_label} pre-stim alpha power')
        if len(good_vals) > 1 and len(bad_vals) > 1:
            _, p_mw = mannwhitneyu(good_vals, bad_vals)
            d = np.abs(np.mean(good_vals) - np.mean(bad_vals)) / \
                np.sqrt((np.var(good_vals) + np.var(bad_vals))/2 + 1e-30)
            ax.text(0.5, 0.95, f'MW p={p_mw:.4f}\nd={d:.2f}',
                    transform=ax.transAxes, ha='center', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Analysis 1: Deep-Layer Pre-stimulus Alpha Power\n'
                 'Predicting L2/3 Alpha Suppression', fontsize=15)
    plt.tight_layout()
    plt.savefig('deep_01_prestim_alpha_power.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# ANALYSIS 2: DEEP ↔ SUPERFICIAL ALPHA COHERENCE
# ============================================================
def analysis_alpha_coherence(all_trials, alpha_changes, good_mask):
    """
    Pre-stimulus alpha-band coherence between deep (L5, L6) and L2/3.
    Does stronger deep-superficial coupling predict better suppression?
    """
    pairs = [
        ('L5 ↔ L2/3', L5_channels, L23_channels),
        ('L6 ↔ L2/3', L6_channels, L23_channels),
        ('L5+L6 ↔ L2/3', L5_channels + L6_channels, L23_channels),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, (pair_label, deep_chs, sup_chs) in enumerate(pairs):
        coh_alpha_pre = []
        coh_alpha_post = []
        coh_full_pre = []  # for spectrum plot
        coh_full_post = []

        for trial in all_trials:
            bp = trial['bipolar_lfp']
            t = trial['time']
            stim = trial['stim_onset_ms']
            n_ch = bp.shape[0]

            pre_mask = (t >= stim - pre_window_ms) & (t < stim)
            post_mask = (t >= stim + post_start_ms) & \
                        (t < stim + post_start_ms + post_window_ms)

            valid_deep = [ch for ch in deep_chs if ch < n_ch]
            valid_sup = [ch for ch in sup_chs if ch < n_ch]
            if not valid_deep or not valid_sup:
                coh_alpha_pre.append(np.nan)
                coh_alpha_post.append(np.nan)
                continue

            deep_pre = np.mean([bp[ch][pre_mask] for ch in valid_deep], axis=0)
            sup_pre = np.mean([bp[ch][pre_mask] for ch in valid_sup], axis=0)
            deep_post = np.mean([bp[ch][post_mask] for ch in valid_deep], axis=0)
            sup_post = np.mean([bp[ch][post_mask] for ch in valid_sup], axis=0)

            nperseg = min(len(deep_pre), 2048)

            f_c, c_pre = coherence(detrend(deep_pre), detrend(sup_pre),
                                    fs=fs, nperseg=nperseg)
            _, c_post = coherence(detrend(deep_post), detrend(sup_post),
                                   fs=fs, nperseg=nperseg)

            alpha_mask = (f_c >= alpha_band[0]) & (f_c <= alpha_band[1])
            coh_alpha_pre.append(np.mean(c_pre[alpha_mask]))
            coh_alpha_post.append(np.mean(c_post[alpha_mask]))
            coh_full_pre.append(c_pre)
            coh_full_post.append(c_post)

        coh_alpha_pre = np.array(coh_alpha_pre)
        coh_alpha_post = np.array(coh_alpha_post)

        # Row 0: coherence spectrum (good vs bad, pre-stim)
        ax = axes[0, col]
        if coh_full_pre:
            good_spec = [coh_full_pre[i] for i in range(len(coh_full_pre)) if good_mask[i]]
            bad_spec = [coh_full_pre[i] for i in range(len(coh_full_pre)) if not good_mask[i]]

            freq_mask = f_c <= 100
            if good_spec:
                mean_g = np.mean(good_spec, axis=0)[freq_mask]
                sem_g = sem(good_spec, axis=0)[freq_mask]
                ax.plot(f_c[freq_mask], mean_g, 'green', lw=2, label='Good')
                ax.fill_between(f_c[freq_mask], mean_g-sem_g, mean_g+sem_g,
                                color='green', alpha=0.15)
            if bad_spec:
                mean_b = np.mean(bad_spec, axis=0)[freq_mask]
                sem_b = sem(bad_spec, axis=0)[freq_mask]
                ax.plot(f_c[freq_mask], mean_b, 'red', lw=2, label='Bad')
                ax.fill_between(f_c[freq_mask], mean_b-sem_b, mean_b+sem_b,
                                color='red', alpha=0.15)

        ax.axvspan(*alpha_band, color='purple', alpha=0.08, label='Alpha')
        ax.axvspan(*gamma_band, color='orange', alpha=0.05, label='Gamma')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Coherence')
        ax.set_title(f'{pair_label}\nPre-stim coherence spectrum')
        ax.legend(fontsize=8)
        ax.set_xlim(1, 100)

        # Row 1: pre-stim alpha coherence vs L2/3 alpha change
        ax = axes[1, col]
        valid = ~np.isnan(coh_alpha_pre)
        ax.scatter(coh_alpha_pre[valid], alpha_changes[valid],
                   c=['green' if g else 'red' for g in good_mask[valid]],
                   edgecolors='k', s=35, alpha=0.7)
        ax.set_xlabel(f'{pair_label} alpha coherence')
        ax.set_ylabel('L2/3 alpha change (%)')
        ax.set_title(f'Pre-stim Alpha Coherence → Suppression')
        if np.sum(valid) > 5:
            r, p = pearsonr(coh_alpha_pre[valid], alpha_changes[valid])
            ax.text(0.05, 0.95, f'r = {r:.2f}, p = {p:.4f}',
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Analysis 2: Deep ↔ Superficial Alpha Coherence', fontsize=15)
    plt.tight_layout()
    plt.savefig('deep_02_alpha_coherence.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# ANALYSIS 3: DEEP ALPHA PHASE → SUPERFICIAL ALPHA AMPLITUDE (PAC)
# ============================================================
def analysis_deep_sup_pac(all_trials, alpha_changes, good_mask):
    """
    Phase-amplitude coupling: does deep-layer alpha phase modulate
    superficial alpha/gamma amplitude?

    Compute modulation index (MI) using Tort et al. (2010) method.
    """
    def modulation_index(phase_sig, amp_sig, n_bins=18):
        """Tort's MI: KL divergence of amplitude distribution over phases."""
        bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        mean_amp = np.zeros(n_bins)
        for b in range(n_bins):
            mask = (phase_sig >= bin_edges[b]) & (phase_sig < bin_edges[b+1])
            if np.sum(mask) > 0:
                mean_amp[b] = np.mean(amp_sig[mask])

        if np.sum(mean_amp) == 0:
            return 0.0, mean_amp

        # Normalize to probability distribution
        p = mean_amp / np.sum(mean_amp)
        p = p[p > 0]  # avoid log(0)
        uniform = 1.0 / n_bins

        # KL divergence
        kl = np.sum(p * np.log(p / uniform))
        mi = kl / np.log(n_bins)
        return mi, mean_amp

    # Compute for each trial
    mi_deep_alpha_sup_alpha = {'L5': [], 'L6': []}
    mi_deep_alpha_sup_gamma = {'L5': [], 'L6': []}
    amp_dist_alpha = {'L5': {'good': [], 'bad': []}, 'L6': {'good': [], 'bad': []}}
    amp_dist_gamma = {'L5': {'good': [], 'bad': []}, 'L6': {'good': [], 'bad': []}}

    deep_defs = {'L5': L5_channels, 'L6': L6_channels}

    for i, trial in enumerate(all_trials):
        bp = trial['bipolar_lfp']
        t = trial['time']
        stim = trial['stim_onset_ms']
        n_ch = bp.shape[0]

        pre_mask = (t >= stim - pre_window_ms) & (t < stim)

        sup_sig = mean_lfp(bp, [ch for ch in L23_channels if ch < n_ch])
        sup_pre = detrend(sup_sig[pre_mask])

        _, sup_alpha_amp, _ = inst_phase_amp(sup_pre, *alpha_band, fs)
        _, sup_gamma_amp, _ = inst_phase_amp(sup_pre, *gamma_band, fs)

        for deep_label, deep_chs in deep_defs.items():
            valid_chs = [ch for ch in deep_chs if ch < n_ch]
            if not valid_chs:
                mi_deep_alpha_sup_alpha[deep_label].append(np.nan)
                mi_deep_alpha_sup_gamma[deep_label].append(np.nan)
                continue

            deep_sig = mean_lfp(bp, valid_chs)
            deep_pre = detrend(deep_sig[pre_mask])

            deep_phase, _, _ = inst_phase_amp(deep_pre, *alpha_band, fs)

            mi_a, amp_a = modulation_index(deep_phase, sup_alpha_amp)
            mi_g, amp_g = modulation_index(deep_phase, sup_gamma_amp)

            mi_deep_alpha_sup_alpha[deep_label].append(mi_a)
            mi_deep_alpha_sup_gamma[deep_label].append(mi_g)

            group = 'good' if good_mask[i] else 'bad'
            amp_dist_alpha[deep_label][group].append(amp_a)
            amp_dist_gamma[deep_label][group].append(amp_g)

    # ---- Plot ----
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))

    n_bins = 18
    bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False) + np.pi/n_bins
    bin_centers_deg = np.rad2deg(bin_centers)

    for col_offset, deep_label in enumerate(['L5', 'L6']):
        mi_alpha = np.array(mi_deep_alpha_sup_alpha[deep_label])
        mi_gamma = np.array(mi_deep_alpha_sup_gamma[deep_label])

        # Row 0, col 0+offset: MI alpha vs alpha change
        ax = axes[0, col_offset * 2]
        valid = ~np.isnan(mi_alpha)
        ax.scatter(mi_alpha[valid], alpha_changes[valid],
                   c=['green' if g else 'red' for g in good_mask[valid]],
                   edgecolors='k', s=35, alpha=0.7)
        ax.set_xlabel(f'{deep_label} α-phase → L2/3 α-amp MI')
        ax.set_ylabel('L2/3 alpha change (%)')
        ax.set_title(f'{deep_label} Alpha→Alpha PAC')
        if np.sum(valid) > 5:
            r, p = pearsonr(mi_alpha[valid], alpha_changes[valid])
            ax.text(0.05, 0.95, f'r={r:.2f}, p={p:.4f}',
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Row 0, col 1+offset: MI gamma vs alpha change
        ax = axes[0, col_offset * 2 + 1]
        valid = ~np.isnan(mi_gamma)
        ax.scatter(mi_gamma[valid], alpha_changes[valid],
                   c=['green' if g else 'red' for g in good_mask[valid]],
                   edgecolors='k', s=35, alpha=0.7)
        ax.set_xlabel(f'{deep_label} α-phase → L2/3 γ-amp MI')
        ax.set_ylabel('L2/3 alpha change (%)')
        ax.set_title(f'{deep_label} Alpha→Gamma PAC')
        if np.sum(valid) > 5:
            r, p = pearsonr(mi_gamma[valid], alpha_changes[valid])
            ax.text(0.05, 0.95, f'r={r:.2f}, p={p:.4f}',
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Row 1: amplitude distribution over phase bins (good vs bad)
        ax = axes[1, col_offset * 2]
        for group, color, label in [('good', 'green', 'Good'), ('bad', 'red', 'Bad')]:
            dists = amp_dist_alpha[deep_label][group]
            if len(dists) > 0:
                mean_d = np.mean(dists, axis=0)
                if np.sum(mean_d) > 0:
                    mean_d = mean_d / np.sum(mean_d)  # normalize
                sem_d = sem(dists, axis=0)
                if np.sum(sem_d) > 0:
                    sem_d = sem_d / np.sum(np.mean(dists, axis=0) + 1e-30)
                ax.plot(bin_centers_deg, mean_d, color=color, lw=2, label=label)
                ax.fill_between(bin_centers_deg, mean_d - sem_d, mean_d + sem_d,
                                color=color, alpha=0.15)
        ax.axhline(1.0/n_bins, color='gray', ls='--', alpha=0.5, label='Uniform')
        ax.set_xlabel(f'{deep_label} alpha phase (deg)')
        ax.set_ylabel('Normalized L2/3 alpha amplitude')
        ax.set_title(f'L2/3 Alpha Amp by {deep_label} Alpha Phase')
        ax.legend(fontsize=8)

        ax = axes[1, col_offset * 2 + 1]
        for group, color, label in [('good', 'green', 'Good'), ('bad', 'red', 'Bad')]:
            dists = amp_dist_gamma[deep_label][group]
            if len(dists) > 0:
                mean_d = np.mean(dists, axis=0)
                if np.sum(mean_d) > 0:
                    mean_d = mean_d / np.sum(mean_d)
                sem_d = sem(dists, axis=0)
                if np.sum(sem_d) > 0:
                    sem_d = sem_d / np.sum(np.mean(dists, axis=0) + 1e-30)
                ax.plot(bin_centers_deg, mean_d, color=color, lw=2, label=label)
                ax.fill_between(bin_centers_deg, mean_d - sem_d, mean_d + sem_d,
                                color=color, alpha=0.15)
        ax.axhline(1.0/n_bins, color='gray', ls='--', alpha=0.5, label='Uniform')
        ax.set_xlabel(f'{deep_label} alpha phase (deg)')
        ax.set_ylabel('Normalized L2/3 gamma amplitude')
        ax.set_title(f'L2/3 Gamma Amp by {deep_label} Alpha Phase')
        ax.legend(fontsize=8)

    plt.suptitle('Analysis 3: Deep Alpha Phase → Superficial Amplitude Coupling (PAC)',
                 fontsize=15)
    plt.tight_layout()
    plt.savefig('deep_03_pac.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# ANALYSIS 4: TIME-RESOLVED DEEP-LAYER FIRING RATES
# ============================================================
def analysis_deep_firing_rates(all_trials, alpha_changes, good_mask):
    """
    Time-resolved firing rates for L5 and L6 cell types,
    split by good vs bad trials.
    """
    layers = ['L5', 'L6']
    cell_types = ['E', 'PV', 'SOM', 'VIP']

    win_pre = 500
    win_post = 700

    fig, axes = plt.subplots(len(layers), len(cell_types),
                              figsize=(22, 5 * len(layers)), sharey='row')

    for row, layer in enumerate(layers):
        for col, ct in enumerate(cell_types):
            ax = axes[row, col]
            mon_key = f"{ct}_rate"

            good_traces, bad_traces = [], []

            for i, trial in enumerate(all_trials):
                rd = trial['rate_data']
                stim = trial['stim_onset_ms']

                if layer not in rd or mon_key not in rd[layer]:
                    continue

                t_ms = rd[layer][mon_key]['t_ms']
                r_hz = rd[layer][mon_key]['rate_hz']

                # Window
                mask = (t_ms >= stim - win_pre) & (t_ms < stim + win_post)
                if np.sum(mask) == 0:
                    continue

                t_rel = t_ms[mask] - stim
                rate_win = r_hz[mask]

                if good_mask[i]:
                    good_traces.append((t_rel, rate_win))
                else:
                    bad_traces.append((t_rel, rate_win))

            # Interpolate to common time grid
            t_common = np.linspace(-win_pre, win_post, 500)

            for traces, color, label in [(good_traces, 'green', 'Good'),
                                          (bad_traces, 'red', 'Bad')]:
                if len(traces) == 0:
                    continue
                interped = []
                for t_rel, rate_win in traces:
                    interped.append(np.interp(t_common, t_rel, rate_win))
                interped = np.array(interped)

                mean_r = np.mean(interped, axis=0)
                sem_r = sem(interped, axis=0)

                ax.plot(t_common, mean_r, color=color, lw=2, label=label)
                ax.fill_between(t_common, mean_r - sem_r, mean_r + sem_r,
                                color=color, alpha=0.15)

            ax.axvline(0, color='black', ls='--', lw=1, label='Stim')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Rate (Hz)')
            ax.set_title(f'{layer} {ct}')
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    plt.suptitle('Analysis 4: Time-Resolved Deep-Layer Firing Rates\n'
                 'Good (α↓) vs Bad Trials', fontsize=15)
    plt.tight_layout()
    plt.savefig('deep_04_firing_rates.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# ANALYSIS 5: L5/L6 SOM RATE AS ALPHA GENERATOR PROXY
# ============================================================
def analysis_som_alpha_link(all_trials, alpha_changes, good_mask):
    """
    SOM interneurons in L5/L6 are hypothesized as key to deep-layer
    alpha generation. Test: does L5/L6 SOM pre-stim rate correlate
    with L2/3 pre-stim alpha power? And does SOM rate change predict
    alpha suppression?
    """
    layers = ['L5', 'L6', 'L23']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, layer in enumerate(layers):
        som_pre_rates = []
        som_post_rates = []
        l23_alpha_pre = []

        for trial in all_trials:
            rd = trial['rate_data']
            stim = trial['stim_onset_ms']
            bp = trial['bipolar_lfp']
            t = trial['time']
            n_ch = bp.shape[0]

            # SOM rate
            mon_key = 'SOM_rate'
            if layer not in rd or mon_key not in rd[layer]:
                som_pre_rates.append(np.nan)
                som_post_rates.append(np.nan)
                l23_alpha_pre.append(np.nan)
                continue

            t_ms = rd[layer][mon_key]['t_ms']
            r_hz = rd[layer][mon_key]['rate_hz']

            pre_r_mask = (t_ms >= stim - pre_window_ms) & (t_ms < stim)
            post_r_mask = (t_ms >= stim + post_start_ms) & \
                          (t_ms < stim + post_start_ms + post_window_ms)

            som_pre_rates.append(np.mean(r_hz[pre_r_mask]) if np.sum(pre_r_mask) > 0 else np.nan)
            som_post_rates.append(np.mean(r_hz[post_r_mask]) if np.sum(post_r_mask) > 0 else np.nan)

            # L2/3 alpha power
            pre_lfp_mask = (t >= stim - pre_window_ms) & (t < stim)
            valid_sup = [ch for ch in L23_channels if ch < n_ch]
            a_pow = []
            for ch in valid_sup:
                a_pow.append(band_power(detrend(bp[ch][pre_lfp_mask]), *alpha_band, fs))
            l23_alpha_pre.append(np.mean(a_pow) if a_pow else np.nan)

        som_pre_rates = np.array(som_pre_rates)
        som_post_rates = np.array(som_post_rates)
        l23_alpha_pre = np.array(l23_alpha_pre)
        som_change = som_post_rates - som_pre_rates

        # Row 0: SOM pre-stim rate vs L2/3 alpha pre-stim power
        ax = axes[0, col]
        valid = ~(np.isnan(som_pre_rates) | np.isnan(l23_alpha_pre))
        if np.sum(valid) > 5:
            ax.scatter(som_pre_rates[valid], l23_alpha_pre[valid],
                       c=['green' if g else 'red' for g in good_mask[valid]],
                       edgecolors='k', s=35, alpha=0.7)
            r, p = pearsonr(som_pre_rates[valid], l23_alpha_pre[valid])
            ax.text(0.05, 0.95, f'r={r:.2f}, p={p:.4f}',
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlabel(f'{layer} SOM pre-stim rate (Hz)')
        ax.set_ylabel('L2/3 pre-stim alpha power')
        ax.set_title(f'{layer} SOM Rate → L2/3 Alpha Power')

        # Row 1: SOM rate change vs alpha suppression
        ax = axes[1, col]
        valid = ~np.isnan(som_change)
        if np.sum(valid) > 5:
            ax.scatter(som_change[valid], alpha_changes[valid],
                       c=['green' if g else 'red' for g in good_mask[valid]],
                       edgecolors='k', s=35, alpha=0.7)
            r, p = pearsonr(som_change[valid], alpha_changes[valid])
            ax.text(0.05, 0.95, f'r={r:.2f}, p={p:.4f}',
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlabel(f'{layer} SOM ΔRate (Hz)')
        ax.set_ylabel('L2/3 alpha change (%)')
        ax.set_title(f'{layer} SOM ΔRate → L2/3 Alpha Suppression')
        ax.axhline(0, color='gray', ls='--', alpha=0.3)
        ax.axvline(0, color='gray', ls='--', alpha=0.3)

    plt.suptitle('Analysis 5: SOM Interneurons as Alpha Generators\n'
                 'Linking deep SOM activity to L2/3 alpha', fontsize=15)
    plt.tight_layout()
    plt.savefig('deep_05_som_alpha.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# ANALYSIS 6: TEMPORAL PRECEDENCE (ALPHA LEAD/LAG)
# ============================================================
def analysis_temporal_precedence(all_trials, alpha_changes, good_mask):
    """
    Cross-correlation of alpha envelopes between deep and superficial.
    Does deep alpha lead superficial alpha? Does this differ for good vs bad?
    """
    pairs = [
        ('L5 → L2/3', L5_channels, L23_channels),
        ('L6 → L2/3', L6_channels, L23_channels),
        ('L6 → L5', L6_channels, L5_channels),
    ]

    max_lag_ms = 100  # ± 100 ms
    max_lag_samples = int(max_lag_ms / 1000 * fs)

    fig, axes = plt.subplots(2, len(pairs), figsize=(6 * len(pairs), 10))

    for col, (pair_label, src_chs, tgt_chs) in enumerate(pairs):
        good_xcorrs = []
        bad_xcorrs = []
        peak_lags = []

        for i, trial in enumerate(all_trials):
            bp = trial['bipolar_lfp']
            t = trial['time']
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

            # Alpha envelopes
            _, src_env, _ = inst_phase_amp(detrend(src_sig[pre_mask]), *alpha_band, fs)
            _, tgt_env, _ = inst_phase_amp(detrend(tgt_sig[pre_mask]), *alpha_band, fs)

            # Normalize
            src_env = (src_env - np.mean(src_env)) / (np.std(src_env) + 1e-10)
            tgt_env = (tgt_env - np.mean(tgt_env)) / (np.std(tgt_env) + 1e-10)

            # Cross-correlation
            xcorr = np.correlate(tgt_env, src_env, mode='full')
            xcorr = xcorr / len(src_env)
            lags = np.arange(-len(src_env) + 1, len(src_env))
            lag_mask = np.abs(lags) <= max_lag_samples
            xcorr_trimmed = xcorr[lag_mask]
            lags_trimmed = lags[lag_mask] / fs * 1000  # convert to ms

            # Positive lag = source leads target
            peak_lag = lags_trimmed[np.argmax(xcorr_trimmed)]
            peak_lags.append(peak_lag)

            if good_mask[i]:
                good_xcorrs.append(xcorr_trimmed)
            else:
                bad_xcorrs.append(xcorr_trimmed)

        peak_lags = np.array(peak_lags)

        # Row 0: average cross-correlation
        ax = axes[0, col]
        for traces, color, label in [(good_xcorrs, 'green', 'Good'),
                                      (bad_xcorrs, 'red', 'Bad')]:
            if len(traces) == 0:
                continue
            min_len = min(len(x) for x in traces)
            traces_arr = np.array([x[:min_len] for x in traces])
            lags_plot = np.linspace(-max_lag_ms, max_lag_ms, min_len)
            mean_xc = np.mean(traces_arr, axis=0)
            sem_xc = sem(traces_arr, axis=0)
            ax.plot(lags_plot, mean_xc, color=color, lw=2, label=label)
            ax.fill_between(lags_plot, mean_xc - sem_xc, mean_xc + sem_xc,
                            color=color, alpha=0.15)

        ax.axvline(0, color='black', ls='--', lw=1)
        ax.set_xlabel('Lag (ms)\n+ = source leads')
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
            ax.text(0.05, 0.95, f'r={r:.2f}, p={p:.4f}',
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlabel('Peak lag (ms, + = source leads)')
        ax.set_ylabel('L2/3 alpha change (%)')
        ax.set_title(f'Peak Lag vs Suppression')
        ax.axvline(0, color='gray', ls='--', alpha=0.5)

    plt.suptitle('Analysis 6: Temporal Precedence — Alpha Envelope Cross-Correlation',
                 fontsize=15)
    plt.tight_layout()
    plt.savefig('deep_06_temporal_precedence.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# ANALYSIS 7: L5/L6 E-CELL RATE CORRELATION WITH L2/3 ALPHA
# ============================================================
def analysis_deep_E_rate_vs_alpha(all_trials, alpha_changes, good_mask):
    """
    Direct test: does L5/L6 E-cell firing rate (which drives SOM→E
    alpha loop) correlate with L2/3 alpha power per trial?
    Both pre-stim and post-stim.
    """
    layers = ['L5', 'L6']
    periods = ['pre', 'post']

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))

    for row, layer in enumerate(layers):
        for col, (period, color_map) in enumerate(
            [('pre', 'viridis'), ('post', 'viridis'),
             ('pre_scatter', None), ('post_scatter', None)]):

            ax = axes[row, col]
            e_rates = []
            l23_alpha = []
            som_rates = []

            for trial in all_trials:
                rd = trial['rate_data']
                stim = trial['stim_onset_ms']
                bp = trial['bipolar_lfp']
                t = trial['time']
                n_ch = bp.shape[0]

                is_pre = 'pre' in period
                if is_pre:
                    t_mask_lfp = (t >= stim - pre_window_ms) & (t < stim)
                else:
                    t_mask_lfp = (t >= stim + post_start_ms) & \
                                 (t < stim + post_start_ms + post_window_ms)

                # E rate
                e_key = 'E_rate'
                som_key = 'SOM_rate'
                if layer not in rd or e_key not in rd[layer]:
                    e_rates.append(np.nan)
                    som_rates.append(np.nan)
                    l23_alpha.append(np.nan)
                    continue

                t_r = rd[layer][e_key]['t_ms']
                r_e = rd[layer][e_key]['rate_hz']

                if som_key in rd[layer]:
                    r_s = rd[layer][som_key]['rate_hz']
                else:
                    r_s = np.zeros_like(r_e)

                if is_pre:
                    r_mask = (t_r >= stim - pre_window_ms) & (t_r < stim)
                else:
                    r_mask = (t_r >= stim + post_start_ms) & \
                             (t_r < stim + post_start_ms + post_window_ms)

                e_rates.append(np.mean(r_e[r_mask]) if np.sum(r_mask) > 0 else np.nan)
                som_rates.append(np.mean(r_s[r_mask]) if np.sum(r_mask) > 0 else np.nan)

                valid_sup = [ch for ch in L23_channels if ch < n_ch]
                a_pow = []
                for ch in valid_sup:
                    a_pow.append(band_power(detrend(bp[ch][t_mask_lfp]), *alpha_band, fs))
                l23_alpha.append(np.mean(a_pow) if a_pow else np.nan)

            e_rates = np.array(e_rates)
            som_rates = np.array(som_rates)
            l23_alpha = np.array(l23_alpha)

            valid = ~(np.isnan(e_rates) | np.isnan(l23_alpha))

            if col < 2:
                # E rate vs L2/3 alpha
                p_label = 'Pre-stim' if col == 0 else 'Post-stim'
                ax.scatter(e_rates[valid], l23_alpha[valid],
                           c=['green' if g else 'red' for g in good_mask[valid]],
                           edgecolors='k', s=35, alpha=0.7)
                ax.set_xlabel(f'{layer} E rate (Hz)')
                ax.set_ylabel('L2/3 alpha power')
                ax.set_title(f'{p_label}: {layer} E → L2/3 Alpha')
                if np.sum(valid) > 5:
                    r, p = pearsonr(e_rates[valid], l23_alpha[valid])
                    ax.text(0.05, 0.95, f'r={r:.2f}, p={p:.4f}',
                            transform=ax.transAxes, va='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                # SOM rate vs L2/3 alpha
                valid_s = ~(np.isnan(som_rates) | np.isnan(l23_alpha))
                p_label = 'Pre-stim' if col == 2 else 'Post-stim'
                ax.scatter(som_rates[valid_s], l23_alpha[valid_s],
                           c=['green' if g else 'red' for g in good_mask[valid_s]],
                           edgecolors='k', s=35, alpha=0.7)
                ax.set_xlabel(f'{layer} SOM rate (Hz)')
                ax.set_ylabel('L2/3 alpha power')
                ax.set_title(f'{p_label}: {layer} SOM → L2/3 Alpha')
                if np.sum(valid_s) > 5:
                    r, p = pearsonr(som_rates[valid_s], l23_alpha[valid_s])
                    ax.text(0.05, 0.95, f'r={r:.2f}, p={p:.4f}',
                            transform=ax.transAxes, va='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Analysis 7: Deep E & SOM Rates vs L2/3 Alpha Power', fontsize=15)
    plt.tight_layout()
    plt.savefig('deep_07_E_SOM_rate_vs_alpha.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# ANALYSIS 8: COMPREHENSIVE SUMMARY DASHBOARD
# ============================================================
def analysis_summary(all_trials, alpha_changes, good_mask):
    """
    Collect all deep-layer features and rank them by predictive power
    for L2/3 alpha suppression.
    """
    n = len(all_trials)
    n_ch_ref = all_trials[0]['bipolar_lfp'].shape[0]

    features = {}

    # Collect features per trial
    l5_alpha_pre = np.full(n, np.nan)
    l6_alpha_pre = np.full(n, np.nan)
    l5_som_pre = np.full(n, np.nan)
    l6_som_pre = np.full(n, np.nan)
    l5_e_pre = np.full(n, np.nan)
    l6_e_pre = np.full(n, np.nan)
    l5_pv_pre = np.full(n, np.nan)
    l6_pv_pre = np.full(n, np.nan)
    l23_alpha_pre = np.full(n, np.nan)
    l5_l23_coh = np.full(n, np.nan)
    l6_l23_coh = np.full(n, np.nan)

    for i, trial in enumerate(all_trials):
        bp = trial['bipolar_lfp']
        t = trial['time']
        stim = trial['stim_onset_ms']
        rd = trial['rate_data']
        n_ch = bp.shape[0]

        pre_mask = (t >= stim - pre_window_ms) & (t < stim)

        # LFP alpha power per layer region
        for chs, arr in [(L5_channels, l5_alpha_pre),
                          (L6_channels, l6_alpha_pre),
                          (L23_channels, l23_alpha_pre)]:
            valid_chs = [ch for ch in chs if ch < n_ch]
            if valid_chs:
                pows = [band_power(detrend(bp[ch][pre_mask]), *alpha_band, fs)
                        for ch in valid_chs]
                arr[i] = np.mean(pows)

        # Firing rates
        for layer, ct, arr in [
            ('L5', 'SOM', l5_som_pre), ('L6', 'SOM', l6_som_pre),
            ('L5', 'E', l5_e_pre), ('L6', 'E', l6_e_pre),
            ('L5', 'PV', l5_pv_pre), ('L6', 'PV', l6_pv_pre),
        ]:
            mon_key = f'{ct}_rate'
            if layer in rd and mon_key in rd[layer]:
                t_r = rd[layer][mon_key]['t_ms']
                r_hz = rd[layer][mon_key]['rate_hz']
                r_mask = (t_r >= stim - pre_window_ms) & (t_r < stim)
                if np.sum(r_mask) > 0:
                    arr[i] = np.mean(r_hz[r_mask])

        # Coherence
        for deep_chs, arr in [(L5_channels, l5_l23_coh), (L6_channels, l6_l23_coh)]:
            valid_d = [ch for ch in deep_chs if ch < n_ch]
            valid_s = [ch for ch in L23_channels if ch < n_ch]
            if valid_d and valid_s:
                d_sig = detrend(mean_lfp(bp, valid_d)[pre_mask])
                s_sig = detrend(mean_lfp(bp, valid_s)[pre_mask])
                nperseg = min(len(d_sig), 2048)
                f_c, coh = coherence(d_sig, s_sig, fs=fs, nperseg=nperseg)
                amask = (f_c >= alpha_band[0]) & (f_c <= alpha_band[1])
                arr[i] = np.mean(coh[amask])

    all_features = {
        'L5 alpha power': l5_alpha_pre,
        'L6 alpha power': l6_alpha_pre,
        'L2/3 alpha power': l23_alpha_pre,
        'L5 SOM rate': l5_som_pre,
        'L6 SOM rate': l6_som_pre,
        'L5 E rate': l5_e_pre,
        'L6 E rate': l6_e_pre,
        'L5 PV rate': l5_pv_pre,
        'L6 PV rate': l6_pv_pre,
        'L5↔L2/3 α coh': l5_l23_coh,
        'L6↔L2/3 α coh': l6_l23_coh,
    }

    # ---- Compute correlations ----
    results = []
    for feat_name, feat_vals in all_features.items():
        valid = ~np.isnan(feat_vals)
        if np.sum(valid) > 10:
            r, p = pearsonr(feat_vals[valid], alpha_changes[valid])
            good_vals = feat_vals[valid & good_mask]
            bad_vals = feat_vals[valid & ~good_mask]
            if len(good_vals) > 1 and len(bad_vals) > 1:
                _, p_mw = mannwhitneyu(good_vals, bad_vals)
                d = np.abs(np.mean(good_vals) - np.mean(bad_vals)) / \
                    np.sqrt((np.var(good_vals) + np.var(bad_vals))/2 + 1e-30)
            else:
                p_mw, d = 1.0, 0.0
            results.append((feat_name, r, p, p_mw, d))

    # Sort by absolute correlation
    results.sort(key=lambda x: abs(x[1]), reverse=True)

    # ---- Plot: bar chart of correlations ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: correlation bar chart
    ax = axes[0]
    names = [r[0] for r in results]
    corrs = [r[1] for r in results]
    p_vals = [r[2] for r in results]
    colors_bar = ['darkgreen' if p < 0.01 else 'green' if p < 0.05
                  else 'orange' if p < 0.1 else 'gray' for p in p_vals]

    y_pos = range(len(names))
    ax.barh(y_pos, corrs, color=colors_bar, edgecolor='k', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Pearson r with L2/3 alpha change (%)')
    ax.set_title('Feature Correlations with Alpha Suppression\n'
                 '(green=p<0.05, dark=p<0.01, gray=ns)')
    ax.axvline(0, color='black', lw=1)

    for i, (name, r, p, p_mw, d) in enumerate(results):
        ax.text(r + 0.02 * np.sign(r), i, f'r={r:.2f}\np={p:.3f}',
                va='center', fontsize=8)

    ax.invert_yaxis()

    # Right: correlation matrix of all features + alpha_change
    ax = axes[1]
    feat_names_short = [r[0][:12] for r in results] + ['α_chg']
    feat_matrix = np.column_stack([all_features[r[0]] for r in results])
    full = np.column_stack([feat_matrix, alpha_changes])

    # Handle NaN for correlation matrix
    valid_all = ~np.any(np.isnan(full), axis=1)
    corr_mat = np.corrcoef(full[valid_all].T)

    im = ax.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(feat_names_short)))
    ax.set_yticks(range(len(feat_names_short)))
    ax.set_xticklabels(feat_names_short, rotation=60, ha='right', fontsize=8)
    ax.set_yticklabels(feat_names_short, fontsize=8)

    for i in range(len(feat_names_short)):
        for j in range(len(feat_names_short)):
            ax.text(j, i, f'{corr_mat[i,j]:.2f}', ha='center', va='center',
                    fontsize=6, color='white' if abs(corr_mat[i,j]) > 0.5 else 'black')

    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title('Feature Correlation Matrix')

    plt.suptitle('Analysis 8: Summary — Deep-Layer Predictors of L2/3 Alpha Suppression',
                 fontsize=15)
    plt.tight_layout()
    plt.savefig('deep_08_summary.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print ranked results
    print("\n" + "=" * 70)
    print("RANKED FEATURES: Predicting L2/3 Alpha Suppression")
    print("=" * 70)
    print(f"{'Feature':<25s} {'r':>7s} {'p':>8s} {'MW p':>8s} {'d':>6s}")
    print("-" * 70)
    for name, r, p, p_mw, d in results:
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"{name:<25s} {r:+.3f}  {p:.4f} {sig:3s}  {p_mw:.4f}  {d:.2f}")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Loading trials...")
    all_trials = load_all_trials(base_path, n_trials)

    print("\nClassifying trials...")
    metrics, alpha_changes, good_mask, bad_mask = classify_trials(
        all_trials, L23_channels, alpha_band,
        pre_window_ms, post_window_ms, post_start_ms, fs, alpha_threshold
    )

    print("\n" + "=" * 60)
    print("ANALYSIS 1: Deep-Layer Pre-stimulus Alpha Power")
    print("=" * 60)
    analysis_deep_prestim_alpha(all_trials, alpha_changes, good_mask)

    print("\n" + "=" * 60)
    print("ANALYSIS 2: Deep ↔ Superficial Alpha Coherence")
    print("=" * 60)
    analysis_alpha_coherence(all_trials, alpha_changes, good_mask)

    print("\n" + "=" * 60)
    print("ANALYSIS 3: Deep Alpha Phase → Superficial Amplitude (PAC)")
    print("=" * 60)
    analysis_deep_sup_pac(all_trials, alpha_changes, good_mask)

    print("\n" + "=" * 60)
    print("ANALYSIS 4: Time-Resolved Deep-Layer Firing Rates")
    print("=" * 60)
    analysis_deep_firing_rates(all_trials, alpha_changes, good_mask)

    print("\n" + "=" * 60)
    print("ANALYSIS 5: SOM as Alpha Generator")
    print("=" * 60)
    analysis_som_alpha_link(all_trials, alpha_changes, good_mask)

    print("\n" + "=" * 60)
    print("ANALYSIS 6: Temporal Precedence (Alpha Lead/Lag)")
    print("=" * 60)
    analysis_temporal_precedence(all_trials, alpha_changes, good_mask)

    print("\n" + "=" * 60)
    print("ANALYSIS 7: Deep E & SOM Rates vs L2/3 Alpha")
    print("=" * 60)
    analysis_deep_E_rate_vs_alpha(all_trials, alpha_changes, good_mask)

    print("\n" + "=" * 60)
    print("ANALYSIS 8: Summary Dashboard")
    print("=" * 60)
    analysis_summary(all_trials, alpha_changes, good_mask)

    print("\nAll deep-layer analyses complete!")
