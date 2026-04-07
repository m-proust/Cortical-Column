"""
Alpha–Gamma Phase-Amplitude Coupling Analysis
==============================================
Replicates Spaak et al. (2012) Figure 2:
  - Detects alpha peaks in infragranular bipolar LFP
  - Aligns time-frequency representations (TFRs) of gamma power to those peaks
  - Plots mean TFR for supragranular, granular, and infragranular compartments
  - Shows percent power modulation relative to epoch mean

Usage:
    python alpha_gamma_coupling.py --trial_dir results/trials_01_04_3 --n_trials 50
"""

import os
import glob
import argparse
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch
from scipy.signal import morlet2, cwt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm


# ──────────────────────────────────────────────────────────
# 1.  LAYER / ELECTRODE MAPPING
# ──────────────────────────────────────────────────────────
# Layer z-ranges from config2.py
LAYER_Z_RANGES = {
    'L23':  (0.45, 1.10),   # supragranular
    'L4AB': (0.14, 0.45),   # granular (upper)
    'L4C':  (-0.14, 0.14),  # granular (lower)
    'L5':   (-0.34, -0.14), # infragranular
    'L6':   (-0.62, -0.34), # infragranular
}

COMPARTMENT_LAYERS = {
    'supragranular': ['L23'],
    'granular':      ['L4AB', 'L4C'],
    'infragranular': ['L5', 'L6'],
}


def classify_bipolar_channels(channel_depths):
    """Assign each bipolar channel to a laminar compartment based on its midpoint depth."""
    labels = []
    for z in channel_depths:
        if z >= LAYER_Z_RANGES['L23'][0]:
            labels.append('supragranular')
        elif z >= LAYER_Z_RANGES['L4C'][0]:
            labels.append('granular')
        else:
            labels.append('infragranular')
    return np.array(labels)


# ──────────────────────────────────────────────────────────
# 2.  SIGNAL PROCESSING HELPERS
# ──────────────────────────────────────────────────────────
def bandpass(sig, lo, hi, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, sig, axis=-1)


def detect_peaks(sig):
    """Return indices of local maxima (positive peaks)."""
    d = np.diff(sig)
    peaks = np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
    return peaks


def morlet_tfr(sig, fs, freqs, n_cycles=5):
    """Compute time-frequency power using Morlet wavelets.

    Parameters
    ----------
    sig : 1-D array
    fs : sampling rate (Hz)
    freqs : array of frequencies to evaluate
    n_cycles : number of cycles (controls frequency resolution)

    Returns
    -------
    power : (n_freqs, n_times) array

    Notes
    -----
    At each frequency f, the wavelet has:
      - Duration: n_cycles / f  (e.g. 5/50Hz = 100ms)
      - Frequency resolution (FWHM): ~f / n_cycles  (e.g. 50/5 = 10Hz)
      - Scale parameter: s = n_cycles * fs / (2*pi*f)
    
    The squared magnitude of the convolution gives instantaneous power.
    """
    n_freqs = len(freqs)
    n_times = len(sig)
    power = np.zeros((n_freqs, n_times))

    for i, f in enumerate(freqs):
        w = n_cycles  # omega_0
        s = w * fs / (2 * np.pi * f)  # scale
        widths = [s]
        coef = cwt(sig, morlet2, widths, w=w)
        power[i] = np.abs(coef[0]) ** 2

    return power


# ──────────────────────────────────────────────────────────
# 3.  HIGH-ALPHA SEGMENT SELECTION  (Spaak et al. approach)
# ──────────────────────────────────────────────────────────
def select_high_alpha_segments(alpha_env, fs, threshold_percentile=75,
                                min_duration_ms=300):
    """Return boolean mask for time points belonging to high-alpha segments.

    Parameters
    ----------
    alpha_env : analytic amplitude envelope of the alpha-filtered signal
    fs : sampling rate
    threshold_percentile : percentile of envelope above which = high alpha
    min_duration_ms : minimum contiguous duration to keep (ms)
    """
    thresh = np.percentile(alpha_env, threshold_percentile)
    mask = alpha_env >= thresh
    # enforce minimum duration
    min_samps = int(min_duration_ms * fs / 1000)
    out = np.zeros_like(mask)
    start = None
    for i in range(len(mask)):
        if mask[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                if (i - start) >= min_samps:
                    out[start:i] = True
                start = None
    if start is not None and (len(mask) - start) >= min_samps:
        out[start:] = True
    return out


# ──────────────────────────────────────────────────────────
# 4.  CORE ANALYSIS: epoch-aligned TFR
# ──────────────────────────────────────────────────────────
def compute_alpha_peak_aligned_tfr(
    bipolar_matrix,
    channel_depths,
    fs=10000,
    alpha_band=(7, 14),
    gamma_freqs=np.arange(15, 201, 2),
    window_ms=300,
    n_cycles=5,
    high_alpha_percentile=75,
    use_high_alpha=True,
    stim_onset_ms=None,
    analysis_period='baseline',
    transient_ms=300,
):
    """
    Main analysis function replicating Spaak et al. Figure 2.

    1. Pick infragranular bipolar channels
    2. Average them to get a single infragranular alpha reference signal
    3. Bandpass 7-14 Hz → detect alpha peaks
    4. For each compartment, compute wavelet TFR
    5. Epoch-align TFR to alpha peaks, average, normalise to % modulation

    Parameters
    ----------
    bipolar_matrix : (n_bipolar_channels, n_samples) array
    channel_depths : list/array of bipolar channel midpoint depths
    fs : sampling rate in Hz
    alpha_band : tuple (lo, hi) Hz for peak detection filter
    gamma_freqs : array of frequencies for the TFR (can extend below gamma)
    window_ms : half-window around alpha peak (ms)
    n_cycles : Morlet wavelet cycles
    high_alpha_percentile : percentile threshold for high-alpha selection
    use_high_alpha : whether to restrict to high-alpha segments
    stim_onset_ms : if set, restrict analysis to a time period
    analysis_period : 'baseline', 'stim', or 'all'
    transient_ms : post-stimulus transient to exclude (ms)

    Returns
    -------
    results : dict with keys per compartment, each containing:
        'tfr_pct' : (n_freqs, n_epoch_samples) percent modulation
        'time_axis_ms' : time axis relative to alpha peak
        'freqs' : frequency axis
        'n_epochs' : number of alpha peaks used
        'alpha_trace' : mean alpha-aligned raw trace (infragranular)
    """
    compartment_labels = classify_bipolar_channels(channel_depths)
    n_channels, n_samples = bipolar_matrix.shape
    window_samp = int(window_ms * fs / 1000)

    # --- determine analysis time range ---
    if stim_onset_ms is not None:
        onset_samp = int(stim_onset_ms * fs / 1000)
        transient_samp = int(transient_ms * fs / 1000)
        if analysis_period == 'baseline':
            t_start, t_end = 0, onset_samp
        elif analysis_period == 'stim':
            t_start, t_end = onset_samp + transient_samp, n_samples
        else:
            t_start, t_end = 0, n_samples
    else:
        t_start, t_end = 0, n_samples

    # --- Step 1: infragranular reference ---
    infra_idx = np.where(compartment_labels == 'infragranular')[0]
    if len(infra_idx) == 0:
        raise ValueError("No infragranular channels found. Check electrode depths.")
    infra_mean = np.mean(bipolar_matrix[infra_idx], axis=0)

    # --- Step 2: alpha filter + envelope ---
    alpha_sig = bandpass(infra_mean, alpha_band[0], alpha_band[1], fs)
    alpha_env = np.abs(hilbert(alpha_sig))

    # --- Step 3: high-alpha mask ---
    if use_high_alpha:
        ha_mask = select_high_alpha_segments(alpha_env, fs,
                                              threshold_percentile=high_alpha_percentile)
    else:
        ha_mask = np.ones(n_samples, dtype=bool)

    # analysis period mask
    period_mask = np.zeros(n_samples, dtype=bool)
    period_mask[t_start:t_end] = True

    # --- Step 4: detect alpha peaks ---
    peaks = detect_peaks(alpha_sig)
    valid_peaks = []
    for p in peaks:
        lo = p - window_samp
        hi = p + window_samp
        if (lo >= 0 and hi < n_samples
                and ha_mask[p]
                and period_mask[lo] and period_mask[hi - 1]):
            valid_peaks.append(p)
    valid_peaks = np.array(valid_peaks)

    if len(valid_peaks) == 0:
        print("  WARNING: no valid alpha peaks found!")
        return None

    # --- Step 5: compute TFR per compartment, epoch-align, average ---
    results = {}
    time_axis = np.arange(-window_samp, window_samp) / fs * 1000  # ms

    for comp_name in ['supragranular', 'granular', 'infragranular']:
        comp_idx = np.where(compartment_labels == comp_name)[0]
        if len(comp_idx) == 0:
            continue

        comp_sig = np.mean(bipolar_matrix[comp_idx], axis=0)
        tfr_full = morlet_tfr(comp_sig, fs, gamma_freqs, n_cycles=n_cycles)

        n_freqs = len(gamma_freqs)
        epoch_len = 2 * window_samp
        tfr_epochs = np.zeros((len(valid_peaks), n_freqs, epoch_len))

        for ei, pk in enumerate(valid_peaks):
            tfr_epochs[ei] = tfr_full[:, pk - window_samp: pk + window_samp]

        mean_tfr = np.mean(tfr_epochs, axis=0)

        # normalise: % modulation relative to epoch-mean power per frequency
        baseline_power = np.mean(mean_tfr, axis=1, keepdims=True)
        baseline_power[baseline_power == 0] = 1e-30
        tfr_pct = (mean_tfr - baseline_power) / baseline_power * 100

        results[comp_name] = {
            'tfr_pct': tfr_pct,
            'time_axis_ms': time_axis,
            'freqs': gamma_freqs,
            'n_epochs': len(valid_peaks),
        }

    # mean alpha-aligned raw infragranular trace
    alpha_epochs = np.zeros((len(valid_peaks), 2 * window_samp))
    for ei, pk in enumerate(valid_peaks):
        alpha_epochs[ei] = infra_mean[pk - window_samp: pk + window_samp]
    results['alpha_trace'] = {
        'mean': np.mean(alpha_epochs, axis=0),
        'time_axis_ms': time_axis,
    }

    results['peak_times_ms'] = valid_peaks / fs * 1000
    results['alpha_band'] = alpha_band

    return results


# ──────────────────────────────────────────────────────────
# 5.  PLOTTING (Spaak et al. Figure 2 style)
# ──────────────────────────────────────────────────────────
def plot_alpha_gamma_coupling(results, title_suffix='', save_path=None):
    """
    4-panel figure mimicking Spaak et al. Figure 2:
      A) Supragranular TFR
      B) Granular TFR
      C) Infragranular TFR
      D) Mean alpha-aligned raw trace (infragranular)

    A dashed horizontal line marks the upper edge of the alpha band
    to indicate the boundary below which coupling is trivially expected
    due to the analysis method.
    """
    fig = plt.figure(figsize=(8, 12))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 0.5], hspace=0.35)

    panel_labels = ['A', 'B', 'C', 'D']
    compartments = ['supragranular', 'granular', 'infragranular']
    compartment_titles = [
        'Supragranular power',
        'Granular power',
        'Infragranular power',
    ]

    # get alpha band for the boundary line
    alpha_hi = 14.0
    if 'alpha_band' in results:
        alpha_hi = results['alpha_band'][1]

    vmax = 0
    for comp in compartments:
        if comp in results:
            vmax = max(vmax, np.max(np.abs(results[comp]['tfr_pct'])))
    if vmax == 0:
        vmax = 20
    vmax = min(vmax, 100)

    for i, (comp, comp_title) in enumerate(zip(compartments, compartment_titles)):
        ax = fig.add_subplot(gs[i])

        if comp in results:
            r = results[comp]
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax.pcolormesh(
                r['time_axis_ms'], r['freqs'], r['tfr_pct'],
                cmap='RdBu_r', norm=norm, shading='auto',
            )
            cb = fig.colorbar(im, ax=ax, label='% power modulation', shrink=0.8)
            ax.set_title(f'{comp_title}  (n={r["n_epochs"]} epochs)',
                         fontsize=11, fontweight='bold')

            # draw alpha band upper boundary
            ax.axhline(alpha_hi, color='white', ls='--', lw=1.2, alpha=0.8)
            ax.text(r['time_axis_ms'][-1] * 0.95, alpha_hi + 2,
                    f'α = {alpha_hi:.0f} Hz', color='white', fontsize=8,
                    ha='right', va='bottom', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.4))
        else:
            ax.text(0.5, 0.5, 'No channels', transform=ax.transAxes,
                    ha='center', va='center')

        ax.set_ylabel('Frequency (Hz)')
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time relative to alpha peak (s)')
            xticks = ax.get_xticks()
            ax.set_xticklabels([f'{x/1000:.1f}' for x in xticks])

        ax.text(-0.08, 1.05, panel_labels[i], transform=ax.transAxes,
                fontsize=14, fontweight='bold')

    # Panel D: mean alpha trace
    ax_d = fig.add_subplot(gs[3])
    if 'alpha_trace' in results:
        at = results['alpha_trace']
        ax_d.plot(at['time_axis_ms'], at['mean'], 'k-', lw=1.2)
        ax_d.axvline(0, color='gray', ls='--', alpha=0.5)
        ax_d.set_xlabel('Time relative to alpha peak (ms)')
        ax_d.set_ylabel('Bipolar LFP (a.u.)')
        ax_d.set_title('Infragranular alpha', fontsize=11, fontweight='bold')
    ax_d.text(-0.08, 1.05, 'D', transform=ax_d.transAxes,
              fontsize=14, fontweight='bold')

    fig.suptitle(f'Alpha–Gamma Coupling (Spaak et al. replication){title_suffix}',
                 fontsize=13, fontweight='bold', y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved figure to {save_path}")

    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────
# 6.  TRIAL LOADING AND AGGREGATION
# ──────────────────────────────────────────────────────────
def load_trial(fpath):
    """Load a single trial .npz file and return a dict."""
    d = np.load(fpath, allow_pickle=True)
    out = {}
    for k in d.files:
        out[k] = d[k]
    return out


def aggregate_trials(trial_dir, n_trials=None, analysis_period='all',
                     use_high_alpha=True, high_alpha_percentile=75,
                     alpha_band=(7, 14), gamma_freqs=None,
                     window_ms=300, fs=10000, transient_ms=300):
    """
    Load trials, compute alpha-peak-aligned TFR for each, then average
    the normalised TFRs across trials.
    """
    if gamma_freqs is None:
        gamma_freqs = np.arange(15, 201, 2)

    files = sorted(glob.glob(os.path.join(trial_dir, 'trial_*.npz')))
    if n_trials is not None:
        files = files[:n_trials]

    if len(files) == 0:
        raise FileNotFoundError(f"No trial files found in {trial_dir}")

    print(f"Found {len(files)} trial files in {trial_dir}")

    acc = {}
    count = 0
    all_peak_times_ms = []

    for fi, fpath in enumerate(files):
        print(f"  Processing {os.path.basename(fpath)} ({fi+1}/{len(files)})...")
        trial = load_trial(fpath)

        bipolar_matrix = trial['bipolar_matrix']
        channel_depths = trial['channel_depths']

        time_ms = trial['time_array_ms']
        dt_ms = time_ms[1] - time_ms[0]
        trial_fs = 1000.0 / dt_ms

        stim_onset = float(trial.get('stim_onset_ms', 0))
        if stim_onset == 0 and 'baseline_ms' in trial:
            stim_onset = float(trial['baseline_ms'])

        # downsample 10kHz → 1kHz
        target_fs = 1000.0
        if trial_fs > target_fs * 1.5:
            ds_factor = int(round(trial_fs / target_fs))
            bipolar_matrix = bipolar_matrix[:, ::ds_factor]
            trial_fs = trial_fs / ds_factor

        res = compute_alpha_peak_aligned_tfr(
            bipolar_matrix,
            channel_depths,
            fs=trial_fs,
            alpha_band=alpha_band,
            gamma_freqs=gamma_freqs,
            window_ms=window_ms,
            n_cycles=5,
            high_alpha_percentile=high_alpha_percentile,
            use_high_alpha=use_high_alpha,
            stim_onset_ms=stim_onset if analysis_period != 'all' else None,
            analysis_period=analysis_period,
            transient_ms=transient_ms,
        )

        if res is None:
            print(f"    Skipped (no valid alpha peaks)")
            continue

        all_peak_times_ms.extend(res['peak_times_ms'].tolist())

        for comp in ['supragranular', 'granular', 'infragranular']:
            if comp not in res:
                continue
            if comp not in acc:
                acc[comp] = {
                    'tfr_sum': np.zeros_like(res[comp]['tfr_pct']),
                    'n_epochs_total': 0,
                    'freqs': res[comp]['freqs'],
                    'time_axis_ms': res[comp]['time_axis_ms'],
                }
            acc[comp]['tfr_sum'] += res[comp]['tfr_pct'] * res[comp]['n_epochs']
            acc[comp]['n_epochs_total'] += res[comp]['n_epochs']

        if 'alpha_trace' in res:
            if 'alpha_trace' not in acc:
                acc['alpha_trace'] = {
                    'sum': np.zeros_like(res['alpha_trace']['mean']),
                    'time_axis_ms': res['alpha_trace']['time_axis_ms'],
                    'count': 0,
                }
            acc['alpha_trace']['sum'] += res['alpha_trace']['mean']
            acc['alpha_trace']['count'] += 1

        count += 1

    if count == 0:
        raise RuntimeError("No trials produced valid results.")

    final = {}
    for comp in ['supragranular', 'granular', 'infragranular']:
        if comp in acc and acc[comp]['n_epochs_total'] > 0:
            final[comp] = {
                'tfr_pct': acc[comp]['tfr_sum'] / acc[comp]['n_epochs_total'],
                'freqs': acc[comp]['freqs'],
                'time_axis_ms': acc[comp]['time_axis_ms'],
                'n_epochs': acc[comp]['n_epochs_total'],
            }

    if 'alpha_trace' in acc and acc['alpha_trace']['count'] > 0:
        final['alpha_trace'] = {
            'mean': acc['alpha_trace']['sum'] / acc['alpha_trace']['count'],
            'time_axis_ms': acc['alpha_trace']['time_axis_ms'],
        }

    final['alpha_band'] = alpha_band

    print(f"\nDone. Aggregated {count} trials.")
    if len(all_peak_times_ms) > 0:
        arr = np.array(all_peak_times_ms)
        print(f"  Alpha peak times (ms): mean={arr.mean():.1f}, "
              f"std={arr.std():.1f}, min={arr.min():.1f}, max={arr.max():.1f}, "
              f"n_peaks={len(arr)}")
    for comp in ['supragranular', 'granular', 'infragranular']:
        if comp in final:
            print(f"  {comp}: {final[comp]['n_epochs']} total alpha-peak epochs")

    return final


# ──────────────────────────────────────────────────────────
# 7.  MAIN
# ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Alpha-gamma coupling analysis (Spaak et al. 2012 replication)')
    parser.add_argument('--trial_dir', type=str, required=True,
                        help='Directory containing trial_*.npz files')
    parser.add_argument('--n_trials', type=int, default=None,
                        help='Max number of trials to load (default: all)')
    parser.add_argument('--period', type=str, default='both',
                        choices=['baseline', 'stim', 'all', 'both'],
                        help='Which time period to analyse (both = baseline + stim separately)')
    parser.add_argument('--no_high_alpha', action='store_true',
                        help='Disable high-alpha segment selection')
    parser.add_argument('--alpha_lo', type=float, default=7.0)
    parser.add_argument('--alpha_hi', type=float, default=14.0)
    parser.add_argument('--gamma_lo', type=float, default=15.0,
                        help='Lower freq for TFR (default: 15 Hz, below alpha to see boundary)')
    parser.add_argument('--gamma_hi', type=float, default=200.0)
    parser.add_argument('--gamma_step', type=float, default=2.0,
                        help='Frequency step for TFR (default: 2 Hz)')
    parser.add_argument('--window_ms', type=float, default=300.0,
                        help='Half-window around alpha peak (ms)')
    parser.add_argument('--percentile', type=float, default=75.0,
                        help='Percentile for high-alpha threshold')
    parser.add_argument('--transient_ms', type=float, default=300.0,
                        help='Duration of post-stimulus transient to exclude (ms)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save figures (default: trial_dir)')
    args = parser.parse_args()

    save_dir = args.save_dir or args.trial_dir
    os.makedirs(save_dir, exist_ok=True)

    gamma_freqs = np.arange(args.gamma_lo, args.gamma_hi + 1, args.gamma_step)

    if args.period == 'both':
        periods = ['baseline', 'stim']
    else:
        periods = [args.period]

    for period in periods:
        print(f"\n{'='*60}")
        print(f"  Analysing period: {period}"
              + (f"  (excluding {args.transient_ms:.0f} ms transient)" if period == 'stim' else ''))
        print(f"{'='*60}")

        final = aggregate_trials(
            trial_dir=args.trial_dir,
            n_trials=args.n_trials,
            analysis_period=period,
            use_high_alpha=not args.no_high_alpha,
            high_alpha_percentile=args.percentile,
            alpha_band=(args.alpha_lo, args.alpha_hi),
            gamma_freqs=gamma_freqs,
            window_ms=args.window_ms,
            transient_ms=args.transient_ms,
        )

        suffix = f'  [{period}]'
        fig_path = os.path.join(save_dir, f'alpha_gamma_coupling_{period}.png')
        plot_alpha_gamma_coupling(final, title_suffix=suffix, save_path=fig_path)

        if not args.no_high_alpha:
            print(f"\n--- Re-running {period} without high-alpha selection ---")
            final_all = aggregate_trials(
                trial_dir=args.trial_dir,
                n_trials=args.n_trials,
                analysis_period=period,
                use_high_alpha=False,
                alpha_band=(args.alpha_lo, args.alpha_hi),
                gamma_freqs=gamma_freqs,
                window_ms=args.window_ms,
                transient_ms=args.transient_ms,
            )
            fig_path2 = os.path.join(save_dir,
                                      f'alpha_gamma_coupling_{period}_all_data.png')
            plot_alpha_gamma_coupling(final_all,
                                      title_suffix=f'{suffix} (all data, no high-α selection)',
                                      save_path=fig_path2)


if __name__ == '__main__':
    main()