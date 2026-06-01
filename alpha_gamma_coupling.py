import os
import glob
import argparse
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch
from scipy.signal import morlet2, cwt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

LAYER_Z_RANGES = {
    'L23':  (0.45, 1.10),
    'L4AB': (0.14, 0.45), 
    'L4C':  (-0.14, 0.14), 
    'L5':   (-0.34, -0.14),
    'L6':   (-0.62, -0.34),
}

COMPARTMENT_LAYERS = {
    'supragranular': ['L23'],
    'granular':      ['L4AB', 'L4C'],
    'infragranular': ['L5', 'L6'],
}

def classify_bipolar_channels(channel_depths):
    labels = []
    for z in channel_depths:
        if z >= LAYER_Z_RANGES['L23'][0]:
            labels.append('supragranular')
        elif z >= LAYER_Z_RANGES['L4C'][0]:
            labels.append('granular')
        else:
            labels.append('infragranular')
    return np.array(labels)

def bandpass(sig, lo, hi, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, sig, axis=-1)


def detect_peaks(sig):
    d = np.diff(sig)
    peaks = np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
    return peaks


def morlet_tfr(sig, fs, freqs, n_cycles=5):
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


def select_alpha_segments(alpha_env, fs, threshold_percentile=75,
                           min_duration_ms=150, period_mask=None,
                           band='high'):

    if period_mask is not None and np.any(period_mask):
        thresh = np.percentile(alpha_env[period_mask], threshold_percentile)
    else:
        thresh = np.percentile(alpha_env, threshold_percentile)
    if band == 'high':
        mask = alpha_env >= thresh
    elif band == 'low':
        mask = alpha_env <= thresh
    else:
        raise ValueError(f"band must be 'high' or 'low', got {band!r}")
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


def compute_alpha_peak_aligned_tfr(
    bipolar_matrix,
    channel_depths,
    fs=10000,
    alpha_band=(7, 12),
    gamma_freqs=np.arange(15, 201, 2),
    window_ms=300,
    n_cycles=5,
    high_alpha_percentile=75,
    low_alpha_percentile=25,
    use_high_alpha=True,
    alpha_band_select='high',
    stim_onset_ms=None,
    analysis_period='baseline',
    transient_ms=300,
    warmup_ms=500,
    min_ha_ms=150,
):
    compartment_labels = classify_bipolar_channels(channel_depths)
    n_channels, n_samples = bipolar_matrix.shape
    window_samp = int(window_ms * fs / 1000)
    warmup_samp = int(warmup_ms * fs / 1000)
    if stim_onset_ms is not None:
        onset_samp = int(stim_onset_ms * fs / 1000)
        transient_samp = int(transient_ms * fs / 1000)
        if analysis_period == 'baseline':
            t_start, t_end = warmup_samp, onset_samp
        elif analysis_period == 'stim':
            t_start, t_end = onset_samp + transient_samp, n_samples
        else:
            t_start, t_end = warmup_samp, n_samples
    else:
        t_start, t_end = warmup_samp, n_samples


    infra_idx = np.where(compartment_labels == 'infragranular')[0]
    if len(infra_idx) == 0:
        raise ValueError("zero infragranular channels found")
    infra_mean = np.mean(bipolar_matrix[infra_idx], axis=0)

    alpha_sig = bandpass(infra_mean, alpha_band[0], alpha_band[1], fs)
    alpha_env = np.abs(hilbert(alpha_sig))

    period_mask = np.zeros(n_samples, dtype=bool)
    period_mask[t_start:t_end] = True

    if use_high_alpha:
        pct = high_alpha_percentile if alpha_band_select == 'high' else low_alpha_percentile
        ha_mask = select_alpha_segments(alpha_env, fs,
                                         threshold_percentile=pct,
                                         min_duration_ms=min_ha_ms,
                                         period_mask=period_mask,
                                         band=alpha_band_select)
    else:
        ha_mask = np.ones(n_samples, dtype=bool)

    peaks = detect_peaks(alpha_sig)
    n_in_period = 0
    n_in_ha = 0
    n_in_window = 0
    valid_peaks = []
    for p in peaks:
        lo = p - window_samp
        hi = p + window_samp
        in_window = lo >= 0 and hi < n_samples
        in_period = in_window and period_mask[lo] and period_mask[hi - 1]
        in_ha = ha_mask[p]
        if in_window:
            n_in_window += 1
        if in_period:
            n_in_period += 1
        if in_ha:
            n_in_ha += 1
        if in_window and in_period and in_ha:
            valid_peaks.append(p)
    valid_peaks = np.array(valid_peaks)

    ha_coverage = 100.0 * np.sum(ha_mask & period_mask) / max(np.sum(period_mask), 1)
    period_dur_ms = (t_end - t_start) / fs * 1000
    print(f"  diag: peaks_total={len(peaks)} in_window={n_in_window} "
          f"in_period={n_in_period} in_ha_mask={n_in_ha} valid={len(valid_peaks)} | "
          f"period={period_dur_ms:.0f}ms ha_coverage_in_period={ha_coverage:.1f}% "
          f"window={window_samp*2/fs*1000:.0f}ms fs={fs:.0f}Hz")

    if len(valid_peaks) == 0:
        print("no valid alpha peaks found")
        return None

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

        baseline_power = np.mean(mean_tfr, axis=1, keepdims=True)
        baseline_power[baseline_power == 0] = 1e-30
        tfr_pct = (mean_tfr - baseline_power) / baseline_power * 100

        results[comp_name] = {
            'tfr_pct': tfr_pct,
            'time_axis_ms': time_axis,
            'freqs': gamma_freqs,
            'n_epochs': len(valid_peaks),
        }

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


def compute_peak_aligned_tfr_raw(
    bipolar_matrix,
    channel_depths,
    fs=1000,
    alpha_band=(7, 14),
    gamma_freqs=np.arange(15, 201, 2),
    window_ms=300,
    n_cycles=5,
    stim_onset_ms=None,
    analysis_period='baseline',
    transient_ms=300,
    warmup_ms=500,
):
    """Per-trial peak-aligned TFR with RAW (un-normalized) power, plus the
    trial's mean alpha power in the analysis window. Used for the
    across-trials median split (Bonnefond & Jensen 2015 Fig 2D style)."""
    compartment_labels = classify_bipolar_channels(channel_depths)
    n_channels, n_samples = bipolar_matrix.shape
    window_samp = int(window_ms * fs / 1000)
    warmup_samp = int(warmup_ms * fs / 1000)
    if stim_onset_ms is not None:
        onset_samp = int(stim_onset_ms * fs / 1000)
        transient_samp = int(transient_ms * fs / 1000)
        if analysis_period == 'baseline':
            t_start, t_end = warmup_samp, onset_samp
        elif analysis_period == 'stim':
            t_start, t_end = onset_samp + transient_samp, n_samples
        else:
            t_start, t_end = warmup_samp, n_samples
    else:
        t_start, t_end = warmup_samp, n_samples

    infra_idx = np.where(compartment_labels == 'infragranular')[0]
    if len(infra_idx) == 0:
        raise ValueError("zero infragranular channels found")
    infra_mean = np.mean(bipolar_matrix[infra_idx], axis=0)

    alpha_sig = bandpass(infra_mean, alpha_band[0], alpha_band[1], fs)
    alpha_env = np.abs(hilbert(alpha_sig))

    trial_alpha_power = float(np.mean(alpha_env[t_start:t_end] ** 2))

    period_mask = np.zeros(n_samples, dtype=bool)
    period_mask[t_start:t_end] = True

    peaks = detect_peaks(alpha_sig)
    valid_peaks = []
    for p in peaks:
        lo = p - window_samp
        hi = p + window_samp
        if lo >= 0 and hi < n_samples and period_mask[lo] and period_mask[hi - 1]:
            valid_peaks.append(p)
    valid_peaks = np.array(valid_peaks)

    if len(valid_peaks) == 0:
        return None

    results = {
        'trial_alpha_power': trial_alpha_power,
        'n_epochs': len(valid_peaks),
        'freqs': gamma_freqs,
        'time_axis_ms': np.arange(-window_samp, window_samp) / fs * 1000,
        'alpha_band': alpha_band,
    }

    for comp_name in ['supragranular', 'granular', 'infragranular']:
        comp_idx = np.where(compartment_labels == comp_name)[0]
        if len(comp_idx) == 0:
            continue
        comp_sig = np.mean(bipolar_matrix[comp_idx], axis=0)
        tfr_full = morlet_tfr(comp_sig, fs, gamma_freqs, n_cycles=n_cycles)

        epoch_len = 2 * window_samp
        n_freqs = len(gamma_freqs)
        tfr_epochs = np.zeros((len(valid_peaks), n_freqs, epoch_len))
        for ei, pk in enumerate(valid_peaks):
            tfr_epochs[ei] = tfr_full[:, pk - window_samp: pk + window_samp]
        results[comp_name] = {
            'tfr_raw_sum': np.sum(tfr_epochs, axis=0),
            'n_epochs': len(valid_peaks),
        }

    return results


def aggregate_trials_median_split(trial_dir, n_trials=None, analysis_period='baseline',
                                   alpha_band=(7, 14), gamma_freqs=None,
                                   window_ms=300, transient_ms=300, warmup_ms=500,
                                   split='median', contrast='logratio'):
    """Bonnefond & Jensen 2015 Fig 2D style: per-trial scalar alpha power,
    split across trials, then contrast high-group vs low-group peak-locked TFR.

    split: 'median' = 50/50 split, 'tertile' = top third vs bottom third.
    contrast:
      - 'logratio' (default): 10 * log10(high / low), in dB. Symmetric,
        dimensionless. Standard for spectral group contrasts.
      - 'normdiff': normalize each group TFR by its per-freq mean, then
        subtract. Compares modulation shape, not magnitude.
      - 'diff': raw arithmetic difference high - low."""
    if gamma_freqs is None:
        gamma_freqs = np.arange(15, 201, 2)

    files = sorted(glob.glob(os.path.join(trial_dir, 'trial_*.npz')))
    if n_trials is not None:
        files = files[:n_trials]
    if len(files) == 0:
        raise FileNotFoundError(f"zero trial files found in {trial_dir}")

    per_trial = []
    for fpath in files:
        trial = load_trial(fpath)
        bipolar_matrix = trial['bipolar_matrix']
        channel_depths = trial['channel_depths']
        time_ms = trial['time_array_ms']
        dt_ms = time_ms[1] - time_ms[0]
        trial_fs = 1000.0 / dt_ms

        stim_onset = float(trial.get('stim_onset_ms', 0))
        if stim_onset == 0 and 'baseline_ms' in trial:
            stim_onset = float(trial['baseline_ms'])

        target_fs = 1000.0
        if trial_fs > target_fs * 1.5:
            ds_factor = int(round(trial_fs / target_fs))
            bipolar_matrix = bipolar_matrix[:, ::ds_factor]
            trial_fs = trial_fs / ds_factor

        res = compute_peak_aligned_tfr_raw(
            bipolar_matrix, channel_depths,
            fs=trial_fs, alpha_band=alpha_band, gamma_freqs=gamma_freqs,
            window_ms=window_ms,
            stim_onset_ms=stim_onset if analysis_period != 'all' else None,
            analysis_period=analysis_period,
            transient_ms=transient_ms, warmup_ms=warmup_ms,
        )
        if res is None:
            print(f"  skipping {os.path.basename(fpath)}: no valid peaks")
            continue
        per_trial.append(res)

    if len(per_trial) < 2:
        raise RuntimeError("need at least 2 trials with valid peaks for median split")

    powers = np.array([t['trial_alpha_power'] for t in per_trial])
    if split == 'tertile':
        hi_thr = np.percentile(powers, 100 * 2 / 3)
        lo_thr = np.percentile(powers, 100 * 1 / 3)
        high_trials = [t for t, p in zip(per_trial, powers) if p >= hi_thr]
        low_trials  = [t for t, p in zip(per_trial, powers) if p <= lo_thr]
        print(f"  tertile split: {len(high_trials)} high (>= {hi_thr:.3g}), "
              f"{len(low_trials)} low (<= {lo_thr:.3g}) | "
              f"power range [{powers.min():.3g}, {powers.max():.3g}]")
    else:
        median = np.median(powers)
        high_trials = [t for t, p in zip(per_trial, powers) if p >= median]
        low_trials  = [t for t, p in zip(per_trial, powers) if p <  median]
        print(f"  median split: {len(high_trials)} high-alpha, "
              f"{len(low_trials)} low-alpha (median={median:.3g}) | "
              f"power range [{powers.min():.3g}, {powers.max():.3g}]")

    def group_mean(trials, comp):
        sums = None
        n_total = 0
        for t in trials:
            if comp not in t:
                continue
            if sums is None:
                sums = np.zeros_like(t[comp]['tfr_raw_sum'])
            sums += t[comp]['tfr_raw_sum']
            n_total += t[comp]['n_epochs']
        if sums is None or n_total == 0:
            return None, 0
        return sums / n_total, n_total

    final = {
        'time_axis_ms': per_trial[0]['time_axis_ms'],
        'freqs': per_trial[0]['freqs'],
        'alpha_band': alpha_band,
        'n_high_trials': len(high_trials),
        'n_low_trials': len(low_trials),
    }
    for comp in ['supragranular', 'granular', 'infragranular']:
        high_mean, n_high = group_mean(high_trials, comp)
        low_mean,  n_low  = group_mean(low_trials,  comp)
        if high_mean is None or low_mean is None:
            continue
        if contrast == 'logratio':
            safe_high = np.where(high_mean > 0, high_mean, 1e-30)
            safe_low  = np.where(low_mean  > 0, low_mean,  1e-30)
            contrast_tfr = 10.0 * np.log10(safe_high / safe_low)
        elif contrast == 'normdiff':
            h_base = np.mean(high_mean, axis=1, keepdims=True)
            l_base = np.mean(low_mean,  axis=1, keepdims=True)
            h_base[h_base == 0] = 1e-30
            l_base[l_base == 0] = 1e-30
            contrast_tfr = (high_mean / h_base) - (low_mean / l_base)
        elif contrast == 'diff':
            contrast_tfr = high_mean - low_mean
        else:
            raise ValueError(f"unknown contrast {contrast!r}")
        final[comp] = {
            'tfr_contrast': contrast_tfr,
            'tfr_high': high_mean,
            'tfr_low': low_mean,
            'n_epochs_high': n_high,
            'n_epochs_low': n_low,
        }
    final['contrast'] = contrast
    final['split'] = split
    return final


def plot_alpha_gamma_coupling(results, title_suffix='', save_path=None,
                              vmax_override=None):
  
    fig = plt.figure(figsize=(8, 12))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 0.5], hspace=0.35)

    panel_labels = ['A', 'B', 'C', 'D']
    compartments = ['supragranular', 'granular', 'infragranular']
    compartment_titles = [
        'Supragranular power',
        'Granular power',
        'Infragranular power',
    ]

    alpha_hi = 14.0
    if 'alpha_band' in results:
        alpha_hi = results['alpha_band'][1]

    if vmax_override is not None:
        vmax = vmax_override
    else:
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

    ax_d = fig.add_subplot(gs[3])
    if 'alpha_trace' in results:
        at = results['alpha_trace']
        ax_d.plot(at['time_axis_ms'], at['mean'], 'k-', lw=1.2)
        ax_d.axvline(0, color='gray', ls='--', alpha=0.5)
        ax_d.set_xlabel('Time relative to alpha peak (ms)')
        ax_d.set_ylabel('Bipolar LFP')
        ax_d.set_title('Infragranular alpha', fontsize=11, fontweight='bold')
    ax_d.text(-0.08, 1.05, 'D', transform=ax_d.transAxes,
              fontsize=14, fontweight='bold')

    fig.suptitle(f'Alpha Gamma Coupling {title_suffix}',
                 fontsize=13, fontweight='bold', y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved figure to {save_path}")

    plt.close(fig)
    return fig


def plot_median_split_contrast(results, title_suffix='', save_path=None):
    """Bonnefond & Jensen 2015 Fig 2D style: high-alpha-trials minus low-alpha-trials,
    peak-locked TFR contrast per cortical compartment."""
    fig = plt.figure(figsize=(8, 10))
    gs = gridspec.GridSpec(3, 1, hspace=0.35)

    compartments = ['supragranular', 'granular', 'infragranular']
    titles = ['Supragranular', 'Granular', 'Infragranular']

    vmax = 0
    for comp in compartments:
        if comp in results:
            vmax = max(vmax, np.max(np.abs(results[comp]['tfr_contrast'])))
    if vmax == 0:
        vmax = 1.0

    for i, (comp, t) in enumerate(zip(compartments, titles)):
        ax = fig.add_subplot(gs[i])
        if comp in results:
            r = results[comp]
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax.pcolormesh(
                results['time_axis_ms'], results['freqs'], r['tfr_contrast'],
                cmap='RdBu_r', norm=norm, shading='auto',
            )
            mode = results.get('contrast', 'diff')
            cbar_label = {
                'logratio': '10·log10(high / low)  [dB]',
                'normdiff': 'high − low (normalized)',
                'diff':     'high − low (raw power)',
            }.get(mode, mode)
            fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
            ax.set_title(f'{t}  (high: {r["n_epochs_high"]} ep / '
                         f'low: {r["n_epochs_low"]} ep)',
                         fontsize=11, fontweight='bold')
            alpha_hi = results.get('alpha_band', (7, 14))[1]
            ax.axhline(alpha_hi, color='white', ls='--', lw=1.2, alpha=0.8)
        else:
            ax.text(0.5, 0.5, 'No channels', transform=ax.transAxes,
                    ha='center', va='center')
        ax.set_ylabel('Frequency (Hz)')
        if i == len(compartments) - 1:
            ax.set_xlabel('Time relative to alpha peak (ms)')

    split_name = results.get('split', 'median')
    mode = results.get('contrast', 'diff')
    fig.suptitle(f'Peak-locked TFR contrast: high vs low alpha trials '
                 f'[{split_name} split, {mode}] '
                 f'(n={results["n_high_trials"]} vs {results["n_low_trials"]}) '
                 f'{title_suffix}',
                 fontsize=12, fontweight='bold', y=0.99)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved figure to {save_path}")
    plt.close(fig)
    return fig


def load_trial(fpath):
    d = np.load(fpath, allow_pickle=True)
    out = {}
    for k in d.files:
        out[k] = d[k]
    return out


def aggregate_trials(trial_dir, n_trials=None, analysis_period='all',
                     use_high_alpha=True, high_alpha_percentile=75,
                     low_alpha_percentile=25, alpha_band_select='high',
                     alpha_band=(7, 14), gamma_freqs=None,
                     window_ms=300, fs=10000, transient_ms=300,
                     warmup_ms=500, min_ha_ms=150):

    if gamma_freqs is None:
        gamma_freqs = np.arange(15, 201, 2)

    files = sorted(glob.glob(os.path.join(trial_dir, 'trial_*.npz')))
    if n_trials is not None:
        files = files[:n_trials]

    if len(files) == 0:
        raise FileNotFoundError(f"zero trial files found in {trial_dir}")

    acc = {}
    count = 0
    all_peak_times_ms = []

    for fi, fpath in enumerate(files):
        trial = load_trial(fpath)

        bipolar_matrix = trial['bipolar_matrix']
        channel_depths = trial['channel_depths']

        time_ms = trial['time_array_ms']
        dt_ms = time_ms[1] - time_ms[0]
        trial_fs = 1000.0 / dt_ms

        stim_onset = float(trial.get('stim_onset_ms', 0))
        if stim_onset == 0 and 'baseline_ms' in trial:
            stim_onset = float(trial['baseline_ms'])

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
            low_alpha_percentile=low_alpha_percentile,
            use_high_alpha=use_high_alpha,
            alpha_band_select=alpha_band_select,
            stim_onset_ms=stim_onset if analysis_period != 'all' else None,
            analysis_period=analysis_period,
            transient_ms=transient_ms,
            warmup_ms=warmup_ms,
            min_ha_ms=min_ha_ms,
        )

        if res is None:
            print(f"  skipping {os.path.basename(fpath)}: no valid alpha peaks")
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
        raise RuntimeError("no trials produced valid results.")

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

    return final


def main():
    parser = argparse.ArgumentParser(
        description='slpha-gamma coupling analysis')
    parser.add_argument('--trial_dir', type=str, required=True,
                        help='Directory containing trial_*.npz files')
    parser.add_argument('--n_trials', type=int, default=None,
                        help='Max number of trials to load (the default is all)')
    parser.add_argument('--period', type=str, default='baseline',
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
    parser.add_argument('--low_percentile', type=float, default=25.0,
                        help='Percentile for low-alpha threshold')
    parser.add_argument('--split', type=str, default='tertile',
                        choices=['median', 'tertile'],
                        help='Across-trials split for contrast (default: tertile)')
    parser.add_argument('--contrast', type=str, default='logratio',
                        choices=['logratio', 'normdiff', 'diff'],
                        help='Group contrast: logratio (default, 10·log10 dB), normdiff, or raw diff')
    parser.add_argument('--transient_ms', type=float, default=300.0,
                        help='Duration of post-stimulus transient to exclude (ms)')
    parser.add_argument('--warmup_ms', type=float, default=500.0,
                        help='Duration of initial network-settling transient to exclude (ms)')
    parser.add_argument('--min_ha_ms', type=float, default=150.0,
                        help='Minimum duration of a high-alpha segment (ms)')
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

        common_kwargs = dict(
            trial_dir=args.trial_dir,
            n_trials=args.n_trials,
            analysis_period=period,
            use_high_alpha=not args.no_high_alpha,
            high_alpha_percentile=args.percentile,
            low_alpha_percentile=args.low_percentile,
            alpha_band=(args.alpha_lo, args.alpha_hi),
            gamma_freqs=gamma_freqs,
            window_ms=args.window_ms,
            transient_ms=args.transient_ms,
            warmup_ms=args.warmup_ms,
            min_ha_ms=args.min_ha_ms,
        )

        print(f"\n=== {period} | HIGH alpha (>= p{args.percentile:.0f}) ===")
        final_high = aggregate_trials(alpha_band_select='high', **common_kwargs)

        print(f"\n=== {period} | LOW alpha (<= p{args.low_percentile:.0f}) ===")
        final_low = aggregate_trials(alpha_band_select='low', **common_kwargs)

        shared_vmax = 0
        for res in (final_high, final_low):
            for comp in ('supragranular', 'granular', 'infragranular'):
                if comp in res:
                    shared_vmax = max(shared_vmax, np.max(np.abs(res[comp]['tfr_pct'])))
        if shared_vmax == 0:
            shared_vmax = 20
        shared_vmax = min(shared_vmax, 100)

        fig_path_high = os.path.join(save_dir, f'alpha_gamma_coupling_{period}_high.png')
        plot_alpha_gamma_coupling(final_high,
                                  title_suffix=f'  [{period} | high alpha]',
                                  save_path=fig_path_high,
                                  vmax_override=shared_vmax)

        fig_path_low = os.path.join(save_dir, f'alpha_gamma_coupling_{period}_low.png')
        plot_alpha_gamma_coupling(final_low,
                                  title_suffix=f'  [{period} | low alpha]',
                                  save_path=fig_path_low,
                                  vmax_override=shared_vmax)

        print(f"\n=== {period} | median-split across trials (Bonnefond & Jensen 2015 style) ===")
        final_contrast = aggregate_trials_median_split(
            trial_dir=args.trial_dir,
            n_trials=args.n_trials,
            analysis_period=period,
            alpha_band=(args.alpha_lo, args.alpha_hi),
            gamma_freqs=gamma_freqs,
            window_ms=args.window_ms,
            transient_ms=args.transient_ms,
            warmup_ms=args.warmup_ms,
            split=args.split,
            contrast=args.contrast,
        )
        fig_path_contrast = os.path.join(
            save_dir,
            f'alpha_gamma_coupling_{period}_contrast_{args.split}_{args.contrast}.png')
        plot_median_split_contrast(final_contrast,
                                    title_suffix=f'[{period}]',
                                    save_path=fig_path_contrast)


if __name__ == '__main__':
    main()