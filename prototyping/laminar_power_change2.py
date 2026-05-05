import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import signal
from scipy.signal import detrend
from scipy.signal.windows import dpss
from fooof import FOOOF, FOOOFGroup
import seaborn as sns


plt.rcParams.update({
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
})
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Paired')


def load_trials(base_path, n_trials):
    all_trials = []
    for trial_idx in range(n_trials):
        fname = f"{base_path}/trial_{trial_idx:03d}.npz"
        data = np.load(fname, allow_pickle=True)

        trial_data = {
            'time': data['time_array_ms'],
            'bipolar_lfp': data['bipolar_matrix'],
            'lfp_matrix': data['lfp_matrix'],
            'rate_data': (data['rate_data'].item()
                          if data['rate_data'].size == 1
                          else data['rate_data']),
            'baseline_ms': float(data['baseline_ms']),
            'stim_onset_ms': float(data['stim_onset_ms']),
            'channel_labels': data['channel_labels'],
            'channel_depths': data['channel_depths'],
            'electrode_positions': data['electrode_positions'],
        }

        if 'lfp_current_matrix' in data.files:
            trial_data['lfp_current_matrix'] = data['lfp_current_matrix']
            trial_data['time_current_ms'] = data['time_current_ms']

        if 'mazzoni_lfp_matrix' in data.files:
            trial_data['mazzoni_lfp_matrix'] = data['mazzoni_lfp_matrix']
            trial_data['mazzoni_time_ms'] = data['mazzoni_time_ms']
            trial_data['mazzoni_layer_names'] = data['mazzoni_layer_names']

        all_trials.append(trial_data)
    return all_trials


def multitaper_psd(data, fs, NW=2, nfft=None):
    data_demeaned = data - np.mean(data)
    if nfft is None:
        nfft = 2 ** int(np.ceil(np.log2(len(data_demeaned))))
    K = int(2 * NW - 1)
    tapers = dpss(len(data_demeaned), NW, K)
    psds = []
    for taper in tapers:
        freqs, psd_single = signal.periodogram(
            data_demeaned * taper, fs=fs, nfft=nfft, scaling='density',
        )
        psds.append(psd_single)
    return freqs, np.mean(psds, axis=0)


def _time_vector_for_key(trial, lfp_key):
    if lfp_key in ('lfp_current_matrix', 'bipolar_lfp_current'):
        if 'time_current_ms' not in trial:
            raise KeyError("trial has no 'time_current_ms'")
        return np.asarray(trial['time_current_ms'])
    return np.asarray(trial['time'])


def _compute_trial_avg_psd(all_trials, lfp_key,
                           pre_window_ms, post_window_ms, post_start_ms,
                           do_detrend, remove_mean):
    """Returns f, psd_pre[ch,f], psd_post[ch,f] averaged across trials."""
    t0 = _time_vector_for_key(all_trials[0], lfp_key)
    fs = 1000.0 / float(np.mean(np.diff(t0)))
    n_channels = all_trials[0][lfp_key].shape[0]

    all_psd_pre, all_psd_post = [], []
    f = None

    for ch in range(n_channels):
        pre_trials, post_trials = [], []
        for trial in all_trials:
            lfp = trial[lfp_key][ch]
            time = _time_vector_for_key(trial, lfp_key)
            stim = trial['stim_onset_ms']

            pre_mask = (time >= stim - pre_window_ms) & (time < stim)
            post_mask = ((time >= stim + post_start_ms) &
                         (time < stim + post_start_ms + post_window_ms))

            pre = lfp[pre_mask].copy()
            post = lfp[post_mask].copy()
            if len(pre) == 0 or len(post) == 0:
                continue
            if np.any(np.isnan(pre)) or np.any(np.isnan(post)):
                continue

            if do_detrend:
                pre = detrend(pre)
                post = detrend(post)
            elif remove_mean:
                pre -= np.mean(pre)
                post -= np.mean(post)

            nfft = 2 ** int(np.ceil(np.log2(min(len(pre), len(post)))))
            f, psd_pre = multitaper_psd(pre, fs=fs, NW=2, nfft=nfft)
            _, psd_post = multitaper_psd(post, fs=fs, NW=2, nfft=nfft)

            pre_trials.append(psd_pre)
            post_trials.append(psd_post)

        if len(pre_trials) == 0:
            all_psd_pre.append(None)
            all_psd_post.append(None)
        else:
            all_psd_pre.append(np.mean(pre_trials, axis=0))
            all_psd_post.append(np.mean(post_trials, axis=0))

    if f is None:
        raise RuntimeError('zero valid trials in any channel')

    n_freq = len(f)
    for i in range(n_channels):
        if all_psd_pre[i] is None:
            all_psd_pre[i] = np.full(n_freq, np.nan)
            all_psd_post[i] = np.full(n_freq, np.nan)

    return f, np.array(all_psd_pre), np.array(all_psd_post), fs


def _fit_fooof_group(psd_matrix, freqs, freq_range, fooof_kwargs):
    """Fit FOOOF on each row. Rows that are all-NaN are skipped (NaN result)."""
    n_ch, _ = psd_matrix.shape
    valid_mask = ~np.any(np.isnan(psd_matrix), axis=1)

    fg = FOOOFGroup(**fooof_kwargs)
    if np.any(valid_mask):
        fg.fit(freqs, psd_matrix[valid_mask], freq_range)
    else:
        raise RuntimeError('no valid channels to fit')

    freqs_fit = fg.freqs

    n_freq_fit = len(freqs_fit)
    aperiodic = np.full((n_ch, n_freq_fit), np.nan)
    full_fit = np.full((n_ch, n_freq_fit), np.nan)
    psd_fit_log = np.full((n_ch, n_freq_fit), np.nan)
    exponents = np.full(n_ch, np.nan)
    offsets = np.full(n_ch, np.nan)
    r_squared = np.full(n_ch, np.nan)

    valid_indices = np.where(valid_mask)[0]
    for out_i, ch in enumerate(valid_indices):
        fm = fg.get_fooof(out_i, regenerate=True)
        aperiodic[ch] = fm._ap_fit
        full_fit[ch] = fm.fooofed_spectrum_
        psd_fit_log[ch] = fm.power_spectrum
        ap_params = fm.get_params('aperiodic_params')
        offsets[ch] = ap_params[0]
        exponents[ch] = ap_params[-1]
        r_squared[ch] = fm.get_params('r_squared')

    return {
        'freqs': freqs_fit,
        'aperiodic_log': aperiodic,
        'full_fit_log': full_fit,
        'psd_log': psd_fit_log,
        'periodic_log': full_fit - aperiodic,
        'exponents': exponents,
        'offsets': offsets,
        'r_squared': r_squared,
        'valid_mask': valid_mask,
    }


def plot_laminar_spectral_profile_fooof(all_trials,
                                        pre_window_ms=1000,
                                        post_window_ms=1000,
                                        post_start_ms=500,
                                        freq_range=(2, 100),
                                        log_freq=True,
                                        remove_mean=True,
                                        do_detrend=True,
                                        lfp_key='bipolar_lfp',
                                        title_suffix=None,
                                        fooof_kwargs=None):
    """Same maps as plot_laminar_spectral_profile, but on the FOOOF
    PERIODIC component (oscillatory power above 1/f), so the % change
    reflects true oscillation changes rather than aperiodic shifts."""
    if fooof_kwargs is None:
        fooof_kwargs = dict(peak_width_limits=(2.0, 12.0),
                            max_n_peaks=6,
                            min_peak_height=0.05,
                            aperiodic_mode='fixed',
                            verbose=False)

    for i, tr in enumerate(all_trials):
        if lfp_key not in tr:
            raise KeyError(
                f"Trial {i} missing key '{lfp_key}'. available: "
                f"{[k for k in tr.keys() if 'lfp' in k or 'matrix' in k]}"
            )

    f, psd_pre, psd_post, _fs = _compute_trial_avg_psd(
        all_trials, lfp_key,
        pre_window_ms, post_window_ms, post_start_ms,
        do_detrend, remove_mean,
    )

    n_channels = psd_pre.shape[0]
    channel_depths = all_trials[0]['channel_depths']
    if lfp_key == 'bipolar_lfp' and len(channel_depths) > n_channels:
        depths = (channel_depths[:-1] + channel_depths[1:]) / 2
    else:
        depths = channel_depths[:n_channels]

    print(f'  fitting FOOOF (pre)  ... {n_channels} channels')
    fit_pre = _fit_fooof_group(psd_pre, f, freq_range, fooof_kwargs)
    print(f'  fitting FOOOF (post) ... {n_channels} channels')
    fit_post = _fit_fooof_group(psd_post, f, freq_range, fooof_kwargs)

    f_fit = fit_pre['freqs']
    periodic_pre = fit_pre['periodic_log']
    periodic_post = fit_post['periodic_log']

    periodic_pre_lin = 10 ** periodic_pre
    periodic_post_lin = 10 ** periodic_post

    pct_change = (periodic_post_lin - periodic_pre_lin) / \
                  np.maximum(periodic_pre_lin, 1e-12) * 100

    periodic_pre_disp = np.flipud(periodic_pre)
    periodic_post_disp = np.flipud(periodic_post)
    pct_change_disp = np.flipud(pct_change)

    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.15], wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    extent = [f_fit[0], f_fit[-1], -0.5, n_channels - 0.5]

    vmin_p = np.nanpercentile([periodic_pre_disp, periodic_post_disp], 5)
    vmax_p = np.nanpercentile([periodic_pre_disp, periodic_post_disp], 95)

    im1 = axes[0].imshow(periodic_pre_disp, aspect='auto', cmap='viridis',
                         extent=extent, origin='upper',
                         vmin=vmin_p, vmax=vmax_p)
    axes[0].set_title('Baseline (periodic, log10)')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Laminar depth')
    plt.colorbar(im1, ax=axes[0], label='log10 periodic power')

    im2 = axes[1].imshow(periodic_post_disp, aspect='auto', cmap='viridis',
                         extent=extent, origin='upper',
                         vmin=vmin_p, vmax=vmax_p)
    axes[1].set_title('Post-stimulus (periodic, log10)')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Laminar depth')
    plt.colorbar(im2, ax=axes[1], label='log10 periodic power')

    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
    im3 = axes[2].imshow(pct_change_disp, aspect='auto', cmap='RdBu_r',
                         extent=extent, origin='upper', norm=norm)
    axes[2].set_title('Stimulus-induced change in periodic (%)')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Laminar depth')
    cbar = plt.colorbar(im3, ax=axes[2], label='% Change (periodic)')
    cbar.set_ticks([-100, -50, 0, 100])

    if log_freq:
        for ax in axes:
            ax.set_xscale('log')

    if title_suffix is None:
        title_suffix = {
            'bipolar_lfp': ' (Bipolar)',
            'lfp_matrix': ' (Kernel method)',
            'lfp_current_matrix': ' (Synaptic-current method)',
        }.get(lfp_key, f' ({lfp_key})')

    plt.suptitle('Laminar FOOOF Periodic Profile' + title_suffix, fontsize=16)
    plt.show()

    return {
        'freqs': f_fit,
        'depths': depths,
        'fit_pre': fit_pre,
        'fit_post': fit_post,
        'pct_change_periodic': pct_change,
    }


def plot_aperiodic_summary(fit_pre, fit_post, depths, title_suffix=''):
    """Per-channel aperiodic exponent and offset (pre vs post)."""
    n_ch = len(fit_pre['exponents'])
    y = np.arange(n_ch)

    fig, axes = plt.subplots(1, 3, figsize=(14, 6),
                             gridspec_kw={'wspace': 0.4})

    axes[0].plot(fit_pre['exponents'], y, 'o-', label='pre', color='steelblue')
    axes[0].plot(fit_post['exponents'], y, 's-', label='post',
                 color='firebrick')
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Aperiodic exponent')
    axes[0].set_ylabel('Channel index (superficial → deep)')
    axes[0].set_title('1/f exponent')
    axes[0].legend()

    axes[1].plot(fit_pre['offsets'], y, 'o-', label='pre', color='steelblue')
    axes[1].plot(fit_post['offsets'], y, 's-', label='post', color='firebrick')
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Aperiodic offset')
    axes[1].set_title('1/f offset')
    axes[1].legend()

    axes[2].plot(fit_pre['r_squared'], y, 'o-', label='pre', color='steelblue')
    axes[2].plot(fit_post['r_squared'], y, 's-', label='post',
                 color='firebrick')
    axes[2].invert_yaxis()
    axes[2].set_xlabel('R²')
    axes[2].set_xlim(0, 1.02)
    axes[2].set_title('Fit quality')
    axes[2].legend()

    plt.suptitle('Aperiodic parameters by depth' + title_suffix, fontsize=14)
    plt.show()


def plot_fooof_fits_per_channel(fit_pre, fit_post, depths,
                                title_suffix='',
                                max_cols=4):
    """One subplot per channel, showing the empirical PSD, full FOOOF fit,
    and aperiodic component, for both pre and post. This is what you
    inspect to see whether FOOOF is fitting reasonably."""
    n_ch = fit_pre['psd_log'].shape[0]
    n_cols = min(max_cols, n_ch)
    n_rows = int(np.ceil(n_ch / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows),
                             squeeze=False)
    f_fit = fit_pre['freqs']

    for ch in range(n_ch):
        r, c = ch // n_cols, ch % n_cols
        ax = axes[r, c]

        if not fit_pre['valid_mask'][ch]:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        ax.plot(f_fit, fit_pre['psd_log'][ch],
                color='steelblue', lw=1.2, alpha=0.9, label='PSD pre')
        ax.plot(f_fit, fit_pre['full_fit_log'][ch],
                color='steelblue', lw=2, ls='--', label='fit pre')
        ax.plot(f_fit, fit_pre['aperiodic_log'][ch],
                color='steelblue', lw=1, ls=':', label='1/f pre')

        ax.plot(f_fit, fit_post['psd_log'][ch],
                color='firebrick', lw=1.2, alpha=0.9, label='PSD post')
        ax.plot(f_fit, fit_post['full_fit_log'][ch],
                color='firebrick', lw=2, ls='--', label='fit post')
        ax.plot(f_fit, fit_post['aperiodic_log'][ch],
                color='firebrick', lw=1, ls=':', label='1/f post')

        depth_str = ''
        try:
            depth_str = f'  d={float(depths[ch]):.0f}'
        except Exception:
            pass
        exp_pre = fit_pre['exponents'][ch]
        exp_post = fit_post['exponents'][ch]
        r2_pre = fit_pre['r_squared'][ch]
        r2_post = fit_post['r_squared'][ch]
        ax.set_title(
            f'ch{ch}{depth_str} | exp {exp_pre:.2f}/{exp_post:.2f} '
            f'| R² {r2_pre:.2f}/{r2_post:.2f}',
            fontsize=9,
        )
        ax.set_xlabel('Hz')
        ax.set_ylabel('log10 power')
        if ch == 0:
            ax.legend(fontsize=7, loc='lower left')

    for ch in range(n_ch, n_rows * n_cols):
        axes[ch // n_cols, ch % n_cols].axis('off')

    plt.suptitle('FOOOF fits — PSD vs. full fit vs. aperiodic'
                 + title_suffix, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


if __name__ == '__main__':
    base_path = 'results/trials2_pop_sweep/L6_SOM'
    n_trials = 10

    all_trials = load_trials(base_path, n_trials)

    print(f'Loaded {len(all_trials)} trials')
    print(f"Time range: {all_trials[0]['time'][0]:.1f} to "
          f"{all_trials[0]['time'][-1]:.1f} ms")
    print(f"Stimulus onset: {all_trials[0]['stim_onset_ms']:.1f} ms")

    common = dict(
        pre_window_ms=500,
        post_window_ms=500,
        post_start_ms=500,
        freq_range=(2, 120),
        log_freq=False,
        remove_mean=True,
        do_detrend=True,
    )

    fooof_kwargs = dict(
        peak_width_limits=(2.0, 12.0),
        max_n_peaks=6,
        min_peak_height=0.05,
        aperiodic_mode='fixed',
        verbose=False,
    )

    for key in ('bipolar_lfp', 'lfp_matrix'):
        out = plot_laminar_spectral_profile_fooof(
            all_trials, lfp_key=key,
            fooof_kwargs=fooof_kwargs, **common,
        )
        plot_aperiodic_summary(out['fit_pre'], out['fit_post'], out['depths'],
                               title_suffix=f' — {key}')
        plot_fooof_fits_per_channel(out['fit_pre'], out['fit_post'],
                                    out['depths'],
                                    title_suffix=f' — {key}')

    if 'lfp_current_matrix' in all_trials[0]:
        out = plot_laminar_spectral_profile_fooof(
            all_trials, lfp_key='lfp_current_matrix',
            fooof_kwargs=fooof_kwargs, **common,
        )
        plot_aperiodic_summary(out['fit_pre'], out['fit_post'], out['depths'],
                               title_suffix=' — lfp_current_matrix')
        plot_fooof_fits_per_channel(out['fit_pre'], out['fit_post'],
                                    out['depths'],
                                    title_suffix=' — lfp_current_matrix')

        for trial in all_trials:
            lfp_cur = trial['lfp_current_matrix']
            trial['bipolar_lfp_current'] = np.diff(lfp_cur, axis=0)

        out = plot_laminar_spectral_profile_fooof(
            all_trials, lfp_key='bipolar_lfp_current',
            title_suffix=' (Bipolar synaptic current)',
            fooof_kwargs=fooof_kwargs, **common,
        )
        plot_aperiodic_summary(out['fit_pre'], out['fit_post'], out['depths'],
                               title_suffix=' — bipolar_lfp_current')
        plot_fooof_fits_per_channel(out['fit_pre'], out['fit_post'],
                                    out['depths'],
                                    title_suffix=' — bipolar_lfp_current')
    else:
        print('\nNo lfp_current_matrix found in trials')
