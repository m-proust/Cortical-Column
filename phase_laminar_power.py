"""Laminar spectral profile per alpha phase.

Loads all phase-locked trials, groups by target_phase_deg (pooling across seeds),
and runs the same multitaper laminar pre/post power analysis as
laminar_power_change.py — one figure per phase.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import detrend
import seaborn as sns

from laminar_power_change import multitaper_psd, _time_vector_for_key

plt.rcParams.update({
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
})
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Paired')


RESULTS_DIR = 'results/trials_phase_07_05'
PHASES = (0, 45, 90, 135, 180, 225, 270, 315)


def load_phase_trials(results_dir, phase_deg):
    pat = os.path.join(results_dir, f"trial_seed*_phase{int(phase_deg):03d}.npz")
    files = sorted(glob.glob(pat))
    trials = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        tr = {
            'time': d['time_array_ms'],
            'bipolar_lfp': d['bipolar_matrix'],
            'lfp_matrix': d['lfp_matrix'],
            'baseline_ms': float(d['baseline_ms']),
            'stim_onset_ms': float(d['stim_onset_ms']),
            'channel_labels': d['channel_labels'],
            'channel_depths': d['channel_depths'],
            'electrode_positions': d['electrode_positions'],
            'phase_deg': float(d['target_phase_deg']),
        }
        if 'lfp_current_matrix' in d.files:
            tr['lfp_current_matrix'] = d['lfp_current_matrix']
            tr['time_current_ms'] = d['time_current_ms']
        trials.append(tr)
    return trials


def compute_laminar_psd(all_trials, lfp_key, pre_window_ms, post_window_ms,
                        post_start_ms, freq_range, do_detrend):
    """Same logic as plot_laminar_spectral_profile but returns arrays only."""
    t0 = _time_vector_for_key(all_trials[0], lfp_key)
    fs = 1000.0 / float(np.mean(np.diff(t0)))
    n_channels = all_trials[0][lfp_key].shape[0]
    channel_depths = all_trials[0]['channel_depths']

    if lfp_key == 'bipolar_lfp' and len(channel_depths) > n_channels:
        depths = (channel_depths[:-1] + channel_depths[1:]) / 2
    else:
        depths = channel_depths[:n_channels]

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
            else:
                pre -= np.mean(pre)
                post -= np.mean(post)
            nfft = 2 ** int(np.ceil(np.log2(min(len(pre), len(post)))))
            f, psd_pre = multitaper_psd(pre, fs=fs, NW=2, nfft=nfft)
            _, psd_post = multitaper_psd(post, fs=fs, NW=2, nfft=nfft)
            pre_trials.append(psd_pre)
            post_trials.append(psd_post)
        if pre_trials:
            all_psd_pre.append(np.mean(pre_trials, axis=0))
            all_psd_post.append(np.mean(post_trials, axis=0))
        else:
            all_psd_pre.append(None)
            all_psd_post.append(None)

    if f is None:
        return None
    n_freq = len(f)
    for i in range(n_channels):
        if all_psd_pre[i] is None:
            all_psd_pre[i] = np.full(n_freq, np.nan)
            all_psd_post[i] = np.full(n_freq, np.nan)
    psd_pre = np.array(all_psd_pre)
    psd_post = np.array(all_psd_post)
    fmask = (f >= freq_range[0]) & (f <= freq_range[1])
    return f[fmask], depths, psd_pre[:, fmask], psd_post[:, fmask]


def plot_phase_grid(results_dir, phases, lfp_key, save_path,
                    pre_window_ms=500, post_window_ms=500,
                    post_start_ms=300, freq_range=(0, 120),
                    do_detrend=True, log_freq=False):
    """One row per phase, three columns: pre, post, %change."""
    n_phases = len(phases)

    # First pass: compute everything to set common color scales
    all_results = {}
    for ph in phases:
        trials = load_phase_trials(results_dir, ph)
        if not trials:
            print(f"  phase {ph}: no trials, skipping")
            continue
        if lfp_key not in trials[0]:
            print(f"  phase {ph}: trial missing key {lfp_key}, skipping")
            continue
        if lfp_key == 'bipolar_lfp_current':
            for trial in trials:
                trial['bipolar_lfp_current'] = np.diff(
                    trial['lfp_current_matrix'], axis=0)
        res = compute_laminar_psd(
            trials, lfp_key, pre_window_ms, post_window_ms,
            post_start_ms, freq_range, do_detrend)
        if res is None:
            continue
        f_plot, depths, psd_pre, psd_post = res
        psd_pre_db = 10 * np.log10(psd_pre + 1e-10)
        psd_post_db = 10 * np.log10(psd_post + 1e-10)
        pct = (psd_post - psd_pre) / psd_pre * 100
        all_results[ph] = {
            'f': f_plot, 'depths': depths,
            'pre_db': np.flipud(psd_pre_db),
            'post_db': np.flipud(psd_post_db),
            'pct': np.flipud(pct),
            'n_trials': len(trials),
        }

    if not all_results:
        print(f"No data for {lfp_key}")
        return

    pres = np.concatenate([r['pre_db'].ravel() for r in all_results.values()])
    posts = np.concatenate([r['post_db'].ravel() for r in all_results.values()])
    pres = pres[np.isfinite(pres)]
    posts = posts[np.isfinite(posts)]
    vmin_db = np.percentile(np.concatenate([pres, posts]), 5)
    vmax_db = np.percentile(np.concatenate([pres, posts]), 95)

    fig, axes = plt.subplots(n_phases, 3, figsize=(15, 2.4 * n_phases),
                             squeeze=False)
    for row, ph in enumerate(phases):
        if ph not in all_results:
            for c in range(3):
                axes[row, c].axis('off')
                axes[row, c].text(0.5, 0.5, f"phase {ph}: no data",
                                  ha='center', va='center')
            continue
        r = all_results[ph]
        f_plot = r['f']
        n_ch = r['pre_db'].shape[0]
        extent = [f_plot[0], f_plot[-1], -0.5, n_ch - 0.5]

        im1 = axes[row, 0].imshow(r['pre_db'], aspect='auto', cmap='viridis',
                                  extent=extent, origin='upper',
                                  vmin=vmin_db, vmax=vmax_db)
        im2 = axes[row, 1].imshow(r['post_db'], aspect='auto', cmap='viridis',
                                  extent=extent, origin='upper',
                                  vmin=vmin_db, vmax=vmax_db)
        norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=500)
        im3 = axes[row, 2].imshow(r['pct'], aspect='auto', cmap='RdBu_r',
                                  extent=extent, origin='upper', norm=norm)

        axes[row, 0].set_ylabel(f"{ph}°  (n={r['n_trials']})\nLaminar depth")
        if row == 0:
            axes[row, 0].set_title('Baseline (dB)')
            axes[row, 1].set_title('Post-stim (dB)')
            axes[row, 2].set_title('% change')
        if row == n_phases - 1:
            for c in range(3):
                axes[row, c].set_xlabel('Frequency (Hz)')
        if log_freq:
            for c in range(3):
                axes[row, c].set_xscale('log')

    fig.colorbar(im1, ax=axes[:, 0].tolist(), label='Power (dB)',
                 shrink=0.6, pad=0.02)
    fig.colorbar(im2, ax=axes[:, 1].tolist(), label='Power (dB)',
                 shrink=0.6, pad=0.02)
    fig.colorbar(im3, ax=axes[:, 2].tolist(), label='% change',
                 shrink=0.6, pad=0.02)

    plt.suptitle(f'Laminar spectral profile by alpha phase ({lfp_key})',
                 fontsize=14, y=1.0)
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  saved {save_path}")


if __name__ == '__main__':
    common = dict(
        pre_window_ms=500,
        post_window_ms=500,
        post_start_ms=300,
        freq_range=(0, 120),
        log_freq=False,
        do_detrend=True,
    )
    out_bipolar = os.path.join(RESULTS_DIR, 'phase_laminar_bipolar.png')
    plot_phase_grid(RESULTS_DIR, PHASES, 'bipolar_lfp', out_bipolar, **common)

    sample = load_phase_trials(RESULTS_DIR, PHASES[0])
    if sample and 'lfp_current_matrix' in sample[0]:
        out_cur = os.path.join(RESULTS_DIR, 'phase_laminar_current.png')
        plot_phase_grid(RESULTS_DIR, PHASES, 'lfp_current_matrix',
                        out_cur, **common)
