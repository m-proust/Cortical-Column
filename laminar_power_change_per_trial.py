import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import detrend
import seaborn as sns

from laminar_power_change import (
    load_trials,
    multitaper_psd,
    _time_vector_for_key,
)


plt.rcParams.update({
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
})
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Paired')


def plot_single_trial_bipolar(trial,
                              trial_idx,
                              pre_window_ms=500,
                              post_window_ms=500,
                              post_start_ms=300,
                              freq_range=(0, 120),
                              log_freq=False,
                              do_detrend=True,
                              remove_mean=True,
                              lfp_key='bipolar_lfp',
                              save_dir=None,
                              show=True):
    if lfp_key not in trial:
        raise KeyError(
            f"Trial {trial_idx} has no key '{lfp_key}'. Available: "
            f"{[k for k in trial.keys() if 'lfp' in k or 'matrix' in k]}"
        )

    time = _time_vector_for_key(trial, lfp_key)
    fs = 1000.0 / float(np.mean(np.diff(time)))
    stim = float(trial['stim_onset_ms'])

    lfp = trial[lfp_key]
    n_channels = lfp.shape[0]
    channel_depths = trial['channel_depths']

    if lfp_key == 'bipolar_lfp' and len(channel_depths) > n_channels:
        depths = (channel_depths[:-1] + channel_depths[1:]) / 2
    else:
        depths = channel_depths[:n_channels]

    pre_mask = (time >= stim - pre_window_ms) & (time < stim)
    post_mask = ((time >= stim + post_start_ms) &
                 (time < stim + post_start_ms + post_window_ms))

    psd_pre_list, psd_post_list = [], []
    f = None
    for ch in range(n_channels):
        pre = lfp[ch][pre_mask].copy()
        post = lfp[ch][post_mask].copy()

        if len(pre) == 0 or len(post) == 0 or \
           np.any(np.isnan(pre)) or np.any(np.isnan(post)):
            psd_pre_list.append(None)
            psd_post_list.append(None)
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
        psd_pre_list.append(psd_pre)
        psd_post_list.append(psd_post)

    if f is None:
        print(f'Trial {trial_idx}: no valid channels, skipping')
        return None

    n_freq = len(f)
    for i in range(n_channels):
        if psd_pre_list[i] is None:
            psd_pre_list[i] = np.full(n_freq, np.nan)
            psd_post_list[i] = np.full(n_freq, np.nan)

    psd_pre = np.array(psd_pre_list)
    psd_post = np.array(psd_post_list)

    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    f_plot = f[freq_mask]
    psd_pre = psd_pre[:, freq_mask]
    psd_post = psd_post[:, freq_mask]

    psd_pre_db = 10 * np.log10(psd_pre + 1e-10)
    psd_post_db = 10 * np.log10(psd_post + 1e-10)
    pct_change = (psd_post - psd_pre) / psd_pre * 100

    psd_pre_db = np.flipud(psd_pre_db)
    psd_post_db = np.flipud(psd_post_db)
    pct_change = np.flipud(pct_change)

    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    extent = [f_plot[0], f_plot[-1], -0.5, n_channels - 0.5]

    finite_vals = np.concatenate([psd_pre_db[np.isfinite(psd_pre_db)],
                                  psd_post_db[np.isfinite(psd_post_db)]])
    if finite_vals.size:
        vmin_db = np.percentile(finite_vals, 5)
        vmax_db = np.percentile(finite_vals, 95)
    else:
        vmin_db, vmax_db = None, None

    im1 = axes[0].imshow(psd_pre_db, aspect='auto', cmap='viridis',
                         extent=extent, origin='upper',
                         vmin=vmin_db, vmax=vmax_db)
    axes[0].set_title('Baseline')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Laminar depth')
    plt.colorbar(im1, ax=axes[0], label='Power (dB)')

    im2 = axes[1].imshow(psd_post_db, aspect='auto', cmap='viridis',
                         extent=extent, origin='upper',
                         vmin=vmin_db, vmax=vmax_db)
    axes[1].set_title('Post-stimulus')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Laminar depth')
    plt.colorbar(im2, ax=axes[1], label='Power (dB)')

    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=500)
    im3 = axes[2].imshow(pct_change, aspect='auto', cmap='RdBu_r',
                         extent=extent, origin='upper', norm=norm)
    axes[2].set_title('Stimulus-induced change (%)')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Laminar depth')
    cbar = plt.colorbar(im3, ax=axes[2], label='% Change')
    cbar.set_ticks([-100, -50, 0, 100])

    if log_freq:
        for ax in axes:
            ax.set_xscale('log')

    plt.suptitle(f'Laminar Spectral Profile (Bipolar) — Trial {trial_idx:03d}',
                 fontsize=16)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, f'trial_{trial_idx:03d}_bipolar.png')
        fig.savefig(out, dpi=130, bbox_inches='tight')
        print(f'  saved {out}')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return f_plot, depths, psd_pre_db, psd_post_db, pct_change


if __name__ == '__main__':
    base_path = 'results/trials3_03-06'
    n_trials = 14
    save_dir = os.path.join(base_path, 'per_trial_bipolar')

    all_trials = load_trials(base_path, n_trials)
    print(f'Loaded {len(all_trials)} trials')

    common = dict(
        pre_window_ms=500,
        post_window_ms=500,
        post_start_ms=300,
        freq_range=(0, 120),
        log_freq=False,
        remove_mean=True,
        do_detrend=True,
        lfp_key='bipolar_lfp',
        save_dir=save_dir,
        show=False,
    )

    for i, tr in enumerate(all_trials):
        print(f'Trial {i:03d}')
        plot_single_trial_bipolar(tr, trial_idx=i, **common)
