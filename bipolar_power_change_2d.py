"""to do : merge this with the laminar_power_change.py script. make one script that plots together with both smoothed (like this script) and not smoothed (like the other script) the bipolar synaptic current and bipolar matrix. so 4 plots : smoothed and non smoothed version of boipolar and bipolar synaptic current.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import zoom, gaussian_filter

from laminar_power_change import load_trials, plot_laminar_spectral_profile


def beautiful_bipolar_2d(all_trials,
                         common,
                         lfp_key='bipolar_lfp',
                         title='Stimulus-induced laminar power change '
                               '(bipolar LFP)',
                         upsample_depth=12,
                         smooth_sigma=(0.8, 1.2),
                         clip_percentile=98,
                         outpath='figures/bipolar_power_change_2d.png'):
    """Compute a bipolar % change matrix then render a smooth 2D heatmap.

    lfp_key      : which matrix to analyze (e.g. 'bipolar_lfp' or
                   'bipolar_lfp_current').
    smooth_sigma : (depth_sigma, freq_sigma) gaussian blur after upsampling.
    """
    # Run the existing pipeline (suppress its pop-up figures) to get the matrix.
    _orig_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        f_plot, depths, _pre_db, _post_db, pct_change = (
            plot_laminar_spectral_profile(all_trials, lfp_key=lfp_key,
                                          **common))
    finally:
        plt.show = _orig_show
        plt.close('all')

    # pct_change is already flipud'd: row 0 = most superficial (top of cortex).
    pct = np.asarray(pct_change)
    depth_axis = np.linspace(depths.max(), depths.min(), pct.shape[0])

    # Smooth: upsample along depth then gaussian blur for a continuous look.
    if upsample_depth > 1 and pct.shape[0] > 1:
        pct_up = zoom(pct, (upsample_depth, 1), order=3)
        d_up = np.linspace(depth_axis[0], depth_axis[-1], pct_up.shape[0])
    else:
        pct_up = pct
        d_up = depth_axis
    pct_up = gaussian_filter(pct_up, sigma=smooth_sigma, mode='nearest')

    # Asymmetric, data-driven diverging color scale centered on 0.
    neg = pct_up[pct_up < 0]
    pos = pct_up[pct_up > 0]
    vmin = min(np.percentile(neg, 100 - clip_percentile) if neg.size else -1.0,
               -1.0)
    vmax = max(np.percentile(pos, clip_percentile) if pos.size else 1.0, 1.0)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9, 6.5), facecolor='white')
    extent = [f_plot[0], f_plot[-1], d_up[-1], d_up[0]]

    im = ax.imshow(pct_up, aspect='auto', cmap='RdBu_r', norm=norm,
                   extent=extent, origin='upper', interpolation='bilinear')

    # Explicit frequency ticks (10, 30, 50, ... Hz) within the plotted range.
    xticks = np.arange(10, f_plot[-1] + 1, 20)
    ax.set_xticks(xticks)

    ax.set_xlabel('Frequency (Hz)', fontsize=13)
    ax.set_ylabel('Cortical depth (μm)', fontsize=13)
    ax.set_title(title, fontsize=15, pad=12)
    ax.tick_params(labelsize=11)
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label('Power change (%)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f'saved {outpath}  (range [{vmin:+.0f}, {vmax:+.0f}]%)')
    return outpath


if __name__ == '__main__':
    base_path = 'results/trials_06_05-fb'
    n_trials = 20
    all_trials = load_trials(base_path, n_trials)
    print(f'Loaded {len(all_trials)} trials')

    common = dict(
        pre_window_ms=500,
        post_window_ms=500,
        post_start_ms=200,
        freq_range=(0, 120),
        log_freq=False,
        remove_mean=True,
        do_detrend=True,
    )

    beautiful_bipolar_2d(all_trials, common)

    # Bipolar synaptic-current LFP: derive it from the saved current matrix
    # (channel-wise difference), same as laminar_power_change.py does.
    if 'lfp_current_matrix' in all_trials[0]:
        for trial in all_trials:
            trial['bipolar_lfp_current'] = np.diff(
                trial['lfp_current_matrix'], axis=0)
        beautiful_bipolar_2d(
            all_trials, common,
            lfp_key='bipolar_lfp_current',
            title='Stimulus-induced laminar power change '
                  '(bipolar synaptic current)',
            outpath='figures/bipolar_current_power_change_2d.png')
    else:
        print('No lfp_current_matrix in trials -- skipping synaptic-current '
              'figure')
