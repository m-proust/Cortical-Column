"""
For each population (E, PV, SOM, VIP), make a figure with one subplot per
cortical layer. Each subplot shows the % power change vs frequency *within
the stimulated layer only* — i.e. the bipolar channels whose midpoint z
falls inside that layer are averaged.

Question this answers: "When I stimulate population X in layer L, what does
the local LFP do? Is it the same effect across layers?"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d

from prototyping.laminar_power_visualizations import (
    LAYER_Z, load_trials, compute_psd_cube, assign_layer)


POPULATIONS = ['E', 'PV', 'SOM', 'VIP']
LAYERS = ['L23', 'L4AB', 'L4C', 'L5', 'L6']

LAYER_CMAP = LinearSegmentedColormap.from_list(
    'blue_violet_pink',
    ['#2563eb', '#7c3aed', '#c026d3', '#ec4899'])
LAYER_COLORS = {n: LAYER_CMAP(i / (len(LAYERS) - 1))
                for i, n in enumerate(LAYERS)}


def within_layer_change(base_path, n_trials, layer_name,
                        lfp_key='bipolar_lfp', **psd_kwargs):
    """Load trials at base_path, return (f, mu, sem) for channels inside
    `layer_name` only.

    If `psd_kwargs['remove_aperiodic_fit']` is True, the metric is the
    *difference of log10 aperiodic-removed power* (post - pre) — i.e.
    extra oscillatory power in log10 units. Otherwise it's % change in
    raw power.

    Returns (None, None, None) if loading fails or no channels match.
    """
    try:
        trials = load_trials(base_path, n_trials)
    except FileNotFoundError:
        return None, None, None

    f, channel_z, psd_pre, psd_post, _ = compute_psd_cube(
        trials, lfp_key=lfp_key, **psd_kwargs)

    keep = np.array([assign_layer(z) == layer_name for z in channel_z])
    if not keep.any():
        return f, None, None

    pre = psd_pre[:, keep, :].mean(1)         # (T, F)
    post = psd_post[:, keep, :].mean(1)

    if psd_kwargs.get('remove_aperiodic_fit', False):
        diff = post - pre                     # already in log10 units
    else:
        diff = (post - pre) / (pre + 1e-12) * 100

    return f, diff.mean(0), diff.std(0) / np.sqrt(diff.shape[0])


def plot_population_grid(sweep_root, population, n_trials=10,
                         smooth_sigma=2.5, ylim=None, **psd_kwargs):
    """One figure for `population`, one subplot per layer."""
    aperiodic = psd_kwargs.get('remove_aperiodic_fit', False)
    ylab = ('Δ log10 oscillatory power' if aperiodic
            else 'Power change (%)')
    suptitle_suffix = ' — aperiodic removed' if aperiodic else ''

    fig, axes = plt.subplots(1, len(LAYERS), figsize=(4 * len(LAYERS), 4.5),
                             sharey=True)

    for ax, layer in zip(axes, LAYERS):
        base_path = f'{sweep_root}/{layer}_{population}'
        f, mu, sem = within_layer_change(
            base_path, n_trials, layer_name=layer, **psd_kwargs)

        if f is None:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
            ax.set_title(layer)
            continue
        if mu is None:
            ax.text(0.5, 0.5, 'no channels\nin layer',
                    transform=ax.transAxes, ha='center', va='center',
                    color='gray')
            ax.set_title(layer)
            continue

        if smooth_sigma and smooth_sigma > 0:
            mu_s = gaussian_filter1d(mu, smooth_sigma, mode='nearest')
            sem_s = gaussian_filter1d(sem, smooth_sigma, mode='nearest')
        else:
            mu_s, sem_s = mu, sem

        color = LAYER_COLORS[layer]
        ax.fill_between(f, mu_s - sem_s, mu_s + sem_s,
                        color=color, alpha=0.2, lw=0)
        ax.plot(f, mu_s, color=color, lw=2.4)
        ax.axhline(0, color='k', lw=0.6, alpha=0.6)
        for x_ref in (10, 40):
            ax.axvline(x_ref, color='gray', lw=0.4, ls='--', alpha=0.5)
        ax.set_title(f'{layer} (stim {population} → record {layer})',
                     fontsize=10)
        ax.set_xlabel('Frequency (Hz)')
        ax.grid(True, alpha=0.3)
        if ylim is not None:
            ax.set_ylim(ylim)

    axes[0].set_ylabel(ylab)
    fig.suptitle(f'Within-layer effect of stimulating {population}'
                 f'{suptitle_suffix}', fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    plt.rcParams.update({
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
        'figure.dpi': 110,
    })

    sweep_root = 'results/trials2_pop_sweep'
    n_trials = 10
    psd_kwargs = dict(pre_window_ms=300, post_window_ms=300,
                      post_start_ms=200, freq_range=(1, 120))

    # Raw % power change
    for pop in POPULATIONS:
        plot_population_grid(sweep_root, pop, n_trials=n_trials, **psd_kwargs)

    # Aperiodic-removed (oscillatory residual)
    for pop in POPULATIONS:
        plot_population_grid(sweep_root, pop, n_trials=n_trials,
                             remove_aperiodic_fit=True,
                             aperiodic_fit_range=(2, 80),
                             **psd_kwargs)

    plt.show()
