import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from config.config2 import CONFIG



def gaussian_kernel(bin_width, smooth_sigma=0.010, nsigma=5):
    sigma_bins = smooth_sigma / bin_width
    half = int(np.ceil(nsigma * sigma_bins))
    t = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (t / sigma_bins) ** 2)
    k /= k.sum()
    return k


def smooth_psth(psth, bin_width, smooth_sigma=0.010):
    k = gaussian_kernel(bin_width, smooth_sigma)
    return np.apply_along_axis(lambda x: np.convolve(x, k, mode='same'), 0, psth)


def build_psth_from_arrays(mua_t_ms, mua_i, n_channels, stim_onset_ms,
                           t_pre_ms, t_post_ms, bin_width_ms):
    t_pre = t_pre_ms / 1000.0 
    t_post = t_post_ms / 1000.0
    bin_width = bin_width_ms / 1000.0
    stim_onset_s = stim_onset_ms / 1000.0

    t_edges = np.arange(-t_pre, t_post + bin_width, bin_width)
    t_centers = t_edges[:-1] + bin_width / 2

    n_bins = len(t_edges) - 1
    spike_times_s = mua_t_ms / 1000.0

    counts = np.zeros((n_bins, n_channels))
    for ch in range(n_channels):
        ch_mask = (mua_i == ch)
        ch_times = spike_times_s[ch_mask] - stim_onset_s
        if len(ch_times) > 0:
            counts[:, ch] = np.histogram(ch_times, bins=t_edges)[0]

    return counts, t_centers, bin_width


def assign_compartments(probe_coords_mm, config):
    z = probe_coords_mm[:, 2]

    layers = config['layers']
    sg_zmin = layers['L23']['coordinates']['z'][0]
    sg_zmax = layers['L23']['coordinates']['z'][1]
    g_zmin = min(layers['L4C']['coordinates']['z'][0],
                 layers['L4AB']['coordinates']['z'][0])
    g_zmax = max(layers['L4C']['coordinates']['z'][1],
                 layers['L4AB']['coordinates']['z'][1])
    ig_zmin = min(layers['L6']['coordinates']['z'][0],
                  layers['L5']['coordinates']['z'][0])
    ig_zmax = max(layers['L6']['coordinates']['z'][1],
                  layers['L5']['coordinates']['z'][1])

    masks = {
        'SG': (z >= sg_zmin) & (z <= sg_zmax),
        'G':  (z >= g_zmin) & (z <= g_zmax),
        'IG': (z >= ig_zmin) & (z <= ig_zmax),
    }
    return masks



def load_trials(trial_dir):
    files = sorted(glob.glob(os.path.join(trial_dir, "trial_*.npz")))
    if not files:
        raise FileNotFoundError(f"No trial files in {trial_dir}")
    trials = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        trials.append(d)
    print(f"Loaded {len(trials)} trials from {trial_dir}")
    return trials



def plot_cleo_psth(trial_dir,
                   t_pre_ms=500,
                   t_post_ms=2000,
                   bin_width_ms=2,
                   smooth_sigma=0.010,
                   config=CONFIG):

    trials = load_trials(trial_dir)

    t0 = trials[0]
    n_channels = int(t0['n_channels'])
    stim_onset_ms = float(t0['stim_onset_ms'])
    probe_coords_mm = t0['probe_coords_mm']

    all_counts = []
    for d in trials:
        counts, t_centers, bin_width = build_psth_from_arrays(
            d['mua_t_ms'], d['mua_i'], n_channels,
            float(d['stim_onset_ms']), t_pre_ms, t_post_ms, bin_width_ms,
        )
        all_counts.append(counts)

    mean_counts = np.mean(all_counts, axis=0)
    psth = mean_counts / bin_width  
    psth = smooth_psth(psth, bin_width, smooth_sigma)

    t_ms = t_centers * 1000.0  
    depths_mm = probe_coords_mm[:, 2]


    fig1, axes1 = plt.subplots(n_channels, 1, figsize=(8, 1.3 * n_channels),
                               sharex=True, constrained_layout=True)
    if n_channels == 1:
        axes1 = [axes1]

    for ch in range(n_channels):
        ax = axes1[ch]
        ax.plot(t_ms, psth[:, ch], 'k-', linewidth=0.8)
        ax.axvline(0, color='r', ls='--', lw=0.8, alpha=0.6)
        ax.set_ylabel(f'Ch{ch}\n({depths_mm[ch]*1000:.0f}μm)', fontsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ch == 0:
            ax.set_title(
                f'Per-channel PSTH (MUA, {len(trials)} trials avg)',
                fontsize=11, fontweight='bold')

    axes1[-1].set_xlabel('Time from stimulus onset (ms)')
    fig1.savefig(os.path.join(trial_dir, "psth_per_channel.png"),
                 dpi=200, bbox_inches='tight')
    print(f"Saved {trial_dir}/psth_per_channel.png")

    masks = assign_compartments(probe_coords_mm, config)
    compartment_colors = {'SG': '#1f77b4', 'G': '#2ca02c', 'IG': '#d62728'}

    fig2, axes2 = plt.subplots(3, 1, figsize=(8, 7),
                               sharex=True, constrained_layout=True)
    for i, comp in enumerate(['SG', 'G', 'IG']):
        ax = axes2[i]
        m = masks[comp]
        n_ch = int(m.sum())
        if n_ch > 0:
            y = psth[:, m].mean(axis=1)
            ax.plot(t_ms, y, color=compartment_colors[comp], linewidth=1.5)
        ax.axvline(0, color='gray', ls='--', lw=0.8)
        ax.set_ylabel('Rate (Hz)')
        ax.set_title(f'{comp} ({n_ch} channels)', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes2[0].set_title(
        f'PSTH by compartment (MUA, {len(trials)} trials avg)\n'
        f'SG ({int(masks["SG"].sum())} ch)',
        fontsize=10)
    axes2[-1].set_xlabel('Time from stimulus onset (ms)')
    fig2.savefig(os.path.join(trial_dir, "psth_by_compartment.png"),
                 dpi=200, bbox_inches='tight')
    print(f"Saved {trial_dir}/psth_by_compartment.png")

    all_lfp = []
    for d in trials:
        lfp = d['lfp']           
        lfp_t_ms = d['lfp_t_ms'] 
        t_rel = lfp_t_ms - float(d['stim_onset_ms'])
        t_mask = (t_rel >= -t_pre_ms) & (t_rel <= t_post_ms)
        all_lfp.append(lfp[t_mask, :])

    min_len = min(a.shape[0] for a in all_lfp)
    all_lfp = np.array([a[:min_len, :] for a in all_lfp])
    mean_lfp = all_lfp.mean(axis=0) 

    t_rel_0 = trials[0]['lfp_t_ms'] - float(trials[0]['stim_onset_ms'])
    t_mask_0 = (t_rel_0 >= -t_pre_ms) & (t_rel_0 <= t_post_ms)
    lfp_t_plot = t_rel_0[t_mask_0][:min_len]

    fig3, axes3 = plt.subplots(3, 1, figsize=(8, 7),
                               sharex=True, constrained_layout=True)
    for i, comp in enumerate(['SG', 'G', 'IG']):
        ax = axes3[i]
        m = masks[comp]
        n_ch = int(m.sum())
        if n_ch > 0:
            y = mean_lfp[:, m].mean(axis=1)
            ax.plot(lfp_t_plot, y, color=compartment_colors[comp], linewidth=0.8)
        ax.axvline(0, color='gray', ls='--', lw=0.8)
        ax.set_ylabel('LFP (a.u.)')
        ax.set_title(f'{comp} ({n_ch} channels)', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes3[0].set_title(
        f'LFP by compartment ({len(trials)} trials avg)\n'
        f'SG ({int(masks["SG"].sum())} ch)',
        fontsize=10)
    axes3[-1].set_xlabel('Time from stimulus onset (ms)')
    fig3.savefig(os.path.join(trial_dir, "lfp_by_compartment.png"),
                 dpi=200, bbox_inches='tight')
    print(f"Saved {trial_dir}/lfp_by_compartment.png")

    plt.show()
    return fig1, fig2, fig3


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-dir", default="results/trials_cleo")
    parser.add_argument("--t-pre", type=float, default=500, help="Pre-stim window (ms)")
    parser.add_argument("--t-post", type=float, default=2000, help="Post-stim window (ms)")
    parser.add_argument("--bin-width", type=float, default=2, help="Bin width (ms)")
    parser.add_argument("--smooth", type=float, default=0.010, help="Gaussian sigma (s)")
    args = parser.parse_args()

    plot_cleo_psth(
        args.trial_dir,
        t_pre_ms=args.t_pre,
        t_post_ms=args.t_post,
        bin_width_ms=args.bin_width,
        smooth_sigma=args.smooth,
    )
