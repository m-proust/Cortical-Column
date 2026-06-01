"""Time-resolved phase effect on firing rate.

For a given layer/pop, build a (phase × time) heatmap of mean rate aligned to
stim onset, and a second heatmap of (rate − mean-over-phases) so the phase-
induced differences pop out. Produces one figure with rows = (L23 E, L4C E,
L5 E, L4C PV).
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import savgol_filter

RESULTS_DIR = 'results/trials_phase_07_05'
PHASES = (0, 45, 90, 135, 180, 225, 270, 315)
WIN_MS = (-200, 600)
SMOOTH_W = 101  # samples (dt=0.1ms → 10ms window)

PANELS = [
    ('L23', 'E'),
    ('L4C', 'E'),
    ('L5', 'E'),
    ('L4C', 'PV'),
]


_T_GRID = None  # canonical grid shared across all phases/layers/pops


def _make_t_grid():
    """Use a fixed dt and window so every phase aligns onto the same grid."""
    global _T_GRID
    if _T_GRID is None:
        dt = 0.1  # ms (matches rate monitor sampling)
        _T_GRID = np.arange(WIN_MS[0], WIN_MS[1] + dt, dt)
    return _T_GRID


def load_phase_seed_rates(phase_deg, layer, pop):
    """Return aligned mean rate (1D) on the canonical t_grid for this phase."""
    pat = os.path.join(RESULTS_DIR,
                       f"trial_seed*_phase{int(phase_deg):03d}.npz")
    files = sorted(glob.glob(pat))
    if not files:
        return None, None
    t_grid = _make_t_grid()
    aligned = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        rd = d['rate_data'].item().get(layer, {})
        key = f"{pop}_rate"
        if key not in rd:
            continue
        t = rd[key]['t_ms']
        r = rd[key]['rate_hz']
        onset = float(d['stim_onset_ms'])
        t_rel = t - onset
        aligned.append(np.interp(t_grid, t_rel, r))
    if not aligned:
        return None, None
    return t_grid, np.array(aligned)


def smooth(x, w=SMOOTH_W):
    if len(x) < w:
        return x
    return savgol_filter(x, w, 3)


def main():
    fig, axes = plt.subplots(len(PANELS), 3,
                             figsize=(15, 2.8 * len(PANELS)),
                             gridspec_kw={'width_ratios': [1, 1, 1.05]},
                             squeeze=False)

    for row, (layer, pop) in enumerate(PANELS):
        phase_means = []
        t_grid_ref = None
        n_seeds_per_phase = []
        for ph in PHASES:
            t_grid, mat = load_phase_seed_rates(ph, layer, pop)
            if mat is None:
                phase_means.append(None)
                n_seeds_per_phase.append(0)
                continue
            if t_grid_ref is None:
                t_grid_ref = t_grid
            mean_r = np.mean(mat, axis=0)
            mean_r = smooth(mean_r)
            phase_means.append(mean_r)
            n_seeds_per_phase.append(mat.shape[0])

        if t_grid_ref is None:
            for c in range(3):
                axes[row, c].axis('off')
            continue

        # Stack into (n_phases, n_t)
        valid_idx = [i for i, m in enumerate(phase_means) if m is not None]
        mat = np.array([phase_means[i] for i in valid_idx])
        phases_used = np.array([PHASES[i] for i in valid_idx])

        # Mean over phases (per-time)
        grand_mean = np.mean(mat, axis=0)
        diff = mat - grand_mean

        # (A) raw rate heatmap
        ax = axes[row, 0]
        extent = [t_grid_ref[0], t_grid_ref[-1],
                  -0.5, len(phases_used) - 0.5]
        im1 = ax.imshow(mat, aspect='auto', cmap='viridis',
                        extent=extent, origin='lower')
        ax.set_yticks(range(len(phases_used)))
        ax.set_yticklabels([f"{p}°" for p in phases_used])
        ax.axvline(0, color='w', ls=':', lw=1.0, alpha=0.8)
        ax.set_ylabel(f'{layer} {pop}\nphase at onset')
        if row == 0:
            ax.set_title('Mean rate (Hz)')
        if row == len(PANELS) - 1:
            ax.set_xlabel('Time from stim onset (ms)')
        plt.colorbar(im1, ax=ax, shrink=0.85)

        # (B) phase-induced deviation heatmap
        ax = axes[row, 1]
        vmax = float(np.max(np.abs(diff)))
        if vmax <= 0 or not np.isfinite(vmax):
            vmax = 1e-6
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im2 = ax.imshow(diff, aspect='auto', cmap='RdBu_r',
                        extent=extent, origin='lower', norm=norm)
        ax.set_yticks(range(len(phases_used)))
        ax.set_yticklabels([f"{p}°" for p in phases_used])
        ax.axvline(0, color='k', ls=':', lw=1.0, alpha=0.8)
        if row == 0:
            ax.set_title('Δ from phase-averaged rate (Hz)')
        if row == len(PANELS) - 1:
            ax.set_xlabel('Time from stim onset (ms)')
        plt.colorbar(im2, ax=ax, shrink=0.85)

        # (C) phase-aligned line plot
        ax = axes[row, 2]
        cmap = plt.cm.twilight
        for ph, r in zip(phases_used, mat):
            ax.plot(t_grid_ref, r, color=cmap(ph / 360.0), lw=1.1,
                    label=f'{ph}°')
        ax.plot(t_grid_ref, grand_mean, color='k', lw=1.6, alpha=0.8,
                label='mean')
        ax.axvline(0, color='k', ls=':', lw=0.8, alpha=0.6)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(f'{layer} {pop} rate (Hz)')
        if row == 0:
            ax.set_title('Mean rate per phase')
            ax.legend(ncol=3, fontsize=6, loc='upper right', framealpha=0.7)
        if row == len(PANELS) - 1:
            ax.set_xlabel('Time from stim onset (ms)')

    fig.suptitle('Time-resolved phase effect on firing rate '
                 '(pooled across seeds)', y=1.0, fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'phase_rate_heatmap.png')
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"saved {out_path}")


if __name__ == '__main__':
    main()
