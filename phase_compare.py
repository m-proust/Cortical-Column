"""Quick look: compare stimulus response across alpha phases for one seed.

For a single seed, plots E-population rate aligned to stim onset for L23, L4C,
L5 across all 8 phases. Also computes the integrated post-stim rate (0-200ms)
per phase to see if there's a phase-modulated gain.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

RESULTS_DIR = "results/trials_phase_07_05"
SEED = 58910
LAYERS = ['L23', 'L4C', 'L5']
WINDOW_MS = (-100, 400)   # around stim onset
GAIN_WINDOW_MS = (0, 200)   # for integrated response


def load_seed_trials(seed):
    pattern = os.path.join(RESULTS_DIR, f"trial_seed{seed}_phase*.npz")
    files = sorted(glob.glob(pattern))
    trials = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        trials.append({
            'phase_deg': float(d['target_phase_deg']),
            'stim_onset_ms': float(d['stim_onset_ms']),
            'rate_data': d['rate_data'].item(),
        })
    return trials


def get_e_rate_aligned(trial, layer, win_ms):
    rd = trial['rate_data'][layer]['E_rate']
    t = rd['t_ms']
    r = rd['rate_hz']
    onset = trial['stim_onset_ms']
    t_rel = t - onset
    mask = (t_rel >= win_ms[0]) & (t_rel <= win_ms[1])
    return t_rel[mask], r[mask]


def smooth(x, w=51):
    if len(x) < w:
        return x
    return savgol_filter(x, w, 3)


def main():
    trials = load_seed_trials(SEED)
    print(f"Loaded {len(trials)} trials for seed {SEED}")

    fig, axes = plt.subplots(len(LAYERS), 1, figsize=(11, 9), sharex=True)
    cmap = plt.cm.twilight
    colors = [cmap(t['phase_deg'] / 360.0) for t in trials]

    for ax, layer in zip(axes, LAYERS):
        for trial, color in zip(trials, colors):
            t_rel, r = get_e_rate_aligned(trial, layer, WINDOW_MS)
            r_s = smooth(r, w=101)
            ax.plot(t_rel, r_s, color=color, lw=1.4,
                    label=f"{trial['phase_deg']:.0f} deg")
        ax.axvline(0, color='k', ls='--', lw=0.8, alpha=0.6)
        ax.set_ylabel(f"{layer} E rate (Hz)")
        ax.grid(True, alpha=0.3)
    axes[0].set_title(f"Stimulus response by alpha phase (seed {SEED})")
    axes[-1].set_xlabel("Time from stim onset (ms)")
    axes[0].legend(ncol=4, fontsize=8, loc='upper right')
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"phase_compare_seed{SEED}.png")
    plt.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")

    # Integrated response per phase (gain curve)
    fig2, axes2 = plt.subplots(1, len(LAYERS), figsize=(13, 3.6), sharey=False)
    for ax, layer in zip(axes2, LAYERS):
        phases = []
        gains = []
        for trial in trials:
            t_rel, r = get_e_rate_aligned(trial, layer, GAIN_WINDOW_MS)
            phases.append(trial['phase_deg'])
            gains.append(np.trapz(r, t_rel))
        order = np.argsort(phases)
        phases = np.array(phases)[order]
        gains = np.array(gains)[order]
        ax.plot(phases, gains, 'o-', lw=1.4)
        ax.set_xlabel("Alpha phase at stim onset (deg)")
        ax.set_ylabel(f"{layer} integrated rate (Hz·ms)")
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.grid(True, alpha=0.3)
    axes2[0].set_title(f"Phase-tuning of stim response (seed {SEED})")
    plt.tight_layout()
    out2 = os.path.join(RESULTS_DIR, f"phase_gain_seed{SEED}.png")
    plt.savefig(out2, dpi=130)
    print(f"Saved {out2}")

    print("\nGain table:")
    print(f"{'phase':>8} | " + " | ".join(f"{l:>10}" for l in LAYERS))
    for i, ph in enumerate(phases):
        row = []
        for layer in LAYERS:
            t_rel, r = get_e_rate_aligned(trials[order[i]], layer, GAIN_WINDOW_MS)
            row.append(np.trapz(r, t_rel))
        print(f"{ph:>8.0f} | " + " | ".join(f"{v:>10.1f}" for v in row))


if __name__ == "__main__":
    main()
