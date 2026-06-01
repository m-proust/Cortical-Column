"""Compare phase rankings under two metrics.

For each layer:
  (A) Δrate ranking: per-phase mean of (post 0–300 ms rate − pre 300 ms rate)
      for the principal E pop in that layer.
  (B) Switch ranking: per-phase mean of switch_min from phase_switch_score.

If best phase agrees → classical "alpha trough drives both rate and switch."
If they disagree → genuine dissociation (rare in the literature).
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

from phase_excitability import per_trial_excitability, load_trial, TARGETS
from phase_switch_score import collect as collect_switch, summarize_metric as summ_switch

RESULTS_DIR = 'results/trials_phase_07_05'
PHASES = (0, 45, 90, 135, 180, 225, 270, 315)
LAYERS_TO_SHOW = ('L23', 'L4AB', 'L4C', 'L5', 'L6')


def collect_drate_per_layer():
    """results[layer] = {phase: [delta_rate per trial]} using E pop only."""
    res = {ln: {ph: [] for ph in PHASES} for ln in LAYERS_TO_SHOW}
    for ph in PHASES:
        pat = os.path.join(RESULTS_DIR,
                           f"trial_seed*_phase{int(ph):03d}.npz")
        for f in sorted(glob.glob(pat)):
            tr = load_trial(f)
            for ln in LAYERS_TO_SHOW:
                m = per_trial_excitability(tr, ln, 'E')
                if m is not None and np.isfinite(m['delta_rate']):
                    res[ln][ph].append(m['delta_rate'])
    return res


def main():
    print("Collecting Δrate per layer per phase...")
    dr = collect_drate_per_layer()

    print("Collecting switch score per layer per phase...")
    sw = collect_switch()

    fig, axes = plt.subplots(2, len(LAYERS_TO_SHOW),
                             figsize=(3.0 * len(LAYERS_TO_SHOW), 6),
                             sharex=True, squeeze=False)

    print()
    print(f"{'layer':>5} | {'best Δrate':>12} | {'best switch':>12} | "
          f"{'worst Δrate':>13} | {'worst switch':>13} | "
          f"{'Spearman ρ':>11} | {'Kendall τ':>10}")
    print('-' * 100)
    for col, ln in enumerate(LAYERS_TO_SHOW):
        phases = np.array(sorted(dr[ln].keys()))
        dr_means = np.array([np.nanmean(dr[ln][p]) if dr[ln][p] else np.nan
                             for p in phases])
        dr_sems = np.array([np.nanstd(dr[ln][p]) / np.sqrt(max(1, len(dr[ln][p])))
                            if dr[ln][p] else np.nan for p in phases])
        _, sw_m, sw_s, _ = summ_switch(sw[ln], 'switch_min')

        best_dr = phases[np.nanargmax(dr_means)]
        worst_dr = phases[np.nanargmin(dr_means)]
        best_sw = phases[np.nanargmax(sw_m)]
        worst_sw = phases[np.nanargmin(sw_m)]

        good = np.isfinite(dr_means) & np.isfinite(sw_m)
        rho, p_rho = spearmanr(dr_means[good], sw_m[good])
        tau, p_tau = kendalltau(dr_means[good], sw_m[good])

        print(f"{ln:>5} | {best_dr:>10}°  | {best_sw:>10}°  | "
              f"{worst_dr:>11}°  | {worst_sw:>11}°  | "
              f"{rho:>+7.3f}({p_rho:.2g}) | {tau:>+6.3f}({p_tau:.2g})")

        # Top row: Δrate vs phase
        ax = axes[0, col]
        ax.errorbar(phases, dr_means, yerr=dr_sems, fmt='o-',
                    color='#1f77b4', lw=1.5, capsize=3)
        ax.axvline(best_dr, color='g', ls=':', lw=1.0, alpha=0.6)
        ax.axvline(worst_dr, color='r', ls=':', lw=1.0, alpha=0.6)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{ln} E')
        if col == 0:
            ax.set_ylabel('Δrate (Hz, post − base)')

        # Bottom row: switch score vs phase
        ax = axes[1, col]
        ax.errorbar(phases, sw_m, yerr=sw_s, fmt='o-',
                    color='#d62728', lw=1.5, capsize=3)
        ax.axvline(best_sw, color='g', ls=':', lw=1.0, alpha=0.6)
        ax.axvline(worst_sw, color='r', ls=':', lw=1.0, alpha=0.6)
        ax.axhline(0, color='k', ls=':', lw=0.6)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Phase (°)')
        if col == 0:
            ax.set_ylabel('Switch score (%)')

    fig.suptitle('Phase rankings: Δrate (top) vs switch score (bottom). '
                 'Green dotted = best phase, red dotted = worst.', y=1.0)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'phase_ranking_compare.png')
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nsaved {out_path}")

    # CSV
    out_csv = os.path.join(RESULTS_DIR, 'phase_ranking_compare.csv')
    with open(out_csv, 'w') as fh:
        fh.write("layer,phase_deg,delta_rate_mean,delta_rate_sem,"
                 "switch_min_mean,switch_min_sem\n")
        for ln in LAYERS_TO_SHOW:
            phases = np.array(sorted(dr[ln].keys()))
            dr_means = np.array([np.nanmean(dr[ln][p]) if dr[ln][p] else np.nan
                                 for p in phases])
            dr_sems = np.array([np.nanstd(dr[ln][p]) / np.sqrt(max(1, len(dr[ln][p])))
                                if dr[ln][p] else np.nan for p in phases])
            _, sw_m, sw_s, _ = summ_switch(sw[ln], 'switch_min')
            for p, drm, drs, swm, sws in zip(phases, dr_means, dr_sems,
                                             sw_m, sw_s):
                fh.write(f"{ln},{int(p)},{drm:.4f},{drs:.4f},"
                         f"{swm:.4f},{sws:.4f}\n")
    print(f"saved {out_csv}")


if __name__ == '__main__':
    main()
