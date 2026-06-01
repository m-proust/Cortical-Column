"""Polar tuning plots for L4C E, L5 E (and a few extras).

Same data as phase_tuning_curves.py, plotted on polar axes so the preferred
phase is visually obvious. Plots Δrate (stim-evoked) so positive values point
outward from origin.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from phase_tuning_curves import (
    PHASES, RESULTS_DIR, collect, summarize, fit_cosine, cosine_model,
)


POPULATIONS_TO_PLOT = [
    ('L4C', 'E'),
    ('L5', 'E'),
    ('L23', 'E'),
    ('L4C', 'PV'),
    ('L5', 'PV'),
    ('L4C', 'SOM'),
]


def main():
    print("Loading trials...")
    results = collect()

    n = len(POPULATIONS_TO_PLOT)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             subplot_kw={'projection': 'polar'},
                             figsize=(4.4 * ncols, 4.2 * nrows),
                             squeeze=False)

    phi_dense_deg = np.linspace(0, 360, 361)
    phi_dense_rad = np.deg2rad(phi_dense_deg)

    for ax_idx, (ln, pop) in enumerate(POPULATIONS_TO_PLOT):
        ax = axes[ax_idx // ncols, ax_idx % ncols]
        phases, _, _, dlt_m, dlt_s = summarize(results[ln][pop])
        phi_rad = np.deg2rad(phases)

        # Close the loop for the line plot
        phi_closed = np.concatenate([phi_rad, [phi_rad[0]]])
        y_closed = np.concatenate([dlt_m, [dlt_m[0]]])

        ax.plot(phi_closed, y_closed, 'o-', color='#2266aa', lw=1.8, ms=6,
                label='Δrate per phase')

        # Error bars as small radial lines
        for p, m, s in zip(phi_rad, dlt_m, dlt_s):
            ax.plot([p, p], [m - s, m + s], color='#2266aa',
                    lw=1.2, alpha=0.7)

        fit = fit_cosine(phases, dlt_m)
        if fit is not None:
            y_fit = cosine_model(phi_dense_deg, fit['a0'], fit['a1'],
                                 fit['phi_pref_deg'])
            ax.plot(phi_dense_rad, y_fit, color='crimson', lw=1.4, alpha=0.85,
                    label=f"cosine fit\nφ*={fit['phi_pref_deg']:.0f}°  "
                          f"depth={fit['mod_depth']:+.2f}")
            # Arrow pointing at preferred phase
            phi_pref_rad = np.deg2rad(fit['phi_pref_deg'])
            r_max = max(np.nanmax(dlt_m + dlt_s),
                        np.nanmax(y_fit)) * 1.05
            ax.annotate('', xy=(phi_pref_rad, r_max), xytext=(phi_pref_rad, 0),
                        arrowprops=dict(arrowstyle='->', color='crimson',
                                        lw=1.4, alpha=0.8))

        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(['0°', '45°', '90°', '135°',
                            '180°', '225°', '270°', '315°'])
        ax.set_title(f'{ln} {pop}  (Δrate, Hz)', y=1.12, fontsize=11)
        ax.grid(True, alpha=0.4)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1),
                  fontsize=7, framealpha=0.7)

    # Hide unused subplots
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis('off')

    fig.suptitle('Polar phase tuning of stim-evoked Δrate '
                 '(0–300 ms post-stim, mean ± SEM across seeds)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'phase_polar_tuning.png')
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"saved {out_path}")


if __name__ == '__main__':
    main()
