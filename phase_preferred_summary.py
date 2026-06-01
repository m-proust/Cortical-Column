"""Cross-layer summary of phase tuning.

Two bar charts:
  1) Preferred phase (φ*) per layer × population, with error bars from
     bootstrap over seeds.
  2) Modulation depth (a1 / |a0|) per layer × population, same bootstrap.

Bootstrap: resample seeds with replacement, refit cosine to mean-over-seeds
tuning curve, repeat N times to get CI on φ* and depth.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from phase_tuning_curves import (
    PHASES, LAYERS, POPS, RESULTS_DIR, collect, fit_cosine,
)

N_BOOT = 500
RNG_SEED = 13
POST_OR_DELTA = 'delta'   # bootstrap on the Δrate tuning curve


def circular_mean_deg(angles_deg):
    a = np.deg2rad(angles_deg)
    mean = np.angle(np.mean(np.exp(1j * a)))
    return float(np.rad2deg(mean) % 360)


def circular_std_deg(angles_deg):
    a = np.deg2rad(angles_deg)
    R = np.abs(np.mean(np.exp(1j * a)))
    if R <= 0 or R >= 1:
        return 0.0 if R >= 1 else 180.0
    return float(np.rad2deg(np.sqrt(-2 * np.log(R))))


def bootstrap_fit(per_seed_dict, n_boot=N_BOOT, rng=None):
    """per_seed_dict: {phase_deg: [{post, base, delta}, ...]}
    Returns dict with phi_pref samples, mod_depth samples, plus point estimate.
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    phases = np.array(sorted(per_seed_dict.keys()))
    # Build per-seed Δrate matrix (n_seeds × n_phases), aligned by seed index
    # (seeds are pooled across phases — we assume seed ordering is consistent).
    n_seeds = min(len(per_seed_dict[ph]) for ph in phases)
    if n_seeds < 3:
        return None
    mat = np.full((n_seeds, len(phases)), np.nan)
    for j, ph in enumerate(phases):
        col = [s[POST_OR_DELTA] for s in per_seed_dict[ph][:n_seeds]]
        mat[:, j] = col

    # Point estimate
    mean_curve = np.nanmean(mat, axis=0)
    point = fit_cosine(phases, mean_curve)

    boots_phi, boots_depth, boots_r2 = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        curve = np.nanmean(mat[idx], axis=0)
        fit = fit_cosine(phases, curve)
        if fit is None:
            continue
        boots_phi.append(fit['phi_pref_deg'])
        boots_depth.append(fit['mod_depth'])
        boots_r2.append(fit['r2'])

    boots_phi = np.array(boots_phi)
    boots_depth = np.array(boots_depth)
    if point is None or len(boots_phi) < 10:
        return None
    return {
        'phi_pref_deg': point['phi_pref_deg'],
        'phi_pref_circstd': circular_std_deg(boots_phi),
        'phi_boot': boots_phi,
        'mod_depth': point['mod_depth'],
        'mod_depth_lo': float(np.percentile(boots_depth, 2.5)),
        'mod_depth_hi': float(np.percentile(boots_depth, 97.5)),
        'r2': point['r2'],
        'n_seeds': n_seeds,
    }


def main():
    print("Loading trials...")
    results = collect()

    fits = {}
    rng = np.random.default_rng(RNG_SEED)
    for ln in LAYERS:
        for pop in POPS:
            f = bootstrap_fit(results[ln][pop], rng=rng)
            if f is not None:
                fits[(ln, pop)] = f

    print(f"Fits succeeded for {len(fits)} / {len(LAYERS) * len(POPS)} "
          f"layer/pop combos")

    # ---------- Figure 1: preferred phase ----------
    fig1, ax1 = plt.subplots(figsize=(10, 4.5))
    pop_colors = {'E': '#d62728', 'PV': '#1f77b4',
                  'SOM': '#2ca02c', 'VIP': '#9467bd'}
    bar_w = 0.18
    x_base = np.arange(len(LAYERS))
    for k, pop in enumerate(POPS):
        xs, ys, errs = [], [], []
        for i, ln in enumerate(LAYERS):
            if (ln, pop) not in fits:
                continue
            xs.append(x_base[i] + (k - 1.5) * bar_w)
            ys.append(fits[(ln, pop)]['phi_pref_deg'])
            errs.append(fits[(ln, pop)]['phi_pref_circstd'])
        ax1.bar(xs, ys, width=bar_w, color=pop_colors[pop], label=pop,
                yerr=errs, capsize=3, alpha=0.85)
    ax1.set_xticks(x_base)
    ax1.set_xticklabels(LAYERS)
    ax1.set_ylabel('Preferred phase φ* (°)')
    ax1.set_ylim(0, 360)
    ax1.set_yticks([0, 90, 180, 270, 360])
    ax1.set_title('Preferred alpha phase per layer × population '
                  '(bootstrap circular SD)')
    ax1.legend(title='Population', loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    out1 = os.path.join(RESULTS_DIR, 'phase_preferred_phase_bars.png')
    plt.savefig(out1, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"saved {out1}")

    # ---------- Figure 2: modulation depth ----------
    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    for k, pop in enumerate(POPS):
        xs, ys, lo, hi = [], [], [], []
        for i, ln in enumerate(LAYERS):
            if (ln, pop) not in fits:
                continue
            xs.append(x_base[i] + (k - 1.5) * bar_w)
            ys.append(fits[(ln, pop)]['mod_depth'])
            lo.append(fits[(ln, pop)]['mod_depth_lo'])
            hi.append(fits[(ln, pop)]['mod_depth_hi'])
        ys = np.array(ys)
        lo = np.array(lo)
        hi = np.array(hi)
        err = np.vstack([ys - lo, hi - ys])
        ax2.bar(xs, ys, width=bar_w, color=pop_colors[pop], label=pop,
                yerr=err, capsize=3, alpha=0.85)
    ax2.set_xticks(x_base)
    ax2.set_xticklabels(LAYERS)
    ax2.set_ylabel('Modulation depth a1 / a0  (Δrate fit)')
    ax2.axhline(0, color='k', lw=0.7, ls=':')
    ax2.set_title('Phase modulation depth per layer × population '
                  '(95% bootstrap CI)')
    ax2.legend(title='Population', loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    out2 = os.path.join(RESULTS_DIR, 'phase_modulation_depth_bars.png')
    plt.savefig(out2, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"saved {out2}")

    # ---------- Figure 3: preferred phase polar scatter ----------
    fig3, ax3 = plt.subplots(subplot_kw={'projection': 'polar'},
                             figsize=(7, 7))
    pop_markers = {'E': 'o', 'PV': 's', 'SOM': '^', 'VIP': 'D'}
    layer_sizes = {ln: 110 + i * 35 for i, ln in enumerate(LAYERS)}
    for (ln, pop), f in fits.items():
        phi = np.deg2rad(f['phi_pref_deg'])
        depth = abs(f['mod_depth'])
        ax3.scatter(phi, depth, s=layer_sizes[ln], color=pop_colors[pop],
                    marker=pop_markers[pop], edgecolor='k', lw=0.6,
                    alpha=0.85, label=f'{ln} {pop}')
    ax3.set_theta_zero_location('E')
    ax3.set_theta_direction(1)
    ax3.set_title('Preferred phase (angle) × |modulation depth| (radius)\n'
                  'per layer × population',
                  pad=20, fontsize=11)
    ax3.legend(loc='center left', bbox_to_anchor=(1.1, 0.5),
               fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.4)
    plt.tight_layout()
    out3 = os.path.join(RESULTS_DIR, 'phase_preferred_polar_scatter.png')
    plt.savefig(out3, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"saved {out3}")

    # ---------- CSV summary ----------
    out_csv = os.path.join(RESULTS_DIR, 'phase_preferred_summary.csv')
    with open(out_csv, 'w') as fh:
        fh.write("layer,pop,n_seeds,phi_pref_deg,phi_pref_circstd_deg,"
                 "mod_depth,mod_depth_ci_lo,mod_depth_ci_hi,r2\n")
        for (ln, pop), f in fits.items():
            fh.write(f"{ln},{pop},{f['n_seeds']},"
                     f"{f['phi_pref_deg']:.2f},{f['phi_pref_circstd']:.2f},"
                     f"{f['mod_depth']:.4f},"
                     f"{f['mod_depth_lo']:.4f},{f['mod_depth_hi']:.4f},"
                     f"{f['r2']:.3f}\n")
    print(f"saved {out_csv}")

    # ---------- Console summary ----------
    print("\n=== Phase preference summary (Δrate fit) ===")
    print(f"{'layer':>5} {'pop':>4} {'n':>3} {'φ*(°)':>8} "
          f"{'±σ(°)':>7} {'depth':>8} {'[CI95%]':>20} {'R²':>5}")
    for (ln, pop), f in sorted(fits.items()):
        ci = f"[{f['mod_depth_lo']:+.2f},{f['mod_depth_hi']:+.2f}]"
        print(f"{ln:>5} {pop:>4} {f['n_seeds']:>3d} "
              f"{f['phi_pref_deg']:>8.1f} {f['phi_pref_circstd']:>7.1f} "
              f"{f['mod_depth']:>+8.3f} {ci:>20} {f['r2']:>5.2f}")


if __name__ == '__main__':
    main()
