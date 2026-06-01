"""Tuning curves: post-stim mean rate vs. alpha phase, per layer/population.

For each (layer, population), compute the mean firing rate in a post-stim window
(per trial), then plot mean +/- SEM across seeds as a function of target phase.
Overlay a cosine fit y = a0 + a1*cos(phi - phi_pref) per layer/pop, and print
modulation depth (a1 / a0) and preferred phase.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

RESULTS_DIR = 'results/trials_phase_07_05'
PHASES = (0, 45, 90, 135, 180, 225, 270, 315)
LAYERS = ('L23', 'L4AB', 'L4C', 'L5', 'L6')
POPS = ('E', 'PV', 'SOM', 'VIP')
POST_WINDOW_MS = (0, 300)
BASELINE_WINDOW_MS = (-300, 0)


def cosine_model(phi_deg, a0, a1, phi_pref_deg):
    return a0 + a1 * np.cos(np.deg2rad(phi_deg - phi_pref_deg))


def load_trial(path):
    d = np.load(path, allow_pickle=True)
    return {
        'seed': int(str(d['network_seed'])),
        'phase_deg': float(d['target_phase_deg']),
        'stim_onset_ms': float(d['stim_onset_ms']),
        'rate_data': d['rate_data'].item(),
    }


def window_mean(rate_entry, onset_ms, win_ms):
    t = rate_entry['t_ms']
    r = rate_entry['rate_hz']
    t_rel = t - onset_ms
    mask = (t_rel >= win_ms[0]) & (t_rel < win_ms[1])
    if not np.any(mask):
        return np.nan
    return float(np.mean(r[mask]))


def fit_cosine(phases_deg, values):
    phases_deg = np.asarray(phases_deg, dtype=float)
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    if good.sum() < 4:
        return None
    p = phases_deg[good]
    v = values[good]
    a0_init = float(np.mean(v))
    a1_init = float((np.max(v) - np.min(v)) / 2)
    phi_init = float(p[np.argmax(v)])
    try:
        popt, pcov = curve_fit(
            cosine_model, p, v,
            p0=[a0_init, a1_init, phi_init],
            bounds=([-np.inf, 0, -360], [np.inf, np.inf, 720]),
            maxfev=5000,
        )
        a0, a1, phi_pref = popt
        phi_pref = phi_pref % 360
        v_fit = cosine_model(p, *popt)
        ss_res = float(np.sum((v - v_fit) ** 2))
        ss_tot = float(np.sum((v - np.mean(v)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        mod_depth = a1 / a0 if a0 != 0 else np.nan
        return {
            'a0': float(a0), 'a1': float(a1),
            'phi_pref_deg': float(phi_pref),
            'r2': float(r2),
            'mod_depth': float(mod_depth),
        }
    except Exception:
        return None


def collect():
    """Returns nested dict: results[layer][pop][phase_deg] = list of per-seed
    (post_mean, baseline_mean, delta) tuples."""
    out = {ln: {p: {ph: [] for ph in PHASES} for p in POPS} for ln in LAYERS}
    for ph in PHASES:
        pat = os.path.join(RESULTS_DIR, f"trial_seed*_phase{int(ph):03d}.npz")
        for f in sorted(glob.glob(pat)):
            tr = load_trial(f)
            for ln in LAYERS:
                rd = tr['rate_data'].get(ln, {})
                for pop in POPS:
                    key = f"{pop}_rate"
                    if key not in rd:
                        continue
                    post = window_mean(rd[key], tr['stim_onset_ms'],
                                       POST_WINDOW_MS)
                    base = window_mean(rd[key], tr['stim_onset_ms'],
                                       BASELINE_WINDOW_MS)
                    out[ln][pop][ph].append({
                        'post': post,
                        'base': base,
                        'delta': post - base if np.isfinite(post)
                                                and np.isfinite(base) else np.nan,
                    })
    return out


def summarize(per_seed_dict):
    """Given {phase: [{post, base, delta}, ...]}, return arrays of mean/sem for
    post and delta."""
    phases = np.array(sorted(per_seed_dict.keys()))
    post_mean = np.array([np.nanmean([s['post'] for s in per_seed_dict[ph]])
                          for ph in phases])
    post_sem = np.array([np.nanstd([s['post'] for s in per_seed_dict[ph]])
                         / np.sqrt(max(1, len(per_seed_dict[ph])))
                         for ph in phases])
    delta_mean = np.array([np.nanmean([s['delta'] for s in per_seed_dict[ph]])
                           for ph in phases])
    delta_sem = np.array([np.nanstd([s['delta'] for s in per_seed_dict[ph]])
                          / np.sqrt(max(1, len(per_seed_dict[ph])))
                          for ph in phases])
    return phases, post_mean, post_sem, delta_mean, delta_sem


def plot_grid(results, value_key, save_path, title_suffix):
    """value_key is 'post' or 'delta'. One subplot per (layer, pop)."""
    n_rows = len(LAYERS)
    n_cols = len(POPS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 2.4 * n_rows),
                             squeeze=False)

    fit_table = []
    phi_dense = np.linspace(0, 360, 361)
    for i, ln in enumerate(LAYERS):
        for j, pop in enumerate(POPS):
            ax = axes[i, j]
            phases, post_m, post_s, dlt_m, dlt_s = summarize(results[ln][pop])
            if value_key == 'post':
                y, ye = post_m, post_s
            else:
                y, ye = dlt_m, dlt_s
            ax.errorbar(phases, y, yerr=ye, fmt='o-', color='#2266aa',
                        lw=1.2, capsize=3, ms=4)
            fit = fit_cosine(phases, y)
            if fit is not None:
                ax.plot(phi_dense, cosine_model(phi_dense, fit['a0'],
                                                fit['a1'],
                                                fit['phi_pref_deg']),
                        color='crimson', lw=1.2, alpha=0.85,
                        label=(f"φ*={fit['phi_pref_deg']:.0f}°, "
                               f"R²={fit['r2']:.2f}\n"
                               f"depth={fit['mod_depth']:+.2f}"))
                ax.legend(fontsize=7, loc='best', framealpha=0.7)
                fit_table.append((ln, pop, value_key, fit))

            ax.set_xticks([0, 90, 180, 270, 360])
            ax.grid(True, alpha=0.3)
            ax.axhline(0 if value_key == 'delta' else
                       np.nanmean(y), color='k', ls=':', lw=0.6, alpha=0.5)
            if i == 0:
                ax.set_title(pop)
            if j == 0:
                ax.set_ylabel(f"{ln}\nrate (Hz)" if value_key == 'post'
                              else f"{ln}\nΔrate (Hz)")
            if i == n_rows - 1:
                ax.set_xlabel('Phase at stim onset (°)')

    fig.suptitle(f'Phase tuning: {title_suffix} '
                 f'(post-stim {POST_WINDOW_MS[0]}-{POST_WINDOW_MS[1]} ms)',
                 fontsize=12, y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"saved {save_path}")
    return fit_table


def main():
    print("Loading trials...")
    results = collect()
    n_seeds = max(len(results[LAYERS[0]][POPS[0]][ph]) for ph in PHASES)
    print(f"max seeds per phase: {n_seeds}")

    fits_post = plot_grid(results, 'post',
                          os.path.join(RESULTS_DIR,
                                       'phase_tuning_post_grid.png'),
                          'absolute post-stim rate')
    fits_delta = plot_grid(results, 'delta',
                           os.path.join(RESULTS_DIR,
                                        'phase_tuning_delta_grid.png'),
                           'stim-evoked Δrate (post - baseline)')

    print("\n=== Cosine fit summary (delta rate) ===")
    print(f"{'layer':>6} {'pop':>4} {'a0(Hz)':>8} {'a1(Hz)':>8} "
          f"{'phi*(deg)':>10} {'depth':>8} {'R^2':>6}")
    for ln, pop, vk, fit in fits_delta:
        print(f"{ln:>6} {pop:>4} {fit['a0']:>8.3f} {fit['a1']:>8.3f} "
              f"{fit['phi_pref_deg']:>10.1f} {fit['mod_depth']:>+8.3f} "
              f"{fit['r2']:>6.2f}")

    out_csv = os.path.join(RESULTS_DIR, 'phase_tuning_fits.csv')
    with open(out_csv, 'w') as fh:
        fh.write("layer,pop,metric,a0,a1,phi_pref_deg,mod_depth,r2\n")
        for table in (fits_post, fits_delta):
            for ln, pop, vk, fit in table:
                fh.write(f"{ln},{pop},{vk},{fit['a0']:.5f},{fit['a1']:.5f},"
                         f"{fit['phi_pref_deg']:.3f},{fit['mod_depth']:.5f},"
                         f"{fit['r2']:.4f}\n")
    print(f"saved {out_csv}")


if __name__ == '__main__':
    main()
