"""Excitability proxies at stim onset, per alpha phase.

For each (layer, pop), measure three "state at stim onset" quantities,
averaged over a short pre-stim window:
  (1) Mean membrane potential V                      → more depolarized = excitable
  (2) Mean inhibitory conductance gI                 → lower gI = disinhibited = excitable
  (3) Pre-stim firing rate (last 50 ms)              → higher rate = excitable

Then compare these excitability proxies against the response strength
(post-stim Δrate, 0–300 ms) on a per-trial basis to see if more-excitable
phases really do produce stronger responses.

Outputs:
  - phase_excitability.png   (3 rows × 3 pop columns + correlation panel)
  - phase_excitability.csv   (per-phase summary)
  - phase_excitability_correlation.csv (per-pop Spearman ρ across trials)
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

RESULTS_DIR = 'results/trials_phase_07_05'
PHASES = (0, 45, 90, 135, 180, 225, 270, 315)

# Populations whose excitability we probe (thalamorecipient + L5 output + L4AB)
TARGETS = [('L4AB', 'E'), ('L4C', 'E'), ('L5', 'E')]

PRE_WIN_MS = 50         # window for V, gI, pre-rate averaging
POST_WIN_MS = (0, 300)  # for response Δrate


def load_trial(path):
    d = np.load(path, allow_pickle=True)
    return {
        'seed': int(str(d['network_seed'])),
        'phase_deg': float(d['target_phase_deg']),
        'stim_onset_ms': float(d['stim_onset_ms']),
        'rate_data': d['rate_data'].item(),
        'state_data': d['state_data'].item(),
    }


def window_mean(t, x, t0, t1):
    mask = (t >= t0) & (t < t1)
    if not np.any(mask):
        return np.nan
    return float(np.mean(x[mask]))


def per_trial_excitability(trial, layer, pop):
    sd = trial['state_data'].get(layer, {}).get(pop, None)
    rd = trial['rate_data'].get(layer, {}).get(f"{pop}_rate", None)
    if sd is None or rd is None:
        return None
    onset = trial['stim_onset_ms']
    t_state = sd['t_ms']
    t_rate = rd['t_ms']
    r = rd['rate_hz']

    v_at = window_mean(t_state, sd['v'], onset - PRE_WIN_MS, onset)
    gI_at = window_mean(t_state, sd['gI'], onset - PRE_WIN_MS, onset)
    gE_at = window_mean(t_state, sd['gE'], onset - PRE_WIN_MS, onset)
    rate_pre = window_mean(t_rate, r, onset - PRE_WIN_MS, onset)
    rate_post = window_mean(t_rate, r,
                            onset + POST_WIN_MS[0], onset + POST_WIN_MS[1])
    rate_base = window_mean(t_rate, r, onset - 300, onset)  # 300ms baseline
    return {
        'v_at_onset': v_at,
        'gI_at_onset': gI_at,
        'gE_at_onset': gE_at,
        'rate_pre_50ms': rate_pre,
        'rate_post_300ms': rate_post,
        'delta_rate': (rate_post - rate_base
                       if np.isfinite(rate_post) and np.isfinite(rate_base)
                       else np.nan),
    }


def collect():
    """results[(layer,pop)][phase] = list of per-trial dicts."""
    res = {t: {ph: [] for ph in PHASES} for t in TARGETS}
    n_seen = 0
    for ph in PHASES:
        pat = os.path.join(RESULTS_DIR,
                           f"trial_seed*_phase{int(ph):03d}.npz")
        for f in sorted(glob.glob(pat)):
            tr = load_trial(f)
            n_seen += 1
            for key in TARGETS:
                ln, pop = key
                m = per_trial_excitability(tr, ln, pop)
                if m is not None:
                    res[key][ph].append(m)
    print(f"  loaded {n_seen} trials")
    return res


def summarize_metric(per_phase, key):
    phases = np.array(sorted(per_phase.keys()))
    means, sems, n = [], [], []
    for ph in phases:
        vals = np.array([s[key] for s in per_phase[ph]
                         if np.isfinite(s[key])])
        if len(vals) == 0:
            means.append(np.nan)
            sems.append(np.nan)
            n.append(0)
        else:
            means.append(float(np.mean(vals)))
            sems.append(float(np.std(vals) / np.sqrt(len(vals))))
            n.append(len(vals))
    return phases, np.array(means), np.array(sems), np.array(n)


def main():
    print("Loading trials...")
    res = collect()

    metrics = [
        ('v_at_onset', 'V at onset (V)', 'mV-ish', 1e3),
        ('gI_at_onset', 'gI at onset (nS)', 'nS', 1e9),
        ('rate_pre_50ms', 'pre-stim rate (Hz)', 'Hz', 1.0),
    ]

    n_rows = len(metrics)
    n_cols = len(TARGETS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.0 * n_cols, 2.8 * n_rows),
                             squeeze=False, sharex=True)

    summary_rows = []
    for col, (ln, pop) in enumerate(TARGETS):
        per_phase = res[(ln, pop)]
        for row, (mkey, label, _, scale) in enumerate(metrics):
            phases, m, s, n = summarize_metric(per_phase, mkey)
            m_disp = m * scale
            s_disp = s * scale
            ax = axes[row, col]
            ax.errorbar(phases, m_disp, yerr=s_disp, fmt='o-',
                        color='#2266aa', lw=1.5, capsize=3, ms=5)
            # Highlight 90/135 (best switch) vs 270/315 (worst)
            for ph_good in (90, 135):
                j = list(phases).index(ph_good)
                ax.axvline(ph_good, color='g', ls=':', lw=0.8, alpha=0.4)
            for ph_bad in (270, 315):
                j = list(phases).index(ph_bad)
                ax.axvline(ph_bad, color='r', ls=':', lw=0.8, alpha=0.4)
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(f'{ln} {pop}')
            if col == 0:
                ax.set_ylabel(label)
            if row == n_rows - 1:
                ax.set_xlabel('Phase at stim onset (°)')

            # Summary row
            for ph, mm, ss, nn in zip(phases, m, s, n):
                summary_rows.append({
                    'layer': ln, 'pop': pop, 'phase_deg': int(ph),
                    'metric': mkey, 'mean': float(mm), 'sem': float(ss),
                    'n': int(nn),
                })

    fig.suptitle(f'Excitability state at stim onset, vs trigger phase '
                 f'(mean ± SEM across trials, pre-window={PRE_WIN_MS} ms)\n'
                 'Green dotted = best switch phases (90°, 135°)   '
                 'Red dotted = worst switch phases (270°, 315°)',
                 y=1.0)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'phase_excitability.png')
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"saved {out_path}")

    # ---- Correlation: per-trial excitability vs response Δrate ----
    fig2, axes2 = plt.subplots(1, len(TARGETS), figsize=(4.4 * len(TARGETS), 4),
                               squeeze=False)
    corr_rows = []
    for col, (ln, pop) in enumerate(TARGETS):
        per_phase = res[(ln, pop)]
        all_v = []
        all_gI = []
        all_rpre = []
        all_dr = []
        all_phase = []
        for ph, entries in per_phase.items():
            for s in entries:
                if not (np.isfinite(s['v_at_onset'])
                        and np.isfinite(s['gI_at_onset'])
                        and np.isfinite(s['delta_rate'])):
                    continue
                all_v.append(s['v_at_onset'])
                all_gI.append(s['gI_at_onset'])
                all_rpre.append(s['rate_pre_50ms'])
                all_dr.append(s['delta_rate'])
                all_phase.append(ph)
        all_v = np.array(all_v)
        all_gI = np.array(all_gI)
        all_rpre = np.array(all_rpre)
        all_dr = np.array(all_dr)
        all_phase = np.array(all_phase)

        # Spearman correlations of each proxy with Δrate
        rho_v, p_v = spearmanr(all_v, all_dr)
        rho_g, p_g = spearmanr(all_gI, all_dr)
        rho_r, p_r = spearmanr(all_rpre, all_dr)

        # Scatter colored by phase
        ax = axes2[0, col]
        cmap = plt.cm.twilight
        ax.scatter(all_v * 1e3, all_dr, c=[cmap(p / 360) for p in all_phase],
                   s=22, alpha=0.85, edgecolor='k', lw=0.3)
        ax.set_xlabel('V at onset (mV-ish, pre-50ms)')
        ax.set_ylabel('Δ firing rate (Hz, post-stim)')
        ax.set_title(f'{ln} {pop}\nρ(V,Δr)={rho_v:+.2f}  '
                     f'ρ(gI,Δr)={rho_g:+.2f}  ρ(r_pre,Δr)={rho_r:+.2f}',
                     fontsize=9)
        ax.grid(True, alpha=0.3)

        corr_rows.append({
            'layer': ln, 'pop': pop,
            'rho_v_vs_dr': rho_v, 'p_v': p_v,
            'rho_gI_vs_dr': rho_g, 'p_gI': p_g,
            'rho_rpre_vs_dr': rho_r, 'p_rpre': p_r,
            'n_trials': len(all_v),
        })

    fig2.suptitle('Per-trial: excitability proxies vs response Δrate '
                  '(color = onset phase)', y=1.02)
    plt.tight_layout()
    out2 = os.path.join(RESULTS_DIR, 'phase_excitability_corr.png')
    plt.savefig(out2, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"saved {out2}")

    # ---- CSV ----
    out_csv = os.path.join(RESULTS_DIR, 'phase_excitability.csv')
    with open(out_csv, 'w') as fh:
        fh.write("layer,pop,phase_deg,metric,mean,sem,n\n")
        for r in summary_rows:
            fh.write(f"{r['layer']},{r['pop']},{r['phase_deg']},"
                     f"{r['metric']},{r['mean']:.6g},{r['sem']:.6g},"
                     f"{r['n']}\n")
    print(f"saved {out_csv}")

    out_csv2 = os.path.join(RESULTS_DIR,
                            'phase_excitability_correlation.csv')
    with open(out_csv2, 'w') as fh:
        fh.write("layer,pop,n_trials,rho_v_vs_dr,p_v,rho_gI_vs_dr,p_gI,"
                 "rho_rpre_vs_dr,p_rpre\n")
        for r in corr_rows:
            fh.write(f"{r['layer']},{r['pop']},{r['n_trials']},"
                     f"{r['rho_v_vs_dr']:.4f},{r['p_v']:.4g},"
                     f"{r['rho_gI_vs_dr']:.4f},{r['p_gI']:.4g},"
                     f"{r['rho_rpre_vs_dr']:.4f},{r['p_rpre']:.4g}\n")
    print(f"saved {out_csv2}")

    # ---- Console: best vs worst phase comparison ----
    print("\n=== Excitability at best (90°,135°) vs worst (270°,315°) phases ===")
    print(f"{'pop':>9} {'metric':>16}  best mean ± sem    worst mean ± sem    "
          "Δ(best-worst)")
    for ln, pop in TARGETS:
        per_phase = res[(ln, pop)]
        for mkey, label, _, scale in metrics:
            phases, m, s, n = summarize_metric(per_phase, mkey)
            idx_best = [list(phases).index(90), list(phases).index(135)]
            idx_worst = [list(phases).index(270), list(phases).index(315)]
            mb = np.mean(m[idx_best]) * scale
            sb = np.mean(s[idx_best]) * scale
            mw = np.mean(m[idx_worst]) * scale
            sw = np.mean(s[idx_worst]) * scale
            print(f"  {ln} {pop:>3} {mkey:>16}  "
                  f"{mb:+9.3f} ± {sb:5.3f}   "
                  f"{mw:+9.3f} ± {sw:5.3f}   "
                  f"Δ={mb-mw:+8.3f}")

    print("\n=== Per-trial Spearman correlations (proxy vs Δrate) ===")
    print(f"{'pop':>9}  {'n':>4}  {'ρ(V)':>8} {'p':>8}   "
          f"{'ρ(gI)':>8} {'p':>8}   {'ρ(rpre)':>8} {'p':>8}")
    for r in corr_rows:
        print(f"  {r['layer']} {r['pop']:>3}  {r['n_trials']:>4d}  "
              f"{r['rho_v_vs_dr']:>+8.3f} {r['p_v']:>8.2g}   "
              f"{r['rho_gI_vs_dr']:>+8.3f} {r['p_gI']:>8.2g}   "
              f"{r['rho_rpre_vs_dr']:>+8.3f} {r['p_rpre']:>8.2g}")


if __name__ == '__main__':
    main()
