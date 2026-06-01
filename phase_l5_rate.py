"""Plot L5 E-rate aligned to stim onset, one trace per phase, averaged across
all available seeds. Also prints alpha-band rate at stim onset per phase
(mean across seeds)."""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

RESULTS_DIR = 'results/trials_phase_07_05'
WIN_MS = (-200, 300)
ALPHA_BAND = (8.0, 13.0)
PHASES = (0, 45, 90, 135, 180, 225, 270, 315)


def bandpass(sig, fs, low, high, order=4):
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig - np.mean(sig)
    sos = butter(order, [low/(0.5*fs), high/(0.5*fs)],
                 btype='band', output='sos')
    return sosfiltfilt(sos, sig)


def load_phase_trials(phase_deg):
    pat = os.path.join(RESULTS_DIR,
                       f"trial_seed*_phase{int(phase_deg):03d}.npz")
    files = sorted(glob.glob(pat))
    out = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        rd = d['rate_data'].item()
        out.append({
            'seed': int(str(d['network_seed'])),
            'phase_deg': float(d['target_phase_deg']),
            'stim_onset_ms': float(d['stim_onset_ms']),
            'L5_E_t': rd['L5']['E_rate']['t_ms'],
            'L5_E_r': rd['L5']['E_rate']['rate_hz'],
        })
    return out


def aligned_segment(trial, win_ms):
    t = trial['L5_E_t']
    r = trial['L5_E_r']
    onset = trial['stim_onset_ms']
    t_rel = t - onset
    mask = (t_rel >= win_ms[0]) & (t_rel <= win_ms[1])
    return t_rel[mask], r[mask]


def resample_common_grid(trials, win_ms):
    """Interpolate each trial's L5 rate onto a common t_rel grid."""
    dt = float(np.mean(np.diff(trials[0]['L5_E_t'])))
    t_grid = np.arange(win_ms[0], win_ms[1] + dt, dt)
    rates = []
    for tr in trials:
        t_rel, r = aligned_segment(tr, win_ms)
        rates.append(np.interp(t_grid, t_rel, r))
    return t_grid, np.array(rates)


def alpha_value_at_onset(trial, fs):
    t = trial['L5_E_t']
    r = trial['L5_E_r']
    onset = trial['stim_onset_ms']
    idx = int(np.argmin(np.abs(t - onset)))
    win = 1500
    i0, i1 = max(0, idx - win), min(len(r), idx + win)
    seg = r[i0:i1]
    seg_alpha = bandpass(seg, fs, *ALPHA_BAND)
    return float(seg_alpha[idx - i0])


def main():
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    cmap = plt.cm.twilight

    bad_phases = {135.0, 180.0}
    summary = []

    n_seeds_per_phase = {}
    for ph in PHASES:
        trials = load_phase_trials(ph)
        if not trials:
            print(f"phase {ph}: no trials")
            continue
        n_seeds_per_phase[ph] = len(trials)
        color = cmap(ph / 360.0)
        ls = '--' if ph in bad_phases else '-'
        lw = 2.2 if ph in bad_phases else 1.4

        t_grid, rates = resample_common_grid(trials, WIN_MS)
        mean_r = np.mean(rates, axis=0)
        sem_r = np.std(rates, axis=0) / np.sqrt(len(rates))

        # Top: raw rate, mean +/- SEM
        axes[0].plot(t_grid, mean_r, color=color, lw=lw, ls=ls,
                     label=f"{ph:.0f}° (n={len(trials)})")
        axes[0].fill_between(t_grid, mean_r - sem_r, mean_r + sem_r,
                             color=color, alpha=0.15, lw=0)

        # Bottom: bandpassed rate, mean across seeds
        dt = float(np.mean(np.diff(trials[0]['L5_E_t'])))
        fs = 1000.0 / dt
        alpha_band_each = np.array([
            bandpass(rates[i], fs, *ALPHA_BAND) for i in range(len(rates))
        ])
        mean_a = np.mean(alpha_band_each, axis=0)
        sem_a = np.std(alpha_band_each, axis=0) / np.sqrt(len(rates))
        axes[1].plot(t_grid, mean_a, color=color, lw=lw, ls=ls)
        axes[1].fill_between(t_grid, mean_a - sem_a, mean_a + sem_a,
                             color=color, alpha=0.15, lw=0)

        # Onset-value summary across seeds
        per_seed_alpha = [alpha_value_at_onset(tr, fs) for tr in trials]
        summary.append({
            'phase': ph,
            'n_seeds': len(trials),
            'mean_alpha_onset': float(np.mean(per_seed_alpha)),
            'sem_alpha_onset': (float(np.std(per_seed_alpha) / np.sqrt(len(per_seed_alpha)))
                                if len(per_seed_alpha) > 1 else float('nan')),
            'per_seed': per_seed_alpha,
        })

    for ax in axes:
        ax.axvline(0, color='k', ls=':', lw=1, alpha=0.7)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('L5 E rate (Hz)')
    n_total = sum(n_seeds_per_phase.values())
    axes[0].set_title(
        f'L5 E rate aligned to stim onset — averaged over seeds '
        f'(total trials: {n_total})')
    axes[1].set_ylabel(f'L5 E rate, {ALPHA_BAND[0]}-{ALPHA_BAND[1]} Hz (Hz)')
    axes[1].set_xlabel('Time from stim onset (ms)')
    axes[0].legend(ncol=4, fontsize=8, loc='upper right')
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, 'phase_L5E_rate_avg.png')
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"saved {out_path}")

    # Summary plot: alpha-band rate value at onset, mean +/- SEM, vs phase
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    phases_arr = np.array([s['phase'] for s in summary])
    means = np.array([s['mean_alpha_onset'] for s in summary])
    sems = np.array([s['sem_alpha_onset'] for s in summary])
    ax2.errorbar(phases_arr, means, yerr=sems, fmt='o-', lw=1.6, capsize=4)
    for s in summary:
        ax2.scatter([s['phase']] * len(s['per_seed']), s['per_seed'],
                    color='gray', alpha=0.5, s=18, zorder=1)
    ax2.axhline(0, color='k', ls=':', lw=0.8)
    ax2.set_xticks(list(PHASES))
    ax2.set_xlabel('Alpha phase at stim onset (deg)')
    ax2.set_ylabel('L5 E α-band rate at onset (Hz)')
    ax2.set_title('Alpha-band L5 E rate at stim onset vs trigger phase')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = os.path.join(RESULTS_DIR, 'phase_L5E_alpha_at_onset.png')
    plt.savefig(out2, dpi=130)
    plt.close()
    print(f"saved {out2}")

    print(f"\n{'phase':>6} | {'n':>3} | {'α-rate at onset (mean ± SEM)':>32}")
    for s in summary:
        sem_str = (f"{s['sem_alpha_onset']:+.3f}"
                   if not np.isnan(s['sem_alpha_onset']) else '   nan')
        print(f"{s['phase']:>6.0f} | {s['n_seeds']:>3d} | "
              f"{s['mean_alpha_onset']:+.3f} ± {sem_str}")


if __name__ == '__main__':
    main()
