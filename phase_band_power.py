"""Band-integrated power change vs. alpha phase at selected laminar channels.

Collapses the laminar PSD grid into two curves per channel: alpha (8–13 Hz) and
gamma (30–80 Hz) % change (post − pre) / pre × 100, plotted against the alpha
phase at stim onset. One panel per reference channel (L4C, L5).
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend

from laminar_power_change import multitaper_psd, _time_vector_for_key
from phase_laminar_power import load_phase_trials
from trials_phase import LAYER_Z_CENTER

RESULTS_DIR = 'results/trials_phase_07_05'
PHASES = (0, 45, 90, 135, 180, 225, 270, 315)
PRE_WINDOW_MS = 500
POST_WINDOW_MS = 500
POST_START_MS = 300

BANDS = {
    'alpha (8–13 Hz)': (8, 13),
    'beta (15–25 Hz)': (15, 25),
    'gamma (30–80 Hz)': (30, 80),
    'high-γ (80–120 Hz)': (80, 120),
}

LAYERS_TO_PROBE = ('L23', 'L4C', 'L5', 'L6')


def pick_bipolar_channel(channel_depths, layer_name, n_bipolar):
    """Bipolar channel midpoint closest to layer center.
    channel_depths is the electrode depth list (len = n_elec); bipolar midpoint
    is mean of adjacent pairs (len = n_elec - 1)."""
    depths = np.asarray(channel_depths)
    mid = (depths[:-1] + depths[1:]) / 2
    mid = mid[:n_bipolar]
    target = LAYER_Z_CENTER[layer_name]
    return int(np.argmin(np.abs(mid - target)))


def per_trial_band_power(trial, ch_idx, win_pre_ms, win_post_ms,
                         post_start_ms, lfp_key='bipolar_lfp'):
    lfp = trial[lfp_key][ch_idx]
    time = _time_vector_for_key(trial, lfp_key)
    stim = trial['stim_onset_ms']
    pre_mask = (time >= stim - win_pre_ms) & (time < stim)
    post_mask = ((time >= stim + post_start_ms) &
                 (time < stim + post_start_ms + win_post_ms))
    pre = lfp[pre_mask].copy()
    post = lfp[post_mask].copy()
    if len(pre) == 0 or len(post) == 0:
        return None
    pre = detrend(pre)
    post = detrend(post)
    fs = 1000.0 / float(np.mean(np.diff(time)))
    nfft = 2 ** int(np.ceil(np.log2(min(len(pre), len(post)))))
    f, psd_pre = multitaper_psd(pre, fs=fs, NW=2, nfft=nfft)
    _, psd_post = multitaper_psd(post, fs=fs, NW=2, nfft=nfft)
    out = {}
    for name, (lo, hi) in BANDS.items():
        band_mask = (f >= lo) & (f <= hi)
        if not np.any(band_mask):
            out[name] = (np.nan, np.nan)
            continue
        p_pre = float(np.trapz(psd_pre[band_mask], f[band_mask]))
        p_post = float(np.trapz(psd_post[band_mask], f[band_mask]))
        out[name] = (p_pre, p_post)
    return out


def collect_band_power():
    """Returns nested dict results[layer][band][phase] = list of per-trial
    (pre, post) tuples."""
    res = {ln: {b: {ph: [] for ph in PHASES} for b in BANDS}
           for ln in LAYERS_TO_PROBE}
    n_bipolar_ref = None
    depths_ref = None
    for ph in PHASES:
        trials = load_phase_trials(RESULTS_DIR, ph)
        if not trials:
            continue
        if n_bipolar_ref is None:
            depths_ref = trials[0]['channel_depths']
            n_bipolar_ref = trials[0]['bipolar_lfp'].shape[0]
        for tr in trials:
            for ln in LAYERS_TO_PROBE:
                ch = pick_bipolar_channel(tr['channel_depths'], ln,
                                          tr['bipolar_lfp'].shape[0])
                bp = per_trial_band_power(
                    tr, ch, PRE_WINDOW_MS, POST_WINDOW_MS, POST_START_MS,
                    'bipolar_lfp')
                if bp is None:
                    continue
                for b in BANDS:
                    res[ln][b][ph].append(bp[b])
    return res


def summarize_pct(per_phase):
    """per_phase: {phase: [(pre, post), ...]} → (phases, mean_pct, sem_pct)."""
    phases = np.array(sorted(per_phase.keys()))
    means, sems = [], []
    for ph in phases:
        rows = per_phase[ph]
        if not rows:
            means.append(np.nan)
            sems.append(np.nan)
            continue
        pcts = np.array([(post - pre) / pre * 100
                         for pre, post in rows
                         if pre > 0 and np.isfinite(pre) and np.isfinite(post)])
        if len(pcts) == 0:
            means.append(np.nan)
            sems.append(np.nan)
        else:
            means.append(float(np.mean(pcts)))
            sems.append(float(np.std(pcts) / np.sqrt(len(pcts))))
    return phases, np.array(means), np.array(sems)


def main():
    print("Loading trials and computing band powers...")
    res = collect_band_power()

    band_colors = {
        'alpha (8–13 Hz)': '#1f77b4',
        'beta (15–25 Hz)': '#2ca02c',
        'gamma (30–80 Hz)': '#d62728',
        'high-γ (80–120 Hz)': '#9467bd',
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    for ax, ln in zip(axes, LAYERS_TO_PROBE):
        for b in BANDS:
            phases, m, s = summarize_pct(res[ln][b])
            ax.errorbar(phases, m, yerr=s, fmt='o-', lw=1.5, capsize=3,
                        ms=5, color=band_colors[b], label=b)
        ax.axhline(0, color='k', ls=':', lw=0.7)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_title(f'{ln} bipolar channel')
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('% power change\n(post − pre) / pre × 100')
    for ax in axes[-2:]:
        ax.set_xlabel('Alpha phase at stim onset (°)')
    axes[0].legend(fontsize=8, loc='best', framealpha=0.7)
    fig.suptitle('Band-integrated power change vs. trigger phase  '
                 f'(pre: 500 ms; post: {POST_START_MS}–'
                 f'{POST_START_MS + POST_WINDOW_MS} ms)',
                 y=1.0)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'phase_band_power.png')
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"saved {out_path}")

    # CSV
    out_csv = os.path.join(RESULTS_DIR, 'phase_band_power.csv')
    with open(out_csv, 'w') as fh:
        fh.write("layer,band,phase_deg,mean_pct,sem_pct,n_trials\n")
        for ln in LAYERS_TO_PROBE:
            for b in BANDS:
                phases, m, s = summarize_pct(res[ln][b])
                for ph, mm, ss in zip(phases, m, s):
                    n = len(res[ln][b][int(ph)])
                    fh.write(f"{ln},{b},{int(ph)},{mm:.4f},{ss:.4f},{n}\n")
    print(f"saved {out_csv}")


if __name__ == '__main__':
    main()
