"""Rank individual trials by quality of low-freq suppression + high-freq enhancement,
then plot the laminar power-change profile of the best ones so you can pick the
lucky network seed to fix in trials.py for a hierarchical run.

Each trial file (written by trials3.py) stores its own 'seed' — that's the
network_seed you can copy back into trials.py for a fixed-network sweep.
"""

import os
import glob
import argparse
import numpy as np

from laminar_power_change import (
    multitaper_psd,
    plot_laminar_spectral_profile,
)
from scipy.signal import detrend


def _trial_psds(lfp_mat, time, stim_onset_ms,
                pre_window_ms, post_window_ms, post_start_ms):
    fs = 1000.0 / float(np.mean(np.diff(time)))
    pre_mask = (time >= stim_onset_ms - pre_window_ms) & (time < stim_onset_ms)
    post_mask = ((time >= stim_onset_ms + post_start_ms) &
                 (time < stim_onset_ms + post_start_ms + post_window_ms))
    nfft = 2 ** int(np.ceil(np.log2(min(int(pre_mask.sum()),
                                        int(post_mask.sum())))))

    n_ch = lfp_mat.shape[0]
    psd_pre = np.zeros((n_ch, nfft // 2 + 1))
    psd_post = np.zeros_like(psd_pre)
    f = None
    for ch in range(n_ch):
        pre = detrend(lfp_mat[ch][pre_mask])
        post = detrend(lfp_mat[ch][post_mask])
        f, psd_pre[ch] = multitaper_psd(pre, fs=fs, NW=2, nfft=nfft)
        _, psd_post[ch] = multitaper_psd(post, fs=fs, NW=2, nfft=nfft)
    return f, psd_pre, psd_post


def _score(f, psd_pre, psd_post, low_band, high_band):
    pre = psd_pre.mean(axis=0)
    post = psd_post.mean(axis=0)
    pct = (post - pre) / pre * 100.0
    low_mask = (f >= low_band[0]) & (f < low_band[1])
    high_mask = (f >= high_band[0]) & (f <= high_band[1])
    low_pct = float(pct[low_mask].mean())
    high_pct = float(pct[high_mask].mean())
    return high_pct - low_pct, low_pct, high_pct


def rank_directory(base_path, lfp_key,
                   pre_window_ms, post_window_ms, post_start_ms,
                   low_band, high_band):
    files = sorted(glob.glob(os.path.join(base_path, "trial_*.npz")))
    if not files:
        raise FileNotFoundError(f"no trial_*.npz under {base_path}")
    rows = []
    for fname in files:
        data = np.load(fname, allow_pickle=True)
        if lfp_key not in data.files:
            continue
        f, psd_pre, psd_post = _trial_psds(
            data[lfp_key], data['time_array_ms'],
            float(data['stim_onset_ms']),
            pre_window_ms, post_window_ms, post_start_ms,
        )
        score, low_pct, high_pct = _score(
            f, psd_pre, psd_post, low_band, high_band,
        )
        seed = int(data['seed']) if 'seed' in data.files else -1
        trial_id = int(data['trial_id']) if 'trial_id' in data.files else -1
        rows.append({
            'fname': fname,
            'trial_id': trial_id,
            'seed': seed,
            'score': score,
            'low_pct': low_pct,
            'high_pct': high_pct,
        })
    rows.sort(key=lambda r: r['score'], reverse=True)
    return rows


def _load_for_plot(fname):
    data = np.load(fname, allow_pickle=True)
    trial = {
        'time': data['time_array_ms'],
        'bipolar_lfp': data['bipolar_matrix'],
        'lfp_matrix': data['lfp_matrix'],
        'baseline_ms': float(data['baseline_ms']),
        'stim_onset_ms': float(data['stim_onset_ms']),
        'channel_labels': data['channel_labels'],
        'channel_depths': data['channel_depths'],
        'electrode_positions': data['electrode_positions'],
    }
    if 'lfp_current_matrix' in data.files:
        trial['lfp_current_matrix'] = data['lfp_current_matrix']
        trial['time_current_ms'] = data['time_current_ms']
    return trial


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str,
                        default="results/trials_19_05_2")
    parser.add_argument("--lfp-key", type=str, default="bipolar_matrix",
                        choices=["bipolar_matrix", "lfp_matrix"])
    parser.add_argument("--top-k", type=int, default=3,
                        help="how many top trials to plot in detail")
    parser.add_argument("--low-band", type=float, nargs=2, default=(1, 20))
    parser.add_argument("--high-band", type=float, nargs=2, default=(30, 80))
    parser.add_argument("--pre-ms", type=float, default=500)
    parser.add_argument("--post-ms", type=float, default=500)
    parser.add_argument("--post-start-ms", type=float, default=500)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    rows = rank_directory(
        args.base_path, args.lfp_key,
        pre_window_ms=args.pre_ms,
        post_window_ms=args.post_ms,
        post_start_ms=args.post_start_ms,
        low_band=tuple(args.low_band),
        high_band=tuple(args.high_band),
    )

    print(f"\n=== Ranked {len(rows)} trials from {args.base_path} ===")
    print(f"low band = {args.low_band}, high band = {args.high_band}, "
          f"lfp_key = {args.lfp_key}")
    print(f"{'rank':>4}  {'trial':>5}  {'seed':>10}  "
          f"{'score':>8}  {'low%':>8}  {'high%':>8}")
    for rank, r in enumerate(rows):
        print(f"{rank:>4}  {r['trial_id']:>5}  {r['seed']:>10}  "
              f"{r['score']:>+8.1f}  {r['low_pct']:>+8.1f}  "
              f"{r['high_pct']:>+8.1f}")

    print("\n>>> Top seed candidates to reuse in trials.py:")
    for r in rows[: args.top_k]:
        print(f"    network_seed={r['seed']}   (trial {r['trial_id']}, "
              f"low={r['low_pct']:+.1f}%, high={r['high_pct']:+.1f}%)")

    if args.no_plot:
        raise SystemExit

    print(f"\nPlotting top {min(args.top_k, len(rows))} trials...")
    for r in rows[: args.top_k]:
        trial = _load_for_plot(r['fname'])
        suffix = (f"  seed={r['seed']}  trial={r['trial_id']}  "
                  f"low={r['low_pct']:+.0f}%  high={r['high_pct']:+.0f}%")
        plot_laminar_spectral_profile(
            [trial],
            pre_window_ms=args.pre_ms,
            post_window_ms=args.post_ms,
            post_start_ms=args.post_start_ms,
            freq_range=(0, 120),
            log_freq=False,
            remove_mean=True,
            do_detrend=True,
            lfp_key='bipolar_lfp',
            title_suffix=suffix,
        )
