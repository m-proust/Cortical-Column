"""
Rank trials by spectral switch quality: suppression below 20 Hz + increase above 20 Hz.

Usage:
    python rank_trials.py results/trials4_05_04
    python rank_trials.py results/trials4_05_04 --n_best 5 --lfp_key lfp_matrix
"""

import argparse
import numpy as np
from scipy.signal import detrend, welch
from pathlib import Path


def score_trial(trial_data, pre_window_ms=300, post_window_ms=300,
                post_start_ms=200, lfp_key='bipolar_lfp', fs=10000,
                ch_start=3, ch_end=13):
    """
    Compute a spectral switch score for a single trial.
    Score = mean % increase above 20 Hz  -  mean % change below 20 Hz.
    Higher is better (more negative below 20, more positive above 20).
    Only uses bipolar channels ch_start to ch_end (inside layers).
    """
    lfp_all = trial_data[lfp_key]
    time = trial_data['time']
    stim = trial_data['stim_onset_ms']
    n_channels = lfp_all.shape[0]

    pre_mask = (time >= stim - pre_window_ms) & (time < stim)
    post_mask = (time >= stim + post_start_ms) & \
                (time < stim + post_start_ms + post_window_ms)

    low_changes = []
    high_changes = []

    for ch in range(ch_start, min(ch_end + 1, n_channels)):
        pre = lfp_all[ch][pre_mask].copy()
        post = lfp_all[ch][post_mask].copy()

        if len(pre) == 0 or len(post) == 0:
            continue
        if np.any(np.isnan(pre)) or np.any(np.isnan(post)):
            continue

        pre = detrend(pre)
        post = detrend(post)

        nperseg = min(len(pre), 100 * min(1024, len(pre) // 4))
        f, psd_pre = welch(pre, fs=fs, nperseg=nperseg, window='hann')
        _, psd_post = welch(post, fs=fs, nperseg=nperseg, window='hann')

        low_mask = (f >= 1) & (f <= 20)
        high_mask = (f > 20) & (f <= 100)

        low_pre = np.mean(psd_pre[low_mask])
        low_post = np.mean(psd_post[low_mask])
        high_pre = np.mean(psd_pre[high_mask])
        high_post = np.mean(psd_post[high_mask])

        low_pct = (low_post - low_pre) / (low_pre + 1e-30) * 100
        high_pct = (high_post - high_pre) / (high_pre + 1e-30) * 100

        low_changes.append(low_pct)
        high_changes.append(high_pct)

    if not low_changes:
        return None

    low_change = np.mean(low_changes)
    high_change = np.mean(high_changes)
    score = high_change - low_change

    return {
        'score': score,
        'low_change': low_change,
        'high_change': high_change,
    }


def main():
    parser = argparse.ArgumentParser(description='Rank trials by spectral switch quality')
    parser.add_argument('trial_dir', type=str, help='Path to trial directory')
    parser.add_argument('--n_best', type=int, default=5, help='Number of top trials to show')
    parser.add_argument('--lfp_key', type=str, default='bipolar_lfp',
                        choices=['bipolar_lfp', 'lfp_matrix'],
                        help='Which LFP signal to use')
    parser.add_argument('--pre_window_ms', type=int, default=300)
    parser.add_argument('--post_window_ms', type=int, default=300)
    parser.add_argument('--post_start_ms', type=int, default=200)
    parser.add_argument('--ch_start', type=int, default=3,
                        help='First bipolar channel to include (0-indexed)')
    parser.add_argument('--ch_end', type=int, default=13,
                        help='Last bipolar channel to include (0-indexed)')
    args = parser.parse_args()

    trial_dir = Path(args.trial_dir)
    trial_files = sorted(trial_dir.glob('trial_*.npz'))

    if not trial_files:
        print(f"No trial files found in {trial_dir}")
        return

    print(f"Found {len(trial_files)} trials in {trial_dir}")
    print(f"Using LFP key: {args.lfp_key}")
    print(f"Windows: pre={args.pre_window_ms}ms, post={args.post_window_ms}ms, "
          f"post_start={args.post_start_ms}ms")
    print(f"Channels: {args.ch_start} to {args.ch_end}")
    print()

    results = []

    for tf in trial_files:
        data = np.load(tf, allow_pickle=True)

        trial_id = int(data['trial_id'])
        seed = int(data['seed']) if 'seed' in data else None

        lfp_key_to_use = args.lfp_key if args.lfp_key in data else 'bipolar_matrix'
        trial_data = {
            'time': data['time_array_ms'],
            args.lfp_key: data[lfp_key_to_use],
            'stim_onset_ms': float(data['stim_onset_ms']),
        }

        result = score_trial(trial_data, pre_window_ms=args.pre_window_ms,
                             post_window_ms=args.post_window_ms,
                             post_start_ms=args.post_start_ms,
                             lfp_key=args.lfp_key,
                             ch_start=args.ch_start,
                             ch_end=args.ch_end)

        if result is None:
            print(f"  trial {trial_id}: SKIPPED (no valid channels)")
            continue

        result['trial_id'] = trial_id
        result['seed'] = seed
        result['file'] = tf.name
        results.append(result)

    if not results:
        print("No valid trials found!")
        return

    # Sort by <20Hz change (most negative = best suppression)
    results.sort(key=lambda r: r['low_change'])

    # Print all trials ranked
    print(f"{'Rank':<5} {'Trial':<8} {'Seed':<12} {'<20Hz %':<12} {'>20Hz %':<12} {'Score':<10}")
    print("-" * 60)
    for i, r in enumerate(results):
        marker = " <-- BEST" if i < args.n_best else ""
        seed_str = str(r['seed']) if r['seed'] is not None else "N/A"
        print(f"{i+1:<5} {r['trial_id']:<8} {seed_str:<12} "
              f"{r['low_change']:<12.1f} {r['high_change']:<12.1f} {r['score']:<10.1f}{marker}")

    print()
    print(f"=== Top {args.n_best} trials (best <20Hz suppression) ===")
    for i, r in enumerate(results[:args.n_best]):
        seed_str = str(r['seed']) if r['seed'] is not None else "N/A"
        print(f"  #{i+1}: trial {r['trial_id']} (seed={seed_str}) — "
              f"<20Hz={r['low_change']:+.1f}%, >20Hz={r['high_change']:+.1f}%, score={r['score']:.1f}")

    # Print worst for comparison
    print(f"\n=== Worst {min(3, len(results))} trials (most <20Hz increase) ===")
    for r in results[-min(3, len(results)):]:
        seed_str = str(r['seed']) if r['seed'] is not None else "N/A"
        print(f"  trial {r['trial_id']} (seed={seed_str}) — "
              f"<20Hz={r['low_change']:+.1f}%, >20Hz={r['high_change']:+.1f}%, score={r['score']:.1f}")


if __name__ == '__main__':
    main()
