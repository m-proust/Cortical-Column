"""Rank trials in results/trials_18_05 by spectral switch quality.

Uses the same windows / detrend / multitaper parameters as
laminar_power_change.py: pre_window 500 ms, post_window 500 ms starting
500 ms after stimulus, detrend, NW=2 multitaper PSD.

For each trial we compute the channel-averaged PSD pre vs post and score:
    low_change  = mean % change over 1-20 Hz   (want negative)
    high_change = mean % change over 20-100 Hz (want positive)
    score       = high_change - low_change      (bigger is better)
"""
import numpy as np
from scipy import signal
from scipy.signal import detrend
from scipy.signal.windows import dpss


BASE = 'results/trials_18_05'
N_TRIALS = 36

SETTLING_MS = 1500   # drop this much from the very start of the simulation
PRE_END_OFFSET_MS = 0  # end the pre-window this many ms BEFORE stim_onset
POST_WIN_MS = 1500
POST_START_MS = 300
VERY_LOW_BAND = (1, 10)
LOW_BAND = (1, 20)
HIGH_BAND = (20, 100)


def multitaper_psd(x, fs, NW=2, nfft=None):
    x = x - np.mean(x)
    if nfft is None:
        nfft = 2 ** int(np.ceil(np.log2(len(x))))
    K = int(2 * NW - 1)
    tapers = dpss(len(x), NW, K)
    psds = []
    for taper in tapers:
        f, p = signal.periodogram(x * taper, fs=fs, nfft=nfft,
                                  scaling='density')
        psds.append(p)
    return f, np.mean(psds, axis=0)


def trial_psd(trial, lfp_key='bipolar_lfp'):
    lfp = trial[lfp_key]
    time = trial['time']
    stim = float(trial['stim_onset_ms'])
    fs = 1000.0 / float(np.mean(np.diff(time)))

    pre_start = SETTLING_MS
    pre_end = stim - PRE_END_OFFSET_MS
    pre_mask = (time >= pre_start) & (time < pre_end)
    post_mask = ((time >= stim + POST_START_MS) &
                 (time < stim + POST_START_MS + POST_WIN_MS))

    n_ch = lfp.shape[0]
    pre_psds, post_psds = [], []
    f_out = None
    for ch in range(n_ch):
        pre = lfp[ch][pre_mask].copy()
        post = lfp[ch][post_mask].copy()
        if len(pre) == 0 or len(post) == 0:
            continue
        if np.any(np.isnan(pre)) or np.any(np.isnan(post)):
            continue
        pre = detrend(pre)
        post = detrend(post)
        nfft = 2 ** int(np.ceil(np.log2(min(len(pre), len(post)))))
        f, ppre = multitaper_psd(pre, fs=fs, NW=2, nfft=nfft)
        _, ppost = multitaper_psd(post, fs=fs, NW=2, nfft=nfft)
        pre_psds.append(ppre)
        post_psds.append(ppost)
        f_out = f

    return f_out, np.array(pre_psds), np.array(post_psds)


def score_trial(f, psd_pre, psd_post):
    # average across channels
    pre = np.mean(psd_pre, axis=0)
    post = np.mean(psd_post, axis=0)
    pct = (post - pre) / pre * 100.0

    vlow_mask = (f >= VERY_LOW_BAND[0]) & (f < VERY_LOW_BAND[1])
    low_mask = (f >= LOW_BAND[0]) & (f < LOW_BAND[1])
    high_mask = (f >= HIGH_BAND[0]) & (f <= HIGH_BAND[1])

    vlow_change = float(np.mean(pct[vlow_mask]))
    low_change = float(np.mean(pct[low_mask]))
    high_change = float(np.mean(pct[high_mask]))
    return vlow_change, low_change, high_change


def main():
    results = []
    for idx in range(N_TRIALS):
        fname = f"{BASE}/trial_{idx:03d}.npz"
        try:
            data = np.load(fname, allow_pickle=True)
        except FileNotFoundError:
            continue
        trial = {
            'time': data['time_array_ms'],
            'bipolar_lfp': data['bipolar_matrix'],
            'lfp_matrix': data['lfp_matrix'],
            'stim_onset_ms': float(data['stim_onset_ms']),
        }
        out_row = {'trial': idx}
        for key in ('bipolar_lfp', 'lfp_matrix'):
            f, ppre, ppost = trial_psd(trial, lfp_key=key)
            vlow, low, high = score_trial(f, ppre, ppost)
            out_row[f'{key}_vlow'] = vlow
            out_row[f'{key}_low'] = low
            out_row[f'{key}_high'] = high
            out_row[f'{key}_score'] = high - low
            out_row[f'{key}_score_vlow'] = high - vlow
        results.append(out_row)

    print(f"\nWindows: pre = [{SETTLING_MS} ms, stim_onset-{PRE_END_OFFSET_MS} ms) "
          f"(settling {SETTLING_MS} ms dropped), "
          f"post = {POST_WIN_MS} ms starting {POST_START_MS} ms after stim")

    # Rank by bipolar score
    results_sorted = sorted(results, key=lambda r: -r['bipolar_lfp_score'])

    print("\n=== Ranking by bipolar LFP (score = high% - low%, bigger=better) ===")
    print(f"{'trial':>5} {'vlow%(1-10)':>12} {'low%(1-20)':>12} {'high%(20-100)':>14} "
          f"{'score(20)':>10} {'score(10)':>10}   "
          f"{'kvlow':>8} {'klow':>8} {'khigh':>8} {'kscore':>8}")
    for r in results_sorted:
        print(f"{r['trial']:>5d} "
              f"{r['bipolar_lfp_vlow']:>+12.1f} "
              f"{r['bipolar_lfp_low']:>+12.1f} "
              f"{r['bipolar_lfp_high']:>+14.1f} "
              f"{r['bipolar_lfp_score']:>+10.1f} "
              f"{r['bipolar_lfp_score_vlow']:>+10.1f}   "
              f"{r['lfp_matrix_vlow']:>+8.1f} "
              f"{r['lfp_matrix_low']:>+8.1f} "
              f"{r['lfp_matrix_high']:>+8.1f} "
              f"{r['lfp_matrix_score']:>+8.1f}")

    print("\n=== Trials with BOTH low(1-20)<0 AND high>0 (bipolar) ===")
    good = [r for r in results
            if r['bipolar_lfp_low'] < 0 and r['bipolar_lfp_high'] > 0]
    good_sorted = sorted(good, key=lambda r: -r['bipolar_lfp_score'])
    for r in good_sorted:
        print(f"  trial {r['trial']:03d}: "
              f"low={r['bipolar_lfp_low']:+.1f}%  "
              f"high={r['bipolar_lfp_high']:+.1f}%  "
              f"score={r['bipolar_lfp_score']:+.1f}")

    print("\n=== Trials with BOTH vlow(1-10)<0 AND high>0 (bipolar) ===")
    good_v = [r for r in results
              if r['bipolar_lfp_vlow'] < 0 and r['bipolar_lfp_high'] > 0]
    good_v_sorted = sorted(good_v, key=lambda r: -r['bipolar_lfp_score_vlow'])
    for r in good_v_sorted:
        print(f"  trial {r['trial']:03d}: "
              f"vlow={r['bipolar_lfp_vlow']:+.1f}%  "
              f"high={r['bipolar_lfp_high']:+.1f}%  "
              f"score={r['bipolar_lfp_score_vlow']:+.1f}")

    print("\n=== Trials passing on BOTH bipolar and kernel (vlow<0, high>0) ===")
    both = [r for r in results
            if r['bipolar_lfp_vlow'] < 0 and r['bipolar_lfp_high'] > 0
            and r['lfp_matrix_vlow'] < 0 and r['lfp_matrix_high'] > 0]
    both_sorted = sorted(both,
                         key=lambda r: -(r['bipolar_lfp_score_vlow']
                                         + r['lfp_matrix_score_vlow']))
    for r in both_sorted:
        print(f"  trial {r['trial']:03d}: "
              f"bipolar vlow={r['bipolar_lfp_vlow']:+.1f}% "
              f"high={r['bipolar_lfp_high']:+.1f}%  ;  "
              f"kernel vlow={r['lfp_matrix_vlow']:+.1f}% "
              f"high={r['lfp_matrix_high']:+.1f}%")


if __name__ == '__main__':
    main()
