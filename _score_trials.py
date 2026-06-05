"""Rank trials in results/trials3_03-06 by spectral-switch cleanliness."""
import numpy as np
from scipy.signal import detrend

from laminar_power_change import load_trials, multitaper_psd, _time_vector_for_key


def trial_score(trial, lfp_key='bipolar_lfp',
                pre_window_ms=500, post_window_ms=500, post_start_ms=300,
                low_band=(2, 20), high_band=(20, 120)):
    time = _time_vector_for_key(trial, lfp_key)
    fs = 1000.0 / float(np.mean(np.diff(time)))
    stim = float(trial['stim_onset_ms'])
    lfp = trial[lfp_key]
    n_ch = lfp.shape[0]

    pre_mask = (time >= stim - pre_window_ms) & (time < stim)
    post_mask = ((time >= stim + post_start_ms) &
                 (time < stim + post_start_ms + post_window_ms))

    low_pcts, high_pcts = [], []
    for ch in range(n_ch):
        pre = lfp[ch][pre_mask].copy()
        post = lfp[ch][post_mask].copy()
        if len(pre) == 0 or len(post) == 0:
            continue
        if np.any(~np.isfinite(pre)) or np.any(~np.isfinite(post)):
            continue
        pre = detrend(pre)
        post = detrend(post)
        nfft = 2 ** int(np.ceil(np.log2(min(len(pre), len(post)))))
        f, psd_pre = multitaper_psd(pre, fs=fs, NW=2, nfft=nfft)
        _, psd_post = multitaper_psd(post, fs=fs, NW=2, nfft=nfft)

        low_m = (f >= low_band[0]) & (f < low_band[1])
        high_m = (f >= high_band[0]) & (f <= high_band[1])
        if not low_m.any() or not high_m.any():
            continue
        low_pre = np.trapz(psd_pre[low_m], f[low_m])
        low_post = np.trapz(psd_post[low_m], f[low_m])
        high_pre = np.trapz(psd_pre[high_m], f[high_m])
        high_post = np.trapz(psd_post[high_m], f[high_m])
        low_pcts.append((low_post - low_pre) / low_pre * 100)
        high_pcts.append((high_post - high_pre) / high_pre * 100)

    low_pcts = np.array(low_pcts)
    high_pcts = np.array(high_pcts)
    # Cleanliness: want low_pct < 0 on ALL channels and high_pct > 0 on ALL channels.
    low_mean = np.mean(low_pcts)
    high_mean = np.mean(high_pcts)
    low_frac_neg = np.mean(low_pcts < 0)
    high_frac_pos = np.mean(high_pcts > 0)
    # Composite: high_mean - low_mean, weighted by fraction of channels going the right way.
    cleanliness = (high_mean - low_mean) * 0.5 * (low_frac_neg + high_frac_pos)
    return dict(low_mean=low_mean, high_mean=high_mean,
                low_frac_neg=low_frac_neg, high_frac_pos=high_frac_pos,
                cleanliness=cleanliness, n_ch=len(low_pcts))


if __name__ == '__main__':
    base = 'results/trials3_03-06'
    trials = load_trials(base, 12)
    rows = []
    for i, tr in enumerate(trials):
        s = trial_score(tr)
        rows.append((i, s))
        print(f"trial {i:03d}  low%={s['low_mean']:+7.1f}  "
              f"high%={s['high_mean']:+7.1f}  "
              f"low_frac<0={s['low_frac_neg']:.2f}  "
              f"high_frac>0={s['high_frac_pos']:.2f}  "
              f"score={s['cleanliness']:+8.1f}")

    rows.sort(key=lambda r: r[1]['cleanliness'], reverse=True)
    print("\n=== ranked by cleanliness ===")
    for i, s in rows:
        print(f"trial {i:03d}  score={s['cleanliness']:+8.1f}  "
              f"low%={s['low_mean']:+7.1f}  high%={s['high_mean']:+7.1f}")
