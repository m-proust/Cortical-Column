"""
Diagnostic script: identify what drives the spurious 1-2 Hz power increase.

Usage:  python diagnostic_low_freq.py
Edit BASE_PATH, N_TRIALS below as needed.
"""

import numpy as np
import warnings
from pathlib import Path
from scipy.signal import welch, detrend, decimate

# == CONFIG ===================================================================
BASE_PATH = "results/trials_05_04"
N_TRIALS = 25

# spectral windows (ms)
PRE_WINDOW_MS = 1000
POST_WINDOW_MS = 1000
POST_START_MS = 500

# the problematic band
LOW_FREQ_BAND = (0.5, 3.0)   # Hz

FS = 10000                    # original sampling rate (Hz)
DS_FACTOR = 10                # downsample to 1000 Hz before spectral analysis
FS_DS = FS // DS_FACTOR       # 1000 Hz after downsampling

GOOD_PERCENTILE = 33
BAD_PERCENTILE = 67
# =============================================================================


def load_trials(base_path, n_trials):
    trials = []
    for i in range(n_trials):
        fname = Path(base_path) / f"trial_{i:03d}.npz"
        if not fname.exists():
            print(f"  [skip] {fname} not found")
            continue
        d = np.load(fname, allow_pickle=True)
        trial = {
            "trial_id": int(d.get("trial_id", i)),
            "seed": int(d.get("seed", 0)),
            "time": d["time_array_ms"],
            "bipolar_lfp": d["bipolar_matrix"],
            "lfp_matrix": d["lfp_matrix"],
            "baseline_ms": float(d["baseline_ms"]),
            "stim_onset_ms": float(d["stim_onset_ms"]),
            "channel_depths": d["channel_depths"],
        }
        for key in ("rate_data", "spike_data"):
            v = d[key]
            trial[key] = v.item() if v.ndim == 0 else v
        trials.append(trial)
    return trials


def band_power(psd, freqs, band):
    """Integrate PSD over a frequency band using trapezoid rule."""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if mask.sum() < 2:
        # not enough bins - just sum what we have
        if mask.sum() == 1:
            df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
            return float(psd[mask][0] * df)
        return np.nan
    return float(np.trapezoid(psd[mask], freqs[mask]))


def safe_welch(signal, fs, target_freq_res=0.5):
    """
    Compute Welch PSD with nperseg chosen for good low-frequency resolution.
    target_freq_res: desired frequency resolution in Hz
    """
    # nperseg = fs / target_freq_res  -> gives bins spaced at target_freq_res
    nperseg = int(fs / target_freq_res)
    nperseg = min(nperseg, len(signal))  # can't exceed signal length

    if nperseg < 64:
        return None, None

    # 50% overlap, Hann window
    f, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                   window='hann', detrend='linear')
    return f, psd


def compute_low_freq_change(trial, lfp_key="bipolar_lfp"):
    """Return per-channel % change in LOW_FREQ_BAND power."""
    lfp_all = trial[lfp_key]
    time = trial["time"]
    stim = trial["stim_onset_ms"]
    n_ch = lfp_all.shape[0]

    changes = []
    for ch in range(n_ch):
        sig = lfp_all[ch]
        pre_mask = (time >= stim - PRE_WINDOW_MS) & (time < stim)
        post_mask = (time >= stim + POST_START_MS) & (
            time < stim + POST_START_MS + POST_WINDOW_MS
        )
        pre = sig[pre_mask].astype(np.float64).copy()
        post = sig[post_mask].astype(np.float64).copy()

        if len(pre) < 100 or len(post) < 100:
            changes.append(np.nan)
            continue

        # remove mean and linear trend manually (avoids scipy detrend overflow)
        pre -= np.mean(pre)
        post -= np.mean(post)

        # downsample from 10kHz to 1kHz for stable low-freq spectral analysis
        if DS_FACTOR > 1 and len(pre) >= DS_FACTOR * 10:
            pre_ds = decimate(pre, DS_FACTOR, ftype='fir', zero_phase=True)
            post_ds = decimate(post, DS_FACTOR, ftype='fir', zero_phase=True)
            fs_use = FS_DS
        else:
            pre_ds = pre
            post_ds = post
            fs_use = FS

        f, psd_pre = safe_welch(pre_ds, fs=fs_use, target_freq_res=0.5)
        _, psd_post = safe_welch(post_ds, fs=fs_use, target_freq_res=0.5)

        if f is None:
            changes.append(np.nan)
            continue

        bp_pre = band_power(psd_pre, f, LOW_FREQ_BAND)
        bp_post = band_power(psd_post, f, LOW_FREQ_BAND)

        if bp_pre is not None and bp_post is not None and bp_pre > 1e-20:
            changes.append(float((bp_post - bp_pre) / bp_pre * 100))
        else:
            changes.append(np.nan)
    return np.array(changes)


# == FIRING-RATE FEATURES =====================================================
def firing_rates_from_spikes(spike_data, time_range):
    rates = {}
    for layer, layer_spikes in spike_data.items():
        for mon_name, mon in layer_spikes.items():
            t = np.asarray(mon["times_ms"], dtype=float)
            idx = np.asarray(mon["spike_indices"], dtype=int)
            mask = (t >= time_range[0]) & (t < time_range[1])
            n_spikes = int(mask.sum())
            duration_s = (time_range[1] - time_range[0]) / 1000.0
            n_neurons = int(idx.max()) + 1 if len(idx) > 0 else 1
            rate = n_spikes / (n_neurons * duration_s) if n_neurons > 0 else 0.0
            pop = mon_name.replace("_spikes", "")
            rates[f"{layer}_{pop}"] = float(rate)
    return rates


def spike_count_variability(spike_data, time_range, bin_ms=50):
    fano = {}
    for layer, layer_spikes in spike_data.items():
        for mon_name, mon in layer_spikes.items():
            t = np.asarray(mon["times_ms"], dtype=float)
            mask = (t >= time_range[0]) & (t < time_range[1])
            t_win = t[mask]
            pop = mon_name.replace("_spikes", "")
            key = f"{layer}_{pop}"
            if len(t_win) < 10:
                fano[key] = np.nan
                continue
            bins = np.arange(time_range[0], time_range[1] + bin_ms, bin_ms)
            counts, _ = np.histogram(t_win, bins=bins)
            fano[key] = float(counts.var() / counts.mean()) if counts.mean() > 0 else np.nan
    return fano


def population_synchrony(spike_data, time_range, bin_ms=5):
    sync = {}
    for layer, layer_spikes in spike_data.items():
        for mon_name, mon in layer_spikes.items():
            if "E" not in mon_name:
                continue
            t = np.asarray(mon["times_ms"], dtype=float)
            mask = (t >= time_range[0]) & (t < time_range[1])
            t_win = t[mask]
            key = f"{layer}_E_sync"
            if len(t_win) < 20:
                sync[key] = np.nan
                continue
            bins = np.arange(time_range[0], time_range[1] + bin_ms, bin_ms)
            counts, _ = np.histogram(t_win, bins=bins)
            mu = counts.mean()
            sync[key] = float(counts.var() / mu) if mu > 0 else np.nan
    return sync


def isi_burst_fraction(spike_data, time_range, burst_threshold_ms=8):
    burst = {}
    for layer, layer_spikes in spike_data.items():
        for mon_name, mon in layer_spikes.items():
            t = np.asarray(mon["times_ms"], dtype=float)
            idx = np.asarray(mon["spike_indices"], dtype=int)
            mask = (t >= time_range[0]) & (t < time_range[1])
            t_win = t[mask]
            idx_win = idx[mask]
            pop = mon_name.replace("_spikes", "")
            key = f"{layer}_{pop}_burst"
            if len(t_win) < 10:
                burst[key] = np.nan
                continue
            isis = []
            for nid in np.unique(idx_win):
                st = np.sort(t_win[idx_win == nid])
                if len(st) > 1:
                    isis.extend(np.diff(st).tolist())
            if len(isis) > 0:
                isis = np.array(isis)
                burst[key] = float((isis < burst_threshold_ms).mean())
            else:
                burst[key] = np.nan
    return burst


def rate_trace_low_freq(rate_data, stim_onset_ms, period="post"):
    """Compute 1-2 Hz power in each population's rate trace."""
    powers = {}
    for layer, layer_rates in rate_data.items():
        for mon_name, mon in layer_rates.items():
            t_ms = np.asarray(mon["t_ms"], dtype=float)
            rate_hz = np.asarray(mon["rate_hz"], dtype=float)

            if period == "post":
                tmask = (t_ms >= stim_onset_ms + POST_START_MS) & (
                    t_ms < stim_onset_ms + POST_START_MS + POST_WINDOW_MS
                )
            else:
                tmask = (t_ms >= stim_onset_ms - PRE_WINDOW_MS) & (
                    t_ms < stim_onset_ms
                )

            r = rate_hz[tmask].copy()
            pop = mon_name.replace("_rate", "")
            key = f"{layer}_{pop}_rate_lowf"

            if len(r) < 200:
                powers[key] = np.nan
                continue

            r -= np.mean(r)

            # downsample rate trace if at 10kHz
            dt_ms = float(np.mean(np.diff(t_ms[tmask])))
            fs_rate = 1000.0 / dt_ms
            if fs_rate > 1500 and len(r) > DS_FACTOR * 10:
                r = decimate(r, DS_FACTOR, ftype='fir', zero_phase=True)
                fs_rate = fs_rate / DS_FACTOR

            f, psd = safe_welch(r, fs=fs_rate, target_freq_res=0.5)
            if f is None:
                powers[key] = np.nan
            else:
                powers[key] = band_power(psd, f, LOW_FREQ_BAND)
    return powers


# == MAIN PIPELINE ============================================================
def main():
    print("=" * 70)
    print("DIAGNOSTIC: What drives the 1-2 Hz power increase?")
    print("=" * 70)

    print(f"\nLoading trials from {BASE_PATH} ...")
    trials = load_trials(BASE_PATH, N_TRIALS)
    n = len(trials)
    print(f"  Loaded {n} trials")

    if n < 6:
        print("  Need at least 6 trials. Aborting.")
        return

    # debug
    t0 = trials[0]
    stim0 = t0['stim_onset_ms']
    print(f"\n  DEBUG - Trial 0:")
    print(f"    time: {t0['time'][0]:.1f} - {t0['time'][-1]:.1f} ms")
    print(f"    stim_onset: {stim0:.1f} ms")
    print(f"    pre window:  {stim0 - PRE_WINDOW_MS:.1f} - {stim0:.1f}")
    print(f"    post window: {stim0 + POST_START_MS:.1f} - {stim0 + POST_START_MS + POST_WINDOW_MS:.1f}")
    print(f"    bipolar shape: {t0['bipolar_lfp'].shape}")
    print(f"    bipolar range: {t0['bipolar_lfp'].min():.2f} to {t0['bipolar_lfp'].max():.2f}")

    # quick welch sanity check on first channel
    ch0 = t0['bipolar_lfp'][0]
    pre_mask = (t0['time'] >= stim0 - PRE_WINDOW_MS) & (t0['time'] < stim0)
    pre_seg = ch0[pre_mask].astype(np.float64).copy()
    pre_seg -= np.mean(pre_seg)
    if DS_FACTOR > 1:
        pre_seg_ds = decimate(pre_seg, DS_FACTOR, ftype='fir', zero_phase=True)
    else:
        pre_seg_ds = pre_seg
    f_test, psd_test = safe_welch(pre_seg_ds, fs=FS_DS, target_freq_res=0.5)
    if f_test is not None:
        bp_test = band_power(psd_test, f_test, LOW_FREQ_BAND)
        print(f"\n  Welch sanity check (ch0 pre, downsampled to {FS_DS} Hz):")
        print(f"    nperseg used: {int(FS_DS / 0.5)}")
        print(f"    freq bins in [{LOW_FREQ_BAND[0]}, {LOW_FREQ_BAND[1]}] Hz: "
              f"{((f_test >= LOW_FREQ_BAND[0]) & (f_test <= LOW_FREQ_BAND[1])).sum()}")
        print(f"    freq resolution: {f_test[1] - f_test[0]:.3f} Hz")
        print(f"    band power: {bp_test:.6g}")
    else:
        print("  Welch sanity check FAILED")

    # == compute per-trial low-freq metric ==
    print("\nComputing 1-2 Hz power change per trial ...")
    trial_metrics = []
    for trial in trials:
        ch_changes = compute_low_freq_change(trial, lfp_key="bipolar_lfp")
        median_change = float(np.nanmedian(ch_changes))
        max_change = float(np.nanmax(ch_changes)) if np.any(np.isfinite(ch_changes)) else np.nan
        n_bad_ch = int(np.nansum(ch_changes > 50))
        trial_metrics.append({
            "trial_id": trial["trial_id"],
            "median_change": median_change,
            "max_change": max_change,
            "n_bad_channels": n_bad_ch,
            "per_channel": ch_changes,
        })

    median_changes = np.array([m["median_change"] for m in trial_metrics])

    n_valid = np.sum(np.isfinite(median_changes))
    print(f"\n  Valid median_changes: {n_valid}/{n}")
    if n_valid < 6:
        print("  Not enough valid trials. Something is still wrong.")
        print(f"  median_changes = {median_changes}")
        return

    print(f"\n  1-2 Hz power change distribution across {n_valid} valid trials:")
    print(f"    min    = {np.nanmin(median_changes):+.1f}%")
    print(f"    25th   = {np.nanpercentile(median_changes, 25):+.1f}%")
    print(f"    median = {np.nanmedian(median_changes):+.1f}%")
    print(f"    75th   = {np.nanpercentile(median_changes, 75):+.1f}%")
    print(f"    max    = {np.nanmax(median_changes):+.1f}%")

    # == split into good vs bad ==
    thresh_good = float(np.nanpercentile(median_changes, GOOD_PERCENTILE))
    thresh_bad = float(np.nanpercentile(median_changes, BAD_PERCENTILE))

    good_idx = [i for i in range(n) if np.isfinite(median_changes[i]) and median_changes[i] <= thresh_good]
    bad_idx = [i for i in range(n) if np.isfinite(median_changes[i]) and median_changes[i] >= thresh_bad]

    print(f"\n  GOOD trials (<={thresh_good:+.1f}%): n={len(good_idx)}, "
          f"ids={[trials[i]['trial_id'] for i in good_idx]}")
    print(f"  BAD  trials (>={thresh_bad:+.1f}%): n={len(bad_idx)}, "
          f"ids={[trials[i]['trial_id'] for i in bad_idx]}")

    if len(good_idx) < 2 or len(bad_idx) < 2:
        print("  Not enough trials in good/bad groups. Adjusting percentiles...")
        thresh_good = float(np.nanpercentile(median_changes, 50))
        thresh_bad = float(np.nanpercentile(median_changes, 50))
        good_idx = [i for i in range(n) if np.isfinite(median_changes[i]) and median_changes[i] <= thresh_good]
        bad_idx = [i for i in range(n) if np.isfinite(median_changes[i]) and median_changes[i] > thresh_good]
        print(f"  Median split: GOOD n={len(good_idx)}, BAD n={len(bad_idx)}")

    # == extract features ==
    print("\nExtracting features from all trials ...")
    all_features = []

    for trial_i, trial in enumerate(trials):
        stim = trial["stim_onset_ms"]
        pre_range = (stim - PRE_WINDOW_MS, stim)
        post_range = (stim + POST_START_MS, stim + POST_START_MS + POST_WINDOW_MS)

        feat = {}
        feat["trial_id"] = trial["trial_id"]
        feat["median_1_2Hz_change"] = float(median_changes[trial_i])

        # -- firing rates --
        rates_pre = firing_rates_from_spikes(trial["spike_data"], pre_range)
        rates_post = firing_rates_from_spikes(trial["spike_data"], post_range)
        all_rate_keys = sorted(set(list(rates_pre.keys()) + list(rates_post.keys())))

        if trial_i == 0:
            print(f"\n  DEBUG - rate keys: {all_rate_keys}")
            print(f"  DEBUG - rates_pre: { {k: round(v,2) for k, v in list(rates_pre.items())[:4]} }")
            print(f"  DEBUG - rates_post: { {k: round(v,2) for k, v in list(rates_post.items())[:4]} }")

        for key in all_rate_keys:
            pre_r = rates_pre.get(key, 0.0)
            post_r = rates_post.get(key, 0.0)
            feat[f"rate_pre_{key}"] = pre_r
            feat[f"rate_post_{key}"] = post_r
            feat[f"rate_change_{key}"] = post_r - pre_r
            feat[f"rate_ratio_{key}"] = post_r / pre_r if pre_r > 0.5 else np.nan

        # -- E/I ratios --
        for layer in trial["spike_data"].keys():
            e_post = rates_post.get(f"{layer}_E", 0.0)
            pv_post = rates_post.get(f"{layer}_PV", 0.0)
            som_post = rates_post.get(f"{layer}_SOM", 0.0)
            i_total = pv_post + som_post

            e_pre = rates_pre.get(f"{layer}_E", 0.0)
            pv_pre = rates_pre.get(f"{layer}_PV", 0.0)
            som_pre = rates_pre.get(f"{layer}_SOM", 0.0)
            i_total_pre = pv_pre + som_pre

            feat[f"EI_ratio_post_{layer}"] = e_post / i_total if i_total > 0.5 else np.nan
            feat[f"EI_ratio_pre_{layer}"] = e_pre / i_total_pre if i_total_pre > 0.5 else np.nan
            if i_total_pre > 0.5 and i_total > 0.5:
                feat[f"EI_ratio_change_{layer}"] = (e_post / i_total) - (e_pre / i_total_pre)
            else:
                feat[f"EI_ratio_change_{layer}"] = np.nan

            feat[f"PV_E_ratio_post_{layer}"] = pv_post / e_post if e_post > 0.5 else np.nan
            feat[f"PV_E_ratio_pre_{layer}"] = pv_pre / e_pre if e_pre > 0.5 else np.nan
            feat[f"SOM_E_ratio_post_{layer}"] = som_post / e_post if e_post > 0.5 else np.nan
            feat[f"SOM_E_ratio_pre_{layer}"] = som_pre / e_pre if e_pre > 0.5 else np.nan

        # -- Fano factor --
        fano_post = spike_count_variability(trial["spike_data"], post_range, bin_ms=50)
        fano_pre = spike_count_variability(trial["spike_data"], pre_range, bin_ms=50)
        for key, val in fano_post.items():
            feat[f"fano_post_{key}"] = val
        for key, val in fano_pre.items():
            feat[f"fano_pre_{key}"] = val
            post_val = fano_post.get(key, np.nan)
            if np.isfinite(val) and np.isfinite(post_val):
                feat[f"fano_change_{key}"] = post_val - val
            else:
                feat[f"fano_change_{key}"] = np.nan

        # -- synchrony --
        sync_post = population_synchrony(trial["spike_data"], post_range, bin_ms=5)
        sync_pre = population_synchrony(trial["spike_data"], pre_range, bin_ms=5)
        for key, val in sync_post.items():
            feat[f"sync_post_{key}"] = val
        for key, val in sync_pre.items():
            feat[f"sync_pre_{key}"] = val
            post_val = sync_post.get(key, np.nan)
            if np.isfinite(val) and np.isfinite(post_val):
                feat[f"sync_change_{key}"] = post_val - val

        # -- burst fraction --
        burst_post = isi_burst_fraction(trial["spike_data"], post_range)
        burst_pre = isi_burst_fraction(trial["spike_data"], pre_range)
        for key, val in burst_post.items():
            feat[f"burst_post_{key}"] = val
        for key, val in burst_pre.items():
            feat[f"burst_pre_{key}"] = val

        # -- 1-2 Hz in rate traces --
        if trial["rate_data"]:
            rlf_post = rate_trace_low_freq(trial["rate_data"], stim, period="post")
            rlf_pre = rate_trace_low_freq(trial["rate_data"], stim, period="pre")
            for key, val in rlf_post.items():
                feat[f"post_{key}"] = val
            for key, val in rlf_pre.items():
                feat[f"pre_{key}"] = val
                post_val = rlf_post.get(key, np.nan)
                if np.isfinite(val) and np.isfinite(post_val) and val > 1e-20:
                    feat[f"change_{key}"] = (post_val - val) / val * 100
                else:
                    feat[f"change_{key}"] = np.nan

        all_features.append(feat)

        if trial_i == 0:
            print(f"  DEBUG - total features: {len(feat)}")

    # == compare good vs bad ==
    print("\n" + "=" * 70)
    print("RESULTS: Features that distinguish GOOD vs BAD trials")
    print("=" * 70)

    all_keys = set()
    for feat in all_features:
        all_keys.update(feat.keys())
    all_keys -= {"trial_id", "median_1_2Hz_change"}

    results = []
    for key in sorted(all_keys):
        good_vals = []
        for i in good_idx:
            v = all_features[i].get(key, np.nan)
            try:
                v = float(v)
                if np.isfinite(v):
                    good_vals.append(v)
            except (TypeError, ValueError):
                pass

        bad_vals = []
        for i in bad_idx:
            v = all_features[i].get(key, np.nan)
            try:
                v = float(v)
                if np.isfinite(v):
                    bad_vals.append(v)
            except (TypeError, ValueError):
                pass

        if len(good_vals) < 2 or len(bad_vals) < 2:
            continue

        good_vals = np.array(good_vals)
        bad_vals = np.array(bad_vals)
        mean_good = float(np.mean(good_vals))
        mean_bad = float(np.mean(bad_vals))

        n_g, n_b = len(good_vals), len(bad_vals)
        var_g = float(np.var(good_vals, ddof=1)) if n_g > 1 else 0.0
        var_b = float(np.var(bad_vals, ddof=1)) if n_b > 1 else 0.0
        pooled_std = np.sqrt(
            (var_g * (n_g - 1) + var_b * (n_b - 1)) / max(n_g + n_b - 2, 1)
        )
        cohens_d = float((mean_bad - mean_good) / pooled_std) if pooled_std > 1e-12 else 0.0

        # Pearson r
        all_mc, all_fv = [], []
        for i in range(n):
            v = all_features[i].get(key, np.nan)
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v) and np.isfinite(median_changes[i]):
                all_mc.append(float(median_changes[i]))
                all_fv.append(v)
        r = float(np.corrcoef(all_mc, all_fv)[0, 1]) if len(all_mc) > 4 else np.nan

        results.append({
            "feature": key,
            "mean_good": mean_good,
            "mean_bad": mean_bad,
            "cohens_d": cohens_d,
            "correlation": r,
            "abs_d": abs(cohens_d),
        })

    results.sort(key=lambda x: x["abs_d"], reverse=True)

    print(f"\n  Total features compared: {len(results)}")

    print(f"\n{'Rank':<5} {'Feature':<45} {'Good':>10} {'Bad':>10} "
          f"{'Cohen d':>9} {'Corr r':>8}")
    print("-" * 92)
    for rank, r in enumerate(results[:40], 1):
        print(f"{rank:<5} {r['feature']:<45} {r['mean_good']:>10.3f} "
              f"{r['mean_bad']:>10.3f} {r['cohens_d']:>+9.2f} "
              f"{r['correlation']:>+8.3f}")

    # == per-layer summary ==
    print("\n" + "=" * 70)
    print("PER-LAYER SUMMARY: Which layers drive the problem?")
    print("=" * 70)
    for layer in ("L23", "L4AB", "L4C", "L5", "L6"):
        layer_results = [r for r in results if layer in r["feature"]]
        if not layer_results:
            continue
        print(f"\n  {layer}:")
        for r in layer_results[:5]:
            print(f"    {r['feature']:<42} d={r['cohens_d']:+.2f}  "
                  f"r={r['correlation']:+.3f}  "
                  f"good={r['mean_good']:.3f}  bad={r['mean_bad']:.3f}")

    # == per-channel breakdown ==
    print("\n" + "=" * 70)
    print("PER-CHANNEL BREAKDOWN: worst bad trials")
    print("=" * 70)
    for i in bad_idx[:3]:
        tid = trials[i]["trial_id"]
        ch_changes = trial_metrics[i]["per_channel"]
        print(f"\n  Trial {tid} (median: {trial_metrics[i]['median_change']:+.1f}%):")
        for ch, val in enumerate(ch_changes):
            if np.isfinite(val):
                bar = "X" * max(0, min(40, int(val / 20)))
            else:
                bar = "?"
            print(f"    ch {ch:2d}: {val:+8.1f}%  {bar}")

    # == correlation ranking ==
    print("\n" + "=" * 70)
    print("TOP CORRELATED FEATURES (Pearson r with 1-2 Hz change):")
    print("=" * 70)
    corr_sorted = sorted(
        [r for r in results if np.isfinite(r.get("correlation", np.nan))],
        key=lambda x: abs(x["correlation"]),
        reverse=True,
    )
    for r in corr_sorted[:20]:
        direction = "-> more low-freq" if r["correlation"] > 0 else "-> less low-freq"
        print(f"  r={r['correlation']:+.3f}  {r['feature']:<45} ({direction})")

    # == save ==
    out_path = Path(BASE_PATH) / "diagnostic_results.npz"
    np.savez_compressed(
        out_path,
        trial_ids=np.array([t["trial_id"] for t in trials]),
        median_changes=median_changes,
        good_idx=np.array(good_idx),
        bad_idx=np.array(bad_idx),
        feature_names=np.array([r["feature"] for r in results], dtype=object),
        cohens_d=np.array([r["cohens_d"] for r in results]),
        correlations=np.array([r["correlation"] for r in results]),
        mean_good=np.array([r["mean_good"] for r in results]),
        mean_bad=np.array([r["mean_bad"] for r in results]),
    )
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("HOW TO INTERPRET:")
    print("=" * 70)
    print("""
  Cohen's d > 0 = feature HIGHER in bad trials
  Cohen's d < 0 = feature LOWER in bad trials
  |d| > 0.8 = large effect, |d| > 1.2 = very large

  r > 0 = feature increases WITH the 1-2 Hz problem

  KEY PATTERNS:
  - rate_change_*_E high      -> E cells over-excited by stimulus
  - EI_ratio_change high      -> inhibition not tracking excitation
  - fano high                 -> bursty firing (adaptation cycles)
  - sync high                 -> population synchrony at slow timescales
  - burst_post high           -> spike bursting
  - *_rate_lowf change high   -> rate trace oscillates at 1-2 Hz
  - PV_E_ratio low            -> PV not keeping up with E
""")


if __name__ == "__main__":
    main()