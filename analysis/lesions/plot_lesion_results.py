"""Lesion-vs-control laminar power change with paired cluster-permutation stats.
Per-lesion cluster heatmaps plus three grouped slide figures.

Run:
    path/to/your/venv/bin/python analysis/lesions/plot_lesion_results.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.transforms import blended_transform_factory
from scipy import stats as sp_stats
from scipy.ndimage import label as nd_label
from scipy.signal import welch

from analysis.spectral.laminar_power_change import (
    load_trials, _time_vector_for_key)

# parameters
LESION_ROOT = "saved_trials/lesions_2026-06-03"
FIG_ROOT = "figures/lesions_2026-06-03/_significance_cluster"
N_PERM = 1000
WINDOW_MS = 3000           # PSD window at end of trial
FREQ_RANGE = (1, 100)
LFP_KEY = "bipolar_lfp"
NPERSEG = 16384
ALPHA = 0.05               # cluster significance
CLUSTER_P_FORMING = 0.01   # cluster-forming threshold
RNG_SEED = 0

BANDS = {
    "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
    "beta": (13, 30), "low_gamma": (30, 50), "high_gamma": (50, 100),
}
BAND_COLORS = {
    "delta": "#f2e6f7", "theta": "#e6f0fa", "alpha": "#e8f5e6",
    "beta": "#fff4d6", "low_gamma": "#fde0d2", "high_gamma": "#f9d2d2",
}
LAYER_CENTRES = {"L2/3": 0.775, "L4AB": 0.295, "L4C": 0.0,  # mm, larger z = superficial
                 "L5": -0.24, "L6": -0.48}
SLIDE_GROUPS = {
    "fig_gain.png": ["L6_L4C"],
    "fig_feedforward.png": ["L4C_L4AB", "L4AB_L23", "L4C_L23"],
    "fig_alpha.png": ["L5_L6", "L4C_L6", "L23_L5"],
}


def count_trials(folder):
    return sum(1 for f in os.listdir(folder)
               if f.startswith("trial_") and f.endswith(".npz"))


def _detrend_linear(x):
    """Remove a least-squares line, robust to flat segments."""
    n = len(x)
    t = np.arange(n, dtype=np.float64)
    tt = t - t.mean()
    denom = (tt * tt).sum()
    slope = (tt * (x - x.mean())).sum() / denom if denom > 0 else 0.0
    return x - (slope * t + (x.mean() - slope * t.mean()))


def per_trial_psd(trials, lfp_key=LFP_KEY, window_ms=WINDOW_MS):
    """psd[n_trial, n_ch, n_freq], freqs, depths, labels."""
    t0 = _time_vector_for_key(trials[0], lfp_key)
    fs = 1000.0 / float(np.mean(np.diff(t0)))
    n_channels = trials[0][lfp_key].shape[0]
    depths = trials[0]["channel_depths"]
    if len(depths) > n_channels:
        depths = (depths[:-1] + depths[1:]) / 2
    else:
        depths = depths[:n_channels]
    labels = trials[0]["channel_labels"][:n_channels]

    psd_list = []
    freqs_ref = None
    for trial in trials:
        time = _time_vector_for_key(trial, lfp_key)
        t_end = float(time[-1])
        mask = (time >= t_end - window_ms) & (time <= t_end)
        trial_psd, valid = [], True
        for ch in range(n_channels):
            seg = trial[lfp_key][ch][mask].copy()
            if len(seg) == 0 or np.any(np.isnan(seg)):
                valid = False
                break
            nps = min(NPERSEG, len(seg))
            f, psd = welch(_detrend_linear(seg), fs=fs, window="hann",
                           nperseg=nps, noverlap=nps // 2, detrend=False)
            if freqs_ref is None:
                freqs_ref = f
            trial_psd.append(psd)
        if valid:
            psd_list.append(np.stack(trial_psd, axis=0))

    if not psd_list:
        raise RuntimeError(f"no valid trials for {lfp_key}")
    psd_all = np.stack(psd_list, axis=0)
    mask_f = (freqs_ref >= FREQ_RANGE[0]) & (freqs_ref <= FREQ_RANGE[1])
    return (psd_all[:, :, mask_f], freqs_ref[mask_f],
            np.asarray(depths), np.asarray(labels))


def _paired_t_map(d):
    """Per-bin paired t-statistic across trials."""
    n = d.shape[0]
    sem = d.std(axis=0, ddof=1) / np.sqrt(n)
    sem = np.where(sem > 0, sem, np.inf)
    return d.mean(axis=0) / sem


_STRUCT = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=int)  # cluster along freq only


def _find_clusters(t_map, t_thresh):
    clusters = []
    for sign, sup in ((+1, t_map > +t_thresh), (-1, t_map < -t_thresh)):
        if not sup.any():
            continue
        lbl, n = nd_label(sup, structure=_STRUCT)
        for k in range(1, n + 1):
            m = lbl == k
            clusters.append({"mask": m, "sign": sign,
                             "mass": float(t_map[m].sum())})
    return clusters


def _max_abs_mass(t_map, t_thresh):
    best = 0.0
    for sup in (t_map > +t_thresh, t_map < -t_thresh):
        if not sup.any():
            continue
        lbl, n = nd_label(sup, structure=_STRUCT)
        if n == 0:
            continue
        sums = np.bincount(lbl.ravel(), weights=t_map.ravel(),
                           minlength=n + 1)[1:]
        best = max(best, float(np.max(np.abs(sums))))
    return best


def cluster_permutation_test(d, n_perm=N_PERM, rng=None):
    """Paired sign-flip cluster test on log2 fold-changes d. Returns t_obs, sig_mask."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    n_trial = d.shape[0]
    t_thresh = float(sp_stats.t.ppf(1 - CLUSTER_P_FORMING / 2, n_trial - 1))

    t_obs = _paired_t_map(d)
    obs_clusters = _find_clusters(t_obs, t_thresh)

    null_max = np.empty(n_perm)
    for p in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=(n_trial, 1, 1))
        null_max[p] = _max_abs_mass(_paired_t_map(d * signs), t_thresh)

    sig_mask = np.zeros_like(t_obs, dtype=bool)
    for c in obs_clusters:
        p_val = (1.0 + float((null_max >= abs(c["mass"])).sum())) / (n_perm + 1)
        if p_val < ALPHA:
            sig_mask |= c["mask"]
    return t_obs, sig_mask


def compute_all_masks(lesion_root, n_perm=N_PERM):
    """Per-lesion log2FC and significance mask vs control, shared rng for reproducibility."""
    ctrl_dir = os.path.join(lesion_root, "control")
    ctrl_trials = load_trials(ctrl_dir, count_trials(ctrl_dir))
    psd_ctrl, freqs, depths, labels = per_trial_psd(ctrl_trials)

    lesion_names = sorted(
        d for d in os.listdir(lesion_root)
        if os.path.isdir(os.path.join(lesion_root, d)) and d != "control")

    rng = np.random.default_rng(RNG_SEED)
    results = {}
    for lname in lesion_names:
        ldir = os.path.join(lesion_root, lname)
        n = count_trials(ldir)
        if n == 0:
            continue
        try:
            psd_les, freqs_l, _, _ = per_trial_psd(load_trials(ldir, n))
        except Exception as exc:
            print(f"[{lname}] failed: {exc}")
            continue
        if not np.allclose(freqs_l, freqs):
            print(f"[{lname}] freq grid mismatch -- skip")
            continue
        n_pair = min(psd_ctrl.shape[0], psd_les.shape[0])
        d = np.log2((psd_les[:n_pair] + 1e-20) / (psd_ctrl[:n_pair] + 1e-20))
        _, sig_mask = cluster_permutation_test(d, n_perm=n_perm, rng=rng)
        results[lname] = {"log2fc": d.mean(axis=0), "sig_mask": sig_mask}
        print(f"[{lname}] {int(sig_mask.sum())}/{sig_mask.size} sig bins")
    return freqs, depths, labels, results


def _band_overlay(ax, x_lo, x_hi):
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for bname, (b_lo, b_hi) in BANDS.items():
        a, b = max(b_lo, x_lo), min(b_hi, x_hi)
        if b <= a:
            continue
        ax.add_patch(plt.Rectangle((a, 1.0), b - a, 0.04, transform=trans,
                                   clip_on=False, facecolor=BAND_COLORS[bname],
                                   edgecolor="k", linewidth=0.3, zorder=10))
        ax.text((a + b) / 2, 1.02, bname, transform=trans, ha="center",
                va="center", fontsize=7, color="#333", zorder=11, clip_on=False)


def _pretty(les):
    src, tgt = les.split("_", 1)
    return f"{src} → {tgt}"


def plot_cluster_heatmap(freqs, depths, labels, log2fc, sig_mask, lesion, out):
    """Two-panel heatmap: all log2FC + significant clusters only."""
    masked = np.where(sig_mask, log2fc, np.nan)
    n_ch = log2fc.shape[0]
    finite = log2fc[np.isfinite(log2fc)]
    vlim = max(float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0, 0.1)
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    extent = [freqs[0], freqs[-1], n_ch - 0.5, -0.5]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    yticklabels = [f"{lab}  z={z:+.2f}" for lab, z in zip(labels, depths)]
    for ax, data, title in (
            (axes[0], log2fc, "log2 FC (lesion / control) -- all bins"),
            (axes[1], masked, f"log2 FC, cluster p<{ALPHA} only")):
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", extent=extent,
                       norm=norm, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_yticks(range(n_ch))
        ax.set_yticklabels(yticklabels, fontsize=7)
        _band_overlay(ax, freqs[0], freqs[-1])
        plt.colorbar(im, ax=ax, label="log2 FC")
    axes[0].set_ylabel("bipolar channel")
    axes[1].set_facecolor("#dddddd")
    fig.suptitle(f"Lesion {lesion}  -- paired cluster permutation "
                 f"(n_perm={N_PERM}, forming p<{CLUSTER_P_FORMING})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _slide_panel(ax, data, freqs, depths, norm, masked=False, show_xlabel=True):
    order = np.argsort(depths)[::-1]   # superficial on top
    data, d_sorted = data[order], depths[order]
    extent = [freqs[0], freqs[-1], d_sorted[-1], d_sorted[0]]
    if masked:
        ax.set_facecolor("#e8e8e8")
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", norm=norm, extent=extent,
                   interpolation="nearest" if masked else "bilinear",
                   origin="upper")
    ax.set_xticks(np.arange(0, freqs[-1] + 1, 10))
    if show_xlabel:
        ax.set_xlabel("Frequency (Hz)", fontsize=10)
    else:
        ax.set_xticklabels([])
    ax.set_yticks(list(LAYER_CENTRES.values()))
    ax.set_yticklabels(list(LAYER_CENTRES.keys()), fontsize=9)
    return im


def plot_slide_group(lesions, freqs, depths, results, out):
    """Stacked log2FC + significant-cluster panels for a lesion group."""
    results = {l: results[l] for l in lesions if l in results}
    if not results:
        print(f"skip {os.path.basename(out)} -- no lesions present")
        return
    lesions = [l for l in lesions if l in results]
    allfc = np.concatenate([np.abs(r["log2fc"]).ravel() for r in results.values()])
    vlim = max(float(np.nanpercentile(allfc, 98)), 0.1)
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)

    n = len(lesions)
    fig, axes = plt.subplots(n, 2, figsize=(9.6, 3.2 * n), squeeze=False)
    for i, les in enumerate(lesions):
        last = i == n - 1
        log2fc, sig = results[les]["log2fc"], results[les]["sig_mask"]
        im = _slide_panel(axes[i, 0], log2fc, freqs, depths, norm, show_xlabel=last)
        _slide_panel(axes[i, 1], np.where(sig, log2fc, np.nan), freqs, depths,
                     norm, masked=True, show_xlabel=last)
        axes[i, 0].set_title(_pretty(les), fontsize=13, fontweight="bold", loc="left")
        axes[i, 1].set_title("significant clusters", fontsize=10, color="0.4", loc="left")
        axes[i, 1].set_yticklabels([])
        axes[i, 0].set_ylabel("cortical layer", fontsize=9)
    fig.subplots_adjust(hspace=0.35, right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cax).set_label("log2 power change", fontsize=9)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def main():
    os.makedirs(FIG_ROOT, exist_ok=True)
    slide_dir = os.path.join(FIG_ROOT, "_slides")
    os.makedirs(slide_dir, exist_ok=True)

    freqs, depths, labels, results = compute_all_masks(LESION_ROOT)
    depths = np.asarray(depths)

    for lname, r in results.items():
        plot_cluster_heatmap(
            freqs, depths, labels, r["log2fc"], r["sig_mask"], lname,
            os.path.join(FIG_ROOT, f"{lname}_cluster_heatmap.png"))

    for out_name, lesions in SLIDE_GROUPS.items():
        plot_slide_group(lesions, freqs, depths, results,
                         os.path.join(slide_dir, out_name))


if __name__ == "__main__":
    main()
