"""View one saved trial: per-layer raster, firing rates and bipolar power spectrum.

Run:
    path/to/your/venv/bin/python analysis/visualisations/view_trial.py
"""
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

# parameters
TRIALS_DIR = "results/trials_06_05-fb"
TRIAL_INDEX = 0
OUT_DIR = "figures/view_trial"
LAYERS = ["L23", "L4AB", "L4C", "L5", "L6"]
PRE_WINDOW_MS = 1000.0    # baseline window ending at stim onset
POST_WINDOW_MS = 1000.0   # stimulus window starting at stim onset
FMAX = 100.0

POP_COLORS = {"E": "#2E8B57", "PV": "#C0392B", "SOM": "#1F4E96", "VIP": "#D4A017"}
POP_ORDER = ["E", "PV", "SOM", "VIP"]
RNG = np.random.default_rng(0)


def smooth_flat(r, t_ms, width_ms=15.0):
    dt = np.median(np.diff(t_ms))
    w = max(1, int(round(width_ms / dt)))
    return np.convolve(r, np.ones(w) / w, mode="same")


def plot_raster(layer, spike_data, t_lo_s, t_hi_s, onset_s, out_dir):
    pop_unique = {}
    for pop in POP_ORDER:
        key = f"{pop}_spikes"
        if key in spike_data:
            ids = np.unique(spike_data[key]["spike_indices"])
            if len(ids):
                pop_unique[pop] = ids
    if not pop_unique:
        print(f"{layer}: no spikes, skipping raster")
        return
    # cap E to the largest interneuron count for a readable raster
    non_e = [len(v) for p, v in pop_unique.items() if p != "E"]
    n_e_cap = max(non_e) if non_e else max(len(v) for v in pop_unique.values())

    fig, ax = plt.subplots(figsize=(11, 9))
    y_offset, yticks, ylabels = 0, [], []
    for pop in POP_ORDER:
        if pop not in pop_unique:
            continue
        times = spike_data[f"{pop}_spikes"]["times_ms"] / 1000.0
        idx = spike_data[f"{pop}_spikes"]["spike_indices"]
        ids = pop_unique[pop]
        chosen = (RNG.choice(ids, size=min(n_e_cap, len(ids)), replace=False)
                  if pop == "E" else ids)
        remap = {nid: y_offset + r for r, nid in enumerate(np.sort(chosen))}
        mask = np.isin(idx, chosen) & (times >= t_lo_s) & (times <= t_hi_s)
        y_sel = np.array([remap[i] for i in idx[mask]])
        ax.scatter(times[mask], y_sel, s=11.0, color=POP_COLORS[pop], alpha=0.9,
                   label=pop, edgecolors="none")
        yticks.append(y_offset + len(chosen) / 2)
        ylabels.append(pop)
        y_offset += len(chosen) + 8

    ax.axvline(onset_s, color="0.3", linestyle="--", lw=1.2, label="stim onset")
    ax.set_xlim(t_lo_s, t_hi_s)
    ax.set_ylim(-5, y_offset)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=12)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_title(layer, fontsize=13)
    ax.legend(fontsize=10, frameon=False, ncol=5, loc="upper right", markerscale=3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"raster_{layer}.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def plot_rates(layer, rate_data, stim_onset_ms, t_hi_ms, out_dir):
    pre_means, post_means, pops = [], [], []
    for pop in POP_ORDER:
        key = f"{pop}_rate"
        if key not in rate_data:
            continue
        t_ms = np.asarray(rate_data[key]["t_ms"])
        r_hz = smooth_flat(np.asarray(rate_data[key]["rate_hz"]), t_ms)
        pre = (t_ms >= stim_onset_ms - PRE_WINDOW_MS) & (t_ms < stim_onset_ms)
        post = (t_ms >= stim_onset_ms) & (t_ms <= t_hi_ms)
        pre_means.append(float(np.mean(r_hz[pre])) if pre.any() else 0.0)
        post_means.append(float(np.mean(r_hz[post])) if post.any() else 0.0)
        pops.append(pop)
    if not pops:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    x, w = np.arange(len(pops)), 0.38
    colors = [POP_COLORS[p] for p in pops]
    ax.bar(x - w / 2, pre_means, width=w, label="pre", color=colors,
           edgecolor="black", linewidth=0.6, alpha=0.55)
    ax.bar(x + w / 2, post_means, width=w, label="post (stim)", color=colors,
           edgecolor="black", linewidth=0.6, alpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels(pops, fontsize=12)
    ax.set_ylabel("Mean firing rate (Hz)", fontsize=13)
    ax.set_title(layer, fontsize=13)
    ax.legend(fontsize=10, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"rates_{layer}.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"{layer}: " + ", ".join(
        f"{p} {pre:.1f}->{post:.1f}Hz"
        for p, pre, post in zip(pops, pre_means, post_means)))


def plot_bipolar_power(data, out_dir):
    """Bipolar LFP power spectrum per channel: pre vs post stimulus."""
    bip = data["bipolar_matrix"]               # (n_ch, n_samples)
    t = np.asarray(data["time_array_ms"])
    labels = data["channel_labels"]
    depths = data["channel_depths"]
    stim_onset_ms = float(data["stim_onset_ms"])
    fs = 1000.0 / float(np.mean(np.diff(t)))

    if len(depths) > bip.shape[0]:
        depths = (depths[:-1] + depths[1:]) / 2

    pre = (t >= stim_onset_ms - PRE_WINDOW_MS) & (t < stim_onset_ms)
    post = (t >= stim_onset_ms) & (t <= stim_onset_ms + POST_WINDOW_MS)
    n_ch = bip.shape[0]
    nps = min(int(min(pre.sum(), post.sum())), 4096)

    def _psd(seg):
        # manual detrend (robust to flat segments), then Welch with detrend off
        seg = seg.astype(np.float64)
        x = np.arange(len(seg))
        xx = x - x.mean()
        denom = (xx * xx).sum()
        slope = (xx * (seg - seg.mean())).sum() / denom if denom > 0 else 0.0
        seg = seg - (slope * x + (seg.mean() - slope * x.mean()))
        return welch(seg, fs=fs, nperseg=nps, detrend=False)

    fig, axes = plt.subplots(n_ch, 1, figsize=(7, 1.8 * n_ch), sharex=True)
    axes = np.atleast_1d(axes)
    for ch in range(n_ch):
        f_pre, p_pre = _psd(bip[ch, pre])
        f_post, p_post = _psd(bip[ch, post])
        m = f_pre <= FMAX
        ax = axes[ch]
        ax.semilogy(f_pre[m], p_pre[m], color="0.4", lw=1.3, label="pre")
        ax.semilogy(f_post[m], p_post[m], color="#e8590c", lw=1.3, label="post")
        ax.set_ylabel(f"{labels[ch]}\nz={depths[ch]:+.2f}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, which="both", alpha=0.25)
        if ch == 0:
            ax.legend(fontsize=8, frameon=False)
    axes[-1].set_xlabel("Frequency (Hz)", fontsize=11)
    fig.suptitle("Bipolar LFP power: pre vs post stimulus", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(os.path.join(out_dir, "bipolar_power.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved bipolar_power.png ({n_ch} channels)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(TRIALS_DIR, "trial_*.npz")))
    if not files:
        raise FileNotFoundError(f"no trial_*.npz in {TRIALS_DIR}")
    trial_file = files[TRIAL_INDEX]
    print(f"Viewing {trial_file}")
    data = np.load(trial_file, allow_pickle=True)

    all_spikes = data["spike_data"].item()
    all_rates = data["rate_data"].item()
    stim_onset_ms = float(data["stim_onset_ms"])
    t_hi_ms = float(data["baseline_ms"]) + float(data["post_ms"])
    t_lo_s = (stim_onset_ms - PRE_WINDOW_MS) / 1000.0
    t_hi_s = t_hi_ms / 1000.0
    onset_s = stim_onset_ms / 1000.0

    for layer in LAYERS:
        if layer in all_spikes:
            plot_raster(layer, all_spikes[layer], t_lo_s, t_hi_s, onset_s, OUT_DIR)
        if layer in all_rates:
            plot_rates(layer, all_rates[layer], stim_onset_ms, t_hi_ms, OUT_DIR)
    plot_bipolar_power(data, OUT_DIR)
    print(f"figures written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
