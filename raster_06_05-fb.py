"""
to do : remove args and comments, simplifiy, and make this a general script for plotting rasters and etc of saved trials
"""
import glob
import numpy as np
import matplotlib.pyplot as plt

# ---------- settings ----------
TRIALS_DIR = "results/trials_06_05-fb"
TRIAL_FILES = sorted(glob.glob(f"{TRIALS_DIR}/trial_*.npz"))
TRIAL_FILE = TRIAL_FILES[0]          # raster shown for a single example trial
LAYERS = ["L23", "L4AB", "L4C", "L5", "L6"]
PRE_WINDOW_MS = 500.0                # how much pre-stimulus baseline to show
RNG = np.random.default_rng(0)

# same colour scheme as the original raster script
POP_COLORS = {
    'E':   '#2E8B57',
    'PV':  '#C0392B',
    'SOM': '#1F4E96',
    'VIP': '#D4A017',
}
POP_ORDER = ['E', 'PV', 'SOM', 'VIP']

# ---------- load ----------
data = np.load(TRIAL_FILE, allow_pickle=True)
all_spikes = data["spike_data"].item()
all_rates = data["rate_data"].item()
baseline_ms = float(data["baseline_ms"])
stim_onset_ms = float(data["stim_onset_ms"])
# show from PRE_WINDOW_MS before onset to the end of recording
t_lo = stim_onset_ms - PRE_WINDOW_MS
t_hi = baseline_ms + float(data["post_ms"])
TWIN = (t_lo / 1000.0, t_hi / 1000.0)   # seconds
ONSET_S = stim_onset_ms / 1000.0


def smooth_flat(r, t_ms, width_ms=15.0):
    dt = np.median(np.diff(t_ms))
    w = max(1, int(round(width_ms / dt)))
    return np.convolve(r, np.ones(w) / w, mode='same')


def make_figures(layer):
    spike_data = all_spikes[layer]
    rate_data = all_rates[layer]

    # distinct neurons per population
    pop_unique = {}
    for pop in POP_ORDER:
        key = f"{pop}_spikes"
        if key in spike_data:
            ids = np.unique(spike_data[key]["spike_indices"])
            if len(ids):
                pop_unique[pop] = ids
    if not pop_unique:
        print(f"{layer}: no spikes, skipping")
        return
    # E is capped to a count similar to the other populations (the largest of
    # the non-E populations); PV/SOM/VIP keep all their neurons.
    non_e = [len(v) for p, v in pop_unique.items() if p != 'E']
    n_e_cap = max(non_e) if non_e else max(len(v) for v in pop_unique.values())

    # ---- 1. raster ----
    fig_r, ax = plt.subplots(figsize=(11, 9))
    y_offset = 0
    yticks, ylabels = [], []
    for pop in POP_ORDER:
        if pop not in pop_unique:
            continue
        times = spike_data[f"{pop}_spikes"]["times_ms"] / 1000.0
        idx = spike_data[f"{pop}_spikes"]["spike_indices"]

        unique_ids = pop_unique[pop]
        if pop == 'E':
            chosen = RNG.choice(unique_ids, size=min(n_e_cap, len(unique_ids)),
                                replace=False)
        else:
            chosen = unique_ids
        n_show = len(chosen)
        remap = {nid: y_offset + r for r, nid in enumerate(np.sort(chosen))}

        mask = np.isin(idx, chosen) & (times >= TWIN[0]) & (times <= TWIN[1])
        y_sel = np.array([remap[i] for i in idx[mask]])
        ax.scatter(times[mask], y_sel, s=11.0, color=POP_COLORS[pop], alpha=0.9,
                   label=pop, edgecolors='none')

        yticks.append(y_offset + n_show / 2)
        ylabels.append(pop)
        y_offset += n_show + 8

    ax.axvline(ONSET_S, color='0.3', linestyle='--', linewidth=1.2,
               label='stim onset')
    ax.set_xlim(TWIN)
    ax.set_ylim(-5, y_offset)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=12)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_title(f'{layer}', fontsize=13)
    ax.legend(fontsize=10, frameon=False, ncol=5, loc='upper right',
              markerscale=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'raster_{layer}_06_05-fb.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'raster_{layer}_06_05-fb.pdf', bbox_inches='tight')
    plt.close(fig_r)

    # ---- 2. mean rate per cell type (pre vs post stimulus, bar) ----
    pre_means, post_means, pops_present = [], [], []
    for pop in POP_ORDER:
        key = f"{pop}_rate"
        if key not in rate_data:
            continue
        t_ms = np.asarray(rate_data[key]['t_ms'])
        r_hz = smooth_flat(np.asarray(rate_data[key]['rate_hz']), t_ms)
        pre = (t_ms >= stim_onset_ms - PRE_WINDOW_MS) & (t_ms < stim_onset_ms)
        post = (t_ms >= stim_onset_ms) & (t_ms <= t_hi)
        pre_means.append(float(np.mean(r_hz[pre])) if pre.any() else 0.0)
        post_means.append(float(np.mean(r_hz[post])) if post.any() else 0.0)
        pops_present.append(pop)

    fig_rate, axr = plt.subplots(figsize=(7, 5))
    x = np.arange(len(pops_present))
    w = 0.38
    axr.bar(x - w / 2, pre_means, width=w, label='pre',
            color=[POP_COLORS[p] for p in pops_present],
            edgecolor='black', linewidth=0.6, alpha=0.55)
    axr.bar(x + w / 2, post_means, width=w, label='post (stim)',
            color=[POP_COLORS[p] for p in pops_present],
            edgecolor='black', linewidth=0.6, alpha=0.95)
    axr.set_xticks(x)
    axr.set_xticklabels(pops_present, fontsize=12)
    axr.set_ylabel('Mean firing rate (Hz)', fontsize=13)
    axr.set_title(f'{layer}', fontsize=13)
    axr.legend(fontsize=10, frameon=False)
    axr.spines['top'].set_visible(False)
    axr.spines['right'].set_visible(False)
    axr.yaxis.grid(True, alpha=0.3, linestyle='--')
    axr.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f'rates_{layer}_06_05-fb.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'rates_{layer}_06_05-fb.pdf', bbox_inches='tight')
    plt.close(fig_rate)

    print(f"{layer}: saved raster + rates  (E capped to {n_e_cap}; "
          f"PV/SOM/VIP all shown)")
    for p, pre_m, post_m in zip(pops_present, pre_means, post_means):
        print(f"    {p:>4}: pre {pre_m:5.2f} Hz -> post {post_m:5.2f} Hz")


for layer in LAYERS:
    make_figures(layer)
