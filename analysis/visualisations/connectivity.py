"""Render connectivity/conductance matrices and interneuron composition from the
config CSVs into figures/connectivity/.

Run:
    path/to/your/venv/bin/python analysis/visualisations/connectivity.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

from config.config import CONFIG

# parameters
OUT_DIR = "figures/connectivity"
PROB_CSV = "config/connection_probabilities.csv"
AMPA_CSV = "config/conductances_AMPA_GABA.csv"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,   # editable text in Illustrator
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.8,
})

LAYERS = ["L23", "L4AB", "L4C", "L5", "L6"]
TYPES = ["E", "PV", "SOM", "VIP"]
LAYER_LABEL = {"L23": "L2/3", "L4AB": "L4A/B", "L4C": "L4C", "L5": "L5", "L6": "L6"}
TYPE_COLOR = {"E": "#c0392b", "PV": "#2471a3", "SOM": "#1e8449", "VIP": "#8e44ad"}

ORDER = [f"{t}_{l}" for l in LAYERS for t in TYPES]
N = len(ORDER)


def load_matrix(path):
    df = pd.read_csv(path, index_col=0)
    order = [f"{t}_{l}" for l in LAYERS for t in TYPES]
    return df.reindex(index=order, columns=order)


def figure_matrix(value_df, title, cbar_label, fname, annotate=False):
    M = value_df.values.astype(float)
    Mmask = np.ma.masked_where(M == 0, M)

    fig, ax = plt.subplots(figsize=(11, 9.5))
    cmap = LinearSegmentedColormap.from_list(
        "conn", ["#fff7ec", "#fdbb84", "#d7301f", "#7f0000"])
    cmap.set_bad("#f2f2f2")  # zero entries = light grey
    im = ax.imshow(Mmask, cmap=cmap, aspect="equal",
                   vmin=0, vmax=np.nanpercentile(M[M > 0], 98))

    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for k in range(4, N, 4):
        ax.axhline(k - 0.5, color="black", linewidth=1.6)
        ax.axvline(k - 0.5, color="black", linewidth=1.6)

    labels = [lab.split("_")[0] for lab in ORDER]
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for tl, lab in zip(ax.get_xticklabels(), labels):
        tl.set_color(TYPE_COLOR[lab])
    for tl, lab in zip(ax.get_yticklabels(), labels):
        tl.set_color(TYPE_COLOR[lab])

    for i, l in enumerate(LAYERS):
        center = i * 4 + 1.5
        ax.text(-2.4, center, LAYER_LABEL[l], rotation=90, va="center",
                ha="center", fontsize=11, fontweight="bold")
        ax.text(center, N + 1.4, LAYER_LABEL[l], va="center",
                ha="center", fontsize=11, fontweight="bold")

    ax.xaxis.tick_top()
    ax.set_title(title, fontsize=15, fontweight="bold", pad=30)

    if annotate:
        hi = np.nanpercentile(M[M > 0], 70)
        for i in range(N):
            for j in range(N):
                v = M[i, j]
                if v > 0:
                    ax.text(j, i, f"{v:.2g}", ha="center", va="center",
                            fontsize=4.5,
                            color="white" if v > hi else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.03)
    cbar.set_label(cbar_label, fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    _save(fig, fname)


def figure_interneuron_proportions(fname):
    pop_colors = {"E": "#2E8B57", "PV": "#C0392B", "SOM": "#1F4E96", "VIP": "#D4A017"}
    inh_pops = ["PV", "SOM", "VIP"]
    layer_names = [n for n in CONFIG["layers"]
                   if "neuron_counts" in CONFIG["layers"][n]]

    proportions = {p: [] for p in inh_pops}
    for layer in layer_names:
        counts = CONFIG["layers"][layer]["neuron_counts"]
        total_inh = sum(counts.get(p, 0) for p in inh_pops)
        for p in inh_pops:
            proportions[p].append(counts.get(p, 0) / total_inh if total_inh > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(layer_names))
    left = np.zeros(len(layer_names))
    for p in inh_pops:
        vals = np.array(proportions[p])
        ax.barh(y, vals, left=left, color=pop_colors[p], edgecolor="white",
                linewidth=1.2, label=p, alpha=0.95, height=0.7)
        for yi, v, l in zip(y, vals, left):
            if v > 0.05:
                ax.text(l + v / 2, yi, f"{v*100:.0f}%", ha="center",
                        va="center", color="white", fontsize=10, fontweight="bold")
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(layer_names, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Relative proportion", fontsize=13)
    ax.set_ylabel("Layer", fontsize=13)
    ax.set_title("Interneuron subtype composition per layer",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{int(t*100)}%" for t in np.linspace(0, 1, 6)])
    ax.legend(title="Interneuron", loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=11, title_fontsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, fname)


def _save(fig, fname):
    base = os.path.join(OUT_DIR, fname)
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {base}.png / .pdf")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    prob = load_matrix(PROB_CSV)
    ampa = load_matrix(AMPA_CSV)   # AMPA for E sources, GABA for I sources

    print("Building connectivity figures...")
    for df, title, cbar, stem in (
            (prob, "Connection probability", "p(connection)", "matrix_prob"),
            (ampa, "Synaptic weight (AMPA / GABA)", "peak conductance (nS)", "matrix_weight")):
        figure_matrix(df, title, cbar, stem, annotate=False)
        figure_matrix(df, title, cbar, f"{stem}_annotated", annotate=True)

    figure_interneuron_proportions("interneuron_proportions")
    print("Done.")


if __name__ == "__main__":
    main()
