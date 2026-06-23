"""
to do : merge this  with the connectivity. py script. have it plot all versions, the one with and without numbers. make the plots be saved in figures/connectivity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, FancyArrow
from matplotlib.collections import PatchCollection
import matplotlib as mpl

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,   # editable text in Illustrator
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.8,
})

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
PROB_CSV = "config/connection_probabilities.csv"
AMPA_CSV = "config/conductances_AMPA_GABA.csv"
NMDA_CSV = "config/conductances_NMDA.csv"

LAYERS = ["L23", "L4AB", "L4C", "L5", "L6"]
TYPES = ["E", "PV", "SOM", "VIP"]
# layer label shown to the reader
LAYER_LABEL = {"L23": "L2/3", "L4AB": "L4A/B", "L4C": "L4C", "L5": "L5", "L6": "L6"}

# populations that receive direct LGN / feedforward thalamic drive
LGN_TARGETS = {("L4C", "E"), ("L4C", "PV"), ("L6", "E"), ("L6", "PV")}

# neuron counts per layer (from config/config.py) - used to scale node sizes
NEURON_COUNTS = {
    "L23":  {"E": 3520, "PV": 260, "SOM": 410, "VIP": 263},
    "L4AB": {"E": 2720, "PV": 340, "SOM": 210, "VIP": 130},
    "L4C":  {"E": 3192, "PV": 320, "SOM": 200, "VIP": 88},
    "L5":   {"E": 1600, "PV": 220, "SOM": 110, "VIP": 70},
    "L6":   {"E": 2040, "PV": 195, "SOM": 110, "VIP": 70},
}

# colours
C_E   = "#c0392b"   # excitatory - red
C_PV  = "#2471a3"   # PV - blue
C_SOM = "#1e8449"   # SOM - green
C_VIP = "#8e44ad"   # VIP - purple
TYPE_COLOR = {"E": C_E, "PV": C_PV, "SOM": C_SOM, "VIP": C_VIP}


def load_matrix(path):
    df = pd.read_csv(path, index_col=0)
    # order rows/cols as Layer x Type
    order = [f"{t}_{l}" for l in LAYERS for t in TYPES]
    df = df.reindex(index=order, columns=order)
    return df


PROB = load_matrix(PROB_CSV)
AMPA = load_matrix(AMPA_CSV)   # AMPA for E sources, GABA for I sources
NMDA = load_matrix(NMDA_CSV)   # NMDA, E sources only

ORDER = [f"{t}_{l}" for l in LAYERS for t in TYPES]
N = len(ORDER)

# Effective connection strength = probability x total synaptic conductance.
# For E sources the excitatory drive is AMPA + NMDA; inhibitory sources use the
# GABA values stored in the AMPA/GABA table (NMDA is zero for I sources).
TOTAL_G = AMPA.fillna(0) + NMDA.fillna(0)
STRENGTH = PROB.fillna(0) * TOTAL_G


def is_excit(label):
    return label.startswith("E_")


# ============================================================================
# FIGURE 1 - Annotated connectivity matrix (heatmap)
# ============================================================================
def figure_matrix(value_df, title, cbar_label, fname, annotate=False):
    M = value_df.values.astype(float)
    Mmask = np.ma.masked_where(M == 0, M)

    fig, ax = plt.subplots(figsize=(11, 9.5))

    cmap = LinearSegmentedColormap.from_list(
        "conn", ["#fff7ec", "#fdbb84", "#d7301f", "#7f0000"])
    cmap.set_bad("#f2f2f2")  # zero entries = light grey

    im = ax.imshow(Mmask, cmap=cmap, aspect="equal",
                   vmin=0, vmax=np.nanpercentile(M[M > 0], 98))

    # gridlines between every cell (thin) and thick lines between layers
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for k in range(4, N, 4):
        ax.axhline(k - 0.5, color="black", linewidth=1.6)
        ax.axvline(k - 0.5, color="black", linewidth=1.6)

    # tick labels = cell type, repeated per layer
    labels = [lab.split("_")[0] for lab in ORDER]
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    # colour the tick labels by cell type
    for tl, lab in zip(ax.get_xticklabels(), labels):
        tl.set_color(TYPE_COLOR[lab])
    for tl, lab in zip(ax.get_yticklabels(), labels):
        tl.set_color(TYPE_COLOR[lab])

    # layer group labels along the edges
    for i, l in enumerate(LAYERS):
        center = i * 4 + 1.5
        ax.text(-2.4, center, LAYER_LABEL[l], rotation=90, va="center",
                ha="center", fontsize=11, fontweight="bold")
        ax.text(center, N + 1.4, LAYER_LABEL[l], va="center",
                ha="center", fontsize=11, fontweight="bold")

    ax.xaxis.tick_top()
    ax.set_title(title, fontsize=15, fontweight="bold", pad=30)

    if annotate:
        for i in range(N):
            for j in range(N):
                v = M[i, j]
                if v > 0:
                    ax.text(j, i, f"{v:.2g}", ha="center", va="center",
                            fontsize=4.5,
                            color="white" if v > np.nanpercentile(M[M>0], 70) else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.03)
    cbar.set_label(cbar_label, fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(f"{fname}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}.png / .pdf")


# ============================================================================
# FIGURE 2 - Laminar circuit schematic
# ============================================================================
def figure_circuit(fname, prob_thresh=0.07):
    """Anatomically arranged: layers stacked top->bottom, 4 cell types per layer.
    Arrows drawn for connections with probability >= prob_thresh; arrow width
    scales with connection probability. Excitatory = solid arrowhead, inhibitory
    = round head."""
    fig, ax = plt.subplots(figsize=(11, 13))

    # vertical position per layer (L2/3 on top)
    y_of_layer = {l: (len(LAYERS) - 1 - i) * 3.0 for i, l in enumerate(LAYERS)}
    # horizontal position per cell type
    x_of_type = {"E": 0.0, "PV": 2.3, "SOM": 4.2, "VIP": 6.0}

    pos = {}   # population -> (x, y)
    for l in LAYERS:
        for t in TYPES:
            pos[f"{t}_{l}"] = (x_of_type[t], y_of_layer[l])

    # layer background bands
    xmin, xmax = -1.6, 7.4
    for l in LAYERS:
        y = y_of_layer[l]
        ax.add_patch(Rectangle((xmin, y - 1.25), xmax - xmin, 2.5,
                               facecolor="#f4f4f4", edgecolor="none", zorder=0))
        ax.text(xmin + 0.12, y + 1.05, LAYER_LABEL[l], fontsize=14,
                fontweight="bold", va="top", color="#444")

    # ---- draw connections ----
    Mp = PROB.values.astype(float)
    maxp = Mp[Mp > 0].max()
    for i, src in enumerate(ORDER):
        for j, tgt in enumerate(ORDER):
            p = Mp[i, j]
            if p < prob_thresh or src == tgt:
                continue
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            excit = is_excit(src)
            col = TYPE_COLOR[src.split("_")[0]]
            lw = 0.4 + 5.0 * (p / maxp)
            alpha = 0.25 + 0.55 * (p / maxp)
            # curve so opposing directions don't overlap
            rad = 0.18 if (i < j) else -0.18
            style = "-|>" if excit else "-"   # excit gets arrowhead
            arr = FancyArrowPatch(
                (x0, y0), (x1, y1),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle=style, mutation_scale=11 if excit else 1,
                lw=lw, color=col, alpha=alpha, zorder=2,
                shrinkA=14, shrinkB=14,
                capstyle="round",
            )
            ax.add_patch(arr)
            if not excit:
                # inhibitory: draw a small filled circle near the target as the "head"
                ux, uy = x1 - x0, y1 - y0
                d = np.hypot(ux, uy)
                ux, uy = ux / d, uy / d
                hx, hy = x1 - ux * 0.42, y1 - uy * 0.42
                ax.add_patch(Circle((hx, hy), 0.10, color=col,
                                    alpha=min(1, alpha + 0.2), zorder=3))

    # ---- draw nodes ----
    for l in LAYERS:
        for t in TYPES:
            x, y = pos[f"{t}_{l}"]
            n = NEURON_COUNTS[l][t]
            r = 0.28 + 0.55 * np.sqrt(n / 3520)
            ax.add_patch(Circle((x, y), r, facecolor=TYPE_COLOR[t],
                                edgecolor="black", lw=1.1, zorder=5))
            ax.text(x, y, t, ha="center", va="center", color="white",
                    fontsize=9, fontweight="bold", zorder=6)
            # LGN input marker
            if (l, t) in LGN_TARGETS:
                ax.add_patch(FancyArrow(x - 1.35, y - 0.95, 0.55, 0.55,
                                        width=0.04, head_width=0.22,
                                        head_length=0.2, length_includes_head=True,
                                        color="#e67e22", zorder=4))

    # LGN label
    ax.text(-1.5, y_of_layer["L4C"] - 1.7, "LGN\nfeedforward",
            color="#e67e22", fontsize=11, fontweight="bold", ha="left", va="top")

    # ---- legend ----
    from matplotlib.lines import Line2D
    leg_pop = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                      markeredgecolor="k", markersize=12, label=t)
               for t, c in TYPE_COLOR.items()]
    leg_conn = [
        Line2D([0], [0], color="grey", lw=3, marker=">", markersize=8,
               label="excitatory"),
        Line2D([0], [0], color="grey", lw=3, marker="o", markersize=8,
               label="inhibitory"),
        FancyArrow(0, 0, 0, 0, color="#e67e22", label="LGN input"),
    ]
    l1 = ax.legend(handles=leg_pop, loc="upper right", fontsize=10,
                   title="Cell type", frameon=False, ncol=4,
                   bbox_to_anchor=(1.0, 1.10))
    ax.add_artist(l1)
    ax.legend(handles=leg_conn[:2], loc="upper left", fontsize=10,
              title="Connection", frameon=False, bbox_to_anchor=(0.0, 1.10))

    ax.set_xlim(xmin - 0.4, xmax + 0.4)
    ax.set_ylim(-1.6, (len(LAYERS) - 1) * 3.0 + 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("V1 cortical column microcircuit",
                 fontsize=16, fontweight="bold", pad=34)
    ax.text(0.5, -0.02,
            f"Arrows shown for connection probability >= {prob_thresh}; "
            "width ~ probability. Node size ~ population size.",
            transform=ax.transAxes, ha="center", fontsize=8, color="#666")

    fig.tight_layout()
    fig.savefig(f"{fname}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}.png / .pdf")


# ============================================================================
# FIGURE 3 - Circular / chord graph
# ============================================================================
def figure_chord(fname, prob_thresh=0.06):
    fig, ax = plt.subplots(figsize=(11, 11))

    angles = np.linspace(90, 90 - 360, N, endpoint=False) * np.pi / 180.0
    R = 1.0
    px = R * np.cos(angles)
    py = R * np.sin(angles)
    node_pos = {ORDER[k]: (px[k], py[k]) for k in range(N)}

    Mp = PROB.values.astype(float)
    maxp = Mp[Mp > 0].max()

    # draw arcs (Bezier toward centre) for connections above threshold
    for i, src in enumerate(ORDER):
        for j, tgt in enumerate(ORDER):
            p = Mp[i, j]
            if p < prob_thresh or src == tgt:
                continue
            x0, y0 = node_pos[src]
            x1, y1 = node_pos[tgt]
            col = TYPE_COLOR[src.split("_")[0]]
            lw = 0.3 + 3.5 * (p / maxp)
            alpha = 0.12 + 0.5 * (p / maxp)
            # pull control point toward centre for a chord look
            cx, cy = (x0 + x1) * 0.18, (y0 + y1) * 0.18
            t = np.linspace(0, 1, 60)
            bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
            by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1
            ax.plot(bx, by, color=col, lw=lw, alpha=alpha, zorder=1,
                    solid_capstyle="round")

    # nodes
    for k, lab in enumerate(ORDER):
        t = lab.split("_")[0]
        ax.scatter(px[k], py[k], s=130, color=TYPE_COLOR[t],
                   edgecolor="black", lw=0.8, zorder=4)
        # outward label
        a = angles[k]
        lx, ly = 1.13 * np.cos(a), 1.13 * np.sin(a)
        ha = "left" if np.cos(a) > 0.01 else ("right" if np.cos(a) < -0.01 else "center")
        rot = np.degrees(a)
        if rot > 90 or rot < -90:
            rot += 180
        ax.text(lx, ly, t, ha=ha, va="center", fontsize=7.5,
                color=TYPE_COLOR[t], rotation=rot, rotation_mode="anchor")

    # layer arcs outside the ring
    for i, l in enumerate(LAYERS):
        a0 = angles[i * 4] + (angles[1] - angles[0]) * 0.5
        a1 = angles[i * 4 + 3] - (angles[1] - angles[0]) * 0.5
        aa = np.linspace(a0, a1, 30)
        ax.plot(1.28 * np.cos(aa), 1.28 * np.sin(aa), color="#333", lw=3,
                solid_capstyle="round")
        am = (a0 + a1) / 2
        rot = np.degrees(am)
        if rot > 90 or rot < -90:
            rot += 180
        ax.text(1.42 * np.cos(am), 1.42 * np.sin(am), LAYER_LABEL[l],
                ha="center", va="center", fontsize=12, fontweight="bold",
                rotation=rot, rotation_mode="anchor")

    # legend
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                  markeredgecolor="k", markersize=12, label=t)
           for t, c in TYPE_COLOR.items()]
    ax.legend(handles=leg, loc="center", fontsize=11, frameon=False,
              title="Cell type", ncol=1, bbox_to_anchor=(0.5, 0.5))

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Cortical column connectivity",
                 fontsize=16, fontweight="bold", y=1.02)
    ax.text(0.5, -0.02,
            f"Chords: connection probability >= {prob_thresh}; "
            "colour = source cell type, width ~ probability.",
            transform=ax.transAxes, ha="center", fontsize=8, color="#666")

    fig.savefig(f"{fname}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}.png / .pdf")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Building connectivity figures...")
    print("[1/3] connectivity matrix (probability)")
    figure_matrix(PROB, "Connection probability",
                  "p(connection)", "connectivity_matrix_prob", annotate=False)
    print("[1b]  connectivity matrix (AMPA/GABA weight)")
    figure_matrix(AMPA, "Synaptic weight (AMPA / GABA)",
                  "peak conductance (nS)", "connectivity_matrix_weight",
                  annotate=False)
    print("[1c]  effective connection strength (prob x conductance)")
    figure_matrix(STRENGTH, "Effective connection strength",
                  "p × g  (nS)", "connectivity_matrix_strength",
                  annotate=False)
    print("[2/3] laminar circuit schematic")
    figure_circuit("connectivity_circuit")
    print("[3/3] chord graph")
    figure_chord("connectivity_chord")
    print("Done.")
