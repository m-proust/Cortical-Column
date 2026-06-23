"""Slide-ready summary figures for the lesion experiment.

Reads the cluster table produced by lesion_significance_cluster.py
(grand_summary_clusters.csv) and renders a small set of clean, presentation-
oriented figures instead of the 30 dense two-panel heatmaps.

The goal is to tell the story in one or two figures:
  Fig 1  Overview matrix: lesion (rows) x frequency band (cols), coloured by
         net signed cluster mass. One glance shows which pathways matter and
         which band each one feeds. Silent lesions stay blank.
  Fig 2  Impact bars: total |cluster mass| per lesion, sorted. Shows the
         load-bearing pathways vs the functionally silent ones.

Run:
    ~/Desktop/venv/bin/python lesion_slide_figures.py \
        figures/lesions_2026-06-03/_significance_cluster
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

BANDS = ["delta", "theta", "alpha", "beta", "low_gamma", "high_gamma"]
BAND_LABEL = {
    "delta": "δ\n1-4", "theta": "θ\n4-8", "alpha": "α\n8-13",
    "beta": "β\n13-30", "low_gamma": "low γ\n30-50", "high_gamma": "high γ\n50-100",
}
# pretty arrow labels for pathways
def pretty(les):
    src, tgt = les.split("_", 1)
    return f"{src} → {tgt}"


def load(summary_dir):
    csv = os.path.join(summary_dir, "grand_summary_clusters.csv")
    df = pd.read_csv(csv)
    # every lesion actually run (incl. silent ones) from heatmap filenames
    run = sorted(
        os.path.basename(f).replace("_cluster_heatmap.png", "")
        for f in glob.glob(os.path.join(summary_dir, "*_cluster_heatmap.png"))
    )
    return df, run


def fig_overview(df, run, out):
    """Lesion x band matrix of net signed cluster mass."""
    # net signed mass per lesion x band
    g = df.groupby(["lesion", "band_hint"])["mass"].sum().reset_index()
    piv = g.pivot(index="lesion", columns="band_hint", values="mass")
    piv = piv.reindex(index=run, columns=BANDS)  # include silent lesions / all bands

    # order rows by total impact (strongest at top)
    order = piv.abs().sum(axis=1).sort_values(ascending=False).index
    piv = piv.reindex(index=order)

    M = piv.values.astype(float)
    finite = M[np.isfinite(M)]
    vmax = np.nanpercentile(np.abs(finite), 98) if finite.size else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    nrow, ncol = M.shape
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * nrow + 1.8))
    masked = np.ma.masked_invalid(M)
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("0.92")  # silent / no-cluster cells -> light grey
    im = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(ncol))
    ax.set_xticklabels([BAND_LABEL[b] for b in BANDS], fontsize=9)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels([pretty(l) for l in piv.index], fontsize=10)
    ax.set_xlabel("Frequency band", fontsize=11)
    ax.tick_params(length=0)

    # thin white gridlines between cells
    ax.set_xticks(np.arange(-.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nrow, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    for s in ax.spines.values():
        s.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("net signed cluster mass\n(blue = power loss)", fontsize=9)

    ax.set_title("Which pathway feeds which rhythm\n"
                 "grey = no significant effect", fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_impact(df, run, out):
    """Total absolute cluster mass per lesion, sorted -> load-bearing vs silent."""
    imp = (df.assign(a=df["mass"].abs()).groupby("lesion")["a"].sum())
    imp = imp.reindex(run).fillna(0.0).sort_values()

    colors = ["#b0b0b0" if v == 0 else "#2166ac" for v in imp.values]
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(imp) + 1.2))
    ax.barh(range(len(imp)), imp.values, color=colors)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels([pretty(l) for l in imp.index], fontsize=10)
    ax.set_xlabel("total |cluster mass|  (overall impact of removing the pathway)",
                  fontsize=10)
    ax.set_title("Most connections are functionally silent;\n"
                 "a few carry the column", fontsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    summary_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "figures/lesions_2026-06-03/_significance_cluster"
    df, run = load(summary_dir)
    out_dir = os.path.join(summary_dir, "_slides")
    os.makedirs(out_dir, exist_ok=True)
    fig_overview(df, run, os.path.join(out_dir, "overview_band_matrix.png"))
    fig_impact(df, run, os.path.join(out_dir, "impact_ranking.png"))


if __name__ == "__main__":
    main()
