"""Objective taxonomy of lesions from cached fold-change maps.

Reads results/<root>/_fc_cache.npz (built by the caching step) and asks, with
no hand-picking: how many *distinct kinds* of lesion effect are there?

Three views:
  1. Feature table  -> CSV: per-lesion scalar descriptors
        peakHz   frequency of strongest power loss
        bw_Hz    half-max bandwidth of the loss  (narrow vs broadband)
        depth_bias = superficial_loss - deep_loss (where it acts)
        strength = mean log2FC over the whole map
  2. Dendrogram     -> figure: hierarchical clustering of full depth-avg spectra
                       (correlation distance) so 'same shape' lesions group.
  3. Scatter map    -> figure: bandwidth vs depth-bias, sized by strength.
                       A 2-axis 'map' of lesion types you can put on one slide.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

CACHE = "results/lesions_2026-06-03/_fc_cache.npz"
OUT = "figures/lesions_2026-06-03/_significance_cluster/_slides"


def arrow(n):
    s, t = n.split("_", 1)
    return f"{s}→{t}"


def load():
    d = np.load(CACHE)
    freqs, depths = d["freqs"], d["depths"]
    names = [n for n in d["_names"]]
    return freqs, depths, names, {n: d[n] for n in names}


def feature_table(freqs, depths, FC):
    rows = {}
    for n, m in FC.items():
        s = m.mean(0)
        peak = s.min()
        thr = peak * 0.5
        band = freqs[s < thr]
        bw = (band.max() - band.min()) if band.size else 0.0
        perch = m.mean(1)
        sup = perch[depths > 0.4].mean()
        deep = perch[depths < -0.2].mean()
        # depth_bias > 0  => loss concentrated superficially; < 0 => deep.
        # (loss values are negative, so the layer with the MORE negative mean
        #  is where the loss is; bias = how much more superficial loss than deep)
        rows[n] = dict(peakHz=float(freqs[np.argmin(s)]), peak=float(peak),
                       bw_Hz=float(bw), strength=float(s.mean()),
                       depth_bias=float(deep - sup), sup=float(sup), deep=float(deep))
    return pd.DataFrame(rows).T.sort_values("strength")


def fig_dendrogram(freqs, FC, out):
    names = list(FC.keys())
    M = np.stack([FC[n].mean(0) for n in names])  # depth-avg spectra
    Z = linkage(M, method="average", metric="correlation")
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(names) + 1.5))
    dendrogram(Z, labels=[arrow(n) for n in names], orientation="left",
               color_threshold=0.5 * Z[:, 2].max(), ax=ax,
               leaf_font_size=10)
    ax.set_xlabel("dissimilarity of spectral shape (correlation distance)", fontsize=10)
    ax.set_title("How many kinds of lesion are there?\n"
                 "branches that join late = genuinely different effects", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
    # also return a flat 3-cluster labelling for annotation
    return dict(zip(names, fcluster(Z, t=3, criterion="maxclust")))


def fig_typemap(tab, out):
    """bandwidth (x) vs depth-bias (y), bubble size = strength. One-slide map."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    x = tab["bw_Hz"].values
    y = tab["depth_bias"].values
    s = (-tab["strength"].values) * 1400 + 30  # stronger loss = bigger
    sc = ax.scatter(x, y, s=s, c=tab["peakHz"].values, cmap="viridis",
                    alpha=0.8, edgecolors="k", linewidths=0.5)
    for n, xi, yi in zip(tab.index, x, y):
        ax.annotate(arrow(n), (xi, yi), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="0.7", lw=0.7)
    ax.set_xlabel("bandwidth of power loss (Hz)   narrow → broadband", fontsize=10)
    ax.set_ylabel("depth bias\ndeep-biased ←   0   → superficial-biased", fontsize=10)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("peak-loss frequency (Hz)", fontsize=9)
    ax.set_title("A map of lesion types\n"
                 "bubble size = overall strength of effect", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)


def main():
    os.makedirs(OUT, exist_ok=True)
    freqs, depths, names, FC = load()
    tab = feature_table(freqs, depths, FC)
    tab.round(3).to_csv(os.path.join(OUT, "lesion_features.csv"))
    print(tab.round(2).to_string())
    clusters = fig_dendrogram(freqs, FC, os.path.join(OUT, "lesion_dendrogram.png"))
    tab["cluster"] = [clusters[n] for n in tab.index]
    fig_typemap(tab, os.path.join(OUT, "lesion_typemap.png"))


if __name__ == "__main__":
    main()
