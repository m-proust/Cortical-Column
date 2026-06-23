"""to do : rmeove comments in the script.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from lesion_significance_cluster import compute_all_masks

ROOT = "results/lesions_2026-06-03"
OUT = "figures/lesions_2026-06-03/_significance_cluster/_slides"

# layer centres (mm), higher z = more superficial (from alpha_gamma_coupling.py)
LAYER_CENTRES = {"L2/3": 0.775, "L4AB": 0.295, "L4C": 0.0,
                 "L5": -0.24, "L6": -0.48}


def pretty(les):
    s, t = les.split("_", 1)
    return f"{s} → {t}"


def _panel(ax, data, freqs, depths, norm, masked=False, show_xlabel=True):
    """One heatmap. depths ascending; we flip so superficial is on top.

    The raw log2FC panel is bilinear-interpolated for a smooth look; the
    significance panel is drawn with `nearest` so the cluster bins stay crisp
    and identical to the original *_cluster_heatmap.png (never blur a mask).
    """
    # depths in this model: larger = more superficial. We want superficial top.
    order = np.argsort(depths)[::-1]          # superficial (high) first
    data = data[order]
    d_sorted = depths[order]
    # extent: y from top(superficial, high mm) to bottom(deep, low mm)
    extent = [freqs[0], freqs[-1], d_sorted[-1], d_sorted[0]]
    interp = "nearest" if masked else "bilinear"
    if masked:
        ax.set_facecolor("#e8e8e8")
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", norm=norm,
                   extent=extent, interpolation=interp, origin="upper")
    ax.set_xticks(np.arange(0, freqs[-1] + 1, 10))
    if show_xlabel:
        ax.set_xlabel("Frequency (Hz)", fontsize=10)
    else:
        ax.set_xticklabels([])
    # label y-axis by layer instead of raw mm
    ax.set_yticks([c for c in LAYER_CENTRES.values()])
    ax.set_yticklabels(list(LAYER_CENTRES.keys()), fontsize=9)
    return im


def make_group(lesions, out_name, freqs, depths, all_results):
    results = {l: all_results[l] for l in lesions}
    # shared colour scale across the group for honest comparison
    allfc = np.concatenate([np.abs(r["log2fc"]).ravel() for r in results.values()])
    vlim = max(float(np.nanpercentile(allfc, 98)), 0.1)
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)

    n = len(lesions)
    fig, axes = plt.subplots(n, 2, figsize=(9.6, 3.2 * n), squeeze=False)
    for i, les in enumerate(lesions):
        last = (i == n - 1)
        log2fc, sig = results[les]["log2fc"], results[les]["sig_mask"]
        im = _panel(axes[i, 0], log2fc, freqs, depths, norm, show_xlabel=last)
        masked = np.where(sig, log2fc, np.nan)
        _panel(axes[i, 1], masked, freqs, depths, norm, masked=True, show_xlabel=last)
        axes[i, 0].set_title(pretty(les), fontsize=13, fontweight="bold", loc="left")
        axes[i, 1].set_title("significant clusters", fontsize=10, color="0.4", loc="left")
        axes[i, 1].set_yticklabels([])
        axes[i, 0].set_ylabel("cortical layer", fontsize=9)
    # reserve a strip on the right for the colorbar so it never overlaps a panel
    fig.subplots_adjust(hspace=0.35, right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("log2 power change", fontsize=9)
    out = os.path.join(OUT, out_name)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def main():
    os.makedirs(OUT, exist_ok=True)
    # one shared computation, identical rng discipline to the original script,
    # so the significance masks match the published *_cluster_heatmap.png exactly.
    freqs, depths, _labels, results = compute_all_masks(ROOT)
    depths = np.asarray(depths)

    make_group(["L6_L4C"], "fig_gain.png", freqs, depths, results)
    make_group(["L4C_L4AB", "L4AB_L23", "L4C_L23"],
               "fig_feedforward.png", freqs, depths, results)
    make_group(["L5_L6", "L4C_L6", "L23_L5"],
               "fig_alpha.png", freqs, depths, results)


if __name__ == "__main__":
    main()
