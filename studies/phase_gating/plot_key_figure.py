"""One-panel summary: alpha phase gates the L4C response to weak (and thalamic) input.

L4C E response (0-20 ms, pulse minus same-state sham) vs L4C's own alpha phase at the
moment the input arrives, as % of that input's average response, for:
  weak current   25 pA into L4C E
  strong current 100 pA into L4C E
  thalamic volley, all 80 LGN sources (onto L4C E + PV and L6)
Onset phases are continuous (onsets on a 25 ms grid land at effectively random
phases), so the curve is a circular kernel smooth (von Mises kernel, ~25 deg sd,
evaluated every 5 deg) of the seed-centred % responses, with a 95% bootstrap band
(onsets resampled within seed); faint dots = plain 45 deg bin means.

Run:  ~/Desktop/venv/bin/python studies/phase_gating/plot_key_figure.py
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(_ROOT, "figures", "exp5_phase")
WIN_BINS = 4                    # 0-20 ms
LAYER = "L4C"
LINES = [  # (label, folder, dose, colour, style)
    ("weak input (25 pA)", "exp5c_restore_pulse_L4C", 25, "#1f4e99", "-"),
    ("strong input (100 pA)", "exp5c_restore_pulse_L4C", 100, "#9aa5b1", "-"),
    ("thalamic volley (80 LGN inputs)", "exp5c_restore_pulse_LGN", 80, "#c0392b", "--"),
]


def load(folder, dose):
    ev, ph, seeds = [], [], []
    for f in sorted(glob.glob(os.path.join(_ROOT, "saved_trials", folder, "seed*.npz"))):
        z = np.load(f, allow_pickle=True)
        li = list(z["layers"]).index(LAYER)
        di = list(z["doses_pa"]).index(dose)
        hz = z["pulse_counts"][:, :, li, :WIN_BINS].sum(-1) / z["n_e"][li] / (WIN_BINS * 5e-3)
        ev.append(hz[:, di] - hz[:, 0])
        ph.append(z[f"phase_{LAYER}"])
        seeds.append(np.full(len(hz), int(z["network_seed"])))
    return np.concatenate(ev), np.concatenate(ph), np.concatenate(seeds)


KERNEL_SD_DEG = 25
N_BOOT = 300
RNG = np.random.default_rng(3)


def centred_pct(y, seeds):
    yc = y.copy()
    for s in np.unique(seeds):
        yc[seeds == s] -= y[seeds == s].mean()
    return 100 * yc / y.mean()


def smooth(yc, ph, grid):
    kappa = 1.0 / np.radians(KERNEL_SD_DEG) ** 2
    w = np.exp(kappa * (np.cos(ph[None, :] - grid[:, None]) - 1))
    return (w * yc[None, :]).sum(1) / w.sum(1)


def smooth_with_band(y, ph, seeds):
    grid = np.radians(np.arange(-180, 181, 5))
    yc = centred_pct(y, seeds)
    m = smooth(yc, ph, grid)
    boots = []
    for _ in range(N_BOOT):
        idx = np.concatenate([RNG.choice(np.flatnonzero(seeds == s), (seeds == s).sum())
                              for s in np.unique(seeds)])
        boots.append(smooth(centred_pct(y[idx], seeds[idx]), ph[idx], grid))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return np.degrees(grid), m, lo, hi


def curve(y, ph, seeds, width=45):
    """% of the average response per phase bin, seed-centred."""
    yc = y.copy()
    for s in np.unique(seeds):
        yc[seeds == s] -= y[seeds == s].mean()
    yc = 100 * yc / y.mean()
    d = np.degrees(ph)
    grid = np.arange(-180, 180, width)
    m, se = [], []
    for g in grid:
        sel = np.abs(np.angle(np.exp(1j * np.radians(d - g)))) <= np.radians(width / 2)
        m.append(yc[sel].mean()); se.append(yc[sel].std() / np.sqrt(sel.sum()))
    return np.r_[grid, 180], np.r_[m, m[0]], np.r_[se, se[0]]


def main():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 11})
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for label, folder, dose, col, ls in LINES:
        y, ph, seeds = load(folder, dose)
        gx, gm, lo, hi = smooth_with_band(y, ph, seeds)
        ax.fill_between(gx, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(gx, gm, ls, color=col, lw=2.8, label=f"{label}")
        x, m, _ = curve(y, ph, seeds)
        ax.plot(x, m, "o", color=col, ms=4, alpha=0.45)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xlim(-180, 180)
    ax.set_xlabel("alpha phase when the input arrives (degrees; 0 = L4C firing peak)")
    ax.set_ylabel("L4C response\n(% change from its average)")
    ax.set_title("Alpha phase gates the L4C response to weak and thalamic inputs",
                 fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    n_seeds = len(glob.glob(os.path.join(_ROOT, "saved_trials", LINES[0][1], "seed*.npz")))
    ax.text(0.01, 0.02, f"{n_seeds} network seeds, {176 * n_seeds} inputs per condition; "
            f"lines: kernel smooth ({KERNEL_SD_DEG}° sd) with 95% bootstrap band; "
            "dots: 45° bin means", transform=ax.transAxes, fontsize=7.5, color="0.4")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"alpha_gates_excitability.{ext}"), dpi=200,
                    bbox_inches="tight")
    print("saved", os.path.join(OUT, "alpha_gates_excitability.png"))


if __name__ == "__main__":
    main()
