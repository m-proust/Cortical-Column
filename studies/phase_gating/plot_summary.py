"""exp5c summary: alpha-phase gating vs input strength, every condition on one axis.

For each condition (current pulse into L4C / L6, LGN volley read out in L4C / L6) and
each dose: modulation of the evoked response (cosine fit on the responding layer's
own alpha phase, seed-centred) as % of the mean evoked response, plotted against the
mean evoked response itself -- so current and synaptic inputs share an x-axis.
Error bars: sem across seeds of the per-seed estimate. The spontaneous (sham)
modulation of each layer is drawn as a dashed line: the ceiling for a vanishing input.

Run:  ~/Desktop/venv/bin/python studies/phase_gating/plot_summary.py
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(_ROOT, "figures", "exp5_phase")
# response window in 5-ms bins (env EXP5C_WIN_BINS): 3 = 0-15 ms, 4 = 0-20 ms. The LGN
# burst in L4C peaks at 15-20 ms, and alpha phase shifts its LATENCY, so 0-15 ms
# inflates the LGN->L4C modulation; 0-20 ms is the conservative measure.
WIN_BINS = int(os.environ.get("EXP5C_WIN_BINS", 4))
CONDITIONS = [  # (label, folder, responding layer, colour, marker)
    ("current -> L4C", "exp5c_restore_pulse_L4C", "L4C", "#1f4e99", "o"),
    ("current -> L6", "exp5c_restore_pulse_L6", "L6", "#c0392b", "o"),
    ("LGN volley, L4C", "exp5c_restore_pulse_LGN", "L4C", "#1f4e99", "s"),
    ("LGN volley, L6", "exp5c_restore_pulse_LGN", "L6", "#c0392b", "s"),
]


def mod_pct(y, ph):
    X = np.column_stack([np.cos(ph), np.sin(ph)])
    X = X - X.mean(0)
    b, c = np.linalg.lstsq(X, y - y.mean(), rcond=None)[0]
    return 100 * np.hypot(b, c) / abs(y.mean())


def per_seed(folder, layer):
    files = sorted(glob.glob(os.path.join(_ROOT, "saved_trials", folder, "seed*.npz")))
    if not files:
        return None
    rows = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        li = list(z["layers"]).index(layer)
        hz = z["pulse_counts"][:, :, li, :WIN_BINS].sum(-1) / z["n_e"][li] / (WIN_BINS * 5e-3)
        ph = z[f"phase_{layer}"]
        sham = mod_pct(hz[:, 0], ph)
        evoked = [(np.mean(hz[:, d] - hz[:, 0]), mod_pct(hz[:, d] - hz[:, 0], ph))
                  for d in range(1, hz.shape[1])]
        rows.append((sham, evoked, z["doses_pa"][1:]))
    return rows


def main():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 5.5))
    lines = ["condition           dose    mean evoked Hz   modulation % (mean +- sem over seeds)"]
    sham_drawn = set()
    for label, folder, layer, col, mk in CONDITIONS:
        rows = per_seed(folder, layer)
        if rows is None:
            lines.append(f"{label}: no data")
            continue
        doses = rows[0][2]
        ev = np.array([[e[0] for e in r[1]] for r in rows])      # seeds x doses
        md = np.array([[e[1] for e in r[1]] for r in rows])
        x, y = ev.mean(0), md.mean(0)
        ye = md.std(0, ddof=1) / np.sqrt(len(rows))
        fill = "white" if mk == "s" else col
        ax.errorbar(x, y, yerr=ye, fmt=f"-{mk}", color=col, mfc=fill, mec=col, ms=7,
                    lw=1.5, capsize=3, label=f"{label} ({len(rows)} seeds)")
        for xi, yi, dz in zip(x, y, doses):
            ax.annotate(f"{int(dz)}", (xi, yi), textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color=col)
        if layer not in sham_drawn:
            s = np.mean([r[0] for r in rows])
            ax.axhline(s, color=col, lw=1, ls="--", alpha=0.6)
            ax.text(0.02, s + 0.6, f"{layer} spontaneous {s:.0f}%",
                    color=col, fontsize=8, transform=ax.get_yaxis_transform())
            sham_drawn.add(layer)
        for dz, xi, yi, e in zip(doses, x, y, ye):
            lines.append(f"{label:18s} {int(dz):5d}   {xi:10.2f}      {yi:5.1f} +- {e:.1f}")
    ax.set_xscale("log")
    ax.set_xlabel(f"mean evoked response of the responding layer, 0-{WIN_BINS * 5} ms (Hz, log)")
    ax.set_ylabel("alpha-phase modulation of the response\n(% of its mean, own-layer phase)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Alpha gates weak inputs, not strong ones "
                 "(numbers = dose: pA or LGN sources)", fontsize=11)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"exp5c_gating_vs_strength_0-{WIN_BINS * 5}ms.{ext}"), dpi=170,
                    bbox_inches="tight")
    report = "\n".join(lines)
    print(report)
    open(os.path.join(OUT, f"exp5c_gating_vs_strength_0-{WIN_BINS * 5}ms.txt"), "w").write(report + "\n")
    print("saved", os.path.join(OUT, f"exp5c_gating_vs_strength_0-{WIN_BINS * 5}ms.png"))


if __name__ == "__main__":
    main()
