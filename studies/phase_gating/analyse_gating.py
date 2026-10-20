"""exp5c: dose-dependent alpha-phase gating of a current pulse (store/restore design).

Data: saved_trials/exp5c_restore_pulse_<LAYER>/seed*.npz (run_pulses.py, formerly
experiments/exp5c_restore_pulse.py): onsets every 25 ms, at each onset doses DOSES_PA (0 pA =
sham from the same stored state), E spike counts per layer in 5-ms bins 0-40 ms;
alpha phase measured afterwards from the run's own unpulsed trajectory.

  evoked(onset, dose) = pulse counts - sham counts, 0-15 ms (target layer E, Hz/cell)

Per dose: y = seed + b cos(phi) + c sin(phi); modulation R in Hz and % of mean; null =
phases shuffled within seed (2000). Phase references: L6 E rate (the generator) and
the target layer's own E rate.
Positive control: the SHAM window itself (spontaneous E firing 0-15 ms) must show the
known alpha-phase modulation -- proves the phase estimate works.
Prediction (from the +-0.4 mV Vm swing): absolute modulation roughly dose-independent
(~0.2-0.3 Hz), so RELATIVE modulation largest for the weakest pulse.

Run:  EXP5C_LAYER=L4C ~/Desktop/venv/bin/python studies/phase_gating/analyse_gating.py
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LAYER = os.environ.get("EXP5C_LAYER", "L4C")        # layer whose response is analysed
# data folder: default = current pulses into LAYER; EXP5C_DIR=exp5c_restore_pulse_LGN
# for the thalamic volley (then LAYER only picks the responding layer)
DIRNAME = os.environ.get("EXP5C_DIR", f"exp5c_restore_pulse_{LAYER}")
D = os.path.join(_ROOT, "saved_trials", DIRNAME)
IS_LGN = DIRNAME.endswith("LGN")
UNIT = "LGN sources" if IS_LGN else "pA"
TAG = f"LGN_{LAYER}" if IS_LGN else LAYER
# (WIN_BINS is defined below; the window tag is appended in main())
OUT = os.path.join(_ROOT, "figures", "exp5_phase")
WIN_BINS = int(os.environ.get("EXP5C_WIN_BINS", 3))   # 5-ms bins: 3 = 0-15 ms
N_PERM = 2000
RNG = np.random.default_rng(11)


def load():
    files = sorted(glob.glob(os.path.join(D, "seed*.npz")))
    if not files:
        raise FileNotFoundError(D)
    rows = {k: [] for k in ("seed", "ph_L6", "ph_own", "amp", "sham", "pulse")}
    for f in files:
        z = np.load(f, allow_pickle=True)
        layers = list(z["layers"])
        li = layers.index(LAYER)
        n = int(z["n_e"][li])
        c = z["pulse_counts"][:, :, li, :WIN_BINS].sum(-1) / n / (WIN_BINS * 5e-3)
        rows["seed"].append(np.full(len(c), int(z["network_seed"])))
        rows["ph_L6"].append(z["phase_L6"])
        rows["ph_own"].append(z[f"phase_{LAYER}"] if f"phase_{LAYER}" in z.files
                              else z["phase_L6"])
        rows["amp"].append(z["amp_L6"])
        rows["sham"].append(c[:, 0])
        rows["pulse"].append(c[:, 1:])
        doses = z["doses_pa"][1:]
    out = {k: np.concatenate(v) for k, v in rows.items()}
    return out, doses


def centre(y, seeds):
    y = y.astype(float).copy()
    for s in np.unique(seeds):
        y[seeds == s] -= y[seeds == s].mean()
    return y


def cos_fit(yc, ph):
    X = np.column_stack([np.cos(ph), np.sin(ph)])
    X = X - X.mean(0)
    b, c = np.linalg.lstsq(X, yc, rcond=None)[0]
    return np.hypot(b, c), np.degrees(np.arctan2(c, b))


def shuffle(x, seeds):
    out = x.copy()
    for s in np.unique(seeds):
        i = np.flatnonzero(seeds == s)
        out[i] = x[RNG.permutation(i)]
    return out


def phase_test(y, ph, seeds):
    yc = centre(y, seeds)
    R, pref = cos_fit(yc, ph)
    null = np.array([cos_fit(yc, shuffle(ph, seeds))[0] for _ in range(N_PERM)])
    return R, pref, (1 + np.sum(null >= R)) / (N_PERM + 1), np.quantile(null, 0.95)


def binned(y, ph, seeds, width=30):
    yc = centre(y, seeds) + y.mean()
    d = np.degrees(ph)
    grid = np.arange(-180, 180, width)
    m, se = [], []
    for g in grid:
        sel = np.abs(np.angle(np.exp(1j * np.radians(d - g)))) <= np.radians(width / 2)
        m.append(yc[sel].mean()); se.append(yc[sel].std() / np.sqrt(sel.sum()))
    return np.r_[grid, 180], np.r_[m, m[0]], np.r_[se, se[0]]


def main():
    global TAG
    TAG = f"{TAG}_0-{WIN_BINS * 5}ms"
    os.makedirs(OUT, exist_ok=True)
    d, doses = load()
    seeds = d["seed"]
    refs = {"L6": d["ph_L6"]}
    if LAYER != "L6":
        refs[LAYER] = d["ph_own"]
    series = [(f"sham (spontaneous, 0 {UNIT})", d["sham"])] + \
             [(f"{int(q)} {UNIT} evoked", d["pulse"][:, i] - d["sham"]) for i, q in enumerate(doses)]
    lines = [f"exp5c {LAYER}: {len(seeds)} onsets, {len(np.unique(seeds))} seeds; "
             f"{LAYER} E, 0-{WIN_BINS * 5} ms; phase 0 = reference E-rate peak", "",
             "series                     mean Hz   sd   | " +
             " | ".join(f"ref {r}: mod Hz (% mean)  detect.  pref   p" for r in refs)]
    res = {}
    for name, y in series:
        out = {r: phase_test(y, ph, seeds) for r, ph in refs.items()}
        res[name] = (y, out)
        sd = centre(y, seeds).std()
        lines.append(f"{name:26s} {y.mean():6.2f}  {sd:5.2f} | " + " | ".join(
            f"{o[0]:5.2f} ({100 * o[0] / abs(y.mean()):5.1f}%)  {o[3]:5.2f}  {o[1]:+5.0f}  {o[2]:.4f}"
            for o in out.values()))
    report = "\n".join(lines)
    print(report)
    open(os.path.join(OUT, f"exp5c_analysis_{TAG}.txt"), "w").write(report + "\n")

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(len(refs), len(series), squeeze=False,
                             figsize=(3.6 * len(series), 3.75 * len(refs) + 0.5),
                             gridspec_kw=dict(hspace=0.5, wspace=0.35))
    for row, (ref, ph) in enumerate(refs.items()):
        for col, (name, _) in enumerate(series):
            y, out = res[name]
            ax = axes[row, col]
            x, m, se = binned(y, ph, seeds)
            ax.fill_between(x, m - se, m + se, color="0.8", lw=0)
            ax.plot(x, m, "k-o", ms=3.5, lw=1.8)
            R, pref, p, mde = out[ref]
            ax.set_xticks([-180, -90, 0, 90, 180])
            ax.set_title(f"{name}\nmod {R:.2f} Hz ({100 * R / abs(y.mean()):.0f}%), p={p:.3g}",
                         fontsize=9, color="k" if p < 0.05 else "0.45")
            if col == 0:
                ax.set_ylabel(f"{LAYER} E rate 0-{WIN_BINS * 5} ms (Hz)\nphase ref: {ref} E rate")
            if row == len(refs) - 1:
                ax.set_xlabel("alpha phase at onset (deg)\n0 = rate peak")
    what = "an LGN volley" if IS_LGN else "a 10 ms current pulse"
    fig.suptitle(f"exp5c: {LAYER} response to {what} vs alpha phase, by dose "
                 f"({len(seeds)} onsets, mean ± sem, 30° bins; seed-centred)", fontsize=11)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"exp5c_phase_by_dose_{TAG}.{ext}"), dpi=160,
                    bbox_inches="tight")
    print("saved", os.path.join(OUT, f"exp5c_phase_by_dose_{TAG}.png"))


if __name__ == "__main__":
    main()
