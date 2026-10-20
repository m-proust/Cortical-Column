"""exp5c: does alpha-phase gating get DEEPER as pre-stimulus alpha amplitude grows?

Pulsed inhibition / the duty-cycle account (Jensen & Mazaheri 2010; Bonnefond &
Jensen) predicts a phase x amplitude interaction on the evoked response. Two readings:

  linear rhythm      modulation (Hz) grows with amplitude, symmetrically:
                     inhibitory-window response FALLS, permissive-window RISES by as much
  pulsed inhibition  inhibitory-window response falls, permissive-window stays FLAT
                     (so the mean evoked response falls with amplitude)

Data: saved_trials/exp5c_restore_pulse_{L4C,L6,LGN} (store/restore, 176 onsets / seed).
Response = pulse - sham E rate in the target layer, 0-20 ms (the conservative window).
Amplitude = PRE-STIMULUS mean alpha envelope, -100..0 ms, recomputed from the run's own
unpulsed trajectory (same 7-16 Hz filter as the phase), / seed median; phase at onset.
Everything is within seed: y centred per seed, amplitude as within-seed rank z in
[-0.5, 0.5] (so terciles and the slope are seed-balanced).

Per dose and phase reference (L6 E rate; target layer's own E rate):
  1. cosine modulation R (Hz, % of tercile mean) in low / mid / high amplitude terciles
  2. interaction: yc ~ cos + sin + z + z*cos + z*sin; "deepening" = (z*cos, z*sin)
     projected on the main-effect direction = change in R (Hz) from lowest to highest
     amplitude. Null: z permuted within seed (2000).
  3. windows +-45 deg around the preferred / anti-preferred phase (from the pooled fit,
     amplitude-blind): slope of response on z in each. Pulsed: inh < 0, perm ~ 0.
Control: phase reliability per tercile (split-half: E spikes of the trajectory thinned
into two random halves, concordance mean cos(phi1 - phi2)). Low-amplitude phase is
noisier, which by itself attenuates low-tercile modulation -- R is reported both raw
and divided by sqrt(concordance) (the attenuation of the full-data phase, approx.).

Run:  ~/Desktop/venv/bin/python studies/phase_gating/phase_x_amplitude.py
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(_ROOT, "figures", "exp5_phase")
WIN_BINS = 4                     # 5-ms bins: 0-20 ms
PRE_MS = 100
PROBE_MS = 6000
N_PERM = 2000
RNG = np.random.default_rng(20260928)
_SOS = butter(4, (7.0, 16.0), btype="bandpass", fs=1000.0, output="sos")

# (label, folder, responding layer)
DATASETS = [("current -> L4C", "exp5c_restore_pulse_L4C", "L4C"),
            ("current -> L6", "exp5c_restore_pulse_L6", "L6"),
            ("LGN -> L4C", "exp5c_restore_pulse_LGN", "L4C"),
            ("LGN -> L6", "exp5c_restore_pulse_LGN", "L6")]


def analytic(t_spk):
    x = np.bincount(t_spk.astype(int), minlength=PROBE_MS)[:PROBE_MS].astype(float)
    return hilbert(sosfiltfilt(_SOS, x - x.mean()))


def load(folder, layer):
    rows = {k: [] for k in ("seed", "sham", "pulse", "ph_L6", "ph_own", "amp_L6",
                            "amp_own", "rel_L6", "rel_own")}
    for f in sorted(glob.glob(os.path.join(_ROOT, "saved_trials", folder, "seed*.npz"))):
        z = np.load(f, allow_pickle=True)
        li = list(z["layers"]).index(layer)
        c = z["pulse_counts"][:, :, li, :WIN_BINS].sum(-1) / int(z["n_e"][li]) \
            / (WIN_BINS * 5e-3)
        on = np.round(z["onsets_ms"]).astype(int)
        rows["seed"].append(np.full(len(on), int(z["network_seed"])))
        rows["sham"].append(c[:, 0])
        rows["pulse"].append(c[:, 1:])
        own = "L6" if layer == "L6" else "L4C"
        for ref, key in (("L6", "L6"), (own, "own")):
            t = z[f"traj_{ref}_E_t"].astype(float)
            a = analytic(t)
            ph = np.angle(a)[on]
            assert np.allclose(np.cos(ph - z[f"phase_{ref}"]), 1, atol=1e-3), "phase mismatch"
            env = np.abs(a)
            pre = np.array([env[o - PRE_MS:o].mean() for o in on])
            rows["ph_" + key].append(ph)
            rows["amp_" + key].append(pre / np.median(pre))
            # split-half phase reliability at each onset
            half = RNG.random(t.size) < 0.5
            d = np.angle(analytic(t[half])[on]) - np.angle(analytic(t[~half])[on])
            rows["rel_" + key].append(np.cos(d))
        doses = z["doses_pa"][1:]
    return {k: np.concatenate(v) for k, v in rows.items()}, doses


def centre(y, seeds):
    y = y.astype(float).copy()
    for s in np.unique(seeds):
        y[seeds == s] -= y[seeds == s].mean()
    return y


def within_rank(a, seeds):
    z = np.empty_like(a)
    for s in np.unique(seeds):
        i = np.flatnonzero(seeds == s)
        z[i] = (np.argsort(np.argsort(a[i])) + 0.5) / i.size - 0.5
    return z


def perm_within(x, seeds):
    out = x.copy()
    for s in np.unique(seeds):
        i = np.flatnonzero(seeds == s)
        out[i] = x[RNG.permutation(i)]
    return out


def cos_R(yc, ph):
    X = np.column_stack([np.cos(ph), np.sin(ph)])
    X = X - X.mean(0)
    b, c = np.linalg.lstsq(X, yc - yc.mean(), rcond=None)[0]
    return np.hypot(b, c), np.arctan2(c, b)


def interaction(yc, ph, z):
    X = np.column_stack([np.cos(ph), np.sin(ph), z, z * np.cos(ph), z * np.sin(ph)])
    X = X - X.mean(0)
    b = np.linalg.lstsq(X, yc - yc.mean(), rcond=None)[0]
    u = b[:2] / np.hypot(*b[:2])
    return float(b[3:5] @ u)          # dR/dz, Hz per full amplitude range


def analyse(y, ph, z, seeds, rel):
    yc = centre(y, seeds)
    R0, pref = cos_R(yc, ph)
    terc = np.digitize(z, [-1 / 6, 1 / 6])
    out = dict(R0=R0, pref=np.degrees(pref), mean=y.mean(), R=[], pct=[], p=[], Rcorr=[],
               rel=[])
    for k in range(3):
        s = terc == k
        R, _ = cos_R(yc[s], ph[s])
        null = [cos_R(yc[s], perm_within(ph[s], seeds[s]))[0] for _ in range(N_PERM // 4)]
        rk = max(rel[s].mean(), 1e-3)
        out["R"].append(R)
        out["pct"].append(100 * R / abs(y[s].mean()))
        out["p"].append((1 + np.sum(np.array(null) >= R)) / (len(null) + 1))
        out["rel"].append(rk)
        out["Rcorr"].append(R / np.sqrt(rk))
    d = interaction(yc, ph, z)
    null = np.array([interaction(yc, ph, perm_within(z, seeds)) for _ in range(N_PERM)])
    out["deep"] = d
    out["deep_p"] = (1 + np.sum(np.abs(null - null.mean()) >= abs(d - null.mean()))) \
        / (N_PERM + 1)
    out["deep_null_mean"] = null.mean()
    # windows around pref / anti-pref (amplitude-blind)
    dist = lambda c: np.abs(np.angle(np.exp(1j * (ph - c))))
    for name, c in (("perm", pref), ("inh", pref + np.pi)):
        s = dist(c) <= np.pi / 4
        zz, yy = z[s], yc[s]
        sl = np.polyfit(zz, yy, 1)[0]
        nul = np.array([np.polyfit(perm_within(zz, seeds[s]), yy, 1)[0]
                        for _ in range(N_PERM // 4)])
        out[f"sl_{name}"] = sl
        out[f"z_{name}"] = (sl - nul.mean()) / nul.std()
    sl = np.polyfit(z, yc, 1)[0]
    nul = np.array([np.polyfit(perm_within(z, seeds), yc, 1)[0] for _ in range(N_PERM // 4)])
    out["sl_all"], out["z_all"] = sl, (sl - nul.mean()) / nul.std()
    out["terc"] = terc
    return out


def curve(y, ph, seeds, sel, width=60, step=15):
    yc = centre(y, seeds) + y.mean()
    grid = np.arange(-180, 181, step)
    m, se = [], []
    for g in grid:
        s = sel & (np.abs(np.angle(np.exp(1j * (ph - np.radians(g))))) <= np.radians(width / 2))
        m.append(yc[s].mean()); se.append(yc[s].std() / np.sqrt(s.sum()))
    return grid, np.array(m), np.array(se)


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = [f"exp5c phase x PRE-STIMULUS amplitude, response 0-{WIN_BINS * 5} ms "
             f"(pulse - sham), amplitude = mean envelope -{PRE_MS}..0 ms",
             "R = cosine modulation (Hz); deep = change in R from lowest to highest "
             "amplitude (Hz), p vs amplitude permuted within seed",
             "window slopes = Hz per full amplitude range, +-45 deg around pref (perm) / "
             "anti-pref (inh). Pulsed: inh<0, perm~0. Linear: inh<0, perm>0 equal size.",
             "rel = split-half phase concordance per tercile (low -> high)", ""]
    plots = []
    for label, folder, layer in DATASETS:
        d, doses = load(folder, layer)
        seeds = d["seed"]
        unit = "LGN" if "LGN" in folder else "pA"
        series = [("sham (spontaneous)", d["sham"])] + \
                 [(f"{int(q)} {unit}", d["pulse"][:, i] - d["sham"]) for i, q in enumerate(doses)]
        refs = [("L6", "L6")] + ([] if layer == "L6" else [(layer, "own")])
        for refname, key in refs:
            ph, z = d["ph_" + key], within_rank(d["amp_" + key], seeds)
            lines.append(f"== {label}  ({len(seeds)} onsets, {len(np.unique(seeds))} seeds)"
                         f"   phase/amplitude ref: {refname} E rate")
            lines.append("  series              mean   R lo / mid / hi (Hz)    % of mean lo/mid/hi"
                         "     R/sqrt(rel) lo->hi   deep (p)          inh-win z  perm-win z  all z")
            for name, y in series:
                o = analyse(y, ph, z, seeds, d["rel_" + key])
                lines.append(
                    f"  {name:18s} {o['mean']:5.2f}   "
                    f"{o['R'][0]:.3f} {o['R'][1]:.3f} {o['R'][2]:.3f}      "
                    f"{o['pct'][0]:5.1f} {o['pct'][1]:5.1f} {o['pct'][2]:5.1f}      "
                    f"{o['Rcorr'][0]:.3f} -> {o['Rcorr'][2]:.3f}     "
                    f"{o['deep']:+.3f} ({o['deep_p']:.3f})    "
                    f"{o['sl_inh']:+.3f} z{o['z_inh']:+4.1f}  {o['sl_perm']:+.3f} z{o['z_perm']:+4.1f}"
                    f"  z{o['z_all']:+4.1f}")
                if refname == layer or (layer == "L6" and refname == "L6"):
                    plots.append((label, refname, name, y, ph, seeds, o))
            rel = [d["rel_" + key][np.digitize(z, [-1 / 6, 1 / 6]) == k].mean() for k in range(3)]
            lines.append(f"  phase reliability lo/mid/hi: {rel[0]:.2f} {rel[1]:.2f} {rel[2]:.2f}"
                         f"   (amp lo/hi median: "
                         f"{np.median(d['amp_' + key][z < -1 / 6]):.2f} / "
                         f"{np.median(d['amp_' + key][z > 1 / 6]):.2f} x seed median)")
            lines.append("")
    report = "\n".join(lines)
    print(report)
    open(os.path.join(OUT, "exp5c_phase_x_amplitude.txt"), "w").write(report + "\n")

    # figure: rows = dataset (own-layer reference), cols = sham + doses
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 8})
    rows = [lab for lab, *_ in DATASETS]
    ncol = max(sum(1 for p in plots if p[0] == r) for r in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(3.3 * ncol, 2.8 * len(rows)),
                             squeeze=False, gridspec_kw=dict(hspace=0.75, wspace=0.35))
    cols = ["#9ecae1", "#4292c6", "#08306b"]
    for r, lab in enumerate(rows):
        ps = [p for p in plots if p[0] == lab]
        for c, (_, ref, name, y, ph, seeds, o) in enumerate(ps):
            ax = axes[r, c]
            for k, tl in enumerate(("low", "mid", "high")):
                g, m, se = curve(y, ph, seeds, o["terc"] == k)
                ax.fill_between(g, m - se, m + se, color=cols[k], alpha=0.2, lw=0)
                ax.plot(g, m, color=cols[k], lw=1.6, label=f"{tl} alpha")
            ax.set_xticks([-180, -90, 0, 90, 180])
            ax.set_title(f"{lab} | {name}\nR {o['R'][0]:.2f}/{o['R'][1]:.2f}/{o['R'][2]:.2f} Hz, "
                         f"deep p={o['deep_p']:.3f}", fontsize=7.5,
                         color="k" if o["deep_p"] < 0.05 else "0.45")
            if c == 0:
                ax.set_ylabel(f"E rate 0-{WIN_BINS * 5} ms (Hz)\nphase: {ref} E")
            if r == len(rows) - 1:
                ax.set_xlabel("alpha phase at onset (deg), 0 = rate peak")
        for c in range(len(ps), ncol):
            axes[r, c].axis("off")
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    fig.suptitle("Does phase gating deepen with pre-stimulus alpha amplitude? "
                 "(terciles within seed; 60° sliding bins, mean ± sem)", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"exp5c_phase_x_amplitude.{ext}"), dpi=160,
                    bbox_inches="tight")
    print("saved", os.path.join(OUT, "exp5c_phase_x_amplitude.png"))


if __name__ == "__main__":
    main()
