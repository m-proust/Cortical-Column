"""Alpha-phase gating of excitability: many weak pulses per run, via store()/restore().

Cosyne abstract Fig. 3B (L4C response to a 25 pA pulse vs the local alpha phase, split by alpha
amplitude) comes from the ten networks of the first command below.

    caffeinate -i ~/Desktop/venv/bin/python studies/phase_gating/run_pulses.py \
        --seeds 58881 20260923 777001 31415 27182 101 202 303 404 505 --out saved_trials/cosyne/pulse_gating_L4C
    caffeinate -i ~/Desktop/venv/bin/python studies/phase_gating/run_pulses.py --layer L6     # current into L6 E
    caffeinate -i ~/Desktop/venv/bin/python studies/phase_gating/run_pulses.py --input lgn    # thalamic volley

Design, per seed (one continuous run, no targeting):
  onsets on a regular grid (every ONSET_STEP_MS after SETTLE_MS): the alpha phase at each is effectively
  random and is measured AFTERWARDS (non-causal 7-16 Hz Hilbert) from the run's own unpulsed trajectory,
  as in human phase studies
  at each onset: store() -> for each dose (0 = sham): input on, run to +WIN_MS, read spikes, restore()
  -> continue to the next onset
RNG: store()/restore() does not rewind Brian's random numbers, so every dose of one onset starts from the
same state but gets its own noise; hence the per-onset sham and the post-hoc phase instead of probe-based
targeting. Only E spike monitors are kept (big monitors would be copied by every store()).

Inputs
  current  10 ms current step into the target layer's E cells; doses in pA
  lgn      a thalamic volley: N_LGN sources, each firing ONE spike at a fixed per-source offset
           (0-LGN_JITTER_MS); dose = number of sources that fire. Wiring from the documented LGN
           design (project notes): L4C E p .12 w 1x, L4C PV p .22 w 2.5x, L6 E p .04 w 1.5x, L6 PV p .07 w 3x
           (w = EXT_AMPA), 1 ms delay. Deterministic, and it recruits PV feedforward inhibition.

Saved per seed (<out>/seed<seed>.npz): onsets, L6 / L4C phase and amplitude at each onset, and per
onset x dose x layer the E spike counts in 5-ms bins 0..WIN_MS. Analysis: analyse_gating.py,
plot_summary.py, plot_key_figure.py, phase_x_amplitude.py.
"""
import argparse
import os
import shutil
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
import numpy as np
import brian2 as b2
from brian2 import ms, pA, SpikeGeneratorGroup, Synapses
from scipy.signal import butter, sosfiltfilt, hilbert
from config.config import CONFIG
from src.column import CorticalColumn
from src.analysis import add_heterogeneity_to_layer

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--input", choices=["current", "lgn"], default="current")
ap.add_argument("--layer", default="L4C", help="target layer of the current pulse (ignored for lgn)")
ap.add_argument("--seeds", type=int, nargs="+", default=[58881, 20260923, 777001, 31415, 27182])
ap.add_argument("--out", help="output folder, relative to the project root "
                              "(default saved_trials/exp5c_restore_pulse_<LAYER> or _LGN)")
ap.add_argument("--max-onsets", type=int, help="cap on onsets per seed, for tests")

# parameters
DOSES_PA      = [0, 10, 25, 50, 100]    # current input; 0 = sham from the same stored state
LGN_DOSES     = [0, 10, 20, 40, 80]     # lgn input: active sources; 0 = sham
N_LGN         = 80
LGN_JITTER_MS = 5.0
LGN_TARGETS   = [("L4C", "E", 0.12, 1.0), ("L4C", "PV", 0.22, 2.5),
                 ("L6", "E", 0.04, 1.5), ("L6", "PV", 0.07, 3.0)]
TARGET_POPS   = ("E",)
PULSE_MS      = 10
WIN_MS        = 40              # recorded after each pulse (5-ms bins)
PROBE_MS      = 6000
SETTLE_MS     = 1500
ONSET_STEP_MS = 25              # onset grid (random alpha phase at each)
LAYERS        = ["L23", "L4AB", "L4C", "L5", "L6"]
N_E           = {L: CONFIG["layers"][L]["neuron_counts"]["E"] for L in LAYERS}
BINS          = np.arange(0, WIN_MS + 1, 5)
CONFIG_FILES  = ["config/config.py", "config/conductances_AMPA_GABA.csv", "config/conductances_NMDA.csv",
                 "config/connection_probabilities.csv", "studies/phase_gating/run_pulses.py"]


def save_config_snapshot(save_dir):
    snapshot_dir = os.path.join(save_dir, "config_snapshot")
    for rel_path in CONFIG_FILES:
        src = os.path.join(_ROOT, rel_path)
        if os.path.exists(src):
            dst = os.path.join(snapshot_dir, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    print(f"Config snapshot saved to {snapshot_dir}")


def build_column(cfg, network_seed):
    np.random.seed(network_seed)
    b2.seed(network_seed)
    b2.start_scope()
    b2.defaultclock.dt = cfg['simulation']['DT']
    column = CorticalColumn(column_id=0, config=cfg)
    for ln, layer in column.layers.items():
        add_heterogeneity_to_layer(layer, cfg)
    return column


def e_spikes(column):
    return {L: (np.array(column.layers[L].monitors["E_spikes"].t / ms),
                np.array(column.layers[L].monitors["E_spikes"].i)) for L in LAYERS}


def grid_onsets():
    """Regular grid: alpha phase at each onset is effectively random and is measured
    afterwards from the pass's own unpulsed trajectory."""
    return list(np.arange(SETTLE_MS, PROBE_MS - WIN_MS - 60, ONSET_STEP_MS))


def pulse_pass(seed, onsets, inp, target_layer, doses):
    column = build_column(CONFIG, seed)
    net = column.network
    # keep only the E spike monitors: nothing else is read, and big monitors would be
    # copied by every store()/restore()
    for L, layer in column.layers.items():
        for name, mon in list(layer.monitors.items()):
            if name != "E_spikes" and mon in net.objects:
                net.remove(mon)
    groups = [column.layers[target_layer].neuron_groups[p] for p in TARGET_POPS]
    if inp == "lgn":
        gen = SpikeGeneratorGroup(N_LGN, [], [] * ms, name="lgn")
        w = CONFIG["synapses"]["Q"]["EXT_AMPA"]
        lgn_syn = []
        for L, pop, prob, scale in LGN_TARGETS:
            sy = Synapses(gen, column.layers[L].neuron_groups[pop],
                          on_pre=f"gE_AMPA_post += {float(w * scale / (1e-9 * 1)):.6f}*nS",
                          delay=1 * ms, name=f"lgn_{L}_{pop}")
            sy.connect(p=prob)
            lgn_syn.append(sy)
        net.add(gen, *lgn_syn)
        # fixed per-seed source order and offsets (own RNG: leaves Brian's untouched)
        lrng = np.random.RandomState(seed + 99)
        order = lrng.permutation(N_LGN)
        offsets = lrng.uniform(0, LGN_JITTER_MS, N_LGN)
    counts = np.zeros((len(onsets), len(doses), len(LAYERS), len(BINS) - 1), np.int32)
    t_now = 0.0
    for k, on in enumerate(onsets):
        net.run((on - t_now) * ms)
        t_now = on
        # store()/restore() does not rewind the random numbers, so every dose of this
        # onset (including the 0 pA sham) starts from the SAME state but gets its own
        # noise. (Overwriting the cython rand buffers to share the noise crashes
        # Brian2 -- do not.)
        net.store("pre")
        np_state = np.random.get_state()
        for di, dose in enumerate(doses):
            if inp == "lgn":
                src = np.sort(order[:int(dose)])
                if src.size:
                    gen.set_spikes(src, (on + offsets[src]) * ms)
                net.run(WIN_MS * ms)
            else:
                for g in groups:
                    g.I = dose * pA
                net.run(PULSE_MS * ms)
                for g in groups:
                    g.I = 0 * pA
                net.run((WIN_MS - PULSE_MS) * ms)
            sp = e_spikes(column)
            for li, L in enumerate(LAYERS):
                t = sp[L][0]
                counts[k, di, li] = np.histogram(t[t >= on] - on, BINS)[0]
            net.restore("pre")
            np.random.set_state(np_state)
        if k % 25 == 0:
            print(f"    onset {k}/{len(onsets)} at {on:.0f} ms", flush=True)
    net.run((PROBE_MS - t_now) * ms)
    return counts, e_spikes(column)


def phase_from_spikes(t_spk, n_ms):
    """Non-causal 7-16 Hz phase/amplitude of a population's E rate (1 ms bins)."""
    x = np.bincount(t_spk.astype(int), minlength=n_ms)[:n_ms].astype(float)
    sos = butter(4, (7.0, 16.0), btype="bandpass", fs=1000.0, output="sos")
    a = hilbert(sosfiltfilt(sos, x - x.mean()))
    return np.angle(a), np.abs(a)


def main(a):
    doses = LGN_DOSES if a.input == "lgn" else DOSES_PA
    out = a.out or (f"saved_trials/exp5c_restore_pulse_{a.layer}" if a.input == "current"
                    else "saved_trials/exp5c_restore_pulse_LGN")
    save = os.path.join(_ROOT, out)
    os.makedirs(save, exist_ok=True)
    save_config_snapshot(save)
    for seed in a.seeds:
        fname = os.path.join(save, f"seed{seed}.npz")
        if os.path.exists(fname):
            print(f"skip (exists) {fname}", flush=True)
            continue
        t0 = time.time()
        onsets = grid_onsets()
        if a.max_onsets:
            onsets = onsets[:a.max_onsets]
        print(f"\n##### seed {seed}: {len(onsets)} onsets x {len(doses)} doses #####", flush=True)
        counts, traj = pulse_pass(seed, onsets, a.input, a.layer, doses)
        # phases from the pass's own unpulsed trajectory (pulse windows were restored away)
        on = np.array(onsets)
        idx = np.round(on).astype(int)
        ph6, amp6 = phase_from_spikes(traj["L6"][0], int(PROBE_MS))
        ph4, amp4 = phase_from_spikes(traj["L4C"][0], int(PROBE_MS))
        np.savez_compressed(
            fname, network_seed=seed, onsets_ms=on, doses_pa=np.array(doses),
            phase_L6=ph6[idx], amp_L6=amp6[idx], phase_L4C=ph4[idx], amp_L4C=amp4[idx],
            layers=np.array(LAYERS), n_e=np.array([N_E[L] for L in LAYERS]),
            bins_ms=BINS, pulse_counts=counts, target_layer=a.layer, input=a.input,
            pulse_ms=PULSE_MS,
            traj_L6_E_t=traj["L6"][0].astype(np.float32),
            traj_L4C_E_t=traj["L4C"][0].astype(np.float32))
        print(f"  saved {fname}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main(ap.parse_args())
