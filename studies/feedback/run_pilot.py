"""Isolated pilot: L6 E->SOM attenuation and LGN input discrimination.

Uses an immutable copy of the user's model. Each invocation is one network.
All branch trials restore network and reseed Brian with the same onset-specific
seed. Native RNG restore failed exact-sham validation with this Cython runtime.
No manual RNG buffers. The continuing sham trajectory uses that seed as well.
Pilot attenuation is UNCOMPENSATED; operating-point shifts must be checked first.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model"
# only the model snapshot (model/) and the helpers below are used by run_feedback.py
OUT = HERE / "pilot_outputs"
os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache" / "mpl_feedback_protocol"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ["LESION_PAIR"] = "NONE"
sys.dont_write_bytecode = True
sys.path.insert(0, str(MODEL))
os.chdir(MODEL)

import numpy as np
import brian2 as b2
from scipy.signal import butter, sosfiltfilt, hilbert
from config.config import CONFIG
from src.column import CorticalColumn
from src.analysis import add_heterogeneity_to_layer

b2.prefs.codegen.target = "cython"
b2.prefs.codegen.runtime.cython.cache_dir = str(Path.home() / "Library" / "Caches" / "cython" / "feedback_protocol")
b2.BrianLogger.log_level_warn()
LAYERS = list(CONFIG["layers"])
POPS = ["E", "PV", "SOM", "VIP"]
BINS = np.arange(0, 61, 5)
LGN_TARGETS = [("L4C", "E", .12, 1.), ("L4C", "PV", .22, 2.5),
               ("L6", "E", .04, 1.5), ("L6", "PV", .07, 3.)]


def atomic_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def fingerprint(column, lgn_syn):
    h = hashlib.sha256()
    for L in LAYERS:
        layer = column.layers[L]
        for p in POPS:
            g = layer.neuron_groups[p]
            for var in ["v", "C", "gL", "tauw", "a", "b", "EL", "DeltaT"]:
                h.update(np.asarray(getattr(g, var)[:]).tobytes())
        for key in sorted(layer.synapses):
            sy = layer.synapses[key]
            for var in ["i", "j", "delay"]:
                h.update(np.asarray(getattr(sy, var)[:]).tobytes())
    for sy in list(column.inter_layer_synapses.values()) + lgn_syn:
        for var in ["i", "j", "delay"]:
            h.update(np.asarray(getattr(sy, var)[:]).tobytes())
    return h.hexdigest()


def snapshot_end(column):
    # Strong sham reproducibility check includes spike identity/time AND state.
    h = hashlib.sha256()
    for L in LAYERS:
        for p in POPS:
            g = column.layers[L].neuron_groups[p]
            for var in ["v", "w", "gE_AMPA", "gE_NMDA", "gPV", "gSOM", "gVIP"]:
                x = np.asarray(getattr(g, var)[:])
                if not np.isfinite(x).all():
                    raise RuntimeError(f"Nonfinite {L}/{p}/{var}")
                h.update(x.tobytes())
            mon = column.layers[L].monitors[p + "_spikes"]
            h.update(np.asarray(mon.t[:]).tobytes())
            h.update(np.asarray(mon.i[:]).tobytes())
    return h.hexdigest()


def response_counts(column, onset):
    result = np.zeros((len(LAYERS), len(BINS)-1), dtype=np.int32)
    for li, L in enumerate(LAYERS):
        ts = np.asarray(column.layers[L].monitors["E_spikes"].t / b2.ms)
        ts = ts[np.searchsorted(ts, onset):] - onset
        result[li] = np.histogram(ts, bins=BINS)[0]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=910081)
    parser.add_argument("--scale", type=float, default=1.)
    parser.add_argument("--onsets", type=int, default=24)
    parser.add_argument("--doses", default="0,10,20,30")
    parser.add_argument("--tag", default="pilot")
    parser.add_argument("--som-current-pa", type=float, default=0.)
    args = parser.parse_args()
    assert 0 <= args.scale <= 1 and args.onsets >= 1
    doses = np.array([int(v) for v in args.doses.split(",")])
    assert doses[0] == 0 and np.all((doses >= 0) & (doses <= 80))
    stem = f"{args.tag}_seed{args.seed}_scale{args.scale:g}_I{args.som_current_pa:g}"
    dest = OUT / "data" / (stem + ".npz")
    status_path = OUT / "logs" / (stem + ".json")
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite {dest}")
    t0 = time.time()
    status = dict(vars(args), status="building", started_unix=t0, pid=os.getpid())
    atomic_json(status_path, status)
    print(json.dumps(status), flush=True)
    cfg = copy.deepcopy(CONFIG)
    for key in ["E_SOM_AMPA", "E_SOM_NMDA"]:
        if key in cfg["layers"]["L6"]["conductance"]:
            cfg["layers"]["L6"]["conductance"][key] *= args.scale
    b2.start_scope()
    b2.seed(args.seed)
    b2.defaultclock.dt = cfg["simulation"]["DT"]
    column = CorticalColumn(column_id=0, config=cfg)
    for layer in column.layers.values():
        add_heterogeneity_to_layer(layer, cfg)
    net = column.network
    for layer in column.layers.values():
        for name, mon in layer.monitors.items():
            if not name.endswith("_spikes"):
                net.remove(mon)
    column.layers["L6"].neuron_groups["SOM"].I = args.som_current_pa * b2.pA
    gen = b2.SpikeGeneratorGroup(80, [], [] * b2.ms, name="pilot_lgn")
    lgn_syn = []
    for L, p, prob, scale in LGN_TARGETS:
        w_ns = float(cfg["synapses"]["Q"]["EXT_AMPA"] / b2.nS) * scale
        sy = b2.Synapses(gen, column.layers[L].neuron_groups[p],
                         on_pre=f"gE_AMPA_post += {w_ns:.9g}*nS",
                         delay=1*b2.ms, name=f"pilot_lgn_{L}_{p}")
        sy.connect(p=prob)
        lgn_syn.append(sy)
    net.add(gen, *lgn_syn)
    wiring_digest = fingerprint(column, lgn_syn)
    rng = np.random.default_rng(args.seed + 99)
    order = rng.permutation(80)
    offsets = rng.uniform(0, 5, 80)
    onsets = 1500 + np.arange(args.onsets) * 137.
    end_ms = max(3000., onsets[-1] + 750.)
    counts = np.zeros((args.onsets, len(doses), len(LAYERS), len(BINS)-1), np.int32)
    status.update(status="simulating", wiring_digest=wiring_digest,
                  build_seconds=time.time()-t0, total_onsets=args.onsets)
    atomic_json(status_path, status)
    print(f"Built full column in {time.time()-t0:.1f}s; digest={wiring_digest}", flush=True)
    net.run(onsets[0] * b2.ms)
    print(f"Settled at 1500 ms after {time.time()-t0:.1f}s wall time", flush=True)
    sham_checks = []
    for k, onset in enumerate(onsets):
        now = float(net.t / b2.ms)
        if onset > now + 1e-6:
            net.run((onset-now)*b2.ms)
        net.store("pre")
        onset_seed = int(np.random.SeedSequence([args.seed, k, 107]).generate_state(1)[0])
        def restore_branch():
            net.restore("pre", restore_random_state=False)
            b2.seed(onset_seed)
        # Two exact sham replicas at first and final onset across full 60ms.
        if k in {0, len(onsets)-1}:
            signatures = []
            for rep in range(2):
                restore_branch()
                gen.set_spikes([], [] * b2.ms)
                net.run(60*b2.ms)
                signatures.append(snapshot_end(column))
            if signatures[0] != signatures[1]:
                raise RuntimeError("Exact full-window sham reproducibility FAILED")
            sham_checks.append(dict(onset_ms=float(onset), identical=True,
                                    signature=signatures[0]))
            print(f"Exact sham check passed at onset {k} (60 ms, all populations)", flush=True)
        for di, dose in enumerate(doses):
            restore_branch()
            src = np.sort(order[:dose])
            gen.set_spikes(src, (onset + offsets[src]) * b2.ms)
            net.run(60*b2.ms)
            counts[k, di] = response_counts(column, onset)
        restore_branch()
        # Advance along the unperturbed path, not along any pulse branch.
        gen.set_spikes([], [] * b2.ms)
        if k in {0, len(onsets)-1}:
            net.run(60*b2.ms)
            if snapshot_end(column) != signatures[0]:
                raise RuntimeError("Continuing unpulsed trajectory differs from matched sham")
            sham_checks[-1]["continuation_identical"] = True
        status.update(status="simulating", completed_onsets=k+1,
                      elapsed_seconds=time.time()-t0, sham_checks=sham_checks)
        atomic_json(status_path, status)
        np.savez_compressed(OUT / "data" / (stem + "_partial.npz"),
                            completed_onsets=k+1, pulse_counts=counts[:k+1],
                            doses=doses, onsets_ms=onsets[:k+1])
        print(f"Onset {k+1}/{len(onsets)} complete; elapsed {time.time()-t0:.1f}s", flush=True)
    net.run((end_ms - float(net.t / b2.ms))*b2.ms)
    duration = int(round(end_ms))
    rate_traces = np.zeros((len(LAYERS),len(POPS),duration),np.float32)
    payload = {}
    baseline_rates = {}
    for li,L in enumerate(LAYERS):
        baseline_rates[L] = {}
        for pi,p in enumerate(POPS):
            mon = column.layers[L].monitors[p+"_spikes"]
            ts = np.asarray(mon.t / b2.ms)
            n = len(column.layers[L].neuron_groups[p])
            rate_traces[li,pi] = np.bincount(ts.astype(int),minlength=duration)[:duration]*1000/n
            baseline_rates[L][p] = float(rate_traces[li,pi,1500:].mean())
        x = rate_traces[li,0].astype(float)
        a = hilbert(sosfiltfilt(butter(4,[7,16],fs=1000,btype="bandpass",output="sos"), x-x.mean()))
        idx = onsets.astype(int)
        payload[f"phase_{L}"] = np.angle(a)[idx]
        payload[f"amp_{L}"] = np.abs(a)[idx]
    np.savez_compressed(dest, **payload, pulse_counts=counts, doses=doses,
                        onsets_ms=onsets, layers=np.array(LAYERS), pops=np.array(POPS),
                        n_e=np.array([cfg['layers'][L]['neuron_counts']['E'] for L in LAYERS]),
                        bins_ms=BINS, baseline_rates_hz=rate_traces,
                        network_seed=args.seed, e_som_scale=args.scale,
                        som_current_pa=args.som_current_pa, wiring_digest=wiring_digest,
                        metadata_json=json.dumps(dict(args=vars(args), sham_checks=sham_checks,
                            branch_rng="Brian seed(onset_seed), shared across doses and sham continuation", phase="posthoc unpulsed 7-16Hz",
                            brian_version=b2.__version__, baseline_rates=baseline_rates,
                            elapsed_seconds=time.time()-t0)))
    status.update(status="complete", elapsed_seconds=time.time()-t0,
                  output=str(dest), baseline_rates=baseline_rates)
    atomic_json(status_path,status)
    print(json.dumps(status),flush=True)


if __name__ == "__main__":
    main()
