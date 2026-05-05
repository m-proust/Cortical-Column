"""
Population stimulus-rate sweep (no LGN / no baseline-stim event).

For every (layer, population) combination, run one simulation per rate in
RATES_HZ. The simulation has a single SIM_MS window during which a
PoissonInput at the swept rate drives the chosen population. We then save
the LFP / spikes / rates to npz so power spectra can be plotted later.

This is purely a parameter scan -- there is no stimulus-onset event, no
baseline window. The 'stim_onset_ms' field is set to 0 so analysis code
can still treat the whole run as the post-stim period.
"""

import os
import shutil
import numpy as np
import brian2 as b2
from brian2 import *
from config.config import CONFIG
from src.column import CorticalColumn
from src.visualization import *
from src.analysis import *
from tools.lfp_kernel import calculate_lfp_kernel_method
from tools.lfp_current_method import calculate_lfp_current_method


LAYERS      = ["L23", "L4AB", "L4C", "L5", "L6"]
POPULATIONS = ["E", "PV", "SOM", "VIP"]

RATES_HZ    = [0, 1, 2, 4, 6, 8, 10, 15, 20]  # rates to sweep
N_STIM      = 30          # # of independent Poisson sources per condition
WEIGHT_MULT = 1.0         # multiplier on EXT_AMPA weight
SIM_MS      = 3000        # length of each simulation
TRANSIENT_MS = 500        # skip this in analysis (saved as field)
NETWORK_SEED = 58910

SAVE_DIR    = "results/pop_stim_sweep"
# ---------------------------------------------------------------------------


CONFIG_FILES = [
    "config/config.py",
    "config/conductances_AMPA_GABA.csv",
    "config/conductances_NMDA.csv",
    "config/connection_probabilities.csv",
    "pop_stim_sweep.py",
]


def save_config_snapshot(save_dir, base_dir=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    snapshot_dir = os.path.join(save_dir, "config_snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)
    for rel_path in CONFIG_FILES:
        src = os.path.join(base_dir, rel_path)
        if os.path.exists(src):
            dst = os.path.join(snapshot_dir, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    print(f"Config snapshot saved to {snapshot_dir}")


def run_one(layer_name, pop_name, rate_hz, sim_ms, network_seed):
    np.random.seed(network_seed)
    b2.seed(network_seed)
    b2.start_scope()
    b2.defaultclock.dt = CONFIG['simulation']['DT']

    column = CorticalColumn(column_id=0, config=CONFIG)
    for ln, layer in column.layers.items():
        add_heterogeneity_to_layer(layer, CONFIG)

    all_monitors = column.get_all_monitors()
    w_ext_AMPA = CONFIG['synapses']['Q']['EXT_AMPA']

    grp = column.layers[layer_name].neuron_groups[pop_name]
    pin = PoissonInput(grp, 'gE_AMPA',
                       N=N_STIM,
                       rate=rate_hz * Hz,
                       weight=WEIGHT_MULT * w_ext_AMPA)
    column.network.add(pin)

    column.network.run(sim_ms * ms)

    spike_monitors, state_monitors, rate_monitors = {}, {}, {}
    isyn_full_monitors, neuron_groups = {}, {}
    for ln, monitors in all_monitors.items():
        spike_monitors[ln] = {k: v for k, v in monitors.items() if 'spikes' in k}
        state_monitors[ln] = {k: v for k, v in monitors.items()
                              if 'state' in k and 'Isyn_full' not in k}
        rate_monitors[ln] = {k: v for k, v in monitors.items() if 'rate' in k}
        isyn_full_monitors[ln] = {k: v for k, v in monitors.items()
                                  if 'Isyn_full' in k}
        neuron_groups[ln] = column.layers[ln].neuron_groups

    electrode_positions = CONFIG['electrode_positions']

    lfp_signals, time_array = calculate_lfp_kernel_method(
        spike_monitors, neuron_groups, CONFIG['layers'],
        electrode_positions, sim_duration_ms=sim_ms,
    )
    bipolar_signals, channel_labels, channel_depths = compute_bipolar_lfp(
        lfp_signals, electrode_positions,
    )

    current_method_monitors = {
        ln: {k.replace('_Isyn_full', '_state'): v for k, v in mons.items()}
        for ln, mons in isyn_full_monitors.items()
    }
    lfp_current_matrix, time_current_ms = calculate_lfp_current_method(
        current_method_monitors, neuron_groups, CONFIG['layers'],
        electrode_positions, dt_ms=0.5, sim_duration_ms=sim_ms,
    )

    spike_data = {}
    for ln, layer_spike_mons in spike_monitors.items():
        spike_data[ln] = {}
        for mon_name, mon in layer_spike_mons.items():
            spike_data[ln][mon_name] = {
                "times_ms": np.array(mon.t / ms),
                "spike_indices": np.array(mon.i),
            }

    rate_data = {}
    for ln, layer_rate_mons in rate_monitors.items():
        rate_data[ln] = {}
        for mon_name, mon in layer_rate_mons.items():
            if len(mon.t) == 0:
                continue
            rate_data[ln][mon_name] = {
                "t_ms":   np.array(mon.t / ms),
                "rate_hz": np.array(mon.rate / Hz),
            }

    n_elec = len(lfp_signals)
    lfp_matrix = np.vstack([lfp_signals[i] for i in range(n_elec)])
    bipolar_matrix = np.vstack([bipolar_signals[i]
                                for i in range(len(bipolar_signals))])

    return {
        "layer":           layer_name,
        "population":      pop_name,
        "rate_hz":         float(rate_hz),
        "n_stim":          N_STIM,
        "weight_mult":     WEIGHT_MULT,
        "network_seed":    network_seed,
        "sim_ms":          sim_ms,
        "transient_ms":    TRANSIENT_MS,
        # The field is named stim_onset_ms so that downstream code
        # (e.g. plot_stim_sweep.py) can still address a 'post-stim' window
        # which here is just everything after the transient.
        "stim_onset_ms":   TRANSIENT_MS,
        "time_array_ms":   np.array(time_array),
        "electrode_positions": np.array(electrode_positions),
        "channel_labels":  np.array(channel_labels, dtype=object),
        "channel_depths":  np.array(channel_depths),
        "rate_data":       rate_data,
        "spike_data":      spike_data,
        "lfp_matrix":      lfp_matrix,
        "bipolar_matrix":  bipolar_matrix,
        "lfp_current_matrix": lfp_current_matrix.astype(np.float32),
        "time_current_ms":    time_current_ms.astype(np.float32),
        "baseline_ms":     0,
        "post_ms":         sim_ms,
    }


def run_all():
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_config_snapshot(SAVE_DIR)

    n_total = len(LAYERS) * len(POPULATIONS) * len(RATES_HZ)
    done = 0

    for layer in LAYERS:
        for pop in POPULATIONS:
            cond_dir = os.path.join(SAVE_DIR, f"{layer}_{pop}")
            os.makedirs(cond_dir, exist_ok=True)
            for rate in RATES_HZ:
                done += 1
                fname = os.path.join(
                    cond_dir, f"{layer}_{pop}_rate_{rate:02d}Hz.npz")
                if os.path.exists(fname):
                    print(f"[{done}/{n_total}] skip (exists) {fname}")
                    continue
                print(f"\n[{done}/{n_total}] === {layer}/{pop} @ {rate} Hz ===")
                data = run_one(layer, pop, rate, SIM_MS, NETWORK_SEED)
                np.savez_compressed(fname, **data)
                print(f"  saved {fname}")


if __name__ == "__main__":
    run_all()
