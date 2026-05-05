"""
Inter-layer connectivity strength sweep.

For p in P_VALUES we run a single simulation in which every inter-layer
synaptic conductance is multiplied by p. p=0 means the layers are
disconnected; p=1 is the canonical model; p>1 over-strengthens inter-layer
coupling. We do NOT change connection probabilities (so the connectivity
graph is preserved across runs); only conductance amplitudes scale.

There is no LGN / no stimulus event; each run is a single SIM_MS window of
the unstimulated network at scaling p.
"""

import os
import shutil
from copy import deepcopy
import numpy as np
import brian2 as b2
from brian2 import *
from config.config import CONFIG
from src.column import CorticalColumn
from src.visualization import *
from src.analysis import *
from tools.lfp_kernel import calculate_lfp_kernel_method
from tools.lfp_current_method import calculate_lfp_current_method


# ---------------------------------------------------------------------------
# USER-EDITABLE
# ---------------------------------------------------------------------------
P_VALUES = list(np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 3))  # 0..1 step .05
SIM_MS         = 3000
TRANSIENT_MS   = 500
NETWORK_SEED   = 58910
SAVE_DIR       = "results/interlayer_p_sweep"
# ---------------------------------------------------------------------------


CONFIG_FILES = [
    "config/config.py",
    "config/conductances_AMPA_GABA.csv",
    "config/conductances_NMDA.csv",
    "config/connection_probabilities.csv",
    "interlayer_p_sweep.py",
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


def scale_inter_layer_conductances(config, p):
    """Return a deep-copied config with every inter-layer conductance * p."""
    new_cfg = deepcopy(config)
    inter = new_cfg.get('inter_layer_conductances', {})
    for key, cond_dict in inter.items():
        for k in list(cond_dict.keys()):
            cond_dict[k] = float(cond_dict[k]) * float(p)
    new_cfg['inter_layer_scaling'] = float(p)
    return new_cfg


def run_one(p, sim_ms, network_seed):
    np.random.seed(network_seed)
    b2.seed(network_seed)
    b2.start_scope()
    b2.defaultclock.dt = CONFIG['simulation']['DT']

    cfg = scale_inter_layer_conductances(CONFIG, p)

    column = CorticalColumn(column_id=0, config=cfg)
    for ln, layer in column.layers.items():
        add_heterogeneity_to_layer(layer, cfg)

    all_monitors = column.get_all_monitors()
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

    electrode_positions = cfg['electrode_positions']

    lfp_signals, time_array = calculate_lfp_kernel_method(
        spike_monitors, neuron_groups, cfg['layers'],
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
        current_method_monitors, neuron_groups, cfg['layers'],
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
        "p":                float(p),
        "sim_ms":           sim_ms,
        "transient_ms":     TRANSIENT_MS,
        "stim_onset_ms":    TRANSIENT_MS,
        "network_seed":     network_seed,
        "time_array_ms":    np.array(time_array),
        "electrode_positions": np.array(electrode_positions),
        "channel_labels":   np.array(channel_labels, dtype=object),
        "channel_depths":   np.array(channel_depths),
        "rate_data":        rate_data,
        "spike_data":       spike_data,
        "lfp_matrix":       lfp_matrix,
        "bipolar_matrix":   bipolar_matrix,
        "lfp_current_matrix": lfp_current_matrix.astype(np.float32),
        "time_current_ms":    time_current_ms.astype(np.float32),
        "baseline_ms":      0,
        "post_ms":          sim_ms,
    }


def run_all():
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_config_snapshot(SAVE_DIR)

    for i, p in enumerate(P_VALUES):
        fname = os.path.join(SAVE_DIR, f"p_{p:.3f}.npz")
        if os.path.exists(fname):
            print(f"[{i+1}/{len(P_VALUES)}] skip (exists) {fname}")
            continue
        print(f"\n[{i+1}/{len(P_VALUES)}] === p = {p:.3f} ===")
        data = run_one(p, SIM_MS, NETWORK_SEED)
        np.savez_compressed(fname, **data)
        print(f"  saved {fname}")


if __name__ == "__main__":
    run_all()
