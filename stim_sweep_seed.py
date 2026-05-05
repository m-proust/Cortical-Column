"""
Background-noise replicates at a single stim rate.

Rationale
---------
The original 23-04 sweep was 21 conditions × 11 rates × 1 seed = 231
trials. This gave dose-response curves but no error bars at any single
rate. Reviewers will need error bars on every claim; right now we have
none.

Better design for the same compute budget: 21 conditions × 1 rate × 10
seeds = 210 trials. Same network instantiation across all trials (only
background-noise seeds differ), so the variance we measure is genuine
trial-to-trial noise, not connectivity-instantiation variance.

Why one rate is enough: the dose-response shape is already known from
the original sweep. What's unknown is whether any specific spectral
result is robust to noise.

Why 5 Hz: peak-rate heatmap analysis showed
  - alpha peaks at 0-3 Hz then drops (so 5 Hz captures alpha-suppression onset)
  - low-gamma peaks at 5 Hz for many conditions (the project's target band)
  - beta starts climbing at 5 Hz (its peak is 6-10 Hz)
  - delta/theta still in dynamic range
  - 5 Hz avoids the saturation regime that some conditions enter at 9-10 Hz
5 Hz is the rate at which conditions are MOST distinguishable, which is
both the right figure rate AND the right replication rate.

Outputs
-------
results/stim_sweep_replicates_5Hz/<condition>/<condition>_rate_05Hz_noise<k>.npz
  for k = 0..9 and each of 21 conditions.

Compatible with all existing analysis scripts; aggregation should
group by noise_seed_idx to compute mean and SEM per condition.
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


# ---------------------------------------------------------------------------
LAYERS = ["L23", "L4AB", "L4C", "L5", "L6"]
POPULATIONS = ["E", "PV", "SOM", "VIP"]

STIM_CONDITIONS = [
    {"name": "L4C_E_PV", "targets": [
        ("L4C", "E",  30, 1.0, 1.0),
        ("L4C", "PV", 40, 2.5, 2.0),
    ]},
]
for _layer in LAYERS:
    for _pop in POPULATIONS:
        STIM_CONDITIONS.append({
            "name": f"{_layer}_{_pop}",
            "targets": [(_layer, _pop, 30, 1.0, 1.0)],
        })

# THE single rate to test, and how many noise seeds.
STIM_RATE_HZ = 5
N_NOISE_SEEDS = 10

# Same network instantiation as 23-04 so connectivity matches.
NETWORK_SEED = 58910

# Offset for the per-trial re-seeds. Different from 23-04's derivation
# so we never accidentally reuse a draw from the original sweep.
NOISE_SEED_BASE = 200000

CONFIG_FILES = [
    "config/config.py",
    "config/conductances_AMPA_GABA.csv",
    "config/conductances_NMDA.csv",
    "config/connection_probabilities.csv",
    "main.py",
    "stim_sweep.py",
    "stim_sweep_replicates_1rate.py",
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


def run_single_trial(config, condition, rate_hz, noise_seed_idx,
                     network_seed=NETWORK_SEED, baseline_ms=2000,
                     stimuli_ms=2000, fs=10000, verbose=True):
    """Network is built with `network_seed` (so connectivity and
    parameter heterogeneity are identical across all repeats).
    Background-noise re-seeds are derived from `noise_seed_idx`,
    making each repeat differ only in background Poisson draws.
    """
    np.random.seed(network_seed)
    b2.seed(network_seed)

    b2.start_scope()
    b2.defaultclock.dt = config['simulation']['DT']
    total_time = baseline_ms + stimuli_ms

    baseline_seed = int(network_seed + NOISE_SEED_BASE + 2 * noise_seed_idx)
    stim_seed     = int(network_seed + NOISE_SEED_BASE + 2 * noise_seed_idx + 1)

    if verbose:
        print(f"\n=== noise_seed_idx={noise_seed_idx} | "
              f"condition={condition['name']} | rate={rate_hz} Hz | "
              f"network_seed={network_seed} | "
              f"baseline_seed={baseline_seed} stim_seed={stim_seed} ===")

    column = CorticalColumn(column_id=0, config=config)
    for layer_name, layer in column.layers.items():
        add_heterogeneity_to_layer(layer, CONFIG)

    all_monitors = column.get_all_monitors()
    w_ext_AMPA = config['synapses']['Q']['EXT_AMPA']

    np.random.seed(baseline_seed); b2.seed(baseline_seed)
    column.network.run(baseline_ms * ms)

    np.random.seed(stim_seed); b2.seed(stim_seed)

    stim_objects = []; stim_rates_meta = {}
    for (layer_name, pop_name, N, w_mult, r_mult) in condition["targets"]:
        grp = column.layers[layer_name].neuron_groups[pop_name]
        target_rate = r_mult * rate_hz
        pin = PoissonInput(grp, 'gE_AMPA',
                           N=N, rate=target_rate * Hz,
                           weight=w_mult * w_ext_AMPA)
        stim_objects.append(pin)
        stim_rates_meta[f"{layer_name}_{pop_name}"] = float(target_rate)
    if stim_objects:
        column.network.add(*stim_objects)
    column.network.run(stimuli_ms * ms)

    # ---- monitor collection (identical to stim_sweep.py) ----
    spike_monitors = {}; state_monitors = {}; rate_monitors = {}
    isyn_full_monitors = {}; neuron_groups = {}
    for ln, mons in all_monitors.items():
        spike_monitors[ln] = {k: v for k, v in mons.items() if 'spikes' in k}
        state_monitors[ln] = {k: v for k, v in mons.items()
                              if 'state' in k and 'Isyn_full' not in k}
        rate_monitors[ln]  = {k: v for k, v in mons.items() if 'rate' in k}
        isyn_full_monitors[ln] = {k: v for k, v in mons.items() if 'Isyn_full' in k}
        neuron_groups[ln] = column.layers[ln].neuron_groups

    electrode_positions = CONFIG['electrode_positions']

    lfp_signals, time_array = calculate_lfp_kernel_method(
        spike_monitors, neuron_groups, CONFIG['layers'],
        electrode_positions, sim_duration_ms=total_time)
    bipolar_signals, channel_labels, channel_depths = compute_bipolar_lfp(
        lfp_signals, electrode_positions)

    from tools.lfp_current_method import calculate_lfp_current_method
    current_method_monitors = {
        ln: {k.replace('_Isyn_full', '_state'): v for k, v in mons.items()}
        for ln, mons in isyn_full_monitors.items()
    }
    lfp_current_matrix, time_current_ms = calculate_lfp_current_method(
        current_method_monitors, neuron_groups, CONFIG['layers'],
        electrode_positions, dt_ms=0.5, sim_duration_ms=total_time)

    spike_data = {}
    for ln, sm in spike_monitors.items():
        spike_data[ln] = {}
        for mn, m in sm.items():
            spike_data[ln][mn] = {"times_ms": np.array(m.t / ms),
                                  "spike_indices": np.array(m.i)}
    lfp_full = {}
    for ln, rm in rate_monitors.items():
        e = rm.get('E_rate')
        if e is not None:
            lfp_full[ln] = np.array(e.smooth_rate(window='gaussian', width=1*ms) / Hz)
    rate_data = {}
    for ln, rm in rate_monitors.items():
        rate_data[ln] = {}
        for mn, m in rm.items():
            if len(m.t) == 0: continue
            rate_data[ln][mn] = {"t_ms": np.array(m.t / ms),
                                 "rate_hz": np.array(m.rate / Hz)}
    state_data = {}
    for ln, sm in state_monitors.items():
        state_data[ln] = {}
        for mn, m in sm.items():
            pn = mn.replace('_state', '')
            state_data[ln][pn] = {"t_ms": np.array(m.t / ms)}
            for var in m.record_variables:
                vals = np.array(getattr(m, var))
                state_data[ln][pn][var] = np.mean(vals, axis=0).astype(np.float32)

    n_elec = len(lfp_signals)
    lfp_matrix = np.vstack([lfp_signals[i] for i in range(n_elec)])
    bipolar_matrix = np.vstack([bipolar_signals[i] for i in range(len(bipolar_signals))])

    return {
        "noise_seed_idx": int(noise_seed_idx),
        "condition_name": condition["name"],
        "condition_targets": np.array(condition["targets"], dtype=object),
        "sweep_rate_hz": float(rate_hz),
        "network_seed": network_seed,
        "baseline_seed": baseline_seed, "stim_seed": stim_seed,
        "stim_rates": stim_rates_meta,
        "time_array_ms": np.array(time_array),
        "electrode_positions": np.array(electrode_positions),
        "channel_labels": np.array(channel_labels, dtype=object),
        "channel_depths": np.array(channel_depths),
        "rate_data": rate_data, "spike_data": spike_data,
        "state_data": state_data, "lfp_full": lfp_full,
        "lfp_matrix": lfp_matrix, "bipolar_matrix": bipolar_matrix,
        "baseline_ms": baseline_ms, "post_ms": stimuli_ms,
        "stim_onset_ms": baseline_ms,
        "lfp_current_matrix": lfp_current_matrix.astype(np.float32),
        "time_current_ms":    time_current_ms.astype(np.float32),
    }


def run_replicates(config, conditions=STIM_CONDITIONS,
                   rate_hz=STIM_RATE_HZ, n_noise_seeds=N_NOISE_SEEDS,
                   network_seed=NETWORK_SEED,
                   save_dir="results/stim_sweep_replicates_5Hz", verbose=True):
    os.makedirs(save_dir, exist_ok=True)
    save_config_snapshot(save_dir)

    total = len(conditions) * n_noise_seeds
    done = 0

    # Outer loop: noise_seed_idx. So if you abort partway you still have
    # at least one full repeat covering every condition.
    for k in range(n_noise_seeds):
        for condition in conditions:
            cond_dir = os.path.join(save_dir, condition["name"])
            os.makedirs(cond_dir, exist_ok=True)
            data = run_single_trial(config, condition, rate_hz,
                                    noise_seed_idx=k,
                                    network_seed=network_seed,
                                    verbose=verbose)
            fname = os.path.join(
                cond_dir,
                f"{condition['name']}_rate_{rate_hz:02d}Hz_noise{k:02d}.npz"
            )
            np.savez_compressed(fname, **data)
            done += 1
            print(f"[{done}/{total}] saved {fname}")


if __name__ == "__main__":
    run_replicates(CONFIG)