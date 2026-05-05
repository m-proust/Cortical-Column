"""
Sweep stimulus strength across specified populations.

For each condition in STIM_CONDITIONS and each rate in STIM_RATES_HZ, run a
trial with a baseline period followed by a stimulus period. Save the same
per-trial data structure as trials.py so existing plotting code works.

Edit STIM_CONDITIONS below to change which layers/populations are stimulated
and their relative strengths (N neurons and weight multiplier on w_ext_AMPA).
Edit STIM_RATES_HZ to change the rate sweep.
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
# USER-EDITABLE STIMULUS SPEC
# ---------------------------------------------------------------------------
# Each condition is a dict:
#   name:    string used in the saved filename
#   targets: list of (layer, population, N, weight_mult, rate_mult) tuples
#            rate_mult scales the swept rate for that target (e.g. 2.0 means
#            PV gets twice the rate of E when both are stimulated together).
#
# The PoissonInput for each target is:
#     PoissonInput(group, 'gE_AMPA',
#                  N=N, rate=rate_mult*rate_hz*Hz, weight=weight_mult*w_ext_AMPA)
#
# rate_hz comes from STIM_RATES_HZ.

LAYERS = ["L23", "L4AB", "L4C", "L5", "L6"]
POPULATIONS = ["E", "PV", "SOM", "VIP"]

STIM_CONDITIONS = [
    {
        "name": "L4C_E_PV",
        "targets": [
            ("L4C", "E",  30, 1.0, 1.0),
            ("L4C", "PV", 40, 2.5, 2.0),
        ],
    },
]

# One condition per (layer, population): stimulate each population of each
# layer in isolation with the swept rate.
for _layer in LAYERS:
    for _pop in POPULATIONS:
        STIM_CONDITIONS.append({
            "name": f"{_layer}_{_pop}",
            "targets": [(_layer, _pop, 30, 1.0, 1.0)],
        })

STIM_RATES_HZ = list(range(0, 11))  # 0, 1, 2, ..., 10 Hz

# ---------------------------------------------------------------------------

CONFIG_FILES = [
    "config/config.py",
    "config/conductances_AMPA_GABA.csv",
    "config/conductances_NMDA.csv",
    "config/connection_probabilities.csv",
    "main.py",
    "stim_sweep.py",
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


def run_single_trial(
    config,
    condition,
    rate_hz,
    trial_id=0,
    network_seed=58880,
    baseline_ms=2000,
    stimuli_ms=2000,
    fs=10000,
    verbose=True,
):
    np.random.seed(network_seed)
    b2.seed(network_seed)

    b2.start_scope()
    b2.defaultclock.dt = config['simulation']['DT']

    total_time = baseline_ms + stimuli_ms

    if trial_id == 0:
        baseline_seed = network_seed
        stim_seed = network_seed
    else:
        baseline_seed = int(network_seed + 2 * trial_id)
        stim_seed = int(network_seed + 2 * trial_id + 1)

    if verbose:
        print(f"\n=== Trial {trial_id} | condition={condition['name']} | "
              f"rate={rate_hz} Hz | network seed {network_seed} ===")
        print("Creating cortical column...")

    column = CorticalColumn(column_id=0, config=config)
    for layer_name, layer in column.layers.items():
        add_heterogeneity_to_layer(layer, CONFIG)

    all_monitors = column.get_all_monitors()
    w_ext_AMPA = config['synapses']['Q']['EXT_AMPA']

    if trial_id != 0:
        np.random.seed(baseline_seed)
        b2.seed(baseline_seed)
    column.network.run(baseline_ms * ms)

    if trial_id != 0:
        np.random.seed(stim_seed)
        b2.seed(stim_seed)

    # Build and attach PoissonInputs for this condition.
    # Keep references alive; track per-target rates for metadata.
    stim_objects = []
    stim_rates_meta = {}
    for (layer_name, pop_name, N, w_mult, r_mult) in condition["targets"]:
        grp = column.layers[layer_name].neuron_groups[pop_name]
        target_rate = r_mult * rate_hz
        pin = PoissonInput(
            grp, 'gE_AMPA',
            N=N,
            rate=target_rate * Hz,
            weight=w_mult * w_ext_AMPA,
        )
        stim_objects.append(pin)
        stim_rates_meta[f"{layer_name}_{pop_name}"] = float(target_rate)

    if stim_objects:
        column.network.add(*stim_objects)

    column.network.run(stimuli_ms * ms)

    if verbose:
        print("Simulation complete")

    spike_monitors = {}
    state_monitors = {}
    rate_monitors = {}
    isyn_full_monitors = {}
    neuron_groups = {}

    for layer_name, monitors in all_monitors.items():
        spike_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'spikes' in k
        }
        state_monitors[layer_name] = {
            k: v for k, v in monitors.items()
            if 'state' in k and 'Isyn_full' not in k
        }
        rate_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'rate' in k
        }
        isyn_full_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'Isyn_full' in k
        }
        neuron_groups[layer_name] = column.layers[layer_name].neuron_groups

    electrode_positions = CONFIG['electrode_positions']

    if verbose:
        print("Computing LFP using kernel method...")

    lfp_signals, time_array = calculate_lfp_kernel_method(
        spike_monitors,
        neuron_groups,
        CONFIG['layers'],
        electrode_positions,
        sim_duration_ms=total_time
    )

    if verbose:
        print("Computing bipolar LFP...")

    bipolar_signals, channel_labels, channel_depths = compute_bipolar_lfp(
        lfp_signals,
        electrode_positions,
    )

    if verbose:
        print("Computing LFP using synaptic current method ...")

    from tools.lfp_current_method import calculate_lfp_current_method

    current_method_monitors = {
        ln: {k.replace('_Isyn_full', '_state'): v for k, v in mons.items()}
        for ln, mons in isyn_full_monitors.items()
    }
    lfp_current_matrix, time_current_ms = calculate_lfp_current_method(
        current_method_monitors,
        neuron_groups,
        CONFIG['layers'],
        electrode_positions,
        dt_ms=0.5,
        sim_duration_ms=total_time,
    )

    spike_data = {}
    for layer_name, layer_spike_mons in spike_monitors.items():
        spike_data[layer_name] = {}
        for mon_name, mon in layer_spike_mons.items():
            spike_data[layer_name][mon_name] = {
                "times_ms": np.array(mon.t / ms),
                "spike_indices": np.array(mon.i),
            }

    lfp_full = {}
    for layer_name, layer_rate_mons in rate_monitors.items():
        e_rate_mon = layer_rate_mons.get('E_rate')
        if e_rate_mon is not None:
            lfp_full[layer_name] = np.array(
                e_rate_mon.smooth_rate(window='gaussian', width=1*ms) / Hz
            )

    rate_data = {}
    for layer_name, layer_rate_mons in rate_monitors.items():
        rate_data[layer_name] = {}
        for mon_name, mon in layer_rate_mons.items():
            if len(mon.t) == 0:
                continue
            t_ms = np.array(mon.t / ms)
            r_hz = np.array(mon.rate / Hz)
            rate_data[layer_name][mon_name] = {"t_ms": t_ms, "rate_hz": r_hz}

    state_data = {}
    for layer_name, layer_state_mons in state_monitors.items():
        state_data[layer_name] = {}
        for mon_name, mon in layer_state_mons.items():
            pop_name = mon_name.replace('_state', '')
            state_data[layer_name][pop_name] = {}
            state_data[layer_name][pop_name]['t_ms'] = np.array(mon.t / ms)
            for var in mon.record_variables:
                vals = np.array(getattr(mon, var))
                state_data[layer_name][pop_name][var] = np.mean(
                    vals, axis=0).astype(np.float32)

    n_elec = len(lfp_signals)
    lfp_matrix = np.vstack([lfp_signals[i] for i in range(n_elec)])
    bipolar_matrix = np.vstack([bipolar_signals[i]
                                for i in range(len(bipolar_signals))])

    data = {
        "trial_id": trial_id,
        "condition_name": condition["name"],
        "condition_targets": np.array(condition["targets"], dtype=object),
        "sweep_rate_hz": float(rate_hz),
        "network_seed": network_seed,
        "baseline_seed": baseline_seed,
        "stim_seed": stim_seed,
        "stim_rates": stim_rates_meta,
        "time_array_ms": np.array(time_array),
        "electrode_positions": np.array(electrode_positions),
        "channel_labels": np.array(channel_labels, dtype=object),
        "channel_depths": np.array(channel_depths),
        "rate_data": rate_data,
        "spike_data": spike_data,
        "state_data": state_data,
        "lfp_full": lfp_full,
        "lfp_matrix": lfp_matrix,
        "bipolar_matrix": bipolar_matrix,
        "baseline_ms": baseline_ms,
        "post_ms": stimuli_ms,
        "stim_onset_ms": baseline_ms,
        "lfp_current_matrix": lfp_current_matrix.astype(np.float32),
        "time_current_ms":    time_current_ms.astype(np.float32),
    }

    if verbose:
        print(f"Trial {trial_id} ({condition['name']} @ {rate_hz} Hz) finished.\n")

    return data


def run_stim_sweep(
    config,
    conditions=STIM_CONDITIONS,
    rates_hz=STIM_RATES_HZ,
    network_seed=58880,
    baseline_ms=2000,
    stimuli_ms=2000,
    fs=10000,
    save_dir="results/stim_sweep",
    verbose=True,
):
    os.makedirs(save_dir, exist_ok=True)
    save_config_snapshot(save_dir)

    trial_id = 0
    for condition in conditions:
        cond_dir = os.path.join(save_dir, condition["name"])
        os.makedirs(cond_dir, exist_ok=True)

        for rate_hz in rates_hz:
            data = run_single_trial(
                config=config,
                condition=condition,
                rate_hz=rate_hz,
                trial_id=trial_id,
                network_seed=network_seed,
                baseline_ms=baseline_ms,
                stimuli_ms=stimuli_ms,
                fs=fs,
                verbose=verbose,
            )
            fname = os.path.join(
                cond_dir, f"{condition['name']}_rate_{rate_hz:02d}Hz.npz"
            )
            np.savez_compressed(fname, **data)
            if verbose:
                print(f"Saved {fname}")
            trial_id += 1


if __name__ == "__main__":
    run_stim_sweep(
        CONFIG,
        conditions=STIM_CONDITIONS,
        rates_hz=STIM_RATES_HZ,
        network_seed=58910,
        baseline_ms=2000,
        stimuli_ms=2000,
        fs=10000,
        save_dir="results/stim_sweep_23_04",
        verbose=True,
    )