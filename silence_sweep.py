"""
Silence each (layer, population) in turn and observe the effect on
spontaneous network dynamics.

For every (layer, population) we run two trials:
  - control:  network runs untouched
  - silenced: at t = baseline_ms we raise Vcut for the entire target
              population to a value the membrane can never reach, so those
              neurons stop spiking. Synaptic currents and membrane
              dynamics still evolve normally, so the only effect on the
              rest of the network is the loss of the population's spike
              output.

There is no LGN / no feedforward stimulus -- this measures how each
population contributes to the resting dynamics. The same data structure
as stim_sweep.py is saved so existing plotting code keeps working;
the "stimulus window" is simply the silenced window.
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


LAYERS = ["L23", "L4AB", "L4C", "L5", "L6"]
POPULATIONS = ["E", "PV", "SOM", "VIP"]

# One condition per (layer, population).
SILENCE_CONDITIONS = []
for _layer in LAYERS:
    for _pop in POPULATIONS:
        SILENCE_CONDITIONS.append({
            "name": f"{_layer}_{_pop}_silence",
            "target": (_layer, _pop),
        })

# For every condition we run both control (no silencing) and silenced.
SILENCE_FLAGS = [False, True]

# Number of independent trials per (condition, flag). Each trial uses a
# distinct base seed; control and silenced within a trial share that seed
# so they form a paired comparison (same network, same Poisson background
# during the baseline window), which kills a lot of trial-to-trial noise.
N_TRIALS = 10

# How we silence: kill the AdEx spike-generation mechanism on the target
# population. We do two things, both per-neuron parameters that take
# effect on the next run() without touching equations:
#   1. DeltaT -> ~0 collapses the exponential term gL*DeltaT*exp(...)
#      so v can no longer run away regardless of how depolarized it gets.
#   2. Vcut -> huge means the threshold v > Vcut never trips even if
#      numerical noise pushes v very high.
# Membrane dynamics and synaptic currents continue normally, so the
# population's contribution to the LFP-from-synaptic-current is
# preserved -- only its spike output disappears.
SILENCE_DELTAT = 1e-6 * mV
SILENCE_VCUT   = 1e6 * mV

# ---------------------------------------------------------------------------

CONFIG_FILES = [
    "config/config.py",
    "config/conductances_AMPA_GABA.csv",
    "config/conductances_NMDA.csv",
    "config/connection_probabilities.csv",
    "main.py",
    "silence_sweep.py",
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
    silenced,
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

    layer_name, pop_name = condition["target"]
    if verbose:
        cond_kind = "SILENCED" if silenced else "control"
        print(f"\n=== Trial {trial_id} | condition={condition['name']} | "
              f"{cond_kind} | network seed {network_seed} ===")
        print("Creating cortical column...")

    column = CorticalColumn(column_id=0, config=config)
    for ln, layer in column.layers.items():
        add_heterogeneity_to_layer(layer, CONFIG)

    all_monitors = column.get_all_monitors()

    grp = column.layers[layer_name].neuron_groups[pop_name]
    n_target = int(len(grp))
    # Snapshot originals so we can describe what we changed.
    deltat_original_mV = float(np.mean(np.array(grp.DeltaT / mV)))
    vcut_original_mV = float(np.mean(np.array(grp.Vcut / mV)))

    if trial_id != 0:
        np.random.seed(baseline_seed)
        b2.seed(baseline_seed)
    column.network.run(baseline_ms * ms)

    if trial_id != 0:
        np.random.seed(stim_seed)
        b2.seed(stim_seed)

    if silenced:
        grp.DeltaT = SILENCE_DELTAT
        grp.Vcut = SILENCE_VCUT
        print(f"  [silencing] {layer_name}/{pop_name}: "
              f"DeltaT -> {float(np.mean(grp.DeltaT/mV)):.2e} mV, "
              f"Vcut -> {float(np.mean(grp.Vcut/mV)):.2e} mV")

    column.network.run(stimuli_ms * ms)

    if verbose:
        print("Simulation complete")

    spike_monitors = {}
    state_monitors = {}
    rate_monitors = {}
    isyn_full_monitors = {}
    neuron_groups = {}

    for ln, monitors in all_monitors.items():
        spike_monitors[ln] = {
            k: v for k, v in monitors.items() if 'spikes' in k
        }
        state_monitors[ln] = {
            k: v for k, v in monitors.items()
            if 'state' in k and 'Isyn_full' not in k
        }
        rate_monitors[ln] = {
            k: v for k, v in monitors.items() if 'rate' in k
        }
        isyn_full_monitors[ln] = {
            k: v for k, v in monitors.items() if 'Isyn_full' in k
        }
        neuron_groups[ln] = column.layers[ln].neuron_groups

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
    for ln, layer_spike_mons in spike_monitors.items():
        spike_data[ln] = {}
        for mon_name, mon in layer_spike_mons.items():
            spike_data[ln][mon_name] = {
                "times_ms": np.array(mon.t / ms),
                "spike_indices": np.array(mon.i),
            }

    lfp_full = {}
    for ln, layer_rate_mons in rate_monitors.items():
        e_rate_mon = layer_rate_mons.get('E_rate')
        if e_rate_mon is not None:
            lfp_full[ln] = np.array(
                e_rate_mon.smooth_rate(window='gaussian', width=1*ms) / Hz
            )

    rate_data = {}
    for ln, layer_rate_mons in rate_monitors.items():
        rate_data[ln] = {}
        for mon_name, mon in layer_rate_mons.items():
            if len(mon.t) == 0:
                continue
            t_ms = np.array(mon.t / ms)
            r_hz = np.array(mon.rate / Hz)
            rate_data[ln][mon_name] = {"t_ms": t_ms, "rate_hz": r_hz}

    state_data = {}
    for ln, layer_state_mons in state_monitors.items():
        state_data[ln] = {}
        for mon_name, mon in layer_state_mons.items():
            pname = mon_name.replace('_state', '')
            state_data[ln][pname] = {}
            state_data[ln][pname]['t_ms'] = np.array(mon.t / ms)
            for var in mon.record_variables:
                vals = np.array(getattr(mon, var))
                state_data[ln][pname][var] = np.mean(
                    vals, axis=0).astype(np.float32)

    n_elec = len(lfp_signals)
    lfp_matrix = np.vstack([lfp_signals[i] for i in range(n_elec)])
    bipolar_matrix = np.vstack([bipolar_signals[i]
                                for i in range(len(bipolar_signals))])

    data = {
        "trial_id": trial_id,
        "condition_name": condition["name"],
        "condition_target": np.array(condition["target"], dtype=object),
        "silenced": bool(silenced),
        "silence_DeltaT_mV": float(SILENCE_DELTAT / mV),
        "silence_Vcut_mV": float(SILENCE_VCUT / mV),
        "DeltaT_original_mV": deltat_original_mV,
        "Vcut_original_mV": vcut_original_mV,
        "n_target_neurons": n_target,
        "network_seed": network_seed,
        "baseline_seed": baseline_seed,
        "stim_seed": stim_seed,
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
        kind = "silenced" if silenced else "control"
        print(f"Trial {trial_id} ({condition['name']} / {kind}) finished.\n")

    return data


def run_silence_sweep(
    config,
    conditions=SILENCE_CONDITIONS,
    silence_flags=SILENCE_FLAGS,
    n_trials=N_TRIALS,
    base_network_seed=58880,
    baseline_ms=2000,
    stimuli_ms=2000,
    fs=10000,
    save_dir="results/silence_sweep",
    verbose=True,
):
    os.makedirs(save_dir, exist_ok=True)
    save_config_snapshot(save_dir)

    trial_id = 0
    for condition in conditions:
        cond_dir = os.path.join(save_dir, condition["name"])
        os.makedirs(cond_dir, exist_ok=True)

        for trial_idx in range(n_trials):
            # Distinct seed per trial; control and silenced share it so they
            # are a paired comparison on the same network instance.
            trial_seed = int(base_network_seed + 1000 * trial_idx)

            for silenced in silence_flags:
                data = run_single_trial(
                    config=config,
                    condition=condition,
                    silenced=silenced,
                    trial_id=trial_id,
                    network_seed=trial_seed,
                    baseline_ms=baseline_ms,
                    stimuli_ms=stimuli_ms,
                    fs=fs,
                    verbose=verbose,
                )
                data["trial_index"] = trial_idx
                tag = "silenced" if silenced else "control"
                fname = os.path.join(
                    cond_dir,
                    f"{condition['name']}_trial{trial_idx:02d}_{tag}.npz",
                )
                np.savez_compressed(fname, **data)
                if verbose:
                    print(f"Saved {fname}")
                trial_id += 1


if __name__ == "__main__":
    run_silence_sweep(
        CONFIG,
        conditions=SILENCE_CONDITIONS,
        silence_flags=SILENCE_FLAGS,
        n_trials=N_TRIALS,
        base_network_seed=58910,
        baseline_ms=2000,
        stimuli_ms=2000,
        fs=10000,
        save_dir="results/silence_sweep_07_05",
        verbose=True,
    )
