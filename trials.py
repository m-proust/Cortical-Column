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


CONFIG_FILES = [
    "config/config.py",
    "config/conductances_AMPA_GABA.csv",
    "config/conductances_NMDA.csv",
    "config/connection_probabilities.csv",
    "main.py",
    "trials3.py",
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
        print(f"\n=== Trial {trial_id}  |  network seed {network_seed}  |  "
              f"baseline seed {baseline_seed}  |  stim seed {stim_seed} ===")
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


    L4C = column.layers['L4C']
    L4C_E_grp = L4C.neuron_groups['E']
    L4C_E_stimAMPA = PoissonInput(L4C_E_grp, 'gE_AMPA',
                                  N=30, rate=5*Hz, weight=w_ext_AMPA)
    L4C_PV_grp = L4C.neuron_groups['PV']
    L4C_PV_stim = PoissonInput(L4C_PV_grp, 'gE_AMPA',
                               N=40, rate=7*Hz, weight=w_ext_AMPA*2.5)

    L6 = column.layers['L6']
    L6_E_grp = L6.neuron_groups['E']
    L6_E_stim = PoissonInput(L6_E_grp, 'gE_AMPA',
                             N=10, rate=5*Hz, weight=w_ext_AMPA*1.5)
    L6_PV_grp = L6.neuron_groups['PV']
    L6_PV_stim = PoissonInput(L6_PV_grp, 'gE_AMPA',
                              N=10, rate=6*Hz, weight=w_ext_AMPA*1.5)

    column.network.add(L6_E_stim, L6_PV_stim)
    column.network.add(L4C_E_stimAMPA, L4C_PV_stim)

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
        "network_seed": network_seed,
        "baseline_seed": baseline_seed,
        "stim_seed": stim_seed,
        "stim_rates": {
            "L4C_E": 5.0, "L4C_PV": 7.0, "L6_E": 5.0, "L6_PV": 6.0,
        },
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
        print(f"Trial {trial_id} finished.\n")

    return data


def run_multiple_trials(
    config,
    n_trials=50,
    network_seed=58880,
    baseline_ms=2000,
    stimuli_ms=2000,
    fs=10000,
    save_dir="results/trials",
    verbose=True,
):
    os.makedirs(save_dir, exist_ok=True)
    save_config_snapshot(save_dir)

    for trial_id in range(n_trials):
        data = run_single_trial(
            config=config,
            trial_id=trial_id,
            network_seed=network_seed,
            baseline_ms=baseline_ms,
            stimuli_ms=stimuli_ms,
            fs=fs,
            verbose=verbose,
        )
        fname = os.path.join(save_dir, f"trial_{trial_id:03d}.npz")
        np.savez_compressed(fname, **data)
        if verbose:
            print(f"Saved trial {trial_id} to {fname}")


if __name__ == "__main__":
    run_multiple_trials(
        CONFIG,
        n_trials=50,
        network_seed=58910,
        baseline_ms=2000,
        stimuli_ms=2000,
        fs=10000,
        save_dir="results/trials_23_04",
        verbose=True,
    )