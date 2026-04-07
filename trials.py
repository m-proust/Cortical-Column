import os
import shutil
import numpy as np
import brian2 as b2
from brian2 import *
from config.config2 import CONFIG
from src.column import CorticalColumn
from src.visualization import *
from src.analysis import *


CONFIG_FILES = [
    "config/config2.py",
    "config/conductances_AMPA2_alpha_v2.csv",
    "config/conductances_NMDA2_alpha_v2.csv",
    "config/connection_probabilities2.csv",
    "main.py",
    "trials.py",
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
    base_seed=None,
    baseline_ms=3000,
    stimuli_ms=3000,
    fs=10000,
    verbose=True,
):
    if base_seed is None:
        base_seed = config['simulation']['RANDOM_SEED']

    trial_seed = int(base_seed + trial_id)
    np.random.seed(trial_seed)
    b2.seed(trial_seed)

    b2.start_scope()
    b2.defaultclock.dt = config['simulation']['DT']

    total_time = baseline_ms + stimuli_ms

    if verbose:
        print(f"\n=== Running trial {trial_id} with seed {trial_seed} ===")
        print("Creating cortical column...")

    column = CorticalColumn(column_id=0, config=config)
    for layer_name, layer in column.layers.items():
        add_heterogeneity_to_layer(layer, CONFIG) # optional

    all_monitors = column.get_all_monitors()

    w_ext_AMPA = config['synapses']['Q']['EXT_AMPA']



   
    column.network.run(baseline_ms * ms)
   
    
    L4C = column.layers['L4C']
    cfg_L4C = CONFIG['layers']['L4C']
   
    
    L4C_E_grp = L4C.neuron_groups['E']
    N_stim_E = 30
    stim_rate_E = 4*Hz  
    L4C_E_stimAMPA = PoissonInput(L4C_E_grp, 'gE_AMPA', 
                                  N=N_stim_E, 
                                  rate=stim_rate_E, 
                                  weight=w_ext_AMPA)  
    
    
    L4C_PV_grp = L4C.neuron_groups['PV']
    N_stim_PV = 30
    stim_rate_PV = 4*Hz 
    L4C_PV_stim = PoissonInput(L4C_PV_grp, 'gE_AMPA', 
                               N=N_stim_PV, 
                               rate=stim_rate_PV, 
                               weight=w_ext_AMPA*2.5)  
    
    
    L6 = column.layers['L6']
    cfg_L6 = CONFIG['layers']['L6']
    L6_PV_grp = L6.neuron_groups['PV']
    N_stim_L6_PV = 10
    stim_rate_L6_PV = 5*Hz  
    
    L6_PV_stim = PoissonInput(L6_PV_grp, 'gE_AMPA',
                             N=N_stim_L6_PV, 
                             rate=stim_rate_L6_PV, 
                             weight=w_ext_AMPA*1.5)
    L6_E_grp = L6.neuron_groups['E']
    N_stim_L6_E = 10
    stim_rate_L6_E = 5*Hz  
    
    L6_E_stim = PoissonInput(L6_E_grp, 'gE_AMPA',
                             N=N_stim_L6_E, 
                             rate=stim_rate_L6_E, 
                             weight=w_ext_AMPA*1.5)



    column.network.add(L6_E_stim, L6_PV_stim)
    column.network.add(L4C_E_stimAMPA, L4C_PV_stim)
    


    column.network.run(stimuli_ms * ms)

    if verbose:
        print("Simulation complete")

    spike_monitors = {}
    state_monitors = {}
    rate_monitors = {}
    neuron_groups = {}

    for layer_name, monitors in all_monitors.items():
        spike_monitors[layer_name] = {k: v for k, v in monitors.items() if 'spikes' in k}
        state_monitors[layer_name] = {k: v for k, v in monitors.items() if 'state' in k}
        rate_monitors[layer_name] = {k: v for k, v in monitors.items() if 'rate' in k}
        neuron_groups[layer_name] = column.layers[layer_name].neuron_groups

    electrode_positions = CONFIG['electrode_positions']

    if verbose:
        print("Computing LFP using kernel method...")

    from tools.lfp_kernel import calculate_lfp_kernel_method
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

    spike_data = {}
    for layer_name, layer_spike_mons in spike_monitors.items():
        spike_data[layer_name] = {}
        for mon_name, mon in layer_spike_mons.items():
            spike_data[layer_name][mon_name] = {
                "times_ms": np.array(mon.t / ms),
                "spike_indices": np.array(mon.i),
            }

    # E population smooth rate as LFP proxy (1ms Gaussian window)
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
            rate_data[layer_name][mon_name] = {
                "t_ms": t_ms,
                "rate_hz": r_hz,
            }

    data = {
        "trial_id": trial_id,
        "seed": trial_seed,
        "time_array_ms": np.array(time_array),
        "electrode_positions": np.array(electrode_positions),
        "channel_labels": np.array(channel_labels, dtype=object),
        "channel_depths": np.array(channel_depths),
        "rate_data": rate_data,
        "spike_data": spike_data,
        "lfp_full": lfp_full,
        "baseline_ms": baseline_ms,
        "post_ms": stimuli_ms,
        "stim_onset_ms": baseline_ms,
    }

    n_elec = len(lfp_signals)
    lfp_matrix = np.vstack([lfp_signals[i] for i in range(n_elec)])
    bipolar_matrix = np.vstack([bipolar_signals[i] for i in range(len(bipolar_signals))])

    data["lfp_matrix"] = lfp_matrix
    data["bipolar_matrix"] = bipolar_matrix

    if verbose:
        print(f"Trial {trial_id} finished.\n")

    return data

def run_multiple_trials(
    config,
    n_trials=10,
    base_seed=None,
    baseline_ms=3000,
    stimuli_ms=3000,
    fs=10000,
    save_dir="results/trials",
    verbose=True,
):
    if base_seed is None:
        base_seed = config['simulation']['RANDOM_SEED']

    os.makedirs(save_dir, exist_ok=True)
    save_config_snapshot(save_dir)

    for trial_id in range(n_trials):
        data = run_single_trial(
            config=config,
            trial_id=trial_id,
            base_seed=base_seed,
            baseline_ms=baseline_ms,
            stimuli_ms=stimuli_ms,
            fs=fs,
            verbose=verbose,
        )

        save_dict = {
            "trial_id": data["trial_id"],
            "seed": data["seed"],
            "time_array_ms": data["time_array_ms"],
            "electrode_positions": data["electrode_positions"],
            "lfp_matrix": data["lfp_matrix"],
            "bipolar_matrix": data["bipolar_matrix"],
            "channel_labels": data["channel_labels"],
            "channel_depths": data["channel_depths"],
            "rate_data": data["rate_data"],
            "spike_data": data["spike_data"],
            "lfp_full": data["lfp_full"],
            "baseline_ms": data["baseline_ms"],
            "post_ms": data["post_ms"],
            "stim_onset_ms": data["stim_onset_ms"],
        }

        fname = os.path.join(save_dir, f"trial_{trial_id:03d}.npz")
        np.savez_compressed(fname, **save_dict)

        if verbose:
            print(f"Saved trial {trial_id} to {fname}")

if __name__ == "__main__":
    run_multiple_trials(
        CONFIG,
        n_trials=50,
        baseline_ms=2000,
        stimuli_ms=2000,
        fs=10000,
        save_dir="results/trials_06_04",
        verbose=True,
    )
