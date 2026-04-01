
import os
import shutil
import numpy as np
import brian2 as b2
from brian2 import *
from config.config2 import CONFIG
from src.column import CorticalColumn
import cleo
from cleo import ephys


CONFIG_FILES = [
    "config/config2.py",
    "config/conductances_AMPA2_alpha_v2.csv",
    "config/conductances_NMDA2_alpha_v2.csv",
    "config/connection_probabilities2.csv",
    "trials_cleo.py",
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
    baseline_ms=2000,
    stimuli_ms=2000,
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
        print(f"\n=== Trial {trial_id} (seed {trial_seed}) ===")
        print("Creating cortical column...")

    column = CorticalColumn(column_id=0, config=config)

    sim = cleo.CLSimulator(column.network)

    rwslfp = ephys.RWSLFPSignalFromPSCs()
    mua = ephys.MultiUnitActivity(threshold_sigma=3.5)

    probe = column.electrode
    probe.add_signals(mua, rwslfp)

    sim.set_io_processor(
        cleo.ioproc.RecordOnlyProcessor(sample_period=1 * b2.ms)
    )

    for layer_name, layer in column.layers.items():
        group = layer.neuron_groups['E']
        sim.inject(
            probe, group,
            Iampa_var_names=['IsynE'],
            Igaba_var_names=['IsynI'],
        )

    if verbose:
        print(f"Running baseline ({baseline_ms} ms)...")
    sim.run(baseline_ms * b2.ms)

    w_ext_AMPA = config['synapses']['Q']['EXT_AMPA']

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
                             weight=w_ext_AMPA)
    L6_E_grp = L6.neuron_groups['E']
    N_stim_L6_E = 10
    stim_rate_L6_E = 5*Hz  
    
    L6_E_stim = PoissonInput(L6_E_grp, 'gE_AMPA',
                             N=N_stim_L6_E, 
                             rate=stim_rate_L6_E, 
                             weight=w_ext_AMPA*2)



    column.network.add(L6_E_stim, L6_PV_stim)
    column.network.add(L4C_E_stimAMPA, L4C_PV_stim)



 
    if verbose:
        print(f"Running stimulus ({stimuli_ms} ms)...")
    sim.run(stimuli_ms * b2.ms)

    if verbose:
        print("Simulation complete, extracting data...")

    mua_t_ms = np.array(mua.t / b2.ms)
    mua_i = np.array(mua.i)

    lfp = np.array(rwslfp.lfp) 
    lfp_t_ms = np.array(rwslfp.t / b2.ms)

    all_monitors = column.get_all_monitors()
    spike_data = {}
    for layer_name, monitors in all_monitors.items():
        spike_data[layer_name] = {}
        for k, v in monitors.items():
            if 'spikes' in k:
                spike_data[layer_name][k] = {
                    "times_ms": np.array(v.t / b2.ms),
                    "spike_indices": np.array(v.i),
                }

    probe_coords_mm = np.array(probe.coords / b2.mm)

    data = {
        "trial_id": trial_id,
        "seed": trial_seed,
        "baseline_ms": baseline_ms,
        "stimuli_ms": stimuli_ms,
        "stim_onset_ms": baseline_ms,
        "mua_t_ms": mua_t_ms,
        "mua_i": mua_i,
        "lfp": lfp,
        "lfp_t_ms": lfp_t_ms,
        "probe_coords_mm": probe_coords_mm,
        "n_channels": probe.n,
        "spike_data": spike_data,
    }

    if verbose:
        print(f"Trial {trial_id} done. MUA spikes: {len(mua_t_ms)}, "
              f"LFP shape: {lfp.shape}")

    return data


def run_multiple_trials(
    config,
    n_trials=10,
    base_seed=None,
    baseline_ms=2000,
    stimuli_ms=2000,
    save_dir="results/trials_cleo",
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
            verbose=verbose,
        )

        fname = os.path.join(save_dir, f"trial_{trial_id:03d}.npz")
        np.savez_compressed(
            fname,
            trial_id=data["trial_id"],
            seed=data["seed"],
            baseline_ms=data["baseline_ms"],
            stimuli_ms=data["stimuli_ms"],
            stim_onset_ms=data["stim_onset_ms"],
            mua_t_ms=data["mua_t_ms"],
            mua_i=data["mua_i"],
            lfp=data["lfp"],
            lfp_t_ms=data["lfp_t_ms"],
            probe_coords_mm=data["probe_coords_mm"],
            n_channels=data["n_channels"],
            spike_data=data["spike_data"],
        )

        if verbose:
            print(f"Saved {fname}")


if __name__ == "__main__":
    run_multiple_trials(
        CONFIG,
        n_trials=50,
        baseline_ms=2000,
        stimuli_ms=2000,
        save_dir="results/trials_cleo",
        verbose=True,
    )
