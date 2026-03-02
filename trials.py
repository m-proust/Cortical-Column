import os
import numpy as np
import brian2 as b2
from brian2 import *
from config.config2 import CONFIG
from src.column import CorticalColumn
from src.visualization import *
from src.analysis import *

def create_thalamic_input(column, CONFIG, baseline_time_ms, w_ext_AMPA, w_ext_NMDA=None):
    objects_to_add = []
  
    
    N_M = 40    
    N_P = 80 
   
    dt_rate = 1.0 
    total_time_ms = baseline_time_ms + 1500  
    n_steps = int(total_time_ms / dt_rate) + 1
 
    
    M_spontaneous = 10.0  
    M_onset_peak  = 120.0 
    M_sustained   = 50.0   
    M_transient_tau = 50.0 
    
    rate_M = np.ones(n_steps) * M_spontaneous
    stim_onset_step = int(baseline_time_ms / dt_rate)
    
    for i in range(stim_onset_step, n_steps):
        t_since_onset = (i - stim_onset_step) * dt_rate  
        transient = (M_onset_peak - M_sustained) * np.exp(-t_since_onset / M_transient_tau)
        rate_M[i] = M_sustained + transient

    
    P_spontaneous = 5.0
    P_onset_peak  = 40.0
    P_sustained   = 25.0
    P_transient_tau = 80.0
    
    rate_P = np.ones(n_steps) * P_spontaneous
    for i in range(stim_onset_step, n_steps):
        t_since_onset = (i - stim_onset_step) * dt_rate
        transient = (P_onset_peak - P_sustained) * np.exp(-t_since_onset / P_transient_tau)
        rate_P[i] = P_sustained + transient
    
    rate_M_array = TimedArray(rate_M * Hz, dt=dt_rate * ms)
    rate_P_array = TimedArray(rate_P * Hz, dt=dt_rate * ms)
    
    LGN_M = PoissonGroup(N_M, rates='rate_M_array(t)',
                         namespace={'rate_M_array': rate_M_array})
    LGN_P = PoissonGroup(N_P, rates='rate_P_array(t)',
                         namespace={'rate_P_array': rate_P_array})
    
    objects_to_add.extend([LGN_M, LGN_P])
   
    
    L4C = column.layers['L4C']
    L4C_E = L4C.neuron_groups['E']
    L4C_PV = L4C.neuron_groups['PV']
    L4C_SOM = L4C.neuron_groups['SOM']

    # --- Weight parameters ---
    # TC AMPA weight onto E cells: 3x background AMPA (conservative; literature: 2-4x)
    w_TC_AMPA_E  = w_ext_AMPA * 3      # ~3.75 nS
    # TC AMPA weight onto PV cells: ~1.5x the E weight (Cruikshank et al. 2007)
    # Note: PV cells have higher convergence (more TC inputs), so per-synapse
    # weight need not be as extreme to achieve strong feedforward inhibition.
    w_TC_AMPA_PV = w_ext_AMPA * 2.5    # ~3.1 nS (reduced: PV should be recruited by E, not directly by TC)
    # TC AMPA weight onto SOM cells (weaker, SOM get less direct TC drive)
    w_TC_AMPA_SOM = w_ext_AMPA * 1.5   # ~1.9 nS

    # NMDA component at TC synapses (ratio ~0.3-0.5 of AMPA; Gil et al. 1999)
    if w_ext_NMDA is not None:
        w_TC_NMDA_E  = w_ext_NMDA * 3
        w_TC_NMDA_PV = w_ext_NMDA * 2   # PV cells have lower NMDA/AMPA ratio
        w_TC_NMDA_SOM = w_ext_NMDA * 4  # SOM cells have higher NMDA expression

    # --- Connection probabilities ---
    # Each E neuron receives ~5-10 M inputs and ~15-30 P inputs on average
    # With N_M=40 and p=0.15 → ~6 M inputs; N_P=80 and p=0.12 → ~10 P inputs
    # PV neurons receive from a broader pool (higher convergence)

    p_M_to_E  = 0.08   # reduced from 0.15 → ~3 M inputs (was ~6)
    p_P_to_E  = 0.06   # reduced from 0.12 → ~5 P inputs (was ~10), total ~8 (was ~16)
    p_M_to_PV = 0.12   # reduced from 0.25 → ~5 M inputs (was ~10)
    p_P_to_PV = 0.08   # reduced from 0.15 → ~6 P inputs (was ~12), total ~11 (was ~22)
    p_M_to_SOM = 0.0   # removed: SOM should not receive direct TC drive
    p_P_to_SOM = 0.0   # removed: SOM activation should come from recurrent E

    # --- Create synapses: Magnocellular → L4C ---

    if w_ext_NMDA is not None:
        on_pre_E  = 'gE_AMPA += w_TC_AMPA_E; gE_NMDA += w_TC_NMDA_E'
        on_pre_PV = 'gE_AMPA += w_TC_AMPA_PV; gE_NMDA += w_TC_NMDA_PV'
        on_pre_SOM = 'gE_AMPA += w_TC_AMPA_SOM; gE_NMDA += w_TC_NMDA_SOM'
        ns_E   = {'w_TC_AMPA_E': w_TC_AMPA_E, 'w_TC_NMDA_E': w_TC_NMDA_E}
        ns_PV  = {'w_TC_AMPA_PV': w_TC_AMPA_PV, 'w_TC_NMDA_PV': w_TC_NMDA_PV}
        ns_SOM = {'w_TC_AMPA_SOM': w_TC_AMPA_SOM, 'w_TC_NMDA_SOM': w_TC_NMDA_SOM}
    else:
        on_pre_E  = 'gE_AMPA += w_TC_AMPA_E'
        on_pre_PV = 'gE_AMPA += w_TC_AMPA_PV'
        on_pre_SOM = 'gE_AMPA += w_TC_AMPA_SOM'
        ns_E   = {'w_TC_AMPA_E': w_TC_AMPA_E}
        ns_PV  = {'w_TC_AMPA_PV': w_TC_AMPA_PV}
        ns_SOM = {'w_TC_AMPA_SOM': w_TC_AMPA_SOM}

    syn_M_to_L4C_E = Synapses(LGN_M, L4C_E, on_pre=on_pre_E, namespace=ns_E)
    syn_M_to_L4C_E.connect(p=p_M_to_E)

    syn_M_to_L4C_PV = Synapses(LGN_M, L4C_PV, on_pre=on_pre_PV, namespace=ns_PV)
    syn_M_to_L4C_PV.connect(p=p_M_to_PV)

    syn_M_to_L4C_SOM = Synapses(LGN_M, L4C_SOM, on_pre=on_pre_SOM, namespace=ns_SOM)
    syn_M_to_L4C_SOM.connect(p=p_M_to_SOM)

    # --- Create synapses: Parvocellular → L4C ---

    syn_P_to_L4C_E = Synapses(LGN_P, L4C_E, on_pre=on_pre_E, namespace=ns_E)
    syn_P_to_L4C_E.connect(p=p_P_to_E)

    syn_P_to_L4C_PV = Synapses(LGN_P, L4C_PV, on_pre=on_pre_PV, namespace=ns_PV)
    syn_P_to_L4C_PV.connect(p=p_P_to_PV)

    syn_P_to_L4C_SOM = Synapses(LGN_P, L4C_SOM, on_pre=on_pre_SOM, namespace=ns_SOM)
    syn_P_to_L4C_SOM.connect(p=p_P_to_SOM)

    objects_to_add.extend([
        syn_M_to_L4C_E, syn_M_to_L4C_PV, syn_M_to_L4C_SOM,
        syn_P_to_L4C_E, syn_P_to_L4C_PV, syn_P_to_L4C_SOM,
    ])
    
    # =========================================================================
    # 4. THALAMOCORTICAL SYNAPSES TO L6
    # =========================================================================
    #
    # L6 receives direct but weaker LGN input (Hubel & Wiesel 1972;
    # Lund & Boothe 1975; Hendrickson et al. 1978).
    # Only a fraction of L6 neurons (~14% corticogeniculate cells) receive
    # suprathreshold direct TC input (Briggs & Usrey 2009).
    # L6 responds early (~35-45 ms; Nowak et al. 1995), consistent with
    # direct LGN drive rather than relayed via L4C.
    #
    # TC synapses in L6 are weaker than in L4C and very sparse.
    
    L6 = column.layers['L6']
    L6_E = L6.neuron_groups['E']
    L6_PV = L6.neuron_groups['PV']
    
    w_TC_AMPA_L6_E  = w_ext_AMPA * 2.0   # weaker than L4C (fewer synapses per axon)
    w_TC_AMPA_L6_PV = w_ext_AMPA * 3.0   # PV still gets stronger drive
    
    if w_ext_NMDA is not None:
        w_TC_NMDA_L6_E  = w_ext_NMDA * 2.0
        w_TC_NMDA_L6_PV = w_ext_NMDA * 1.5
        on_pre_L6_E  = 'gE_AMPA += w_TC_AMPA_L6_E; gE_NMDA += w_TC_NMDA_L6_E'
        on_pre_L6_PV = 'gE_AMPA += w_TC_AMPA_L6_PV; gE_NMDA += w_TC_NMDA_L6_PV'
        ns_L6_E  = {'w_TC_AMPA_L6_E': w_TC_AMPA_L6_E, 'w_TC_NMDA_L6_E': w_TC_NMDA_L6_E}
        ns_L6_PV = {'w_TC_AMPA_L6_PV': w_TC_AMPA_L6_PV, 'w_TC_NMDA_L6_PV': w_TC_NMDA_L6_PV}
    else:
        on_pre_L6_E  = 'gE_AMPA += w_TC_AMPA_L6_E'
        on_pre_L6_PV = 'gE_AMPA += w_TC_AMPA_L6_PV'
        ns_L6_E  = {'w_TC_AMPA_L6_E': w_TC_AMPA_L6_E}
        ns_L6_PV = {'w_TC_AMPA_L6_PV': w_TC_AMPA_L6_PV}

    # Very sparse: only ~5% of L6 E cells get direct LGN input
    # (subset of corticogeniculate neurons; Briggs & Usrey 2009)
    p_M_to_L6_E  = 0.04
    p_P_to_L6_E  = 0.03
    p_M_to_L6_PV = 0.03
    p_P_to_L6_PV = 0.02

    syn_M_to_L6_E = Synapses(LGN_M, L6_E, on_pre=on_pre_L6_E, namespace=ns_L6_E)
    syn_M_to_L6_E.connect(p=p_M_to_L6_E)

    syn_P_to_L6_E = Synapses(LGN_P, L6_E, on_pre=on_pre_L6_E, namespace=ns_L6_E)
    syn_P_to_L6_E.connect(p=p_P_to_L6_E)

    syn_M_to_L6_PV = Synapses(LGN_M, L6_PV, on_pre=on_pre_L6_PV, namespace=ns_L6_PV)
    syn_M_to_L6_PV.connect(p=p_M_to_L6_PV)

    syn_P_to_L6_PV = Synapses(LGN_P, L6_PV, on_pre=on_pre_L6_PV, namespace=ns_L6_PV)
    syn_P_to_L6_PV.connect(p=p_P_to_L6_PV)
    
    objects_to_add.extend([
        syn_M_to_L6_E, syn_P_to_L6_E,
        syn_M_to_L6_PV, syn_P_to_L6_PV,
    ])

    return objects_to_add





def run_single_trial(
    config,
    trial_id=0,
    base_seed=None,
    baseline_ms=1000,    
    post_ms=500,      
    fs=10000,
    verbose=True,
):
    if base_seed is None:
        base_seed = config['simulation']['RANDOM_SEED']

    trial_seed = int(base_seed + trial_id)
    np.random.seed(trial_seed)
    # trial_seed = base_seed
    b2.seed(trial_seed)

    b2.start_scope()
    b2.defaultclock.dt = config['simulation']['DT']

    if verbose:
        print(f"\n=== Running trial {trial_id} with seed {trial_seed} ===")
        print("Creating cortical column...")

    column = CorticalColumn(column_id=0, config=config)

    # add heterogeneity (this can be commented out)
    # for layer_name, layer in column.layers.items():
    #     add_heterogeneity_to_layer(layer, config)

    all_monitors = column.get_all_monitors()


    ##############################################
    w_ext_AMPA = CONFIG['synapses']['Q']['EXT_AMPA']
    w_ext_NMDA = CONFIG['synapses']['Q']['EXT_NMDA']
    
   
    
   
    # L23 = column.layers['L23']
    # cfg_L23 = CONFIG['layers']['L23']
   
    
    # L23_SOM_grp = L23.neuron_groups['SOM']
    # N_stim_SOM = int(cfg_L23['poisson_inputs']['SOM']['N'])
    # stim_rate_SOM = 10*Hz  
    # L23_SOM_stimNMDA= PoissonInput(L23_SOM_grp, 'gE_NMDA', 
    #                               N=N_stim_SOM, 
    #                               rate=stim_rate_SOM, 
    #                               weight=w_ext_NMDA)  
    # column.network.add(L23_SOM_stimNMDA)

   

    column.network.run(baseline_ms * ms)
   
    # column.network.remove(L23_SOM_stimNMDA)
   
    
   
    thalamic_objects = create_thalamic_input(
        column, CONFIG,
        baseline_time_ms=baseline_ms,
        w_ext_AMPA=w_ext_AMPA,
        w_ext_NMDA=w_ext_NMDA,
    )
    column.network.add(*thalamic_objects)



    column.network.run(post_ms * ms)

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

    total_sim_ms = baseline_ms + post_ms

    if verbose:
        print("Computing LFP using kernel method...")

    lfp_signals, time_array = calculate_lfp_kernel_method(
        spike_monitors,
        neuron_groups,
        config['layers'],
        electrode_positions,
        fs=fs,
        sim_duration_ms=total_sim_ms,  
    )

    # from lfp_mazzoni_method import calculate_lfp_mazzoni

    # # Compute LFP
    # lfp_signals, time_array = calculate_lfp_mazzoni(
    #     spike_monitors,
    #     neuron_groups,
    #     CONFIG['layers'],
    #     electrode_positions,
    #     fs=10000,
    #     sim_duration_ms=total_sim_ms
    # )


    if verbose:
        print("Computing CSD from monopolar LFP...")

    csd, csd_depths, csd_sort_idx = compute_csd_from_lfp(
        lfp_signals,
        electrode_positions,
        sigma=0.3,
        vaknin=True,
    )

    if verbose:
        print("Computing bipolar LFP...")

    bipolar_signals, channel_labels, channel_depths = compute_bipolar_lfp(
        lfp_signals,
        electrode_positions,
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
    spike_data = {}
    for layer_name, layer_spk_mons in spike_monitors.items():
        spike_data[layer_name] = {}
        for mon_name, mon in layer_spk_mons.items():
            spike_data[layer_name][mon_name] = {
                'times_ms': np.array(mon.t / ms),
                'indices':  np.array(mon.i),
            }


    data = {
        "trial_id": trial_id,
        "seed": trial_seed,
        "time_array_ms": np.array(time_array),
        "electrode_positions": np.array(electrode_positions),
        "csd": np.array(csd),
        "csd_depths": np.array(csd_depths),
        "csd_sort_idx": np.array(csd_sort_idx),
        "channel_labels": np.array(channel_labels, dtype=object),
        "channel_depths": np.array(channel_depths),
        "rate_data": rate_data,
        "spike_data" : spike_data,
        "baseline_ms": baseline_ms,
        "post_ms": post_ms,
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
    baseline_ms=1000,
    post_ms=500,
    fs=10000,
    save_dir="results/trials",
    verbose=True,
):
    if base_seed is None:
        base_seed = config['simulation']['RANDOM_SEED']

    os.makedirs(save_dir, exist_ok=True)

    for trial_id in range(n_trials):
        data = run_single_trial(
            config=config,
            trial_id=trial_id,
            base_seed=base_seed,
            baseline_ms=baseline_ms,
            post_ms=post_ms,
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
            "csd": data["csd"],
            "csd_depths": data["csd_depths"],
            "csd_sort_idx": data["csd_sort_idx"],
            "channel_labels": data["channel_labels"],
            "channel_depths": data["channel_depths"],
            "rate_data": data["rate_data"],
            "spike_data": data["spike_data"],
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
        post_ms=1500,
        fs=10000,
        save_dir="results/27_02_new_thalamic_input",
        verbose=True,
    )
