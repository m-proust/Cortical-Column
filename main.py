import numpy as np
import brian2 as b2
from brian2 import *
from brian2tools import *
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
    
    # =========================================================================
    # 5. PRINT DIAGNOSTIC SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("THALAMIC INPUT CONFIGURATION")
    print("=" * 60)
    print(f"LGN pool: {N_M} M-cells + {N_P} P-cells = {N_M + N_P} total")
    print(f"\nM-cell rates: spontaneous={M_spontaneous}Hz, "
          f"onset_peak={M_onset_peak}Hz, sustained={M_sustained}Hz, "
          f"tau={M_transient_tau}ms")
    print(f"P-cell rates: spontaneous={P_spontaneous}Hz, "
          f"onset_peak={P_onset_peak}Hz, sustained={P_sustained}Hz, "
          f"tau={P_transient_tau}ms")
    
    print(f"\nL4C connections:")
    print(f"  E:   M(p={p_M_to_E}, ~{N_M*p_M_to_E:.0f} inputs) + "
          f"P(p={p_P_to_E}, ~{N_P*p_P_to_E:.0f} inputs) = "
          f"~{N_M*p_M_to_E + N_P*p_P_to_E:.0f} TC inputs/neuron")
    print(f"       AMPA weight = {w_TC_AMPA_E}")
    print(f"  PV:  M(p={p_M_to_PV}, ~{N_M*p_M_to_PV:.0f} inputs) + "
          f"P(p={p_P_to_PV}, ~{N_P*p_P_to_PV:.0f} inputs) = "
          f"~{N_M*p_M_to_PV + N_P*p_P_to_PV:.0f} TC inputs/neuron")
    print(f"       AMPA weight = {w_TC_AMPA_PV}")
    print(f"  SOM: M(p={p_M_to_SOM}) + P(p={p_P_to_SOM})")
    
    print(f"\nL6 connections (sparse):")
    print(f"  E:   ~{N_M*p_M_to_L6_E + N_P*p_P_to_L6_E:.1f} TC inputs/neuron")
    print(f"  PV:  ~{N_M*p_M_to_L6_PV + N_P*p_P_to_L6_PV:.1f} TC inputs/neuron")
    print("=" * 60 + "\n")
    
    return objects_to_add






def main():
    np.random.seed(CONFIG['simulation']['RANDOM_SEED'])
    b2.start_scope()
    b2.defaultclock.dt = CONFIG['simulation']['DT']

    
    baseline_time = 2000 # In ms, time during which to run the baseline simulation
    stimuli_time = 1500 # In ms, time during which to run the simulation after adding the stimuli

    print(" Creating cortical column...")
    column = CorticalColumn(column_id=0, config=CONFIG)
 
    
    # for layer_name, layer in column.layers.items():
    #     add_heterogeneity_to_layer(layer, CONFIG) # optional
    
    all_monitors = column.get_all_monitors()
    
   

    w_ext_AMPA = CONFIG['synapses']['Q']['EXT_AMPA']
    w_ext_NMDA = CONFIG['synapses']['Q']['EXT_NMDA']
    

    column.network.run(baseline_time * ms)

    thalamic_objects = create_thalamic_input(
        column, CONFIG,
        baseline_time_ms=baseline_time,
        w_ext_AMPA=w_ext_AMPA,
        w_ext_NMDA=w_ext_NMDA,
    )
    column.network.add(*thalamic_objects)

   
    

    
    
    # L6 = column.layers['L6']
    # cfg_L6 = CONFIG['layers']['L6']
    # L6_PV_grp = L6.neuron_groups['PV']
    # N_stim_L6_PV = int(cfg_L6['poisson_inputs']['PV']['N'])
    # stim_rate_L6_PV = 15*Hz  
    
    # L6_PV_stim = PoissonInput(L6_PV_grp, 'gE_AMPA',
    #                          N=N_stim_L6_PV, 
    #                          rate=stim_rate_L6_PV, 
    #                          weight=w_ext_AMPA*3)
    # L6_E_grp = L6.neuron_groups['E']
    # N_stim_L6_E = int(cfg_L6['poisson_inputs']['E']['N'])
    # stim_rate_L6_E = 15*Hz  
    
    # L6_E_stim = PoissonInput(L6_E_grp, 'gE_AMPA',
    #                          N=N_stim_L6_E, 
    #                          rate=stim_rate_L6_E, 
    #                          weight=w_ext_AMPA)



    # column.network.add(L6_E_stim, L6_PV_stim)
    # column.network.add(L4C_E_stimAMPA, L4C_PV_stim)


    column.network.run(stimuli_time* ms)

    print("Simulation complete")
    
    spike_monitors = {}
    state_monitors = {}
    rate_monitors = {}
    neuron_groups = {}
    
    for layer_name, monitors in all_monitors.items():
        spike_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'spikes' in k
        }
        state_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'state' in k
        }
        rate_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'rate' in k
        }
        neuron_groups[layer_name] = column.layers[layer_name].neuron_groups
    
    for layer_name, monitors in all_monitors.items():
        spike_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'spikes' in k
        }
        state_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'state' in k
        }
        rate_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'rate' in k
        }

    electrode_positions = CONFIG['electrode_positions']

    # print("Computing LFP using kernel method...")
    # lfp_signals, time_array = calculate_lfp_kernel_method(
    #     spike_monitors,
    #     neuron_groups,
    #     CONFIG['layers'],
    #     electrode_positions,
    #     fs=10000,
    #     sim_duration_ms=baseline_time + stimuli_time
    # )
    from lfp_mazzoni_method import calculate_lfp_mazzoni

    # Compute LFP
    lfp_signals, time_array = calculate_lfp_mazzoni(
        spike_monitors,
        neuron_groups,
        CONFIG['layers'],
        electrode_positions,
        fs=10000,
        sim_duration_ms=baseline_time + stimuli_time
    )
    print("Computing bipolar LFP...")
    bipolar_signals, channel_labels, channel_depths = compute_bipolar_lfp(
        lfp_signals,
        electrode_positions
    )

    fig_raster = plot_raster(spike_monitors, baseline_time, stimuli_time, CONFIG['layers'])

    fig_power_lfp = plot_lfp_power_comparison_kernel(
                        lfp_signals,
                        time_array,
                        electrode_positions,
                        baseline_time=baseline_time,
                        pre_stim_duration=1000,
                        post_stim_duration=1000,
                        transient_skip=500
                    )

    fig_power_bipolar = plot_bipolar_power_comparison_kernel(
                        bipolar_signals,
                        channel_labels,
                        channel_depths,
                        time_array,
                        baseline_time=baseline_time,
                        pre_stim_duration=1000,
                        post_stim_duration=1000,
                        transient_skip=500
                    )


    fig_rate = plot_rate(rate_monitors, CONFIG['layers'], baseline_time, stimuli_time,
                 smooth_window=15*ms, 
                 ylim_max=80,      
                 show_stats=True)  
    fig_lfp = plot_lfp_comparison(lfp_signals, bipolar_signals, time_array, electrode_positions, 
                        channel_labels, channel_depths, figsize=(18, 12), time_range=(1000, 3500))
    
    plt.show()


if __name__ == "__main__":
    main()