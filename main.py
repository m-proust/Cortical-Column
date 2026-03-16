import numpy as np
import brian2 as b2
from brian2 import *
from brian2tools import *
from config.config2 import CONFIG
from src.column import CorticalColumn
from src.visualization import *
from src.analysis import *


def main():
    np.random.seed(CONFIG['simulation']['RANDOM_SEED'])
    b2.start_scope()
    b2.defaultclock.dt = CONFIG['simulation']['DT']

    
    baseline_time = 2000 # In ms, time during which to run the baseline simulation
    stimuli_time = 2000 # In ms, time during which to run the simulation after adding the stimuli

    print(" Creating cortical column...")
    column = CorticalColumn(column_id=0, config=CONFIG)
 
    
    # for layer_name, layer in column.layers.items():
    #     add_heterogeneity_to_layer(layer, CONFIG) # optional
    
    all_monitors = column.get_all_monitors()
    
   

    w_ext_AMPA = CONFIG['synapses']['Q']['EXT_AMPA']
    # w_ext_NMDA = CONFIG['synapses']['Q']['EXT_NMDA']
    
   
    
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

   

    column.network.run(baseline_time * ms)
   
    # column.network.remove(L23_SOM_stimNMDA)
    # L23 = column.layers['L23']
    # cfg_L23 = CONFIG['layers']['L23']
   
    
    # L23_VIP_grp = L23.neuron_groups['VIP']
    # N_stim_VIP = int(cfg_L23['poisson_inputs']['VIP']['N'])
    # stim_rate_VIP = 6*Hz  
    # L23_VIP_stim= PoissonInput(L23_VIP_grp, 'gE_AMPA', 
    #                               N=N_stim_VIP, 
    #                               rate=stim_rate_VIP, 
    #                               weight=w_ext_AMPA)  
    # column.network.add(L23_VIP_stim)

   
    


    L4C = column.layers['L4C']
    cfg_L4C = CONFIG['layers']['L4C']
   
    
    L4C_E_grp = L4C.neuron_groups['E']
    N_stim_E = 40
    stim_rate_E = 20*Hz  
    L4C_E_stimAMPA = PoissonInput(L4C_E_grp, 'gE_AMPA', 
                                  N=N_stim_E, 
                                  rate=stim_rate_E, 
                                  weight=w_ext_AMPA*2)  
    
    
    L4C_PV_grp = L4C.neuron_groups['PV']
    N_stim_PV = 30
    stim_rate_PV = 20*Hz 
    L4C_PV_stim = PoissonInput(L4C_PV_grp, 'gE_AMPA', 
                               N=N_stim_PV, 
                               rate=stim_rate_PV, 
                               weight=w_ext_AMPA*2)  
    
    
    L6 = column.layers['L6']
    cfg_L6 = CONFIG['layers']['L6']
    L6_PV_grp = L6.neuron_groups['PV']
    N_stim_L6_PV = int(cfg_L6['poisson_inputs']['PV']['N'])
    stim_rate_L6_PV = 10*Hz  
    
    L6_PV_stim = PoissonInput(L6_PV_grp, 'gE_AMPA',
                             N=N_stim_L6_PV, 
                             rate=stim_rate_L6_PV, 
                             weight=w_ext_AMPA*2)
    L6_E_grp = L6.neuron_groups['E']
    N_stim_L6_E = int(cfg_L6['poisson_inputs']['E']['N'])
    stim_rate_L6_E = 10*Hz  
    
    L6_E_stim = PoissonInput(L6_E_grp, 'gE_AMPA',
                             N=N_stim_L6_E, 
                             rate=stim_rate_L6_E, 
                             weight=w_ext_AMPA*2)



    column.network.add(L6_E_stim, L6_PV_stim)
    column.network.add(L4C_E_stimAMPA, L4C_PV_stim)


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

    print("Computing LFP using kernel method...")
    from lfp_kernel import calculate_lfp_kernel_method
    lfp_signals, time_array = calculate_lfp_kernel_method(
        spike_monitors,
        neuron_groups,
        CONFIG['layers'],
        electrode_positions,
        sim_duration_ms=baseline_time + stimuli_time
    )
    # from lfp_mazzoni_method import calculate_lfp_mazzoni

    # # Compute LFP
    # lfp_signals, time_array = calculate_lfp_mazzoni(
    #     spike_monitors,
    #     neuron_groups,
    #     CONFIG['layers'],
    #     electrode_positions,
    #     fs=10000,
    #     sim_duration_ms=baseline_time + stimuli_time
    # )
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

    # ---- Pairwise Phase Consistency (PPC) ----
    print("Computing Pairwise Phase Consistency...")
    fig_ppc = plot_ppc_by_layer(
        spike_monitors,
        bipolar_signals,
        channel_depths,
        time_array,
        CONFIG['layers'],
        freq_range=(1, 100),
        n_freqs=50,
        fs=10000,
        baseline_time=baseline_time,
        stimuli_time=stimuli_time,
        min_spikes_per_neuron=5,
        bandwidth=4.0,
        smooth_sigma=1.5,
    )

    plt.show()


if __name__ == "__main__":
    main()