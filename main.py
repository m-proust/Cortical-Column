import numpy as np
import brian2 as b2
from brian2 import *
from brian2tools import *
from config.config2 import CONFIG
from src.column import CorticalColumn
from src.visualization import *
from src.analysis import *
from lfp_mazzoni_perlayer import calculate_lfp_mazzoni_perlayer


def main():
    np.random.seed(CONFIG['simulation']['RANDOM_SEED'])
    b2.start_scope()
    b2.defaultclock.dt = CONFIG['simulation']['DT']

    
    baseline_time = 2000 # In ms, time during which to run the baseline simulation
    stimuli_time = 2500 # In ms, time during which to run the simulation after adding the stimuli

    print(" Creating cortical column...")
    column = CorticalColumn(column_id=0, config=CONFIG)
 
    
    for layer_name, layer in column.layers.items():
        add_heterogeneity_to_layer(layer, CONFIG) # optional
    
    all_monitors = column.get_all_monitors()
    
   

    w_ext_AMPA = CONFIG['synapses']['Q']['EXT_AMPA']
   

   

    column.network.run(baseline_time * ms)
   
    
   

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
    stim_rate_L6_PV = 4*Hz  
    
    L6_PV_stim = PoissonInput(L6_PV_grp, 'gE_AMPA',
                             N=N_stim_L6_PV, 
                             rate=stim_rate_L6_PV, 
                             weight=w_ext_AMPA)
    L6_E_grp = L6.neuron_groups['E']
    N_stim_L6_E = 10
    stim_rate_L6_E = 4*Hz  
    
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
    lfp_signals, time_array = calculate_lfp_kernel_method(
        spike_monitors,
        neuron_groups,
        CONFIG['layers'],
        electrode_positions,
        # fs=10000,
        sim_duration_ms=baseline_time + stimuli_time
    )

    print("Computing bipolar LFP...")
    bipolar_signals, channel_labels, channel_depths = compute_bipolar_lfp(
        lfp_signals,
        electrode_positions
    )

    print("Computing LFP using Mazzoni per-layer method...")
    mazzoni_lfp, mazzoni_time = calculate_lfp_mazzoni_perlayer(
        state_monitors,
        CONFIG['layers'],
        fs=10000,
        dt_ms=1.0,
    )

    fig_raster = plot_raster(spike_monitors, baseline_time, stimuli_time, CONFIG['layers'])

    fig_power_lfp = plot_lfp_power_comparison_kernel(
                        lfp_signals,
                        time_array,
                        electrode_positions,
                        baseline_time=baseline_time,
                        pre_stim_duration=300,
                        post_stim_duration=300,
                        transient_skip=300
                    )

    fig_power_bipolar = plot_bipolar_power_comparison_kernel(
                        bipolar_signals,
                        channel_labels,
                        channel_depths,
                        time_array,
                        baseline_time=baseline_time,
                        pre_stim_duration=300,
                        post_stim_duration=300,
                        transient_skip=300
                    )


    fig_power_mazzoni = plot_lfp_power_comparison_mazzoni_perlayer(
                            mazzoni_lfp,
                            mazzoni_time,
                            baseline_time=baseline_time,
                            pre_stim_duration=1000,
                            post_stim_duration=1000,
                            transient_skip=500,
                            fs=10000,
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