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
    stimuli_time = 1500 # In ms, time during which to run the simulation after adding the stimuli

    print(" Creating cortical column...")
    column = CorticalColumn(column_id=0, config=CONFIG)
 
    
    # for layer_name, layer in column.layers.items():
    #     add_heterogeneity_to_layer(layer, CONFIG) # optional
    
    all_monitors = column.get_all_monitors()
    
   

    w_ext_AMPA = CONFIG['synapses']['Q']['EXT_AMPA']
  

   
        

    from lgn_to_brian2_v2 import make_lgn_inputs

    column.network.run(baseline_time * ms)

    lgn = make_lgn_inputs(
        column, CONFIG,
        npz_path='lgn_spikes_grating.npz',
        baseline_time_ms=baseline_time,        # 2000
        stimuli_time_ms=stimuli_time,           # 1500
        lgn_t_start=500.0,                      # gray_screen_dur = 0.5s = 500ms
        layers_to_connect=['L4C', 'L6'],
        drive_scale=2
    )

    for obj_list in lgn.values():
        column.network.add(*obj_list)

    column.network.run(stimuli_time * ms)

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
    from lfp_kernel import calculate_lfp_kernel_method
    print("Computing LFP using kernel method...")
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
    
    plt.show()


if __name__ == "__main__":
    main()