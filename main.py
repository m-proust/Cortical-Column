"""Run one simulation and show plots.

Simulates a baseline epoch, then a stimulus epoch and shows
rasters, rates and LFP/bipolar power. 

Run:
    path/to/your/venv/bin/python main.py

You can change
    STIM_PROFILE  -- "feedforward" or "feedback", or any other stimulus you may have added (in stim_profiles.py)
    baseline_time -- ms of resting simulation before the stimulus
    stimuli_time  -- ms of simulation with the stimulus on
"""
import numpy as np
import brian2 as b2
from brian2 import *
from brian2tools import *
from config.config import CONFIG
from src.column import CorticalColumn
from src.visualization import *
from src.analysis import *
from tools.lfp_kernel import calculate_lfp_kernel_method
from stim_profiles import build_epoch

STIM_PROFILE = "feedback"


def main():
    seed = 58925
    np.random.seed(seed)
    b2.seed(seed)
    b2.start_scope()
    b2.defaultclock.dt = CONFIG['simulation']['DT']

    baseline_time = 2000  # ms of resting simulation
    stimuli_time = 2500   # ms of simulation with stimulus on

    print("Creating cortical column...")
    column = CorticalColumn(column_id=0, config=CONFIG)

    for layer_name, layer in column.layers.items():
        add_heterogeneity_to_layer(layer, CONFIG)

    all_monitors = column.get_all_monitors()
    w_ext_AMPA = CONFIG['synapses']['Q']['EXT_AMPA']
    w_ext_NMDA = CONFIG['synapses']['Q'].get('EXT_NMDA', w_ext_AMPA)

    # ---- baseline epoch ----
    base_inputs = build_epoch(STIM_PROFILE, "baseline", column, w_ext_AMPA, w_ext_NMDA)
    column.network.add(*base_inputs)
    column.network.run(baseline_time * ms)

    # ---- stimulus epoch ----
    stim_inputs = build_epoch(STIM_PROFILE, "stim", column, w_ext_AMPA, w_ext_NMDA)
    column.network.add(*stim_inputs)
    column.network.run(stimuli_time * ms)

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

    print("Computing LFP using kernel method...")
    lfp_signals, time_array = calculate_lfp_kernel_method(
        spike_monitors,
        neuron_groups,
        CONFIG['layers'],
        electrode_positions,
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
                        pre_stim_duration=500,
                        post_stim_duration=500,
                        transient_skip=200
                    )

    fig_power_bipolar = plot_bipolar_power_comparison_kernel(
                        bipolar_signals,
                        channel_labels,
                        channel_depths,
                        time_array,
                        baseline_time=baseline_time,
                        pre_stim_duration=1500,
                        post_stim_duration=1500,
                        transient_skip=200
                    )


    fig_rate = plot_rate(rate_monitors, CONFIG['layers'], baseline_time, stimuli_time,
                 smooth_window=15*ms,
                 ylim_max=80,
                 show_stats=True)
    fig_lfp = plot_lfp_comparison(lfp_signals, bipolar_signals, time_array, electrode_positions,
                        channel_labels, channel_depths, figsize=(18, 12), time_range=(1000, 3500))

    

    fig_mean_rates = plot_mean_rates_bar(rate_monitors, CONFIG['layers'],
                                         baseline_time, stimuli_time,
                                         transient_skip=300)




    plt.show()


if __name__ == "__main__":
    main()