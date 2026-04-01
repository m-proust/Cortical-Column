import numpy as np
import brian2 as b2
from brian2 import *
from brian2tools import *
from config.config2 import CONFIG
from src.column import CorticalColumn
from src.visualization import *

import matplotlib.pyplot as plt
import numpy as np

def plot_raster(spike_data, baseline_time, stimuli_time, layer_configs, figsize=(15, 10)):
    fig, axes = plt.subplots(len(spike_data), 1, figsize=figsize)
    
    # Ensure axes is iterable if only one subplot
    if len(spike_data) == 1:
        axes = [axes]
    
    for i, (layer_name, monitors) in enumerate(spike_data.items()):
        ax = axes[i]
        config = layer_configs[layer_name]
        
        # Excitatory neurons
        if 'E_spikes' in monitors:
            ax.scatter(monitors['E_spikes']["times_ms"] / 1000,
                       monitors['E_spikes']["spike_indices"],
                       color='green', s=0.5, alpha=0.6, label="E")
        
        # SOM neurons
        if 'SOM_spikes' in monitors:
            ax.scatter(monitors['SOM_spikes']["times_ms"] / 1000,
                       monitors['SOM_spikes']["spike_indices"] + config['neuron_counts']['E'],
                       color='blue', s=0.5, alpha=0.8, label="SOM")
        
        # PV neurons
        if 'PV_spikes' in monitors:
            offset = config['neuron_counts']['E']
            if 'SOM_spikes' in monitors:
                offset += config['neuron_counts']['SOM']
            
            ax.scatter(monitors['PV_spikes']["times_ms"] / 1000,
                       monitors['PV_spikes']["spike_indices"] + offset,
                       color='red', s=0.5, alpha=0.8, label="PV")
        
        # VIP neurons
        if 'VIP_spikes' in monitors:
            if layer_name == 'L1':
                offset = 0
            else:
                offset = config['neuron_counts']['E']
                offset += config['neuron_counts'].get('SOM', 0)
                offset += config['neuron_counts'].get('PV', 0)
            
            ax.scatter(monitors['VIP_spikes']["times_ms"] / 1000,
                       monitors['VIP_spikes']["spike_indices"] + offset,
                       color='gold', s=0.5, alpha=0.8, label="VIP")
        
        # Axis settings
        x_lim = (baseline_time + stimuli_time) / 1000
        ax.set_xlim(0.3, x_lim)
        ax.set_ylabel('Neuron index')
        ax.set_title(f'{layer_name} Spike Raster Plot')
        ax.legend()
    
    plt.tight_layout()
    return fig

baseline_time = 2000 # In ms, time during which to run the baseline simulation
stimuli_time = 2000 # In ms, time during which to run the simulation after adding the stimuli

fname = f"results/trials_28_03/trial_005.npz"
data = np.load(fname, allow_pickle=True)

trial_data = {
    'seed' : data["seed"],
    'time': data["time_array_ms"],
    'bipolar_lfp': data["bipolar_matrix"],
    'lfp_matrix': data["lfp_matrix"],
    'csd': data["csd"],
    'rate_data': data["rate_data"].item() if data["rate_data"].size == 1 else data["rate_data"],
    'spike_data': data["spike_data"].item() if data["spike_data"].size == 1 else data["spike_data"],
    'baseline_ms': float(data["baseline_ms"]),
    'stim_onset_ms': float(data["stim_onset_ms"]),
    'channel_labels': data["channel_labels"],
    'channel_depths': data["channel_depths"],
    'electrode_positions': data["electrode_positions"]
}
spike_data = trial_data['spike_data']

fig_raster = plot_raster(spike_data, baseline_time, stimuli_time, CONFIG['layers'])


plt.show()
