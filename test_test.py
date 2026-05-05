import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import signal
from scipy.signal import detrend
from scipy.signal.windows import dpss
import seaborn as sns
from config.config import CONFIG
from src.visualization import *
from src.analysis import *

from brian2 import ms

plt.rcParams.update({
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
})
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Paired')



def plot_rate_from_data(rate_data, layer_configs, figsize=(10, 12),
                        smooth_window_ms=15.0, ylim_max=80, show_stats=True):
    layer_names = list(layer_configs.keys()) if isinstance(layer_configs, dict) else list(rate_data.keys())
    n_layers = len(layer_names)

    fig, axes = plt.subplots(n_layers, 1, sharex=True, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    pop_colors = {'E': 'royalblue', 'PV': 'darkorange', 'SOM': 'forestgreen', 'VIP': 'gold'}

    for ax, layer_name in zip(axes, layer_names):
        layer_rates = rate_data.get(layer_name, {})
        stats_text = []

        for pop_key in sorted(layer_rates.keys()):
            entry = layer_rates[pop_key]
            t = np.asarray(entry['t_ms'])
            r = np.asarray(entry['rate_hz'])

            if t.size > 1:
                dt_ms = np.median(np.diff(t))
                win = max(1, int(round(smooth_window_ms / dt_ms)))
                if win > 1:
                    kernel = np.ones(win) / win
                    r = np.convolve(r, kernel, mode='same')

            pop_name = pop_key.split('_')[0] if '_' in pop_key else pop_key
            color = pop_colors.get(pop_name, 'gray')
            ax.plot(t, r, label=pop_name, color=color, linewidth=1.5, alpha=0.8)

            if show_stats:
                pre_mask = (t >= 1000) & (t < 2000)
                post_mask = (t >= 2000)
                if np.sum(pre_mask) > 0:
                    stats_text.append(f"{pop_name} pre: {np.mean(r[pre_mask]):.1f}Hz")
                if np.sum(post_mask) > 0:
                    stats_text.append(f"{pop_name} post: {np.mean(r[post_mask]):.1f}Hz")

        if ylim_max is not None:
            ax.set_ylim(0, ylim_max)

        ax.axvline(500, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Stimulus')
        ax.set_ylabel("Rate (Hz)", fontsize=12)
        ax.set_title(f"Layer {layer_name} — Population Rates", fontsize=14)
        ax.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.9)

        if show_stats and stats_text:
            ax.text(0.02, 0.98, '\n'.join(stats_text), transform=ax.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (ms)", fontsize=12)
    fig.tight_layout(h_pad=1.0)
    return fig


def plot_raster2(spike_monitors, baseline_time, stimuli_time, layer_configs, figsize=(15, 10)):
    fig, axes = plt.subplots(len(spike_monitors), 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()  # Ensure axes is 1D for easy indexing

    for i, (layer_name, monitors) in enumerate(spike_monitors.items()):
        ax = axes[i]
        config = layer_configs[layer_name]
  
        if 'E_spikes' in monitors:
            ax.scatter(monitors['E_spikes']["times_ms"], monitors['E_spikes']['spike_indices'],
                        color='green', s=0.5, alpha=0.6, label="E")
        
        if 'SOM_spikes' in monitors:
            ax.scatter(monitors['SOM_spikes']["times_ms"], 
                        monitors['SOM_spikes']['spike_indices'] + config['neuron_counts']['E'],
                        color='blue', s=0.5, alpha=0.8, label="SOM")
        
        if 'PV_spikes' in monitors:
            if 'SOM_spikes' in monitors:
                ax.scatter(monitors['PV_spikes']["times_ms"],
                        monitors['PV_spikes']['spike_indices'] + config['neuron_counts']['E'] + config['neuron_counts']['SOM'],
                        color='red', s=0.5, alpha=0.8, label="PV")
            else:
                ax.scatter(monitors['PV_spikes']["times_ms"],
                            monitors['PV_spikes']['spike_indices'] + config['neuron_counts']['E'],
                            color='red', s=0.5, alpha=0.8, label="PV")
            ########TO IMPROVE###############
        if 'VIP_spikes' in monitors:
            if layer_name == 'L1':
                ax.scatter(monitors['VIP_spikes']["times_ms"], monitors['VIP_spikes']['spike_indices'],
                        color='gold', s=0.5, alpha=0.8, label="VIP")
            else :
                ax.scatter(monitors['VIP_spikes']["times_ms"],
                        monitors['VIP_spikes']['spike_indices'] + config['neuron_counts']['E'] + config['neuron_counts']['SOM'] + config['neuron_counts']['PV'],
                        color='gold', s=0.5, alpha=0.8, label="VIP")
        x_lim = (baseline_time + stimuli_time)/1000
        # ax.set_xlim(2000, 7000)
        ax.set_ylabel('Neuron index')
        ax.set_title(f'{layer_name} Spike Raster Plot')
        ax.legend()
    
    plt.tight_layout()
    return fig




fname = 'results/stim_sweep_23_04/L4AB_PV/L4AB_PV_rate_10Hz.npz'
data = np.load(fname, allow_pickle=True)

trial_data = {
    'time': data['time_array_ms'],
    'bipolar_lfp': data['bipolar_matrix'],
    'lfp_matrix': data['lfp_matrix'],
    'rate_data': (data['rate_data'].item()
                    if data['rate_data'].size == 1
                    else data['rate_data']),
    "spike_data": data['spike_data'],
    'baseline_ms': float(data['baseline_ms']),
    'stim_onset_ms': float(data['stim_onset_ms']),
    'channel_labels': data['channel_labels'],
    'channel_depths': data['channel_depths'],
    'electrode_positions': data['electrode_positions'],
}


if 'lfp_current_matrix' in data.files:
    trial_data['lfp_current_matrix'] = data['lfp_current_matrix']
    trial_data['time_current_ms'] = data['time_current_ms']

if 'mazzoni_lfp_matrix' in data.files:
    trial_data['mazzoni_lfp_matrix'] = data['mazzoni_lfp_matrix']
    trial_data['mazzoni_time_ms'] = data['mazzoni_time_ms']
    trial_data['mazzoni_layer_names'] = data['mazzoni_layer_names']

spike_data = trial_data['spike_data']

if isinstance(spike_data, np.ndarray) and spike_data.dtype == object:
    spike_monitors = spike_data.item()
else:
    spike_monitors = spike_data

fig_raster = plot_raster2(spike_monitors, 2000, 2500, CONFIG['layers'])

rate_data = trial_data['rate_data']
if isinstance(rate_data, np.ndarray) and rate_data.dtype == object:
    rate_data = rate_data.item()

fig_rate = plot_rate_from_data(rate_data, CONFIG['layers'],
                               smooth_window_ms=15.0, ylim_max=80, show_stats=True)

# import tikzplotlib

# tikzplotlib.save("essai_tickz.tex")

plt.show()
