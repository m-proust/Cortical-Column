"""Plotting functions for rasters, rates and LFP power spectra."""
import matplotlib.pyplot as plt
import numpy as np
from brian2 import *
from scipy import signal as scipy_signal


POP_COLORS = {
    'E':   '#2E8B57',
    'PV':  '#C0392B',
    'SOM': '#1F4E96',
    'VIP': '#D4A017',
}


def plot_raster(spike_monitors, baseline_time, stimuli_time, layer_configs, figsize=(15, 10)):
    fig, axes = plt.subplots(len(spike_monitors), 1, figsize=figsize)
    if len(spike_monitors) == 1:
        axes = [axes]

    for i, (layer_name, monitors) in enumerate(spike_monitors.items()):
        ax = axes[i]
        config = layer_configs[layer_name]

        if 'E_spikes' in monitors:
            ax.scatter(monitors['E_spikes'].t/second, monitors['E_spikes'].i,
                        color=POP_COLORS['E'], s=0.5, alpha=0.55, label="E")

        if 'SOM_spikes' in monitors:
            ax.scatter(monitors['SOM_spikes'].t/second,
                        monitors['SOM_spikes'].i + config['neuron_counts']['E'],
                        color=POP_COLORS['SOM'], s=0.5, alpha=0.7, label="SOM")

        if 'PV_spikes' in monitors:
            if 'SOM_spikes' in monitors:
                ax.scatter(monitors['PV_spikes'].t/second,
                        monitors['PV_spikes'].i + config['neuron_counts']['E'] + config['neuron_counts']['SOM'],
                        color=POP_COLORS['PV'], s=0.5, alpha=0.7, label="PV")
            else:
                ax.scatter(monitors['PV_spikes'].t/second,
                            monitors['PV_spikes'].i + config['neuron_counts']['E'],
                            color=POP_COLORS['PV'], s=0.5, alpha=0.7, label="PV")
        if 'VIP_spikes' in monitors:
            if layer_name == 'L1':
                ax.scatter(monitors['VIP_spikes'].t/second, monitors['VIP_spikes'].i,
                        color=POP_COLORS['VIP'], s=0.5, alpha=0.7, label="VIP")
            else :
                ax.scatter(monitors['VIP_spikes'].t/second,
                        monitors['VIP_spikes'].i + config['neuron_counts']['E'] + config['neuron_counts']['SOM'] + config['neuron_counts']['PV'],
                        color=POP_COLORS['VIP'], s=0.5, alpha=0.7, label="VIP")
        x_lim = (baseline_time + stimuli_time)/1000
        ax.set_xlim(0.3, x_lim)
        ax.set_ylabel('Neuron index')
        ax.set_title(f'{layer_name} Spike Raster Plot')
        ax.legend()
    
    plt.tight_layout()
    return fig



def plot_rate(rate_monitors, layer_configs, baseline_time, stim_time, figsize=(10, 12), smooth_window=10*ms, 
              ylim_max=None, show_stats=True):
    layer_names = list(layer_configs.keys()) if isinstance(layer_configs, dict) else list(rate_monitors.keys())
    n_layers = len(layer_names) if layer_names else len(rate_monitors)
    
    if n_layers == 0:
        fig = plt.figure(figsize=figsize)
        fig.suptitle("Population Rates (no data)")
        return fig
    
    fig, axes = plt.subplots(n_layers, 1, sharex=True, figsize=figsize)
    if n_layers == 1:
        axes = [axes]
    
    pop_colors = POP_COLORS

    for ax, layer_name in zip(axes, layer_names):
        layer_rates = rate_monitors.get(layer_name, {})
        plotted_any = False
        stats_text = []

        for pop_key in sorted(layer_rates.keys()):
            mon = layer_rates[pop_key]
            try:
                t = mon.t / ms
                r = mon.smooth_rate(window='flat', width=smooth_window) / Hz

                pop_name = pop_key.split('_')[0] if '_' in pop_key else pop_key
                color = pop_colors.get(pop_name, 'gray')
                
                ax.plot(t, r, label=pop_name, color=color, linewidth=1.5, alpha=0.8)
                # ax.set_xlim(0, 1000)
                
                if show_stats:
                    pre_mask = (t >= 1000) & (t < 2000)
                    post_mask = (t >= 2000)
                    
                    if np.sum(pre_mask) > 0:
                        mean_pre = np.mean(r[pre_mask])
                        stats_text.append(f"{pop_name} pre: {mean_pre:.1f}Hz")
                    
                    if np.sum(post_mask) > 0:
                        mean_post = np.mean(r[post_mask])
                        stats_text.append(f"{pop_name} post: {mean_post:.1f}Hz")
                
                plotted_any = True
                
            except Exception as e:
                ax.text(0.01, 0.9, f"Error plotting {pop_key}: {e}", 
                       transform=ax.transAxes, fontsize=8, color="red")
        
        if ylim_max is not None:
            ax.set_ylim(0, ylim_max)
        
        stim_time=stim_time/ms
        ax.axvline(500, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Stimulus')
        
        ax.set_ylabel("Rate (Hz)", fontsize=12)
        title = f"Layer {layer_name} — Population Rates"
        ax.set_title(title, fontsize=14)
        
        if plotted_any:
            ax.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.9)
            
        if show_stats and stats_text:
            stats_str = '\n'.join(stats_text)
            ax.text(0.02, 0.98, stats_str, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Time (ms)", fontsize=12)
    fig.tight_layout(h_pad=1.0)
    return fig



def plot_lfp_comparison(lfp_signals, bipolar_signals, time_array, electrode_positions,
                        channel_labels, channel_depths, figsize=(18, 12), time_range=(0, 1000)):

    n_monopolar = len(lfp_signals)
    n_bipolar = len(bipolar_signals)
    
    fig, (ax_mono, ax_bipo) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    
    time_mask = (time_array >= time_range[0]) & (time_array <= time_range[1])
    time_plot = time_array[time_mask]
    
    offset_mono = 0
    spacing_mono = 6 
    
    for i in range(n_monopolar):
        lfp = lfp_signals[i][time_mask]
        if np.std(lfp) > 0:
            lfp_norm = (lfp - np.mean(lfp)) / np.std(lfp)
        else:
            lfp_norm = lfp
        
        ax_mono.plot(time_plot, lfp_norm + offset_mono, 'b-', linewidth=0.8, alpha=0.8)
        ex, ey, ez = electrode_positions[i]
        ax_mono.text(time_range[0] - 50, offset_mono, f'Ch{i}\nz={ez:.2f}', 
                    ha='right', va='center', fontsize=9)
        offset_mono += spacing_mono
    
    ax_mono.set_xlabel('Time (ms)', fontsize=12)
    ax_mono.set_ylabel('Channels (monopolar)', fontsize=12)
    ax_mono.set_title('Monopolar LFP', fontsize=14, fontweight='bold')
    ax_mono.set_xlim(time_range)
    ax_mono.grid(True, alpha=0.3)
    ax_mono.spines['left'].set_visible(False)
    ax_mono.set_yticks([])
    
    offset_bipo = 0
    spacing_bipo = 6
    
    for i, (ch_idx, lfp) in enumerate(bipolar_signals.items()):
        lfp_plot = lfp[time_mask]
        if np.std(lfp_plot) > 0:
            lfp_norm = (lfp_plot - np.mean(lfp_plot)) / np.std(lfp_plot)
        else:
            lfp_norm = lfp_plot
        
        ax_bipo.plot(time_plot, lfp_norm + offset_bipo, 'r-', linewidth=0.8, alpha=0.8)
        ax_bipo.text(time_range[0] - 50, offset_bipo, 
                    f'{channel_labels[i]}\nz={channel_depths[i]:.2f}', 
                    ha='right', va='center', fontsize=9)
        offset_bipo += spacing_bipo
    
    ax_bipo.set_xlabel('Time (ms)', fontsize=12)
    ax_bipo.set_title('Bipolar LFP', fontsize=14, fontweight='bold')
    ax_bipo.set_xlim(time_range)
    ax_bipo.grid(True, alpha=0.3)
    ax_bipo.spines['left'].set_visible(False)
    ax_bipo.set_yticks([])
    
    plt.tight_layout()
    return fig

def plot_lfp_power_comparison_kernel(lfp_signals, time_array, electrode_positions,
                                      baseline_time=1000, pre_stim_duration=1000,
                                      post_stim_duration=1000, transient_skip=500,
                                      fs=10000, fmax=100, figsize=(12, 8)):

    n_electrodes = len(lfp_signals)

    fig, axes = plt.subplots(n_electrodes, 1, figsize=figsize, sharex=True)
    if n_electrodes == 1:
        axes = [axes]

    dt = time_array[1] - time_array[0]

    pre_start_idx = int((baseline_time - pre_stim_duration) / dt)
    pre_end_idx = int(baseline_time / dt)
    post_start_idx = int((baseline_time + transient_skip) / dt)
    post_end_idx = int((baseline_time + transient_skip + post_stim_duration) / dt)

    for i, elec_idx in enumerate(range(lfp_signals.shape[0])):
        lfp = lfp_signals[elec_idx]
        ax = axes[i]
        ex, ey, ez = electrode_positions[elec_idx]

        lfp_pre = lfp[pre_start_idx:pre_end_idx]
        lfp_post = lfp[post_start_idx:post_end_idx]

        nperseg = 100*min(1024, len(lfp_pre) // 4)
        freq_pre, psd_pre = scipy_signal.welch(lfp_pre, fs=fs, nperseg=nperseg, window='hann')
        freq_post, psd_post = scipy_signal.welch(lfp_post, fs=fs, nperseg=nperseg, window='hann')

        freq_mask = freq_pre <= fmax

        ax.plot(freq_pre[freq_mask], psd_pre[freq_mask], 'b-', linewidth=1.5,
                label='Pre-stim', alpha=0.9)
        ax.plot(freq_post[freq_mask], psd_post[freq_mask], 'r--', linewidth=1.5,
                label='Post-stim', alpha=0.9)

        peak_idx_pre = np.argmax(psd_pre[freq_mask])
        peak_idx_post = np.argmax(psd_post[freq_mask])

        ax.axvline(freq_pre[freq_mask][peak_idx_pre], color='b', linestyle=':', alpha=0.5,
                   label=f'Pre peak: {freq_pre[freq_mask][peak_idx_pre]:.1f} Hz')
        ax.axvline(freq_post[freq_mask][peak_idx_post], color='r', linestyle=':', alpha=0.5,
                   label=f'Post peak: {freq_post[freq_mask][peak_idx_post]:.1f} Hz')

        ax.set_ylabel(f'Elec {i}\nz={ez:.2f}mm', fontsize=9)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

    axes[-1].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[0].set_title('LFP Power Spectrum: Pre vs Post Stimulation (Kernel Method)',
                      fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig


def oscillatory_peak(freq, psd, min_prominence=0.05):
    """Frequency of the largest peak above the 1/f trend, or None if too weak."""
    pos = freq > 0
    f, p = freq[pos], psd[pos]

    slope, intercept = np.polyfit(np.log10(f), np.log10(p), 1)
    residual = np.log10(p) - (slope * np.log10(f) + intercept)

    idx = np.argmax(residual)
    if residual[idx] < min_prominence:  
        return None
    return f[idx]


def plot_bipolar_power_comparison_kernel(bipolar_signals, channel_labels, channel_depths, time_array,
                                         baseline_time=1000, pre_stim_duration=1000,
                                         post_stim_duration=1000, transient_skip=500,
                                         fs=10000, fmax=100, figsize=(14, 20)):
    n_channels = len(bipolar_signals)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]

    dt = time_array[1] - time_array[0]
    pre_start_idx = int((baseline_time - pre_stim_duration) / dt)
    pre_end_idx = int(baseline_time / dt)
    post_start_idx = int((baseline_time + transient_skip) / dt)
    post_end_idx = int((baseline_time + transient_skip + post_stim_duration) / dt)

    for i, (ch_idx, lfp) in enumerate(bipolar_signals.items()):
        ax = axes[i]
        depth = channel_depths[ch_idx]
        label = channel_labels[ch_idx]

        lfp_pre = lfp[pre_start_idx:pre_end_idx]
        lfp_post = lfp[post_start_idx:post_end_idx]

        nperseg = len(lfp_pre) // 2
        nfft = int(2 ** np.ceil(np.log2(4 * nperseg)))
        freq, psd_pre = scipy_signal.welch(lfp_pre, fs=fs, nperseg=nperseg, nfft=nfft, window='hann')
        _, psd_post = scipy_signal.welch(lfp_post, fs=fs, nperseg=nperseg, nfft=nfft, window='hann')

        mask = freq <= fmax
        f = freq[mask]
        p_pre = psd_pre[mask]
        p_post = psd_post[mask]

        ax.plot(f, p_pre, 'b-', linewidth=1.5, label='Pre-stim', alpha=0.9)
        ax.plot(f, p_post, 'r--', linewidth=1.5, label='Post-stim', alpha=0.9)

        pk_pre = oscillatory_peak(f, p_pre)
        pk_post = oscillatory_peak(f, p_post)
        if pk_pre is not None:
            ax.axvline(pk_pre, color='b', linestyle=':', alpha=0.5, label=f'Pre peak: {pk_pre:.1f} Hz')
        if pk_post is not None:
            ax.axvline(pk_post, color='r', linestyle=':', alpha=0.5, label=f'Post peak: {pk_post:.1f} Hz')

        ax.set_ylabel(f'{label}\nz={depth:.2f}mm', fontsize=9)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')

    axes[-1].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[0].set_title('Bipolar LFP Power Spectrum: Pre vs Post Stimulation',
                      fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_mean_rates_bar(rate_monitors, layer_configs, baseline_time, stimuli_time,
                        transient_skip=300, figsize=(12, 6)):
  
    layer_names = list(layer_configs.keys()) if isinstance(layer_configs, dict) else list(rate_monitors.keys())

    pop_colors = POP_COLORS
    pop_order = ['E', 'PV', 'SOM', 'VIP']

    total_time = baseline_time + stimuli_time

    means = {p: [] for p in pop_order}
    sems = {p: [] for p in pop_order}

    for layer_name in layer_names:
        layer_rates = rate_monitors.get(layer_name, {})
        layer_pops = {}
        for pop_key, mon in layer_rates.items():
            pop_name = pop_key.split('_')[0] if '_' in pop_key else pop_key
            layer_pops[pop_name] = mon

        for pop in pop_order:
            if pop in layer_pops:
                mon = layer_pops[pop]
                t = mon.t / ms
                r = mon.smooth_rate(window='flat', width=15*ms) / Hz
                mask = (t >= transient_skip) & (t <= total_time)
                if np.sum(mask) > 0:
                    means[pop].append(float(np.mean(r[mask])))
                    sems[pop].append(float(np.std(r[mask]) / np.sqrt(np.sum(mask))))
                else:
                    means[pop].append(np.nan)
                    sems[pop].append(0.0)
            else:
                means[pop].append(np.nan)
                sems[pop].append(0.0)

    n_layers = len(layer_names)
    x = np.arange(n_layers)
    pops_present = [p for p in pop_order if not all(np.isnan(means[p]))]
    n_pops = len(pops_present)
    bar_width = 0.8 / max(n_pops, 1)

    fig, ax = plt.subplots(figsize=figsize)

    for i, pop in enumerate(pops_present):
        offset = (i - (n_pops - 1) / 2) * bar_width
        vals = np.array(means[pop])
        errs = np.array(sems[pop])
        ax.bar(x + offset, vals, bar_width,
               yerr=errs, capsize=3,
               color=pop_colors.get(pop, 'gray'),
               edgecolor='black', linewidth=0.6,
               label=pop, alpha=0.92,
               error_kw={'elinewidth': 0.8, 'ecolor': '0.2'})

    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, fontsize=12)
    ax.set_ylabel('Mean firing rate (Hz)', fontsize=13)
    ax.set_xlabel('Layer', fontsize=13)
    ax.set_title(f'Mean population firing rates',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Population', frameon=True, framealpha=0.95,
              fontsize=11, title_fontsize=11, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig
