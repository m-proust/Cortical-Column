"""
Plotting functions
"""
import matplotlib.pyplot as plt
import numpy as np
from .analysis import *
from brian2 import *
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy import signal as scipy_signal



def plot_raster(spike_monitors, baseline_time, stimuli_time, layer_configs, figsize=(15, 10)):
    fig, axes = plt.subplots(len(spike_monitors), 1, figsize=figsize)
    if len(spike_monitors) == 1:
        axes = [axes]
    
    for i, (layer_name, monitors) in enumerate(spike_monitors.items()):
        ax = axes[i]
        config = layer_configs[layer_name]
        
        if 'E_spikes' in monitors:
            ax.scatter(monitors['E_spikes'].t/second, monitors['E_spikes'].i,
                        color='green', s=0.5, alpha=0.6, label="E")
        
        if 'SOM_spikes' in monitors:
            ax.scatter(monitors['SOM_spikes'].t/second, 
                        monitors['SOM_spikes'].i + config['neuron_counts']['E'],
                        color='blue', s=0.5, alpha=0.8, label="SOM")
        
        if 'PV_spikes' in monitors:
            if 'SOM_spikes' in monitors:
                ax.scatter(monitors['PV_spikes'].t/second,
                        monitors['PV_spikes'].i + config['neuron_counts']['E'] + config['neuron_counts']['SOM'],
                        color='red', s=0.5, alpha=0.8, label="PV")
            else:
                ax.scatter(monitors['PV_spikes'].t/second,
                            monitors['PV_spikes'].i + config['neuron_counts']['E'],
                            color='red', s=0.5, alpha=0.8, label="PV")
            ########TO IMPROVE###############
        if 'VIP_spikes' in monitors:
            if layer_name == 'L1':
                ax.scatter(monitors['VIP_spikes'].t/second, monitors['VIP_spikes'].i,
                        color='gold', s=0.5, alpha=0.8, label="VIP")
            else :
                ax.scatter(monitors['VIP_spikes'].t/second,
                        monitors['VIP_spikes'].i + config['neuron_counts']['E'] + config['neuron_counts']['SOM'] + config['neuron_counts']['PV'],
                        color='gold', s=0.5, alpha=0.8, label="VIP")
        x_lim = (baseline_time + stimuli_time)/1000
        ax.set_xlim(0.3, x_lim)
        ax.set_ylabel('Neuron index')
        ax.set_title(f'{layer_name} Spike Raster Plot')
        ax.legend()
    
    plt.tight_layout()
    return fig



def plot_lfp(state_monitors, layer_configs, figsize=(15, 12)):
    fig, axes = plt.subplots(len(state_monitors), 1, figsize=figsize)
    if len(state_monitors) == 1:
        axes = [axes]
    
    for i, (layer_name, monitors) in enumerate(state_monitors.items()):
        if 'E_state' in monitors:
            time_stable, lfp_stable = process_lfp(monitors['E_state'])
            
            ax = axes[i]
            ax.plot(time_stable, lfp_stable, 'b-', linewidth=0.5)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('LFP (norm)')
            ax.set_title(f'{layer_name} Local Field Potential')
            ax.set_xlim(0, 1000)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_lfp_power_comparison(state_monitors, layer_configs, baseline_time=1000,
                               pre_stim_duration=1000, post_stim_duration=1000,
                               transient_skip=500, figsize=(12, 10)):
    """
    Plot power spectrum comparison before and after stimulation.

    Parameters
    ----------
    transient_skip : float
        Time in ms to skip after stimulus onset to avoid transient effects.
    """
    fig, axes = plt.subplots(len(state_monitors), 1, figsize=figsize)
    if len(state_monitors) == 1:
        axes = [axes]

    colors = ['b', 'g', 'purple', 'orange', 'red']

    for i, (layer_name, monitors) in enumerate(state_monitors.items()):
        if 'E_state' in monitors:
            monitor = monitors['E_state']

            layer_cfg = layer_configs[layer_name]
            neuron_params = {
                'E_E': 0.0,
                'E_I': -70.0,
                'g_L': layer_cfg.get('g_L', 10.0)
            }

            lfp_full, time_array = calculate_lfp_mazzoni(
                monitor, neuron_params, method='weighted'
            )

            dt = time_array[1] - time_array[0]
            pre_start_idx = int((baseline_time - pre_stim_duration) / dt)
            pre_end_idx = int(baseline_time / dt)
            # Skip transient period after stimulus onset
            post_start_idx = int((baseline_time + transient_skip) / dt)
            post_end_idx = int((baseline_time + transient_skip + post_stim_duration) / dt)
            
            lfp_pre = lfp_full[pre_start_idx:pre_end_idx]
            lfp_post = lfp_full[post_start_idx:post_end_idx]
            
            fs = 1000.0 / dt  
            nperseg = 100*min(1024, len(lfp_pre) // 4)
            
            freq_pre, psd_pre = scipy_signal.welch(
                lfp_pre, fs=fs, nperseg=nperseg, window='hann'
            )
            freq_post, psd_post = scipy_signal.welch(
                lfp_post, fs=fs, nperseg=nperseg, window='hann'
            )
            
            ax = axes[i]
            color = colors[i % len(colors)]
            
            ax.plot(freq_pre[:100], psd_pre[:100], color=color,
                   label=f'{layer_name} Pre-stim', linewidth=2.5, linestyle='-')
            
            ax.plot(freq_post[:100], psd_post[:100], color=color,
                   label=f'{layer_name} Post-stim', linewidth=2.5, 
                   linestyle='--', alpha=0.8)
            
            ax.set_ylabel('Power (a.u.)', fontsize=16)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlim(0,220)
            ax.set_ylim(0.000009,0.002)
            ax.grid(True, alpha=0.3)
            
            peak_idx_pre = np.argmax(psd_pre[:100])
            peak_idx_post = np.argmax(psd_post[:100])
            
            ax.axvline(freq_pre[peak_idx_pre], color=color, linestyle=':', 
                      alpha=0.5, linewidth=1.5, label=f'Pre peak: {freq_pre[peak_idx_pre]:.1f} Hz')
            ax.axvline(freq_post[peak_idx_post], color='r', linestyle=':', 
                      alpha=0.5, linewidth=1.5, label=f'Post peak: {freq_post[peak_idx_post]:.1f} Hz')
            
            ax.legend(fontsize=11, loc='upper right')
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.set_title(f'{layer_name} - LFP Power Spectrum (Mazzoni Method)', 
                        fontsize=14, fontweight='bold')
    
    axes[-1].set_xlabel('Frequency (Hz)', fontsize=16)
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
    
    pop_colors = {'E': 'royalblue', 'PV': 'darkorange', 'SOM': 'forestgreen'}
    
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
                    pre_mask = (t >= 200) & (t < 500)
                    post_mask = (t >= 500)
                    
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



def plot_peak_freq_track(state_monitors, layer_configs, f_gamma=(20, 80), fmax=100,
                            win_ms=250, step_ms=25, light_window=(2.0, 4.0), figsize=(12, 4)):

    import matplotlib.pyplot as plt
    figs = []
    for layer_name, monitors in state_monitors.items():
        if 'E_state' not in monitors:
            continue
        time_ms, lfp = process_lfp(monitors['E_state'])
        t_spec, f, Sxx = compute_spectrogram(time_ms, lfp, fmax=fmax,
                                                            win_ms=win_ms, step_ms=step_ms)
        peak_f, peak_p = peak_frequency_track(f, Sxx, f_gamma=f_gamma)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(t_spec, peak_f, lw=2)
        ax.set_title(f'{layer_name} Gamma Peak Frequency ({f_gamma[0]}–{f_gamma[1]} Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Peak freq (Hz)')
        if light_window is not None:
            on, off = light_window
            ax.axvline(on, color='k', ls='--', lw=1)
            ax.axvline(off, color='k', ls='--', lw=1)
        ax.set_ylim(f_gamma[0], f_gamma[1])
        ax.grid(alpha=0.3)
        figs.append(fig)
    return figs

    
def plot_lfp_kernel(lfp_signals, time_array, electrode_positions, figsize=(15, 10)):
    n_electrodes = len(lfp_signals)
    
    fig, axes = plt.subplots(n_electrodes, 1, figsize=figsize, sharex=True)
    if n_electrodes == 1:
        axes = [axes]
    
    for i, (elec_idx, lfp) in enumerate(lfp_signals.items()):
        ax = axes[i]
        ex, ey, ez = electrode_positions[elec_idx]
        
        # if np.std(lfp) > 0:
        #     lfp_norm = (lfp - np.mean(lfp)) / np.std(lfp)
        # else:
        lfp_norm = lfp
        
        ax.plot(time_array, lfp_norm, 'b-', linewidth=0.5)
        ax.set_ylabel('LFP (norm)')
        #ax.set_title(f'Electrode {i} at z={ez:.3f} mm')
        ax.grid(True, alpha=0.3)
        
        ax.set_xlim(200, 1000)
        ax.set_ylim(-250, 100)
    
    axes[-1].set_xlabel('Time (ms)')
    plt.tight_layout()
    return fig

def plot_bipolar_lfp(bipolar_signals, channel_labels, channel_depths, time_array, 
                     figsize=(14, 10), time_range=(0, 1000)):

    n_channels = len(bipolar_signals)
    
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    time_mask = (time_array >= time_range[0]) & (time_array <= time_range[1])
    time_plot = time_array[time_mask]
    
    for i, (ch_idx, lfp) in enumerate(bipolar_signals.items()):
        ax = axes[i]
        lfp_plot = lfp[time_mask]
        
        if np.std(lfp_plot) > 0:
            lfp_norm = (lfp_plot - np.mean(lfp_plot)) / np.std(lfp_plot)
        else:
            lfp_norm = lfp_plot
        
        ax.plot(time_plot, lfp_norm, 'k-', linewidth=0.8)
        ax.set_ylabel(f'{channel_labels[i]} (z={channel_depths[i]:.3f} mm)', fontsize=10)
        ax.set_xlim(-200,500)
        
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        if i < n_channels - 1:
            ax.set_xticklabels([])
    
    axes[-1].set_xlabel('Time (ms)', fontsize=12)
    axes[0].set_title('Bipolar LFP', fontsize=14, loc='left', fontweight='bold')
    
    plt.tight_layout()
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

        ax.set_ylabel(f'{label}\nz={depth:.2f}mm', fontsize=9)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')

    axes[-1].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[0].set_title('Bipolar LFP Power Spectrum: Pre vs Post Stimulation',
                      fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_bipolar_power_spectra(bipolar_signals, channel_labels, channel_depths, time_array,
                               fs=10000, fmax=100, ncols=3, figsize=None):
 
    freq, psds = compute_bipolar_power_spectrum(
        bipolar_signals, time_array, fs=fs, fmax=fmax
    )

    n_channels = len(psds)
    
    nrows = int(np.ceil(n_channels / ncols))
    if figsize is None:
        figsize =  (16, 20)

    fig, axes = plt.subplots(15, 1, sharex=True, sharey=True, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for ax in axes[n_channels:]:
        ax.axis('off')

    for (ch_idx, psd), ax in zip(psds.items(), axes[:n_channels]):
        ax.plot(freq, psd, linewidth=1.5, alpha=0.9)

        peak_idx = np.argmax(psd)
        ax.plot(freq[peak_idx], psd[peak_idx], 'o',
                markersize=4, markeredgecolor='white', markeredgewidth=1)

        ax.set_ylabel(
            f"{channel_labels[ch_idx]}\n(z={channel_depths[ch_idx]:.3f} mm)",
            fontsize=9
        )


        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, fmax)

    fig.text(0.5, 0.04, 'Frequency (Hz)', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Power Spectral Density (a.u.)',
             va='center', rotation='vertical', fontsize=12)

    fig.suptitle('Bipolar LFP Power Spectra', fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0.06, 0.06, 1.0, 0.95])

    return fig


def plot_lfp_power_comparison_mazzoni_perlayer(lfp_signals, time_array,
                                                baseline_time=1000,
                                                pre_stim_duration=1000,
                                                post_stim_duration=1000,
                                                transient_skip=500,
                                                fs=10000, fmax=100,
                                                figsize=(12, 10)):
    """
    Plot per-layer LFP power spectra (Mazzoni RWS method) comparing
    pre- vs post-stimulus periods.

    Parameters
    ----------
    lfp_signals : dict {layer_name: 1D np.ndarray}
        Output of calculate_lfp_mazzoni_perlayer().
    time_array : np.ndarray
        Time vector in ms.
    """
    layer_names = list(lfp_signals.keys())
    n_layers = len(layer_names)

    fig, axes = plt.subplots(n_layers, 1, figsize=figsize, sharex=True)
    if n_layers == 1:
        axes = [axes]

    dt = time_array[1] - time_array[0]

    pre_start_idx = int((baseline_time - pre_stim_duration) / dt)
    pre_end_idx = int(baseline_time / dt)
    post_start_idx = int((baseline_time + transient_skip) / dt)
    post_end_idx = int((baseline_time + transient_skip + post_stim_duration) / dt)

    for i, layer_name in enumerate(layer_names):
        lfp = lfp_signals[layer_name]
        ax = axes[i]

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

        ax.set_ylabel(f'{layer_name}', fontsize=11)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

    axes[-1].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[0].set_title('Per-Layer LFP Power Spectrum: Pre vs Post Stimulation (Mazzoni RWS)',
                      fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig
