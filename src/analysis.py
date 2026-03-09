"""
Analysis functions.
"""
import numpy as np
import pywt
from scipy.signal import welch, spectrogram
from brian2 import *
from scipy.ndimage import gaussian_filter1d
from spectrum import pmtm
from numpy.fft import rfft, rfftfreq

def calculate_lfp(monitor, neuron_type='E'):
    """Calculate LFP using current inputs into E neurons, Mazzoni method"""
    ge = np.array(monitor.gE/nS)  
    gi = np.array(monitor.gI/nS)  
    V = np.array(monitor.v/mV)
    
    I_AMPA = np.abs(ge * (0 - V))  # Ee = 0mV
    I_GABA = np.abs(gi * (-80 - V))  # Ei = -80mV
    
    total_current = np.sum(I_AMPA + I_GABA, axis=0)
    
    return total_current


def process_lfp(monitor, start_time_ms=300):
    lfp = calculate_lfp(monitor)
    lfp_time = np.array(monitor.t/ms)
    
    start_idx = np.argmax(lfp_time >= start_time_ms)
    lfp_stable = lfp[start_idx:]
    time_stable = lfp_time[start_idx:]
    if np.std(lfp_stable) != 0 :
        lfp_stable = (lfp_stable - np.mean(lfp_stable))/np.std(lfp_stable)
    return time_stable, lfp_stable


def calculate_lfp_mazzoni(state_monitor, neuron_params, method='weighted'):

    gE = np.array(state_monitor.gE / nS)
    gI = np.array(state_monitor.gI / nS)
    
    V = np.array(state_monitor.v / mV)
    
    E_E = neuron_params.get('E_E', 0.0)      
    E_I = neuron_params.get('E_I', -70.0)   
    I_excitatory = np.mean(gE * (V - E_E), axis=0)
    I_inhibitory = np.mean(gI * (V - E_I), axis=0)
    
    time = state_monitor.t / ms
    dt = time[1] - time[0] if len(time) > 1 else 1.0
    
    if method == 'weighted':
        alpha = 1.65        
        delay_ms = 6.0    
        delay_samples = int(delay_ms / dt)
        
        I_exc_delayed = np.zeros_like(I_excitatory)
        if delay_samples < len(I_excitatory):
            I_exc_delayed[delay_samples:] = I_excitatory[:-delay_samples]
        
        lfp_raw = alpha * np.abs(I_exc_delayed) - np.abs(I_inhibitory)
        
    elif method == 'absolute':
        lfp_raw = np.abs(I_excitatory) + np.abs(I_inhibitory)
        
    else:
        raise ValueError("method must be 'weighted' or 'absolute'")
    

    lfp_mean = np.mean(lfp_raw)
    lfp_std = np.std(lfp_raw)
    
    if lfp_std > 0:
        lfp_zscore = (lfp_raw - lfp_mean) / lfp_std
    else:
        lfp_zscore = lfp_raw - lfp_mean
    
    return lfp_zscore, time

def compute_power_spectrum(lfp_signal, fs=10000, nperseg=None):


    if nperseg is None:
        nperseg = min(4096, len(lfp_signal) // 4)

    freq, psd = welch(lfp_signal, fs=fs, nperseg=nperseg,
                    noverlap=nperseg//2, window='hann')
    return freq, psd


def compute_normalized_fft_magnitude(lfp_signal, fs=10000):

    signal_centered = lfp_signal - np.mean(lfp_signal)

    energy = np.sqrt(np.sum(signal_centered**2))
    if energy > 0:
        signal_normalized = signal_centered / energy
    else:
        signal_normalized = signal_centered

    n = len(signal_normalized)
    fft_result = rfft(signal_normalized)
    magnitude = np.abs(fft_result)

    freq = rfftfreq(n, d=1.0/fs)

    return freq, magnitude
    

def peak_frequency_track(f_hz, Sxx, f_gamma=(20, 80)):
    fmask = (f_hz >= f_gamma[0]) & (f_hz <= f_gamma[1])
    if not np.any(fmask):
        return np.full(Sxx.shape[1], np.nan), np.full(Sxx.shape[1], np.nan)
    Sg = Sxx[fmask, :]
    idx = np.argmax(Sg, axis=0)
    freqs_in_band = f_hz[fmask]
    peak_freq = freqs_in_band[idx]
    peak_pow  = Sg[idx, np.arange(Sg.shape[1])]
    return peak_freq, peak_pow


def add_heterogeneity_to_layer(layer, config):
    for pop_name, neuron_group in layer.neuron_groups.items():
        n = len(neuron_group)
        base = config['intrinsic_params'][pop_name]
        
        neuron_group.C = base['C'] * np.abs(1 + np.random.randn(n) * 0.15)
        neuron_group.gL = base['gL'] * np.abs(1 + np.random.randn(n) * 0.12)
        neuron_group.tauw = base['tauw'] * np.abs(1 + np.random.randn(n) * 0.15)
        neuron_group.b = base['b'] * np.abs(1 + np.random.randn(n) * 0.20)
        neuron_group.a = base['a'] * np.abs(1 + np.random.randn(n) * 0.15)



def calculate_lfp_kernel_method(spike_monitors, neuron_groups, layer_configs,
                                electrode_positions, dt_ms=0.1, sim_duration_ms=2000):

    lambda_space = 0.22 
    v_axon = 0.2  
    base_delay = 10.4   

    sigma_i = 2.1       

    sigma_e = 3.15  


    depth_table = {
        'rel_depths_mm': np.array([-0.4, 0.0, 0.4, 0.8]),
        'i_amplitudes':  np.array([-0.2, 3.0, -1.2, 0.3]),  
        'e_amplitudes':  np.array([-0.16, 0.48, 0.24, -0.08]), 
    }


    pop_lfp_weight = {
        'E':   1.0,
        'PV':  1.0,
        'SOM': 0.05,  
        'VIP': 0.02,
    }

    def get_depth_amplitude(rel_depth_mm, cell_type):
        depths = depth_table['rel_depths_mm']
        if cell_type == 'inhibitory':
            amps = depth_table['i_amplitudes']
        else:
            amps = depth_table['e_amplitudes']
        return np.interp(rel_depth_mm, depths, amps)

    n_samples = int(sim_duration_ms / dt_ms)
    time_array = np.arange(n_samples) * dt_ms 

    kernel_half_width_ms = 30.0  
    kernel_samples = int(2 * kernel_half_width_ms / dt_ms)
    kernel_t = np.arange(kernel_samples) * dt_ms - kernel_half_width_ms 

    def make_kernel_template(sigma):
        """Gaussian kernel template, peak at t=0, unit amplitude."""
        return np.exp(-kernel_t**2 / (2 * sigma**2))

    template_i = make_kernel_template(sigma_i)
    template_e = make_kernel_template(sigma_e)

    lfp_signals = np.zeros((len(electrode_positions), n_samples))

    for layer_name, layer_spike_mons in spike_monitors.items():
        layer_config = layer_configs[layer_name]
        z_range = layer_config['coordinates']['z']
        layer_center_z = np.mean(z_range)

        for pop_name in ['E', 'PV', 'SOM', 'VIP']:
            spike_key = f'{pop_name}_spikes'
            if spike_key not in layer_spike_mons:
                continue

            spike_mon = layer_spike_mons[spike_key]
            neuron_grp = neuron_groups[layer_name][pop_name]

            is_excitatory = (pop_name == 'E')
            sigma = sigma_e if is_excitatory else sigma_i
            template = template_e if is_excitatory else template_i
            type_weight = pop_lfp_weight[pop_name]

            if type_weight < 0.01:
                continue 

            spike_times_ms = np.array(spike_mon.t / ms)
            spike_indices = np.array(spike_mon.i)
            neuron_x = np.array(neuron_grp.x / mm)
            neuron_y = np.array(neuron_grp.y / mm)
            neuron_z = np.array(neuron_grp.z / mm)

            n_neurons = len(neuron_x)

            for elec_idx, (ex, ey, ez) in enumerate(electrode_positions):
                rel_depth = ez - layer_center_z
                cell_type = 'excitatory' if is_excitatory else 'inhibitory'
                A_depth = get_depth_amplitude(rel_depth, cell_type)

          
                distances = np.sqrt(
                    (neuron_x - ex)**2 + (neuron_y - ey)**2 + (neuron_z - ez)**2
                )

                A_spatial = np.exp(-distances / lambda_space)  

                delays = base_delay + distances / v_axon 

      
                delay_bins = np.round(delays / dt_ms).astype(int)
                unique_delays = np.unique(delay_bins)

                for d_bin in unique_delays:
                    neuron_mask = (delay_bins == d_bin)
                    neuron_ids = np.where(neuron_mask)[0]

                    weighted_rate = np.zeros(n_samples)
                    for nid in neuron_ids:
                        spike_mask = (spike_indices == nid)
                        if not np.any(spike_mask):
                            continue
                        st = spike_times_ms[spike_mask]
                        bins = np.clip((st / dt_ms).astype(int), 0, n_samples - 1)
                        np.add.at(weighted_rate, bins, A_spatial[nid])

                    convolved = np.convolve(weighted_rate, template, mode='same')

                    shift = int(d_bin)
                    if shift > 0 and shift < n_samples:
                        lfp_signals[elec_idx, shift:] += (
                            A_depth * type_weight * convolved[:-shift]
                        )
                    else:
                        lfp_signals[elec_idx] += A_depth * type_weight * convolved

    return lfp_signals, time_array



def compute_bipolar_lfp(lfp_signals, electrode_positions):

    n_electrodes = len(lfp_signals)
    bipolar_signals = {}
    channel_labels = []
    channel_depths = []
    
    for i in range(n_electrodes - 1):
        bipolar_signals[i] = lfp_signals[i+1] - lfp_signals[i]
        
        channel_labels.append(f'Ch{i+1}-Ch{i}')
        
        z_avg = (electrode_positions[i][2] + electrode_positions[i+1][2]) / 2
        channel_depths.append(z_avg)
    
    return bipolar_signals, channel_labels, channel_depths



def compute_bipolar_power_spectrum(bipolar_signals, time_array, fs=10000, 
                                   fmax=100, method='welch'):
   
    psds = {}
    
    for ch_idx, lfp in bipolar_signals.items():
        freq, psd = compute_power_spectrum(lfp, fs=fs, method=method)
        
        freq_mask = freq <= fmax
        psds[ch_idx] = psd[freq_mask]
    
    return freq[freq_mask], psds



def compute_csd_from_lfp(lfp_signals, electrode_positions, sigma=0.3, vaknin=True):
    n_electrodes = len(lfp_signals)

    z = np.array([pos[2] for pos in electrode_positions], dtype=float)

    sort_idx = np.argsort(z)
    depths_sorted = z[sort_idx]

    V = np.vstack([lfp_signals[int(i)] for i in sort_idx])

    dz = np.diff(depths_sorted)
    h_mm = np.mean(dz)
    if h_mm <= 0:
        raise ValueError("Electrode depths are not strictly ordered; check electrode_positions.")

    h = h_mm 

    csd = np.zeros_like(V)

    for ch in range(n_electrodes):
        if ch == 0:  
            V_plus = V[ch + 1]
            if vaknin:
                V_minus = V[ch + 1]
            else:
                V_minus = V[ch]
        elif ch == n_electrodes - 1: 
            V_minus = V[ch - 1]
            if vaknin:
                V_plus = V[ch - 1]
            else:
                V_plus = V[ch]
        else:
            V_minus = V[ch - 1]
            V_plus = V[ch + 1]

        csd[ch] = -sigma * (V_plus - 2 * V[ch] + V_minus) / (h ** 2)

    return csd, depths_sorted, sort_idx

def plot_csd(csd, time_array, depths, time_range=(300, 800),
             figsize=(8, 10), vlim=None, cmap='seismic'):

    t0, t1 = time_range
    time_mask = (time_array >= t0) & (time_array <= t1)
    t_plot = time_array[time_mask]
    csd_plot = csd[:, time_mask]

    if vlim is None:
        vmax = np.max(np.abs(csd_plot))
    else:
        vmax = float(vlim)

    fig, ax = plt.subplots(figsize=figsize)


    im = ax.imshow(
        csd_plot,
        aspect='auto',
        origin='lower',
        extent=[t_plot[0], t_plot[-1], depths[0], depths[-1]],
        vmin=-vmax,
        vmax=vmax,
        cmap=cmap,
    )

    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Depth (mm)', fontsize=12)
    ax.set_title('Laminar Current Source Density', fontsize=14, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('CSD (a.u.)', fontsize=12)

    plt.tight_layout()
    return fig

def plot_rate_fft(rate_monitors, fmax=150, method='welch'):

    layer_names = list(rate_monitors.keys())
    n_layers = len(layer_names)

    if n_layers == 0:
        raise ValueError("rate_monitors is empty – no rate data available.")

    fig, axes = plt.subplots(
        n_layers, 1,
        figsize=(8, 2.5 * n_layers),
        sharex=True,
        sharey=True
    )
    if n_layers == 1:
        axes = [axes]

    for ax, layer_name in zip(axes, layer_names):
        layer_rate_mons = rate_monitors[layer_name]

        if len(layer_rate_mons) == 0:
            ax.text(0.5, 0.5, f"No rate data for {layer_name}",
                    transform=ax.transAxes, ha='center', va='center')
            ax.axis('off')
            continue

        times = None
        global_rate = None
        n_pops = 0

        for mon_name, mon in layer_rate_mons.items():
            if len(mon.t) == 0:
                continue

            t_ms = np.array(mon.t / ms)
            r_hz = np.array(mon.rate / Hz)

            if times is None:
                times = t_ms
                global_rate = r_hz.copy()
            else:
                L = min(len(times), len(t_ms), len(global_rate), len(r_hz))
                times = times[:L]
                global_rate = global_rate[:L] + r_hz[:L]

            n_pops += 1

        if n_pops == 0 or times is None:
            ax.text(0.5, 0.5, f"No valid rate data for {layer_name}",
                    transform=ax.transAxes, ha='center', va='center')
            ax.axis('off')
            continue

        global_rate /= n_pops

        signal = global_rate - np.mean(global_rate)

        dt_ms = np.median(np.diff(times))
        if dt_ms <= 0:
            ax.text(0.5, 0.5, f"Invalid dt for {layer_name}",
                    transform=ax.transAxes, ha='center', va='center')
            ax.axis('off')
            continue

        fs = 1000.0 / dt_ms 

        freq, psd = compute_power_spectrum(signal, fs=fs, method=method)

        mask = freq <= fmax
        freq_plot = freq[mask]
        psd_plot = psd[mask]

        ax.plot(freq_plot, psd_plot, linewidth=1.5, alpha=0.9)

        if len(psd_plot) > 0 and np.any(psd_plot > 0):
            peak_idx = np.argmax(psd_plot)
            ax.plot(freq_plot[peak_idx], psd_plot[peak_idx], 'o',
                    markersize=4, markeredgecolor='white', markeredgewidth=1)
            ax.set_title(
                f'{layer_name} – peak: {freq_plot[peak_idx]:.1f} Hz',
                fontsize=10
            )
        else:
            ax.set_title(f'{layer_name} – no clear peak', fontsize=10)

        ax.grid(True, alpha=0.3)
        ax.set_ylabel('PSD (a.u.)', fontsize=9)

    axes[-1].set_xlabel('Frequency (Hz)', fontsize=11)
    fig.suptitle('Global population rate power spectrum by layer',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.04, 0.04, 1.0, 0.95])

    return fig

