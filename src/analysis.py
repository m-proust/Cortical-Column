"""
Analysis functions.
"""
import numpy as np
import pywt
from scipy.signal import welch, spectrogram
from brian2 import *
from spectrum import pmtm

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
        
        def vary(base_val, sigma=0.15):
            factors = np.clip(1 + np.random.randn(n) * sigma, 0.5, 1.5)
            return base_val * factors
        
        neuron_group.C    = vary(base['C'], 0.15)
        neuron_group.gL   = vary(base['gL'], 0.12)
        neuron_group.tauw = vary(base['tauw'], 0.15)
        neuron_group.b    = vary(base['b'], 0.20)
        neuron_group.a    = vary(base['a'], 0.15)
        base_EL = base['EL']
        neuron_group.EL = base_EL + np.random.randn(n) * 2*mV  
        neuron_group.DeltaT = vary(base['DeltaT'], 0.10)

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


