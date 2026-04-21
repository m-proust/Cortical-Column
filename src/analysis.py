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


def record_trial_metadata(column):
    """Record randomized parameters for each trial (synapse counts, intrinsic params, initial V).

    Returns a lightweight dict with summary stats per projection and per-neuron intrinsic params.
    """
    metadata = {
        'synapse_counts': {},
        'synapse_mean_delay_ms': {},
        'intrinsic_params': {},
        'initial_v_mean_mV': {},
    }

    # Intra-layer synapses
    for layer_name, layer in column.layers.items():
        for syn_name, syn in layer.synapses.items():
            key = f"{layer_name}_{syn_name}"
            metadata['synapse_counts'][key] = len(syn)
            if len(syn) > 0:
                metadata['synapse_mean_delay_ms'][key] = float(np.mean(syn.delay / ms))

        # Per-neuron intrinsic params (only summary stats to save space)
        for pop_name, grp in layer.neuron_groups.items():
            key = f"{layer_name}_{pop_name}"
            metadata['intrinsic_params'][key] = {
                'C_mean': float(np.mean(grp.C / pF)),
                'C_std': float(np.std(grp.C / pF)),
                'gL_mean': float(np.mean(grp.gL / nS)),
                'gL_std': float(np.std(grp.gL / nS)),
                'tauw_mean': float(np.mean(grp.tauw / ms)),
                'tauw_std': float(np.std(grp.tauw / ms)),
                'b_mean': float(np.mean(grp.b / pA)),
                'b_std': float(np.std(grp.b / pA)),
                'a_mean': float(np.mean(grp.a / nS)),
                'a_std': float(np.std(grp.a / nS)),
                'EL_mean': float(np.mean(grp.EL / mV)),
                'EL_std': float(np.std(grp.EL / mV)),
            }
            metadata['initial_v_mean_mV'][key] = float(np.mean(grp.v / mV))

    # Inter-layer synapses
    for syn_name, syn in column.inter_layer_synapses.items():
        metadata['synapse_counts'][syn_name] = len(syn)
        if len(syn) > 0:
            metadata['synapse_mean_delay_ms'][syn_name] = float(np.mean(syn.delay / ms))

    return metadata


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


