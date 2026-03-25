"""
Analysis functions.
"""
import numpy as np
import pywt
from scipy.signal import welch, spectrogram, hilbert, butter, filtfilt
from brian2 import *
from scipy.ndimage import gaussian_filter1d
from spectrum import pmtm
from numpy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

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


def compute_ppc(spike_times_ms, lfp_signal, time_array, fs, freq, bandwidth=2.0):
    """
    Compute Pairwise Phase Consistency (Vinck et al., 2010) at a single frequency.

    For each spike, extract the LFP phase at that frequency using a narrow bandpass
    + Hilbert transform. Then compute PPC as the mean cosine similarity of all
    spike-phase pairs.

    Returns PPC value (float) or np.nan if fewer than 2 spikes.
    """
    if len(spike_times_ms) < 2:
        return np.nan

    # Narrow bandpass around target frequency
    f_lo = max(freq - bandwidth / 2, 0.5)
    f_hi = freq + bandwidth / 2
    nyq = fs / 2.0
    if f_hi >= nyq:
        return np.nan

    b, a = butter(3, [f_lo / nyq, f_hi / nyq], btype='band')
    filtered = filtfilt(b, a, lfp_signal)

    # Instantaneous phase via Hilbert transform
    analytic = hilbert(filtered)
    inst_phase = np.angle(analytic)

    # Get phase at each spike time
    spike_indices = np.searchsorted(time_array, spike_times_ms)
    spike_indices = spike_indices[(spike_indices >= 0) & (spike_indices < len(inst_phase))]

    if len(spike_indices) < 2:
        return np.nan

    phases = inst_phase[spike_indices]
    n = len(phases)

    # PPC = (1/(n*(n-1))) * sum_{i!=j} cos(phi_i - phi_j)
    # Efficiently: PPC = (|sum(exp(j*phi))|^2 - n) / (n*(n-1))
    resultant = np.sum(np.exp(1j * phases))
    ppc = (np.abs(resultant) ** 2 - n) / (n * (n - 1))

    return ppc


def _get_inst_phase(lfp_signal, fs, freq, bandwidth=2.0):
    """Bandpass-filter the LFP and return instantaneous phase via Hilbert transform."""
    f_lo = max(freq - bandwidth / 2, 0.5)
    f_hi = freq + bandwidth / 2
    nyq = fs / 2.0
    if f_hi >= nyq:
        return None
    b, a = butter(3, [f_lo / nyq, f_hi / nyq], btype='band')
    filtered = filtfilt(b, a, lfp_signal)
    analytic = hilbert(filtered)
    return np.angle(analytic)


def compute_ppc_per_neuron(spike_mon, time_window, lfp_signal, time_array, fs, freq,
                           min_spikes_per_neuron=2, bandwidth=2.0, inst_phase=None):
    """
    Compute PPC averaged across single neurons (Vinck et al., 2010).

    For each neuron that has >= min_spikes_per_neuron in the time window,
    compute its individual PPC from its own spike phases. Then return the
    average PPC across neurons, weighted by n_i*(n_i-1) (number of spike pairs).

    This avoids inflating PPC by pooling spikes from many neurons whose
    aggregate train is rhythmic even if individual cells fire sparsely.

    Parameters
    ----------
    spike_mon : Brian2 SpikeMonitor
    time_window : tuple (t_start_ms, t_end_ms)
    lfp_signal : 1D array, LFP segment matching time_array
    time_array : 1D array, time points in ms
    fs : sampling rate in Hz
    freq : target frequency in Hz
    min_spikes_per_neuron : minimum spikes per neuron to include
    bandwidth : bandpass bandwidth in Hz
    inst_phase : pre-computed instantaneous phase array (optional, for speed)

    Returns
    -------
    ppc : float or np.nan
    """
    if inst_phase is None:
        inst_phase = _get_inst_phase(lfp_signal, fs, freq, bandwidth)
    if inst_phase is None:
        return np.nan

    t_start, t_end = time_window
    all_times = np.array(spike_mon.t / ms)
    all_indices = np.array(spike_mon.i)

    mask = (all_times >= t_start) & (all_times < t_end)
    times = all_times[mask]
    neuron_ids = all_indices[mask]

    if len(times) < 2:
        return np.nan

    # Map spike times to LFP sample indices (once for all neurons)
    spike_samples = np.searchsorted(time_array, times)
    valid = (spike_samples >= 0) & (spike_samples < len(inst_phase))
    spike_samples = spike_samples[valid]
    neuron_ids = neuron_ids[valid]
    phases_all = inst_phase[spike_samples]

    # Per-neuron PPC, weighted average
    unique_neurons = np.unique(neuron_ids)
    weighted_ppc_sum = 0.0
    weight_sum = 0

    for nid in unique_neurons:
        neuron_mask = neuron_ids == nid
        phases = phases_all[neuron_mask]
        n = len(phases)
        if n < min_spikes_per_neuron:
            continue
        resultant = np.sum(np.exp(1j * phases))
        ppc_i = (np.abs(resultant) ** 2 - n) / (n * (n - 1))
        w = n * (n - 1)
        weighted_ppc_sum += ppc_i * w
        weight_sum += w

    if weight_sum == 0:
        return np.nan

    return weighted_ppc_sum / weight_sum


def _find_closest_bipolar_channel(layer_z_range, channel_depths):
    """Find the bipolar channel index whose depth is closest to the layer center."""
    layer_center = (layer_z_range[0] + layer_z_range[1]) / 2.0
    distances = [abs(d - layer_center) for d in channel_depths]
    return int(np.argmin(distances))


def plot_ppc_by_layer(spike_monitors, bipolar_signals, channel_depths, time_array,
                      layer_configs, freq_range=(1, 100), n_freqs=50, fs=10000,
                      baseline_time=2000, stimuli_time=1500, min_spikes_per_neuron=5,
                      bandwidth=4.0, smooth_sigma=1.5):
    """
    Plot PPC spectra for each cell type in each layer (one subplot per layer).

    Uses the bipolar LFP channel closest to each layer's center depth.
    Computes PPC separately for baseline and stimulus periods.
    """
    freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
    layer_names = list(layer_configs.keys())
    n_layers = len(layer_names)

    cell_types = ['E', 'PV', 'SOM', 'VIP']
    colors = {'E': '#1f77b4', 'PV': '#d62728', 'SOM': '#2ca02c', 'VIP': '#ff7f0e'}

    fig, axes = plt.subplots(n_layers, 2, figsize=(14, 3 * n_layers),
                             sharex=True, sharey=False)
    if n_layers == 1:
        axes = axes[np.newaxis, :]

    # Time masks for baseline and stimulus
    # Skip first 500ms transient from each period
    base_mask = (time_array >= 500) & (time_array < baseline_time)
    stim_mask = (time_array >= baseline_time + 500) & (time_array < baseline_time + stimuli_time)

    for row, layer_name in enumerate(layer_names):
        layer_cfg = layer_configs[layer_name]
        z_range = layer_cfg['coordinates']['z']
        ch_idx = _find_closest_bipolar_channel(z_range, channel_depths)

        bipolar_lfp = bipolar_signals[ch_idx]

        # Baseline and stimulus LFP segments
        lfp_base = bipolar_lfp[base_mask]
        time_base = time_array[base_mask]
        lfp_stim = bipolar_lfp[stim_mask]
        time_stim = time_array[stim_mask]

        layer_spike_mons = spike_monitors[layer_name]

        for cell_type in cell_types:
            spike_key = f'{cell_type}_spikes'
            if spike_key not in layer_spike_mons:
                continue

            spike_mon = layer_spike_mons[spike_key]

            ppc_base = np.full(n_freqs, np.nan)
            ppc_stim = np.full(n_freqs, np.nan)

            base_window = (time_base[0], time_base[-1])
            stim_window = (time_stim[0], time_stim[-1])

            for fi, f in enumerate(freqs):
                phase_base = _get_inst_phase(lfp_base, fs, f, bandwidth)
                phase_stim = _get_inst_phase(lfp_stim, fs, f, bandwidth)

                ppc_base[fi] = compute_ppc_per_neuron(
                    spike_mon, base_window, lfp_base, time_base, fs, f,
                    min_spikes_per_neuron=min_spikes_per_neuron,
                    bandwidth=bandwidth, inst_phase=phase_base)
                ppc_stim[fi] = compute_ppc_per_neuron(
                    spike_mon, stim_window, lfp_stim, time_stim, fs, f,
                    min_spikes_per_neuron=min_spikes_per_neuron,
                    bandwidth=bandwidth, inst_phase=phase_stim)

            # Smooth PPC spectra to reduce noise
            if smooth_sigma > 0:
                valid_b = ~np.isnan(ppc_base)
                valid_s = ~np.isnan(ppc_stim)
                if np.sum(valid_b) > 3:
                    ppc_base[valid_b] = gaussian_filter1d(ppc_base[valid_b], smooth_sigma)
                if np.sum(valid_s) > 3:
                    ppc_stim[valid_s] = gaussian_filter1d(ppc_stim[valid_s], smooth_sigma)

            # Plot baseline (left column)
            axes[row, 0].plot(freqs, ppc_base, color=colors[cell_type],
                              label=cell_type, linewidth=1.5, alpha=0.85)
            # Plot stimulus (right column)
            axes[row, 1].plot(freqs, ppc_stim, color=colors[cell_type],
                              label=cell_type, linewidth=1.5, alpha=0.85)

        ch_label = f'bip ch{ch_idx} (z={channel_depths[ch_idx]:.2f}mm)'
        axes[row, 0].set_title(f'{layer_name} — Baseline — {ch_label}', fontsize=10)
        axes[row, 1].set_title(f'{layer_name} — Stimulus — {ch_label}', fontsize=10)

        for col in range(2):
            axes[row, col].set_ylabel('PPC', fontsize=9)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend(fontsize=8, loc='upper right')
            axes[row, col].set_ylim(bottom=-0.01)

    axes[-1, 0].set_xlabel('Frequency (Hz)', fontsize=11)
    axes[-1, 1].set_xlabel('Frequency (Hz)', fontsize=11)
    fig.suptitle('Pairwise Phase Consistency (Vinck et al. 2010)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

