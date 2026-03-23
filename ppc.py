
import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from brian2 import second

def get_layer_electrode_map(layer_configs, electrode_positions):

    electrode_z = np.array([pos[2] for pos in electrode_positions])
    layer_electrode = {}

    for layer_name, cfg in layer_configs.items():
        z_range = cfg['coordinates']['z']
        layer_center_z = (z_range[0] + z_range[1]) / 2.0
        closest_idx = np.argmin(np.abs(electrode_z - layer_center_z))
        layer_electrode[layer_name] = closest_idx

    return layer_electrode



def bandpass(signal, fs, low, high, order=4):
    lo = max(low, 0.5)
    hi = min(high, fs / 2 - 1)
    if lo >= hi:
        return np.zeros_like(signal)
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return filtfilt(b, a, signal)


def _filter_edge_samples(fs, low, order=4):

    if low <= 0:
        low = 0.5
    settle_time = 3.0 * (2 * order) / (2 * np.pi * low)
    return int(np.ceil(settle_time * fs))




def _rayleigh_pvalue(phases):

    n = len(phases)
    if n < 2:
        return 1.0
    R = np.abs(np.sum(phases)) / n
    Z = n * R**2
    p = np.exp(-Z) * (1 + (2*Z - Z**2) / (4*n) - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4) / (288*n**2))
    return max(p, 0.0)


def _ppc_one_neuron(spike_times_sec, hilbert_trimmed, t_start, dt,
                    n_valid, min_spikes=20, rayleigh_alpha=None):
   
    spikes = spike_times_sec[spike_times_sec >= t_start]
    if len(spikes) < min_spikes:
        return np.nan, 0

    spike_indices = np.round((spikes - t_start) / dt).astype(int)

    valid_mask = (spike_indices >= 0) & (spike_indices < n_valid)
    spike_indices = spike_indices[valid_mask]

    if len(spike_indices) < min_spikes:
        return np.nan, 0

    analytic_at_spikes = hilbert_trimmed[spike_indices]

    magnitudes = np.abs(analytic_at_spikes)
    nonzero = magnitudes > 0
    if np.sum(nonzero) < min_spikes:
        return np.nan, 0
    z = analytic_at_spikes[nonzero] / magnitudes[nonzero]

    dof = len(z)
    if dof < 2:
        return np.nan, 0

    if rayleigh_alpha is not None:
        p = _rayleigh_pvalue(z)
        if p >= rayleigh_alpha:
            return np.nan, 0

    sinSum = np.sum(z.imag)
    cosSum = np.sum(z.real)
    ppc = (cosSum**2 + sinSum**2 - dof) / (dof * (dof - 1))
    return ppc, dof


def compute_ppc_at_band(spike_monitor, lfp, fs, dt,
                        band=(25, 45), t_discard=1.0,
                        min_spikes=20, rayleigh_alpha=None):
  
    filtered = bandpass(lfp, fs, band[0], band[1])
    hilbert_signal = hilbert(filtered)

    n_samples = len(lfp)
    time_array = np.arange(n_samples) * dt

    n_edge = _filter_edge_samples(fs, band[0])
    n_skip_start = max(int(t_discard / dt), n_edge)

    n_skip_end = n_edge

    if n_skip_start + n_skip_end >= n_samples:
        return 0.0, []

    hilbert_trimmed = hilbert_signal[n_skip_start: n_samples - n_skip_end]
    t_start = time_array[n_skip_start]
    n_valid = len(hilbert_trimmed)

    trains = spike_monitor.spike_trains()
    per_neuron = []

    for idx in trains:
        st = np.array(trains[idx] / second)
        ppc_val, dof = _ppc_one_neuron(
            st, hilbert_trimmed, t_start, dt, n_valid,
            min_spikes=min_spikes, rayleigh_alpha=rayleigh_alpha
        )
        if not np.isnan(ppc_val) and np.isfinite(ppc_val):
            per_neuron.append((ppc_val, dof))

    if not per_neuron:
        return 0.0, per_neuron

    ppcs = np.array([p for p, _ in per_neuron])
    dofs = np.array([d for _, d in per_neuron])
    weights = dofs * (dofs - 1)
    weighted_ppc = np.average(ppcs, weights=weights)

    return weighted_ppc, per_neuron

def compute_ppc_spectrum(spike_monitor, lfp, fs, dt,
                         t_discard=1.0, freq_range=(5, 100),
                         freq_step=2, Q=4.0, min_spikes=20,
                         rayleigh_alpha=None):
   
    center_freqs = np.arange(freq_range[0], freq_range[1] + freq_step,
                             freq_step)
    ppc_values = np.zeros(len(center_freqs))

    for fi, fc in enumerate(center_freqs):
        bw = fc / Q
        lo = fc - bw / 2
        hi = fc + bw / 2
        mean_ppc, _ = compute_ppc_at_band(
            spike_monitor, lfp, fs, dt,
            band=(lo, hi), t_discard=t_discard,
            min_spikes=min_spikes, rayleigh_alpha=rayleigh_alpha
        )
        ppc_values[fi] = mean_ppc

    return center_freqs, ppc_values



def _cell_type_from_key(key):
    key_upper = key.upper()
    for ct in ['VIP', 'SOM', 'PV']:
        if ct in key_upper:
            return ct
    if 'E' in key.split('_')[0].upper() and len(key.split('_')[0]) <= 2:
        return 'E'
    return key.split('_')[0]

def compute_all_ppc(spike_monitors, lfp_signals, lfp_time_array,
                    layer_configs, electrode_positions,
                    t_discard=2.0,
                    freq_range=(5, 100), freq_step=2, Q=4.0,
                    min_spikes=20, rayleigh_alpha=None):
  
    layer_electrode = get_layer_electrode_map(layer_configs, electrode_positions)

    if len(lfp_time_array) < 2:
        raise ValueError("LFP time array must have at least 2 samples.")
    dt_lfp = (lfp_time_array[1] - lfp_time_array[0]) / 1000.0  # ms -> s
    fs = 1.0 / dt_lfp
    dt = dt_lfp

    print(f"LFP: {lfp_signals.shape[0]} electrodes, "
          f"{lfp_signals.shape[1]} samples, fs={fs:.0f} Hz, dt={dt*1e3:.3f} ms")
    print(f"Analysis: Q={Q}, min_spikes={min_spikes}, "
          f"t_discard={t_discard}s, rayleigh_alpha={rayleigh_alpha}")
    print(f"Layer -> electrode mapping:")
    for layer_name, elec_idx in layer_electrode.items():
        z_center = (layer_configs[layer_name]['coordinates']['z'][0] +
                    layer_configs[layer_name]['coordinates']['z'][1]) / 2
        elec_z = electrode_positions[elec_idx][2]
        dist = abs(z_center - elec_z)
        print(f"  {layer_name}: center z={z_center:.2f}, "
              f"electrode {elec_idx} (z={elec_z:.2f}, dist={dist:.2f})")

    results = {}
    for layer_name in spike_monitors:
        results[layer_name] = {}
        elec_idx = layer_electrode[layer_name]
        lfp = lfp_signals[elec_idx]

        for key, sm in spike_monitors[layer_name].items():
            ct = _cell_type_from_key(key)
            n_neurons = len(sm.spike_trains())
            print(f"  {layer_name} {ct} ({n_neurons} neurons, "
                  f"electrode {elec_idx})...", end=" ")

            freqs, spec = compute_ppc_spectrum(
                sm, lfp, fs, dt,
                t_discard=t_discard,
                freq_range=freq_range, freq_step=freq_step,
                Q=Q, min_spikes=min_spikes,
                rayleigh_alpha=rayleigh_alpha,
            )

            results[layer_name][ct] = {'ppc_spectrum': (freqs, spec)}

            if len(spec) > 0 and np.any(spec > 0):
                peak_idx = np.argmax(spec)
                print(f"peak PPC = {spec[peak_idx]:.6f} at {freqs[peak_idx]:.1f} Hz")
            else:
                print("no valid PPC")

    return results


def plot_ppc_spectra(results, save_path=None, smooth_sigma=0.7):

    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    colors = {'E': '#2ca02c', 'PV': '#d62728',
              'SOM': '#1f77b4', 'VIP': '#ff7f0e'}
    layers = list(results.keys())
    n = len(layers)
    fig, axes = plt.subplots(n, 1, figsize=(7, 3 * n), squeeze=False)

    for i, layer in enumerate(layers):
        ax = axes[i, 0]
        for ct in ['E', 'PV', 'SOM', 'VIP']:
            if ct in results[layer] and 'ppc_spectrum' in results[layer][ct]:
                freqs, vals = results[layer][ct]['ppc_spectrum']
                if len(freqs) > 0:
                    if smooth_sigma > 0:
                        plot_vals = gaussian_filter1d(vals, sigma=smooth_sigma)
                    else:
                        plot_vals = vals
                    ax.plot(freqs, plot_vals, color=colors[ct], label=ct, lw=2)
                    if smooth_sigma > 0:
                        ax.fill_between(freqs, vals, alpha=0.08, color=colors[ct])
        ax.set_ylabel('PPC')
        ax.set_title(layer)
        ax.legend(loc='upper right')
        ax.set_xlim(freqs[0] if len(freqs) > 0 else 0,
                    freqs[-1] if len(freqs) > 0 else 100)
        if i == n - 1:
            ax.set_xlabel('Frequency (Hz)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()