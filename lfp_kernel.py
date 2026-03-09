"""
LFP calculation using the kernel-based method from:
Telenczuk, Telenczuk & Destexhe (2020), J Neuroscience Methods 344:108871.

  - Eq. 4: Gaussian kernel   uLFP(x,t) = A(x) * exp(-(t - t_p)^2 / (2*sigma^2))
  - Eq. 5: Peak time          t_p = t_0 + d + |x - x_0| / v_a
  - Eq. 6: Amplitude decay    A(x) = A_0 * exp(-|x - x_0| / lambda)
  - Eq. 8: LFP sum            V_e(x,t) = sum_k K_e(x, t - t_{e,k}) + sum_l K_i(x, t - t_{i,l})
  - Table 1: Depth-dependent amplitude profiles for hippocampal uLFPs

Parameters from the paper:
  - Fig. 4A constrained fit: v_a = 166 mm/s, d = 10.4 ms, A_0 = -3.4 uV,
    lambda = 0.34 mm, sigma_i = 2.1 ms
  - Section 3.3: sigma_e ~ 1.5 * sigma_i = 3.15 ms
  - Table 1: depth-dependent amplitude profiles (see DEPTH_TABLE below)
  - Fig. 2: experimental lambda ~ 0.22-0.25 mm (raw measurements)
"""

import numpy as np
from brian2 import ms, mm


LAMBDA_SPACE_MM = 0.34


V_AXON_MM_PER_MS = 0.166

BASE_DELAY_MS = 10.4

SIGMA_INHIB_MS = 2.1

SIGMA_EXCIT_MS = 3.15

KERNEL_HALF_WIDTH_MS = 25.0


DEPTH_TABLE = {
    'rel_depths_mm': np.array([-0.4, 0.0, 0.4, 0.8]),
    'i_amplitudes_uV': np.array([-0.2, 3.0, -1.2, 0.3]),
    'e_amplitudes_uV': np.array([-0.16, 0.48, 0.24, -0.08]),
}


_A0_FIT_INHIB = -3.4  
_A_DEPTH_INHIB_SUPERFICIAL = -1.2  
DEPTH_SCALE_FACTOR = _A0_FIT_INHIB / _A_DEPTH_INHIB_SUPERFICIAL


def _get_depth_amplitude(rel_depth_mm, is_excitatory):
  
    depths = DEPTH_TABLE['rel_depths_mm']
    if is_excitatory:
        amps = DEPTH_TABLE['e_amplitudes_uV']
    else:
        amps = DEPTH_TABLE['i_amplitudes_uV']

    raw_amp = np.interp(rel_depth_mm, depths, amps)
    return DEPTH_SCALE_FACTOR * raw_amp


def _build_kernel_template(sigma_ms, dt_ms):
  
    n_samples = int(2 * KERNEL_HALF_WIDTH_MS / dt_ms)
    kernel_t = np.arange(n_samples) * dt_ms - KERNEL_HALF_WIDTH_MS
    return np.exp(-kernel_t**2 / (2 * sigma_ms**2))


def _compute_lateral_distance(neuron_x, neuron_y, elec_x, elec_y):

    return np.sqrt((neuron_x - elec_x)**2 + (neuron_y - elec_y)**2)


def _compute_3d_distance(neuron_x, neuron_y, neuron_z, elec_x, elec_y, elec_z):
 
    return np.sqrt(
        (neuron_x - elec_x)**2 +
        (neuron_y - elec_y)**2 +
        (neuron_z - elec_z)**2
    )


def calculate_lfp_kernel_method(spike_monitors, neuron_groups, layer_configs,
                                electrode_positions, dt_ms=0.1,
                                sim_duration_ms=2000):
    """
    Calculate LFP from spike trains using the kernel-based method.

    The LFP at electrode position x is computed as (Eq. 8):
        V_e(x, t) = sum_k K_e(x, t - t_{e,k}) + sum_l K_i(x, t - t_{i,l})

    where each kernel K combines:
        - A depth-dependent amplitude A_depth(z) from Table 1
        - A lateral exponential decay exp(-d_lateral / lambda) from Eq. 6
        - A Gaussian temporal profile from Eq. 4
        - An axonal propagation delay from Eq. 5

    """
    n_samples = int(sim_duration_ms / dt_ms)
    n_electrodes = len(electrode_positions)
    time_array = np.arange(n_samples) * dt_ms

    template_inhib = _build_kernel_template(SIGMA_INHIB_MS, dt_ms)
    template_excit = _build_kernel_template(SIGMA_EXCIT_MS, dt_ms)

    lfp_signals = np.zeros((n_electrodes, n_samples))

    for layer_name, layer_spike_mons in spike_monitors.items():
        if layer_name not in layer_configs:
            continue

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
            template = template_excit if is_excitatory else template_inhib

            spike_times_ms = np.array(spike_mon.t / ms)
            spike_indices = np.array(spike_mon.i)

            if len(spike_times_ms) == 0:
                continue

            neuron_x = np.array(neuron_grp.x / mm)
            neuron_y = np.array(neuron_grp.y / mm)
            neuron_z = np.array(neuron_grp.z / mm)

            for elec_idx, (ex, ey, ez) in enumerate(electrode_positions):

        
                rel_depth_mm = ez - layer_center_z
                A_depth = _get_depth_amplitude(rel_depth_mm, is_excitatory)

                if abs(A_depth) < 1e-6:
                    continue


                d_lateral = _compute_lateral_distance(
                    neuron_x, neuron_y, ex, ey
                )
                A_spatial = np.exp(-d_lateral / LAMBDA_SPACE_MM)

                amplitudes = A_depth * A_spatial 

                d_3d = _compute_3d_distance(
                    neuron_x, neuron_y, neuron_z, ex, ey, ez
                )
                delays_ms = BASE_DELAY_MS + d_3d / V_AXON_MM_PER_MS
                delay_bins = np.round(delays_ms / dt_ms).astype(int)

                unique_delays = np.unique(delay_bins)

                for d_bin in unique_delays:
                    neuron_ids = np.where(delay_bins == d_bin)[0]
                    if len(neuron_ids) == 0:
                        continue


                    weighted_rate = np.zeros(n_samples)
                    for nid in neuron_ids:
                        spike_mask = (spike_indices == nid)
                        if not np.any(spike_mask):
                            continue
                        st = spike_times_ms[spike_mask]
                        bins = np.clip(
                            (st / dt_ms).astype(int), 0, n_samples - 1
                        )
                        np.add.at(weighted_rate, bins, amplitudes[nid])

                    convolved = np.convolve(weighted_rate, template,
                                            mode='same')

             
                    shift = int(d_bin)
                    if shift > 0 and shift < n_samples:
                        lfp_signals[elec_idx, shift:] += (
                            convolved[:n_samples - shift]
                        )
                    elif shift == 0:
                        lfp_signals[elec_idx] += convolved
               

    return lfp_signals, time_array


def compute_bipolar_lfp(lfp_signals, electrode_positions):

    n_electrodes = lfp_signals.shape[0]
    n_samples = lfp_signals.shape[1]

    bipolar_signals = np.zeros((n_electrodes - 1, n_samples))
    channel_labels = []
    channel_depths = []

    for i in range(n_electrodes - 1):
        bipolar_signals[i] = lfp_signals[i + 1] - lfp_signals[i]
        channel_labels.append(f'ch{i}-ch{i+1}')
        z_mid = (electrode_positions[i][2] +
                 electrode_positions[i + 1][2]) / 2.0
        channel_depths.append(z_mid)

    return bipolar_signals, channel_labels, channel_depths