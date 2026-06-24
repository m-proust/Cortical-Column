"""Network analysis helpers: intrinsic heterogeneity and bipolar LFP."""
import numpy as np
from brian2 import *


def add_heterogeneity_to_layer(layer, config, scale=1.0):
    """Jitter intrinsic params of every neuron group. scale multiplies each sigma."""
    for pop_name, neuron_group in layer.neuron_groups.items():
        n = len(neuron_group)
        base = config['intrinsic_params'][pop_name]

        def vary(base_val, sigma):
            factors = np.clip(1 + np.random.randn(n) * sigma * scale, 0.5, 1.5)
            return base_val * factors

        neuron_group.C    = vary(base['C'], 0.15)
        neuron_group.gL   = vary(base['gL'], 0.12)
        neuron_group.tauw = vary(base['tauw'], 0.15)
        neuron_group.b    = vary(base['b'], 0.20)
        neuron_group.a    = vary(base['a'], 0.15)
        neuron_group.EL   = base['EL'] + np.random.randn(n) * 2 * mV * scale
        neuron_group.DeltaT = vary(base['DeltaT'], 0.10)


def compute_bipolar_lfp(lfp_signals, electrode_positions):
    """Successive differences between adjacent electrodes (rejects common signal)."""
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
