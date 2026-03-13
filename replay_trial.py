"""
Replay a specific optimization trial with full plots (raster, LFP, power spectra).

USAGE:
  # Replay best trial from the study:
  python replay_trial.py

  # Replay a specific trial number:
  python replay_trial.py --trial 29

  # Use a JSON params file instead:
  python replay_trial.py --json best_params.json
"""

import argparse
import json
import time
import numpy as np
import pandas as pd
from copy import deepcopy

import brian2 as b2
from brian2 import *
from brian2tools import *

from config.config2 import CONFIG as BASE_CONFIG
from src.column import CorticalColumn
from src.visualization import *
from src.analysis import *
from lfp_kernel import calculate_lfp_kernel_method
from lgn_to_brian2_v2 import make_lgn_inputs_split
from tools.utils import load_connectivity_from_csv

# Reuse the scaling functions from optimize_v1
from optimize_v1 import (
    apply_conductance_scaling,
    SIM_CONFIG,
)


def load_params_from_study(trial_number=None):
    """Load params from the Optuna study DB."""
    import optuna
    study = optuna.load_study(
        study_name=SIM_CONFIG['study_name'],
        storage=SIM_CONFIG['db_path'],
    )
    if trial_number is not None:
        trial = study.trials[trial_number]
        print(f"Loading trial {trial_number} (score={trial.value:.4f})")
    else:
        trial = study.best_trial
        print(f"Loading best trial #{trial.number} (score={trial.value:.4f})")

    params = dict(trial.params)

    # Print spectral details if available
    spectral = trial.user_attrs.get('spectral_details', {})
    if spectral:
        print(f"  Low-freq ratio: {spectral.get('low_freq_ratio', '?')}")
        print(f"  Gamma ratio:    {spectral.get('gamma_ratio', '?')}")

    return params


def load_params_from_json(json_path):
    """Load params from a JSON file."""
    with open(json_path) as f:
        params = json.load(f)
    print(f"Loaded params from {json_path}")
    return params


def build_config(params):
    """Apply optimization params to build a modified CONFIG."""
    config = deepcopy(BASE_CONFIG)

    # 1) Scale conductances
    base_ampa = pd.read_csv('config/conductances_AMPA2_alpha_v2.csv', index_col=0)
    base_nmda = pd.read_csv('config/conductances_NMDA2_alpha_v2.csv', index_col=0)

    scaled_ampa, scaled_nmda = apply_conductance_scaling(
        base_ampa, base_nmda, params
    )

    tmp_ampa = '/tmp/replay_ampa.csv'
    tmp_nmda = '/tmp/replay_nmda.csv'
    scaled_ampa.to_csv(tmp_ampa)
    scaled_nmda.to_csv(tmp_nmda)

    csv_layer_configs, inter_conns, inter_conds = load_connectivity_from_csv(
        'config/connection_probabilities2.csv', tmp_ampa, tmp_nmda
    )

    # 2) Apply to config
    for layer_name in config['layers']:
        if layer_name in csv_layer_configs:
            config['layers'][layer_name]['connection_prob'] = \
                csv_layer_configs[layer_name]['connection_prob']
            config['layers'][layer_name]['conductance'] = \
                csv_layer_configs[layer_name]['conductance']

    # 3) Scale Poisson backgrounds
    bg_nmda_scale = params.get('bg_nmda_scale', 1.0)
    bg_ampa_scale = params.get('bg_ampa_scale', 1.0)
    bg_rate = params.get('bg_rate_hz', 5.0)

    for layer_name, layer_cfg in config['layers'].items():
        layer_cfg['input_rate'] = bg_rate * Hz

        if 'poisson_inputs' in layer_cfg:
            for key, pdict in layer_cfg['poisson_inputs'].items():
                if 'NMDA' in key:
                    pdict['N'] = max(1, int(round(pdict['N'] * bg_nmda_scale)))
                else:
                    pdict['N'] = max(1, int(round(pdict['N'] * bg_ampa_scale)))

    # 4) Inter-layer connections
    config['inter_layer_connections'] = inter_conns
    config['inter_layer_conductances'] = inter_conds

    return config


def main():
    parser = argparse.ArgumentParser(description='Replay an optimization trial with plots')
    parser.add_argument('--trial', type=int, default=None,
                        help='Trial number to replay (default: best trial)')
    parser.add_argument('--json', type=str, default=None,
                        help='Load params from JSON file instead of study DB')
    args = parser.parse_args()

    # Load params
    if args.json:
        params = load_params_from_json(args.json)
    else:
        params = load_params_from_study(args.trial)

    print("\nParameters:")
    for k, v in sorted(params.items()):
        print(f"  {k}: {v:.4f}")

    # Build config
    config = build_config(params)

    # Run simulation
    gray_ms = SIM_CONFIG['gray_duration_ms']
    grating_ms = SIM_CONFIG['grating_duration_ms']
    total_ms = gray_ms + grating_ms

    np.random.seed(SIM_CONFIG['random_seed'])
    b2.start_scope()
    b2.defaultclock.dt = config['simulation']['DT']

    print("\nBuilding cortical column...")
    column = CorticalColumn(column_id=0, config=config)
    all_monitors = column.get_all_monitors()

    print("Connecting LGN inputs...")
    lgn = make_lgn_inputs_split(
        column, config,
        npz_path=SIM_CONFIG['npz_path'],
        total_lgn_duration_ms=total_ms,
        layers_to_connect=['L4C', 'L6'],
        gray_drive_scale=0.6,
        grating_drive_scale=1.2,
        gray_duration_ms=gray_ms,
    )
    for obj_list in lgn.values():
        column.network.add(*obj_list)

    print(f"Running simulation ({total_ms}ms)...")
    t_start = time.time()
    column.network.run(total_ms * ms)
    print(f"Simulation took {time.time() - t_start:.1f}s")

    # Extract monitors
    spike_monitors = {}
    state_monitors = {}
    rate_monitors = {}
    neuron_groups = {}

    for layer_name, monitors in all_monitors.items():
        spike_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'spikes' in k
        }
        state_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'state' in k
        }
        rate_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'rate' in k
        }
        neuron_groups[layer_name] = column.layers[layer_name].neuron_groups

    # Print firing rates
    print("\nFiring rates:")
    for layer_name, monitors in spike_monitors.items():
        for mon_key, mon in monitors.items():
            n_spikes = len(mon.t)
            n_neurons = mon.source.N if hasattr(mon, 'source') else 1
            rate = n_spikes / (n_neurons * total_ms / 1000.0) if n_neurons > 0 else 0
            print(f"  {layer_name}_{mon_key}: {rate:.1f} Hz")

    # Compute LFP
    print("\nComputing LFP...")
    electrode_positions = config['electrode_positions']
    lfp_signals, time_array = calculate_lfp_kernel_method(
        spike_monitors, neuron_groups,
        config['layers'], electrode_positions,
        sim_duration_ms=total_ms
    )

    print("Computing bipolar LFP...")
    bipolar_signals, channel_labels, channel_depths = compute_bipolar_lfp(
        lfp_signals, electrode_positions
    )

    # --- Plots (same as main.py) ---
    fig_raster = plot_raster(spike_monitors, gray_ms, grating_ms, config['layers'])

    fig_power_lfp = plot_lfp_power_comparison_kernel(
        lfp_signals, time_array, electrode_positions,
        baseline_time=gray_ms,
        pre_stim_duration=500,
        post_stim_duration=500,
        transient_skip=300
    )

    fig_power_bipolar = plot_bipolar_power_comparison_kernel(
        bipolar_signals, channel_labels, channel_depths, time_array,
        baseline_time=gray_ms,
        pre_stim_duration=500,
        post_stim_duration=500,
        transient_skip=500
    )

    fig_rate = plot_rate(rate_monitors, config['layers'], gray_ms, grating_ms,
                         smooth_window=15*ms, ylim_max=80, show_stats=True)

    fig_lfp = plot_lfp_comparison(
        lfp_signals, bipolar_signals, time_array, electrode_positions,
        channel_labels, channel_depths, figsize=(18, 12),
        time_range=(1000, 3500)
    )

    plt.show()


if __name__ == '__main__':
    main()
