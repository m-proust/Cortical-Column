import os
import numpy as np
import brian2 as b2
from brian2 import *
from config.config2 import CONFIG
from src.column import CorticalColumn
from src.analysis import calculate_lfp_kernel_method, compute_power_spectrum
import copy
import argparse


def run_single_rate(
    config,
    targets,
    stim_rate_hz,
    trial_id=0,
    base_seed=None,
    baseline_ms=1000,
    stim_ms=1500,
    fs=10000,
    verbose=True,
):
   

    if base_seed is None:
        base_seed = config['simulation']['RANDOM_SEED']

    trial_seed = int(base_seed + trial_id)
    np.random.seed(trial_seed)
    b2.seed(trial_seed)

    b2.start_scope()
    b2.defaultclock.dt = config['simulation']['DT']

    if verbose:
        parts = []
        for key, scale in targets.items():
            if len(key) == 2:
                layer, pop = key
                input_type = 'AMPA'
            else:
                layer, pop, input_type = key
            parts.append(f"{pop}_{layer}[{input_type}](x{scale})")
        targets_str = ", ".join(parts)
        print(f"\n=== Stimulating {targets_str} at {stim_rate_hz} Hz ===")
        print("Creating cortical column...")

    column = CorticalColumn(column_id=0, config=config)
    all_monitors = column.get_all_monitors()

    w_ext_AMPA = config['synapses']['Q']['EXT_AMPA']

    w_ext_NMDA = config['synapses']['Q']['EXT_NMDA']
    input_type_map = {
        'AMPA': ('gE_AMPA', w_ext_AMPA),
        'NMDA': ('gE_NMDA', w_ext_NMDA),
        'PV': ('gPV', w_ext_AMPA),    
        'SOM': ('gSOM', w_ext_AMPA),
        'VIP': ('gVIP', w_ext_AMPA),
    }

    if stim_rate_hz > 0:
        for target_key, weight_scale in targets.items():
       
            if len(target_key) == 2:
                target_layer, target_pop = target_key
                input_type = 'AMPA'
            else:
                target_layer, target_pop, input_type = target_key

            if input_type not in input_type_map:
                raise ValueError(f"Unknown input_type '{input_type}'. Must be one of: {list(input_type_map.keys())}")

            conductance_var, base_weight = input_type_map[input_type]

            target_layer_obj = column.layers[target_layer]
            cfg_target = config['layers'][target_layer]
            target_grp = target_layer_obj.neuron_groups[target_pop]
            N_stim = int(cfg_target['poisson_inputs'][target_pop]['N'])

            stim_input = PoissonInput(target_grp, conductance_var,
                                      N=N_stim,
                                      rate=stim_rate_hz * Hz,
                                      weight=base_weight * weight_scale)
            column.network.add(stim_input)

            if verbose:
                print(f"Added stimulation: {target_pop}_{target_layer} -> {conductance_var} at {stim_rate_hz} Hz (weight x{weight_scale})")

    if verbose:
        print(f"Running baseline ({baseline_ms} ms) for stabilization...")
    column.network.run(baseline_ms * ms)

    if verbose:
        print(f"Running analysis period ({stim_ms} ms)...")
    column.network.run(stim_ms * ms)

    if verbose:
        print("Simulation complete. Computing LFP...")

    spike_monitors = {}
    neuron_groups = {}

    for layer_name, monitors in all_monitors.items():
        spike_monitors[layer_name] = {k: v for k, v in monitors.items() if 'spikes' in k}
        neuron_groups[layer_name] = column.layers[layer_name].neuron_groups

    electrode_positions = config['electrode_positions']
    total_sim_ms = baseline_ms + stim_ms

    from lfp_mazzoni_method import calculate_lfp_mazzoni


    lfp_signals, time_array = calculate_lfp_mazzoni(
        spike_monitors,
        neuron_groups,
        config['layers'],
        electrode_positions,
        fs=fs,
        sim_duration_ms=total_sim_ms,
    )

    analysis_start_ms = baseline_ms + 500
    analysis_end_ms = baseline_ms + 1500

    start_idx = int(analysis_start_ms * fs / 1000)
    end_idx = int(analysis_end_ms * fs / 1000)


    window_samples = end_idx - start_idx
    nperseg = window_samples // 2

    power_spectra = {}
    freqs = None

    for elec_idx in range(len(electrode_positions)):
        lfp_window = lfp_signals[elec_idx][start_idx:end_idx]
        freq, psd = compute_power_spectrum(lfp_window, fs=fs, nperseg=nperseg)
        power_spectra[elec_idx] = psd
        if freqs is None:
            freqs = freq

    n_elec = len(lfp_signals)
    lfp_matrix = np.vstack([lfp_signals[i] for i in range(n_elec)])

    psd_matrix = np.vstack([power_spectra[i] for i in range(n_elec)])

    data = {
        "targets": targets,
        "stim_rate_hz": stim_rate_hz,
        "trial_id": trial_id,
        "seed": trial_seed,
        "time_array_ms": np.array(time_array),
        "electrode_positions": np.array(electrode_positions),
        "lfp_matrix": lfp_matrix,
        "frequencies": freqs,
        "psd_matrix": psd_matrix,
        "baseline_ms": baseline_ms,
        "stim_ms": stim_ms,
        "analysis_start_ms": analysis_start_ms,
        "analysis_end_ms": analysis_end_ms,
        "fs": fs,
    }

    if verbose:
        print(f"Done. PSD matrix shape: {psd_matrix.shape}")

    return data


def run_rate_sweep(
    config,
    targets,
    rate_values,
    base_seed=None,
    baseline_ms=1000,
    stim_ms=1500,
    fs=10000,
    save_dir="results/input_sweeps",
    verbose=True,
):
   

    if base_seed is None:
        base_seed = config['simulation']['RANDOM_SEED']

    os.makedirs(save_dir, exist_ok=True)

    all_psd_matrices = []
    all_rates = []
    freqs = None
    electrode_positions = None

    for i, rate in enumerate(rate_values):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Rate sweep: {i+1}/{len(rate_values)} - {rate} Hz")
            print(f"{'='*60}")

        data = run_single_rate(
            config=config,
            targets=targets,
            stim_rate_hz=rate,
            trial_id=i,
            base_seed=base_seed,
            baseline_ms=baseline_ms,
            stim_ms=stim_ms,
            fs=fs,
            verbose=verbose,
        )

        all_psd_matrices.append(data['psd_matrix'])
        all_rates.append(rate)

        if freqs is None:
            freqs = data['frequencies']
            electrode_positions = data['electrode_positions']

    psd_stack = np.stack(all_psd_matrices, axis=0)

    rate_min = int(min(rate_values))
    rate_max = int(max(rate_values))
   
    targets_str_parts = []
    target_layers = []
    target_pops = []
    input_types = []
    weight_scales = []

    for target_key, scale in targets.items():
        if len(target_key) == 2:
            layer, pop = target_key
            input_type = 'AMPA'
        else:
            layer, pop, input_type = target_key

        targets_str_parts.append(f"{pop}{layer}_{input_type}")
        target_layers.append(layer)
        target_pops.append(pop)
        input_types.append(input_type)
        weight_scales.append(scale)

    targets_str = "_".join(targets_str_parts)
    fname = f"sweep_{targets_str}_{rate_min}-{rate_max}Hz.npz"
    fpath = os.path.join(save_dir, fname)


    target_layers = np.array(target_layers)
    target_pops = np.array(target_pops)
    input_types = np.array(input_types)
    weight_scales = np.array(weight_scales)

    np.savez_compressed(
        fpath,
        target_layers=target_layers,
        target_pops=target_pops,
        input_types=input_types,
        weight_scales=weight_scales,
        rate_values=np.array(rate_values),
        frequencies=freqs,
        psd_stack=psd_stack,
        electrode_positions=electrode_positions,
        baseline_ms=baseline_ms,
        stim_ms=stim_ms,
        fs=fs,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"Sweep complete! Saved to: {fpath}")
        print(f"PSD stack shape: {psd_stack.shape}")
        print(f"  - {len(rate_values)} rates")
        print(f"  - {psd_stack.shape[1]} electrodes")
        print(f"  - {psd_stack.shape[2]} frequency bins")
        print(f"{'='*60}")

    return fpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run input sweep simulations')
    parser.add_argument('--layer', type=str, required=True,
                        help='Target layer (e.g., L4C, L23, L5, L6, L1)')
    parser.add_argument('--pop', type=str, required=True,
                        help='Target population(s), comma-separated (e.g., E or E,PV)')
    parser.add_argument('--input-type', type=str, default='AMPA',
                        choices=['AMPA', 'NMDA', 'PV', 'SOM', 'VIP'],
                        help='Input type (default: AMPA)')
    parser.add_argument('--weight-scale', type=float, default=1.0,
                        help='Weight scale multiplier (default: 1.0)')
    parser.add_argument('--rate-min', type=float, default=0,
                        help='Minimum stimulation rate in Hz (default: 0)')
    parser.add_argument('--rate-max', type=float, default=20,
                        help='Maximum stimulation rate in Hz (default: 20)')
    parser.add_argument('--rate-step', type=float, default=1,
                        help='Step size for rate sweep in Hz (default: 1)')
    parser.add_argument('--baseline-ms', type=float, default=1000,
                        help='Baseline duration in ms (default: 1000)')
    parser.add_argument('--stim-ms', type=float, default=1500,
                        help='Stimulation duration in ms (default: 1500)')
    parser.add_argument('--save-dir', type=str, default='results/input_sweeps2',
                        help='Directory to save results (default: results/input_sweeps2)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    rate_values = np.arange(args.rate_min, args.rate_max + args.rate_step/2, args.rate_step)

    populations = [p.strip() for p in args.pop.split(',')]
    targets = {
        (args.layer, pop, args.input_type): args.weight_scale
        for pop in populations
    }

    pop_str = '+'.join(populations)
    print(f"\n{'='*70}")
    print(f"Starting sweep: {pop_str} in {args.layer} with {args.input_type} input")
    print(f"Weight scale: {args.weight_scale}")
    print(f"Rate range: {args.rate_min}-{args.rate_max} Hz (step: {args.rate_step})")
    print(f"Save directory: {args.save_dir}")
    print(f"{'='*70}\n")
    
    result_path = run_rate_sweep(
        config=CONFIG,
        targets=targets,
        rate_values=rate_values,
        baseline_ms=args.baseline_ms,
        stim_ms=args.stim_ms,
        fs=10000,
        save_dir=args.save_dir,
        verbose=not args.quiet,
    )

    print(f"\n{'='*70}")
    print(f"Results saved to: {result_path}")
    print(f"{'='*70}\n")