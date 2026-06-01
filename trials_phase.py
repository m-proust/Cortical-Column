"""
Phase-locked trial runner.

For each seed, run a baseline-only "probe" simulation, compute the bipolar
LFP, and extract the L5 alpha (8-13 Hz) instantaneous phase from the
L5-centered bipolar channel. Find the first time after t_min_ms at which
each target phase occurs, then run one stimulus trial per target phase with
stim onset locked to that time.

Reference signal:
    Bipolar LFP channel whose midpoint depth is closest to the center of
    layer L5 (z=-0.24). Bandpass filtered 8-13 Hz, Hilbert phase.
    Convention: 0 deg = peak of the bandpassed alpha oscillation
    (cosine convention).

Reproducibility note:
    Each (seed, phase) combination is a fully independent simulation. The
    probe run is *only* used to discover the per-seed phase->time mapping;
    the actual stim trial re-builds the network with the same network_seed
    and re-runs the baseline so the dynamics up to stim onset match the
    probe run. This relies on Brian2 being deterministic given the same
    seed, defaultclock.dt, and network construction order.
"""

import os
import shutil
import numpy as np
import brian2 as b2
from brian2 import *
from scipy.signal import butter, filtfilt, hilbert

from config.config import CONFIG
from src.column import CorticalColumn
from src.visualization import *
from src.analysis import *
from tools.lfp_kernel import calculate_lfp_kernel_method


CONFIG_FILES = [
    "config/config.py",
    "config/conductances_AMPA_GABA.csv",
    "config/conductances_NMDA.csv",
    "config/connection_probabilities.csv",
    "main.py",
    "trials_phase.py",
]


REFERENCE_LAYER = 'L5'
ALPHA_BAND = (8.0, 13.0)
PHASE_LOG_LAYERS = ['L23', 'L4AB', 'L4C', 'L5', 'L6']
TRANSIENT_MS = 500  # discard before filtering / phase extraction

# z-center of each layer (mm), from config['layers'][...]['coordinates']['z']
LAYER_Z_CENTER = {
    'L23':  0.775,
    'L4AB': 0.295,
    'L4C':  0.0,
    'L5':  -0.24,
    'L6':  -0.48,
}


def save_config_snapshot(save_dir, base_dir=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    snapshot_dir = os.path.join(save_dir, "config_snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)
    for rel_path in CONFIG_FILES:
        src = os.path.join(base_dir, rel_path)
        if os.path.exists(src):
            dst = os.path.join(snapshot_dir, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    print(f"Config snapshot saved to {snapshot_dir}")


def _bandpass(sig, fs, low, high, order=4):
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig - np.mean(sig)
    nyq = 0.5 * fs
    sos = butter(order, [low / nyq, high / nyq], btype='band', output='sos')
    from scipy.signal import sosfiltfilt
    return sosfiltfilt(sos, sig)


def _build_column(config, network_seed):
    np.random.seed(network_seed)
    b2.seed(network_seed)
    b2.start_scope()
    b2.defaultclock.dt = config['simulation']['DT']
    column = CorticalColumn(column_id=0, config=config)
    for _, layer in column.layers.items():
        add_heterogeneity_to_layer(layer, config)
    return column


def _add_stimulus(column, config):
    """Attach the standard L4C + L6 PoissonInput stimulus. Returns inputs so
    caller can keep references. Mirrors trials.run_single_trial."""
    w_ext_AMPA = config['synapses']['Q']['EXT_AMPA']

    L4C = column.layers['L4C']
    L6 = column.layers['L6']

    L4C_E_grp = L4C.neuron_groups['E']
    L4C_PV_grp = L4C.neuron_groups['PV']
    L6_E_grp = L6.neuron_groups['E']
    L6_PV_grp = L6.neuron_groups['PV']

    L4C_E_stim = PoissonInput(L4C_E_grp, 'gE_AMPA',
                              N=30, rate=5*Hz, weight=w_ext_AMPA)
    L4C_PV_stim = PoissonInput(L4C_PV_grp, 'gE_AMPA',
                               N=40, rate=7*Hz, weight=w_ext_AMPA*2.5)
    L6_E_stim = PoissonInput(L6_E_grp, 'gE_AMPA',
                             N=10, rate=5*Hz, weight=w_ext_AMPA*1.5)
    L6_PV_stim = PoissonInput(L6_PV_grp, 'gE_AMPA',
                              N=10, rate=6*Hz, weight=w_ext_AMPA*1.5)

    column.network.add(L4C_E_stim, L4C_PV_stim, L6_E_stim, L6_PV_stim)
    return [L4C_E_stim, L4C_PV_stim, L6_E_stim, L6_PV_stim]


def _hilbert_phase(sig, dt_ms, band=ALPHA_BAND):
    """Return instantaneous phase (radians) of bandpassed signal."""
    fs = 1000.0 / dt_ms
    filt = _bandpass(sig, fs, band[0], band[1])
    analytic = hilbert(filt)
    return np.angle(analytic), filt


def _find_phase_crossings(phase, t_ms, target_phases_rad, t_min_ms):
    """For each target phase, return the first t_ms >= t_min_ms where the
    instantaneous phase passes through the target in the ascending direction
    (phase increasing through tgt). Robust to wrap discontinuities."""
    out = {}
    idx0 = int(np.searchsorted(t_ms, t_min_ms))
    # Use a unit phasor projection: as phase ascends through tgt, the signal
    # cos(phase - tgt) crosses through zero from + to -? No — better: use
    # sin(phase - tgt). When phase ascends through tgt: sin(phase - tgt)
    # crosses 0 from negative to positive. AND |phase - tgt| must be small,
    # to reject the ambiguous opposite-side crossing.
    for tgt in target_phases_rad:
        s = np.sin(phase - tgt)
        c = np.cos(phase - tgt)
        # Ascending zero-crossings of sin where cos > 0 (i.e. near tgt, not tgt+pi)
        cross = (s[:-1] < 0) & (s[1:] >= 0) & (c[:-1] > 0)
        cross_idx = np.where(cross)[0]
        cross_idx = cross_idx[cross_idx >= idx0]
        if len(cross_idx) == 0:
            out[tgt] = None
            continue
        i = cross_idx[0]
        s0, s1 = s[i], s[i + 1]
        frac = -s0 / (s1 - s0) if (s1 - s0) != 0 else 0.0
        out[tgt] = float(t_ms[i] + frac * (t_ms[i + 1] - t_ms[i]))
    return out


def _pick_layer_bipolar_channel(channel_depths, layer_name):
    """Return the index of the bipolar channel whose midpoint is closest to
    the center of the given layer."""
    target_z = LAYER_Z_CENTER[layer_name]
    return int(np.argmin(np.abs(np.array(channel_depths) - target_z)))


def run_probe(config, network_seed, probe_ms=2600, verbose=True):
    """Run baseline only, compute bipolar LFP, return phase trace of the
    L5-centered bipolar channel + per-layer bipolar phase traces."""
    if verbose:
        print(f"  [probe] seed={network_seed}, duration={probe_ms} ms")

    column = _build_column(config, network_seed)
    column.network.run(probe_ms * ms)

    all_monitors = column.get_all_monitors()
    spike_monitors = {}
    neuron_groups = {}
    for layer_name, monitors in all_monitors.items():
        spike_monitors[layer_name] = {k: v for k, v in monitors.items() if 'spikes' in k}
        neuron_groups[layer_name] = column.layers[layer_name].neuron_groups

    electrode_positions = config['electrode_positions']
    lfp_signals, time_array = calculate_lfp_kernel_method(
        spike_monitors, neuron_groups, config['layers'],
        electrode_positions, sim_duration_ms=probe_ms,
    )
    bipolar_signals, channel_labels, channel_depths = compute_bipolar_lfp(
        lfp_signals, electrode_positions,
    )

    t_ms_full = np.array(time_array)  # dt = 0.1 ms from kernel method
    dt_ms = float(t_ms_full[1] - t_ms_full[0])

    # Discard transient before filtering
    start_idx = int(TRANSIENT_MS / dt_ms)
    t_ms = t_ms_full[start_idx:]

    ref_ch = _pick_layer_bipolar_channel(channel_depths, REFERENCE_LAYER)
    ref_signal = np.array(bipolar_signals[ref_ch])[start_idx:]
    ref_phase, ref_filt = _hilbert_phase(ref_signal, dt_ms)

    layer_phases = {}
    for layer_name in PHASE_LOG_LAYERS:
        ch = _pick_layer_bipolar_channel(channel_depths, layer_name)
        sig = np.array(bipolar_signals[ch])[start_idx:]
        ph, filt = _hilbert_phase(sig, dt_ms)
        layer_phases[layer_name] = {
            'phase': ph, 'filt': filt, 'channel_idx': ch,
            'channel_depth': float(channel_depths[ch]),
            'channel_label': channel_labels[ch],
        }

    if verbose:
        print(f"    L5 reference: bipolar ch {ref_ch} "
              f"({channel_labels[ref_ch]}, z={channel_depths[ref_ch]:.3f}); "
              f"alpha-band amp = {np.std(ref_filt):.3g}")

    return {
        'ref_t_ms': t_ms,
        'ref_signal': ref_signal,
        'ref_phase': ref_phase,
        'ref_filt': ref_filt,
        'ref_channel_idx': ref_ch,
        'ref_channel_label': channel_labels[ref_ch],
        'ref_channel_depth': float(channel_depths[ref_ch]),
        'dt_ms': dt_ms,
        'layer_phases': layer_phases,
        'channel_labels': channel_labels,
        'channel_depths': channel_depths,
    }


def run_phase_locked_trial(
    config,
    trial_id,
    network_seed,
    stim_onset_ms,
    target_phase_deg,
    post_stim_ms=2000,
    verbose=True,
):
    """Re-run with same seed; baseline up to stim_onset_ms then add stimulus."""
    if verbose:
        print(f"  [trial {trial_id}] seed={network_seed}, "
              f"phase={target_phase_deg:.1f} deg, onset={stim_onset_ms:.2f} ms")

    column = _build_column(config, network_seed)
    column.network.run(stim_onset_ms * ms)
    _add_stimulus(column, config)
    column.network.run(post_stim_ms * ms)

    total_time = stim_onset_ms + post_stim_ms

    all_monitors = column.get_all_monitors()
    spike_monitors, state_monitors, rate_monitors, isyn_full_monitors = {}, {}, {}, {}
    neuron_groups = {}
    for layer_name, monitors in all_monitors.items():
        spike_monitors[layer_name] = {k: v for k, v in monitors.items() if 'spikes' in k}
        state_monitors[layer_name] = {
            k: v for k, v in monitors.items()
            if 'state' in k and 'Isyn_full' not in k
        }
        rate_monitors[layer_name] = {k: v for k, v in monitors.items() if 'rate' in k}
        isyn_full_monitors[layer_name] = {k: v for k, v in monitors.items() if 'Isyn_full' in k}
        neuron_groups[layer_name] = column.layers[layer_name].neuron_groups

    electrode_positions = config['electrode_positions']

    lfp_signals, time_array = calculate_lfp_kernel_method(
        spike_monitors, neuron_groups, config['layers'],
        electrode_positions, sim_duration_ms=total_time,
    )
    bipolar_signals, channel_labels, channel_depths = compute_bipolar_lfp(
        lfp_signals, electrode_positions,
    )

    from tools.lfp_current_method import calculate_lfp_current_method
    current_method_monitors = {
        ln: {k.replace('_Isyn_full', '_state'): v for k, v in mons.items()}
        for ln, mons in isyn_full_monitors.items()
    }
    lfp_current_matrix, time_current_ms = calculate_lfp_current_method(
        current_method_monitors, neuron_groups, config['layers'],
        electrode_positions, dt_ms=0.5, sim_duration_ms=total_time,
    )

    spike_data = {}
    for layer_name, layer_spike_mons in spike_monitors.items():
        spike_data[layer_name] = {}
        for mon_name, mon in layer_spike_mons.items():
            spike_data[layer_name][mon_name] = {
                "times_ms": np.array(mon.t / ms),
                "spike_indices": np.array(mon.i),
            }

    rate_data = {}
    lfp_full = {}
    for layer_name, layer_rate_mons in rate_monitors.items():
        rate_data[layer_name] = {}
        for mon_name, mon in layer_rate_mons.items():
            if len(mon.t) == 0:
                continue
            rate_data[layer_name][mon_name] = {
                "t_ms": np.array(mon.t / ms),
                "rate_hz": np.array(mon.rate / Hz),
            }
        e_rate_mon = layer_rate_mons.get('E_rate')
        if e_rate_mon is not None:
            lfp_full[layer_name] = np.array(
                e_rate_mon.smooth_rate(window='gaussian', width=1*ms) / Hz
            )

    state_data = {}
    for layer_name, layer_state_mons in state_monitors.items():
        state_data[layer_name] = {}
        for mon_name, mon in layer_state_mons.items():
            pop_name = mon_name.replace('_state', '')
            state_data[layer_name][pop_name] = {'t_ms': np.array(mon.t / ms)}
            for var in mon.record_variables:
                vals = np.array(getattr(mon, var))
                state_data[layer_name][pop_name][var] = np.mean(
                    vals, axis=0).astype(np.float32)

    n_elec = len(lfp_signals)
    lfp_matrix = np.vstack([lfp_signals[i] for i in range(n_elec)])
    bipolar_matrix = np.vstack([bipolar_signals[i] for i in range(len(bipolar_signals))])

    return {
        "trial_id": trial_id,
        "network_seed": network_seed,
        "target_phase_deg": float(target_phase_deg),
        "stim_onset_ms": float(stim_onset_ms),
        "post_stim_ms": int(post_stim_ms),
        "baseline_ms": float(stim_onset_ms),
        "reference_layer": REFERENCE_LAYER,
        "alpha_band_hz": np.array(ALPHA_BAND),
        "stim_rates": {"L4C_E": 5.0, "L4C_PV": 7.0, "L6_E": 5.0, "L6_PV": 6.0},
        "time_array_ms": np.array(time_array),
        "electrode_positions": np.array(electrode_positions),
        "channel_labels": np.array(channel_labels, dtype=object),
        "channel_depths": np.array(channel_depths),
        "rate_data": rate_data,
        "spike_data": spike_data,
        "state_data": state_data,
        "lfp_full": lfp_full,
        "lfp_matrix": lfp_matrix,
        "bipolar_matrix": bipolar_matrix,
        "lfp_current_matrix": lfp_current_matrix.astype(np.float32),
        "time_current_ms": time_current_ms.astype(np.float32),
    }


def run_phase_experiment(
    config,
    seeds,
    target_phases_deg=(0, 45, 90, 135, 180, 225, 270, 315),
    t_min_ms=2000,
    probe_ms=2600,
    post_stim_ms=2000,
    save_dir="results/trials_phase",
    verbose=True,
):
    """For each seed, run a probe, extract phase->time mapping, run one trial
    per target phase. Saves one .npz per (seed, phase) plus a per-seed probe
    summary."""
    os.makedirs(save_dir, exist_ok=True)
    save_config_snapshot(save_dir)

    target_phases_rad = np.deg2rad(np.array(target_phases_deg))

    for seed in seeds:
        if verbose:
            print(f"\n=== seed {seed} ===")

        probe = run_probe(config, seed, probe_ms=probe_ms, verbose=verbose)
        phase_to_t = _find_phase_crossings(
            probe['ref_phase'], probe['ref_t_ms'],
            target_phases_rad, t_min_ms,
        )

        # Save probe summary (small) for diagnostics
        probe_path = os.path.join(save_dir, f"probe_seed_{seed}.npz")
        layer_phase_at_target = {
            ph_deg: {ln: None for ln in PHASE_LOG_LAYERS}
            for ph_deg in target_phases_deg
        }
        for ph_deg, tgt_rad in zip(target_phases_deg, target_phases_rad):
            t_target = phase_to_t[tgt_rad]
            if t_target is None:
                continue
            idx = int(np.argmin(np.abs(probe['ref_t_ms'] - t_target)))
            for ln, lp in probe['layer_phases'].items():
                layer_phase_at_target[ph_deg][ln] = float(lp['phase'][idx])

        np.savez_compressed(
            probe_path,
            seed=seed,
            ref_t_ms=probe['ref_t_ms'].astype(np.float32),
            ref_signal=probe['ref_signal'].astype(np.float32),
            ref_phase=probe['ref_phase'].astype(np.float32),
            ref_filt=probe['ref_filt'].astype(np.float32),
            ref_channel_label=probe['ref_channel_label'],
            ref_channel_depth=probe['ref_channel_depth'],
            dt_ms=probe['dt_ms'],
            target_phases_deg=np.array(target_phases_deg),
            phase_to_onset_ms=np.array(
                [phase_to_t[r] if phase_to_t[r] is not None else np.nan
                 for r in target_phases_rad]),
            layer_phase_at_target=np.array(layer_phase_at_target, dtype=object),
            reference_layer=REFERENCE_LAYER,
            alpha_band_hz=np.array(ALPHA_BAND),
        )
        if verbose:
            print(f"  probe saved to {probe_path}")
            for ph_deg, tgt_rad in zip(target_phases_deg, target_phases_rad):
                t_on = phase_to_t[tgt_rad]
                print(f"    phase {ph_deg:>4} deg -> "
                      f"{'SKIP' if t_on is None else f'{t_on:.2f} ms'}")

        # Run one stim trial per target phase
        for trial_idx, (ph_deg, tgt_rad) in enumerate(
                zip(target_phases_deg, target_phases_rad)):
            t_on = phase_to_t[tgt_rad]
            if t_on is None:
                if verbose:
                    print(f"  skipping phase {ph_deg} deg (no crossing found)")
                continue

            data = run_phase_locked_trial(
                config=config,
                trial_id=trial_idx,
                network_seed=seed,
                stim_onset_ms=t_on,
                target_phase_deg=ph_deg,
                post_stim_ms=post_stim_ms,
                verbose=verbose,
            )
            fname = os.path.join(
                save_dir,
                f"trial_seed{seed}_phase{int(ph_deg):03d}.npz",
            )
            np.savez_compressed(fname, **data)
            if verbose:
                print(f"    saved {fname}")


if __name__ == "__main__":
    SEEDS = [58910 + 100 * i for i in range(15)]   # 15 seeds
    PHASES = (0, 45, 90, 135, 180, 225, 270, 315)  # 8 phases

    run_phase_experiment(
        CONFIG,
        seeds=SEEDS,
        target_phases_deg=PHASES,
        t_min_ms=2000,
        probe_ms=2600,
        post_stim_ms=2000,
        save_dir="results/trials_phase_07_05",
        verbose=True,
    )
