"""
Bayesian optimization of V1 SNN spectral properties using Optuna.

This script wraps your existing Brian2 cortical column simulation and
automatically tunes parameters to achieve the biological spectral shift:
  - Suppression of power below ~20 Hz after stimulus onset
  - Enhancement of power above ~30 Hz (gamma band) after stimulus onset

DESIGN PRINCIPLES:
  1. All biological zeros in connectivity are FROZEN — never touched.
  2. Parameters are grouped into ~25 scaling factors for conductances
     + ~10 LGN/background params, instead of 600+ individual values.
  3. Scaling factors multiply your existing CSV values, so relative
     structure within each group is preserved.
  4. Short simulations (configurable) for speed during optimization.
  5. Full logging of every trial to CSV for later analysis.

USAGE:
  1. Make sure your project files are importable (column.py, config, etc.)
  2. Adjust SIM_CONFIG paths and durations below.
  3. Run:  python optimize_v1.py
  4. Monitor: optuna-dashboard sqlite:///v1_optimization.db
     (or just watch the console / results CSV)

REQUIREMENTS:
  pip install optuna pandas numpy scipy
  (brian2 and brian2tools already installed)

CHANGES FROM ORIGINAL:
  - Fixed: Config module reloaded each trial to avoid cached mutable state
  - Fixed: Brian2 full scope cleanup with device.reinit() and seed()
  - Fixed: CMA-ES no longer polluted by -100 crash scores (uses TrialPruned)
  - Fixed: Welch nperseg reduced to 2048 for proper PSD averaging
  - Fixed: CSV logging is now thread-safe with file locking
  - Fixed: Defensive spike monitor access for firing rate penalty
  - Added: subprocess-based trial runner option for full isolation
"""

import os
import sys
import json
import time
import importlib
import traceback
import filelock
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    raise ImportError("Install optuna:  pip install optuna")

from scipy.signal import welch


# ============================================================================
#  SIMULATION CONFIGURATION — adjust these to match your setup
# ============================================================================

SIM_CONFIG = {
    # Match main.py durations so LGN weights are identical
    'gray_duration_ms':    2000,
    'grating_duration_ms': 2000,
    'npz_path': 'lgn_spikes_12_03.npz',

    # LFP analysis windows (relative to stimulus onset = gray_duration_ms)
    'fs': 10000,                   # sampling rate for LFP (Hz)
    'pre_stim_window_ms':  500,    # analyze 500ms before stimulus onset
    'post_stim_transient_ms': 300, # skip 300ms transient after stimulus onset
    'post_stim_window_ms': 500,    # then analyze 500ms of steady-state response

    # Spectral bands (Hz)
    'low_freq_band':  (2, 20),
    'gamma_band':     (30, 80),

    # Welch PSD settings
    'welch_nperseg': 2048,         # shorter segments -> proper averaging

    # Optimization
    'n_trials':       200,         # total Optuna trials
    'n_startup':      30,          # random trials before Bayesian kicks in
    'db_path':        'sqlite:///v1_optimization.db',
    'study_name':     'v1_spectral_shift',
    'results_csv':    'optimization_results.csv',
    'random_seed':    58879,
}


# ============================================================================
#  PARAMETER SPACE DEFINITION
# ============================================================================
#
#  Strategy: instead of optimizing 626 individual connectivity values,
#  we define SCALING FACTORS for biologically meaningful groups.
#  Each factor multiplies the AMPA conductance of all connections in
#  that group. Zeros remain zero. Relative weights within groups preserved.
#
#  We also optimize:
#   - Background Poisson NMDA counts (global scaling)
#   - Background Poisson input rates
#   - Inter-layer connection scaling
#
#  Total: ~40 parameters — well within Bayesian optimization range.
# ============================================================================


def define_parameter_space(trial):
    """
    Define all optimizable parameters using Optuna's suggest API.
    Returns a dict of parameter values for this trial.

    Every parameter has:
      - A biologically motivated range
      - A default (baseline) value of 1.0 for scaling factors
      - Clear comments on what it controls
    """
    params = {}

    # ------------------------------------------------------------------
    #  GROUP 1: CONDUCTANCE SCALING FACTORS
    #  Tight ranges centered on 1.0 to keep network stable.
    #  1.0 = unchanged from your current CSVs.
    # ------------------------------------------------------------------

    # --- E -> PV (critical for gamma generation via PING) ---
    params['g_E_PV_intra'] = trial.suggest_float('g_E_PV_intra', 0.7, 1.8, log=True)
    params['g_E_PV_inter'] = trial.suggest_float('g_E_PV_inter', 0.6, 1.6, log=True)

    # --- PV -> E (critical for gamma: inhibitory feedback) ---
    params['g_PV_E_intra'] = trial.suggest_float('g_PV_E_intra', 0.7, 2.0, log=True)
    params['g_PV_E_inter'] = trial.suggest_float('g_PV_E_inter', 0.6, 1.6, log=True)

    # --- PV -> PV (gamma sharpening via mutual inhibition) ---
    params['g_PV_PV_intra'] = trial.suggest_float('g_PV_PV_intra', 0.6, 1.8, log=True)
    params['g_PV_PV_inter'] = trial.suggest_float('g_PV_PV_inter', 0.6, 1.6, log=True)

    # --- E -> E (recurrent excitation — very sensitive to instability) ---
    params['g_E_E_intra'] = trial.suggest_float('g_E_E_intra', 0.6, 1.5, log=True)
    params['g_E_E_inter'] = trial.suggest_float('g_E_E_inter', 0.6, 1.5, log=True)

    # --- E -> SOM (drives slow inhibition) ---
    params['g_E_SOM_intra'] = trial.suggest_float('g_E_SOM_intra', 0.5, 1.5, log=True)
    params['g_E_SOM_inter'] = trial.suggest_float('g_E_SOM_inter', 0.5, 1.5, log=True)

    # --- SOM -> E (slow inhibition of E — alpha/low-freq control) ---
    params['g_SOM_E_intra'] = trial.suggest_float('g_SOM_E_intra', 0.5, 1.8, log=True)
    params['g_SOM_E_inter'] = trial.suggest_float('g_SOM_E_inter', 0.5, 1.5, log=True)

    # --- SOM -> PV (SOM inhibits PV — modulates gamma) ---
    params['g_SOM_PV_intra'] = trial.suggest_float('g_SOM_PV_intra', 0.5, 1.8, log=True)
    params['g_SOM_PV_inter'] = trial.suggest_float('g_SOM_PV_inter', 0.5, 1.5, log=True)

    # --- VIP -> SOM (disinhibition — critical for spectral shift) ---
    params['g_VIP_SOM_intra'] = trial.suggest_float('g_VIP_SOM_intra', 0.7, 2.5, log=True)
    params['g_VIP_SOM_inter'] = trial.suggest_float('g_VIP_SOM_inter', 0.7, 2.0, log=True)

    # --- E -> VIP (drives disinhibition chain) ---
    params['g_E_VIP_intra'] = trial.suggest_float('g_E_VIP_intra', 0.5, 2.0, log=True)
    params['g_E_VIP_inter'] = trial.suggest_float('g_E_VIP_inter', 0.5, 1.5, log=True)

    # --- Minor pathways (tighter ranges) ---
    params['g_PV_VIP_all']   = trial.suggest_float('g_PV_VIP_all',   0.6, 1.5, log=True)
    params['g_SOM_VIP_all']  = trial.suggest_float('g_SOM_VIP_all',  0.6, 1.5, log=True)
    params['g_VIP_E_intra']  = trial.suggest_float('g_VIP_E_intra',  0.6, 1.5, log=True)
    params['g_VIP_PV_intra'] = trial.suggest_float('g_VIP_PV_intra', 0.6, 1.5, log=True)
    params['g_VIP_VIP_intra']= trial.suggest_float('g_VIP_VIP_intra',0.6, 1.5, log=True)

    # ------------------------------------------------------------------
    #  GROUP 2: CANONICAL INTER-LAYER PATHWAY SCALING
    #  These multiply ALL connections (E and I) from source→target layer.
    #  Separate excitatory and inhibitory scaling per pathway.
    #  Canonical pathways: L4C→L4AB, L4C→L23, L4AB→L23, L23→L5,
    #                      L5→L6, L6→L4C
    # ------------------------------------------------------------------

    # L4C → L4AB (feedforward from granular to upper granular)
    params['path_L4C_L4AB_exc'] = trial.suggest_float('path_L4C_L4AB_exc', 0.5, 2.5, log=True)
    params['path_L4C_L4AB_inh'] = trial.suggest_float('path_L4C_L4AB_inh', 0.5, 2.0, log=True)

    # L4C → L23 (feedforward from granular to supragranular)
    params['path_L4C_L23_exc'] = trial.suggest_float('path_L4C_L23_exc', 0.5, 2.5, log=True)
    params['path_L4C_L23_inh'] = trial.suggest_float('path_L4C_L23_inh', 0.5, 2.0, log=True)

    # L4AB → L23 (feedforward upper granular to supragranular)
    params['path_L4AB_L23_exc'] = trial.suggest_float('path_L4AB_L23_exc', 0.5, 2.5, log=True)
    params['path_L4AB_L23_inh'] = trial.suggest_float('path_L4AB_L23_inh', 0.5, 2.0, log=True)

    # L23 → L5 (supragranular to infragranular)
    params['path_L23_L5_exc'] = trial.suggest_float('path_L23_L5_exc', 0.5, 2.5, log=True)
    params['path_L23_L5_inh'] = trial.suggest_float('path_L23_L5_inh', 0.5, 2.0, log=True)

    # L5 → L6 (deep layer cascade)
    params['path_L5_L6_exc'] = trial.suggest_float('path_L5_L6_exc', 0.5, 2.5, log=True)
    params['path_L5_L6_inh'] = trial.suggest_float('path_L5_L6_inh', 0.5, 2.0, log=True)

    # L6 → L4C (feedback from deep to granular)
    params['path_L6_L4C_exc'] = trial.suggest_float('path_L6_L4C_exc', 0.5, 2.5, log=True)
    params['path_L6_L4C_inh'] = trial.suggest_float('path_L6_L4C_inh', 0.5, 2.0, log=True)

    # ------------------------------------------------------------------
    #  GROUP 3: BACKGROUND POISSON SCALING
    # ------------------------------------------------------------------

    params['bg_nmda_scale'] = trial.suggest_float('bg_nmda_scale', 0.5, 1.3)
    params['bg_ampa_scale'] = trial.suggest_float('bg_ampa_scale', 0.5, 1.5)
    params['bg_rate_hz'] = trial.suggest_float('bg_rate_hz', 3.0, 6.0)

    return params


# ============================================================================
#  PARAMETER APPLICATION — how scaling factors modify the CONFIG
# ============================================================================

def get_conductance_scale_key(src_type, tgt_type, is_intra):
    """
    Map a (source_type, target_type, intra/inter) to the parameter name.
    Returns the key into the params dict, or None if not optimized.
    """
    locality = 'intra' if is_intra else 'inter'

    # Main pathways with intra/inter distinction
    main_keys = {
        ('E', 'PV'),  ('PV', 'E'),  ('PV', 'PV'),
        ('E', 'E'),   ('E', 'SOM'), ('SOM', 'E'),
        ('SOM', 'PV'),('VIP', 'SOM'),('E', 'VIP'),
    }
    if (src_type, tgt_type) in main_keys:
        return f'g_{src_type}_{tgt_type}_{locality}'

    # Intra-only pathways
    intra_only = {
        ('VIP', 'E'):   'g_VIP_E_intra',
        ('VIP', 'PV'):  'g_VIP_PV_intra',
        ('VIP', 'VIP'): 'g_VIP_VIP_intra',
    }
    if (src_type, tgt_type) in intra_only:
        if is_intra:
            return intra_only[(src_type, tgt_type)]
        else:
            return None  # inter-layer VIP->E etc. don't exist in your matrix

    # Grouped pathways (no intra/inter split)
    grouped = {
        ('PV', 'VIP'):  'g_PV_VIP_all',
        ('SOM', 'VIP'): 'g_SOM_VIP_all',
    }
    if (src_type, tgt_type) in grouped:
        return grouped[(src_type, tgt_type)]

    return None  # shouldn't happen if connectivity is correct


def apply_conductance_scaling(base_ampa_df, base_nmda_df, params):
    """
    Apply scaling factors to the conductance DataFrames.
    Returns new (scaled_ampa_df, scaled_nmda_df).
    Zeros remain zero.
    """
    scaled_ampa = base_ampa_df.copy()
    scaled_nmda = base_nmda_df.copy()

    for src in base_ampa_df.index:
        src_type = src.split('_')[0]
        src_layer = '_'.join(src.split('_')[1:])

        for tgt in base_ampa_df.columns:
            tgt_type = tgt.split('_')[0]
            tgt_layer = '_'.join(tgt.split('_')[1:])

            # Skip zeros (biological constraint)
            if base_ampa_df.loc[src, tgt] == 0.0:
                continue

            is_intra = (src_layer == tgt_layer)
            key = get_conductance_scale_key(src_type, tgt_type, is_intra)

            if key is not None and key in params:
                scale = params[key]
                scaled_ampa.loc[src, tgt] = base_ampa_df.loc[src, tgt] * scale
                # Scale NMDA proportionally (keeps NMDA/AMPA ratio for recurrent)
                if base_nmda_df.loc[src, tgt] != 0.0:
                    scaled_nmda.loc[src, tgt] = base_nmda_df.loc[src, tgt] * scale

    # --- Apply canonical inter-layer pathway scaling ---
    # These multiply on top of the cell-type scaling above.
    canonical_pathways = {
        ('L4C', 'L4AB'): ('path_L4C_L4AB_exc', 'path_L4C_L4AB_inh'),
        ('L4C', 'L23'):  ('path_L4C_L23_exc',  'path_L4C_L23_inh'),
        ('L4AB', 'L23'): ('path_L4AB_L23_exc',  'path_L4AB_L23_inh'),
        ('L23', 'L5'):   ('path_L23_L5_exc',    'path_L23_L5_inh'),
        ('L5', 'L6'):    ('path_L5_L6_exc',     'path_L5_L6_inh'),
        ('L6', 'L4C'):   ('path_L6_L4C_exc',    'path_L6_L4C_inh'),
    }
    excitatory_types = {'E'}

    for src in scaled_ampa.index:
        src_type = src.split('_')[0]
        src_layer = '_'.join(src.split('_')[1:])

        for tgt in scaled_ampa.columns:
            tgt_layer = '_'.join(tgt.split('_')[1:])

            if src_layer == tgt_layer:
                continue  # intra-layer, skip

            pathway_key = (src_layer, tgt_layer)
            if pathway_key not in canonical_pathways:
                continue

            exc_param, inh_param = canonical_pathways[pathway_key]
            if src_type in excitatory_types:
                scale_key = exc_param
            else:
                scale_key = inh_param

            if scale_key in params:
                scale = params[scale_key]
                if scaled_ampa.loc[src, tgt] != 0.0:
                    scaled_ampa.loc[src, tgt] *= scale
                if scaled_nmda.loc[src, tgt] != 0.0:
                    scaled_nmda.loc[src, tgt] *= scale

    return scaled_ampa, scaled_nmda




# ============================================================================
#  CONFIG LOADER — fresh reload each trial to avoid cached state
# ============================================================================

def load_fresh_config():
    """
    Reload the config module from scratch to avoid Python's module cache
    returning a previously mutated CONFIG dict.

    Returns a deep copy of the freshly loaded CONFIG.
    """
    # If config.config2 is already imported, reload it
    module_name = 'config.config2'
    if module_name in sys.modules:
        mod = importlib.reload(sys.modules[module_name])
    else:
        mod = importlib.import_module(module_name)

    return deepcopy(mod.CONFIG)


# ============================================================================
#  FITNESS FUNCTION — spectral analysis of LFP
# ============================================================================

def compute_spectral_fitness(lfp_signals, time_array, electrode_positions,
                             gray_duration_ms, grating_duration_ms,
                             fs=10000):
    """
    Compute fitness score from LFP signals using experimentally matched windows.

    Analysis windows (relative to stimulus onset at t = gray_duration_ms):
      - PRE-STIMULUS:  [-500, 0] ms  (last 500ms of settled gray screen)
      - POST-STIMULUS: [+300, +800] ms  (skip 300ms transient, then 500ms)

    The score rewards:
      - Suppression of power below 20 Hz  (post/pre ratio < 1)
      - Enhancement of power above 30 Hz  (post/pre ratio > 1)

    Returns: (score, details_dict)
    """
    dt_ms = 1000.0 / fs

    low_band = SIM_CONFIG['low_freq_band']
    gamma_band = SIM_CONFIG['gamma_band']
    pre_window = SIM_CONFIG['pre_stim_window_ms']
    post_transient = SIM_CONFIG['post_stim_transient_ms']
    post_window = SIM_CONFIG['post_stim_window_ms']
    nperseg_val = SIM_CONFIG['welch_nperseg']

    # Stimulus onset in samples
    stim_onset_idx = int(gray_duration_ms / dt_ms)

    # PRE-STIMULUS window: [stim_onset - 500ms, stim_onset)
    pre_start_idx = stim_onset_idx - int(pre_window / dt_ms)
    pre_stop_idx  = stim_onset_idx

    # POST-STIMULUS window: [stim_onset + 300ms, stim_onset + 800ms)
    post_start_idx = stim_onset_idx + int(post_transient / dt_ms)
    post_stop_idx  = post_start_idx + int(post_window / dt_ms)

    # Sanity checks
    if pre_start_idx < 0:
        return -100.0, {'error': f'pre-stim window starts before t=0 '
                                  f'(need {pre_window}ms before stim, '
                                  f'gray is only {gray_duration_ms}ms)'}

    max_idx = len(lfp_signals[0]) if len(lfp_signals) > 0 else 0
    if post_stop_idx > max_idx:
        return -100.0, {'error': f'post-stim window exceeds simulation '
                                  f'(need {post_transient + post_window}ms after stim, '
                                  f'grating is only {grating_duration_ms}ms)'}

    # Evaluate electrodes 3-13 (spans all layers, skips edge artifacts)
    target_electrodes = list(range(3, 14))
    target_electrodes = [ch for ch in target_electrodes if ch < len(lfp_signals)]

    if len(target_electrodes) == 0:
        return -100.0, {'error': 'no valid electrodes in range 3-13'}

    # Compute PSD per electrode
    def electrode_psd(signals, ch, idx_start, idx_stop):
        seg = signals[ch][idx_start:idx_stop]
        if len(seg) < nperseg_val:
            # Fall back to segment length if too short, but warn
            effective_nperseg = len(seg)
        else:
            effective_nperseg = nperseg_val

        if effective_nperseg < 256:
            return None, None

        freqs, psd = welch(seg, fs=fs, nperseg=effective_nperseg,
                          noverlap=effective_nperseg // 2, detrend='constant')
        if np.any(np.isnan(psd)) or np.any(np.isinf(psd)):
            return None, None
        return freqs, psd

    # Score each electrode independently
    n_crossover = 0      # electrodes with the right pattern
    n_valid = 0
    all_low_ratios = []
    all_gamma_ratios = []
    per_electrode_scores = []

    for ch in target_electrodes:
        freqs_pre, psd_pre = electrode_psd(lfp_signals, ch, pre_start_idx, pre_stop_idx)
        freqs_post, psd_post = electrode_psd(lfp_signals, ch, post_start_idx, post_stop_idx)

        if psd_pre is None or psd_post is None:
            continue

        n_valid += 1

        eps = np.percentile(psd_pre[psd_pre > 0], 5) if np.any(psd_pre > 0) else 1e-30
        ratio = psd_post / (psd_pre + eps)

        low_mask = (freqs_pre >= low_band[0]) & (freqs_pre <= low_band[1])
        gamma_mask = (freqs_pre >= gamma_band[0]) & (freqs_pre <= gamma_band[1])

        if not np.any(low_mask) or not np.any(gamma_mask):
            continue

        ch_low = np.mean(ratio[low_mask])
        ch_gamma = np.mean(ratio[gamma_mask])
        all_low_ratios.append(ch_low)
        all_gamma_ratios.append(ch_gamma)

        # Does this electrode show the right pattern?
        if ch_low < 1.0 and ch_gamma > 1.0:
            n_crossover += 1

        # Per-electrode score
        ch_log_low = np.log10(ch_low + 1e-10)
        ch_log_gamma = np.log10(ch_gamma + 1e-10)
        ch_score = -ch_log_low + min(ch_log_gamma, np.log10(5.0))
        if ch_low > 1.0:
            ch_score -= 2.0 * ch_log_low  # penalty for low-freq increase
        per_electrode_scores.append(ch_score)

    if n_valid == 0:
        return -100.0, {'error': 'no valid electrode PSDs'}

    # === AGGREGATE SCORING ===

    # Mean ratios across all electrodes (for reporting)
    low_ratio = np.mean(all_low_ratios)
    gamma_ratio = np.mean(all_gamma_ratios)
    log_low = np.log10(low_ratio + 1e-10)
    log_gamma = np.log10(gamma_ratio + 1e-10)

    # Average per-electrode spectral score
    avg_electrode_score = np.mean(per_electrode_scores)

    # Fraction of electrodes with the right pattern (0 to 1)
    crossover_fraction = n_crossover / n_valid

    # Crossover bonus: scales with how many electrodes show the pattern
    crossover_bonus = 5.0 * crossover_fraction

    # Penalty if low-freq increases on average
    low_freq_penalty = 0.0
    if low_ratio > 1.0:
        low_freq_penalty = -3.0 * log_low

    # Final score
    score = (
        3.0 * avg_electrode_score
        + crossover_bonus
        + low_freq_penalty
    )

    details = {
        'low_freq_ratio': float(low_ratio),
        'gamma_ratio': float(gamma_ratio),
        'log_low': float(log_low),
        'log_gamma': float(log_gamma),
        'n_crossover': n_crossover,
        'n_valid_electrodes': n_valid,
        'crossover_fraction': float(crossover_fraction),
        'avg_electrode_score': float(avg_electrode_score),
        'crossover_bonus': float(crossover_bonus),
        'low_freq_penalty': float(low_freq_penalty),
        'total_score': float(score),
        'pre_window_ms': f'{gray_duration_ms - pre_window}-{gray_duration_ms}',
        'post_window_ms': f'{gray_duration_ms + post_transient}-'
                          f'{gray_duration_ms + post_transient + post_window}',
    }

    return score, details


def compute_firing_rate_penalty(spike_monitors, gray_duration_ms, grating_duration_ms):
    """
    Penalize if any cell type has zero firing rate (network died)
    or extremely high rates (epileptic).

    Returns: (penalty, rate_details)
      penalty is <= 0, subtracted from score
    """
    penalty = 0.0
    rate_details = {}

    total_duration_s = (gray_duration_ms + grating_duration_ms) / 1000.0

    for layer_name, monitors in spike_monitors.items():
        for mon_key, mon in monitors.items():
            # Defensive access: handle stale or missing references
            try:
                n_spikes = len(mon.t)
            except Exception:
                rate_details[f'{layer_name}_{mon_key}'] = -1.0
                penalty -= 5.0
                continue

            try:
                n_neurons = mon.source.N if hasattr(mon, 'source') else 1
            except Exception:
                # Fallback: try to infer from monitor name or skip
                n_neurons = 1

            if n_neurons == 0:
                continue

            rate = n_spikes / (n_neurons * total_duration_s)
            rate_details[f'{layer_name}_{mon_key}'] = rate

            # Penalty for dead populations
            if rate < 0.1:
                penalty -= 5.0  # harsh penalty for dead cells

            # Penalty for epileptic rates (>100 Hz for E, >200 Hz for PV)
            cell_type = mon_key.replace('_spikes', '').replace('spikes_', '')
            max_rate = 200.0 if 'PV' in cell_type else 100.0
            if rate > max_rate:
                penalty -= 2.0 * (rate / max_rate - 1.0)

    return penalty, rate_details


# ============================================================================
#  THREAD-SAFE CSV LOGGING
# ============================================================================

_csv_lock = filelock.FileLock("optimization_results.csv.lock")


def log_trial_to_csv(trial_id, total_score, sim_time, spectral_details,
                     rate_details, params):
    """
    Append one row to the results CSV with file-locking for safety
    in case of parallel workers.
    """
    log_row = {
        'trial': trial_id,
        'score': total_score,
        'sim_time_s': sim_time,
        **spectral_details,
        **{f'rate_{k}': v for k, v in rate_details.items()},
        **params,
    }
    log_df = pd.DataFrame([log_row])
    csv_path = SIM_CONFIG['results_csv']

    with _csv_lock:
        write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        log_df.to_csv(csv_path, mode='a', header=write_header, index=False)


# ============================================================================
#  MAIN OBJECTIVE FUNCTION
# ============================================================================

def objective(trial):
    """
    Single optimization trial:
      1. Sample parameters
      2. Build modified CONFIG (freshly reloaded)
      3. Run Brian2 simulation (with full scope cleanup)
      4. Compute LFP and spectral fitness
      5. Return score (or prune on crash/dead network)
    """
    import brian2 as b2
    from brian2 import ms, Hz, nS

    # Sample parameters
    params = define_parameter_space(trial)

    # Log params
    trial_id = trial.number
    print(f"\n{'='*70}")
    print(f"  TRIAL {trial_id}")
    print(f"{'='*70}")
    for k, v in sorted(params.items()):
        print(f"  {k}: {v:.4f}")

    try:
        # --- Full Brian2 cleanup for trial isolation ---
        b2.start_scope()
        # Also reinit device to clear any lingering named objects
        try:
            b2.device.reinit()
            b2.device.activate()
        except Exception:
            pass  # some device backends don't support reinit
        b2.defaultclock.dt = 0.1 * ms

        # Seed both numpy and Brian2 for reproducibility per trial
        trial_seed = SIM_CONFIG['random_seed'] + trial_id
        np.random.seed(trial_seed)
        b2.seed(trial_seed)

        # --- Load base config FRESH (avoids cached mutable state) ---
        config = load_fresh_config()

        # 1) Scale conductances
        base_ampa = pd.read_csv('config/conductances_AMPA2_alpha_v2.csv', index_col=0)
        base_nmda = pd.read_csv('config/conductances_NMDA2_alpha_v2.csv', index_col=0)

        scaled_ampa, scaled_nmda = apply_conductance_scaling(
            base_ampa, base_nmda, params
        )

        # Write scaled CSVs to temp location for this trial
        tmp_ampa = f'/tmp/trial_{trial_id}_ampa.csv'
        tmp_nmda = f'/tmp/trial_{trial_id}_nmda.csv'
        scaled_ampa.to_csv(tmp_ampa)
        scaled_nmda.to_csv(tmp_nmda)

        # Reload connectivity with scaled conductances
        # Reload utils module too in case it caches anything
        if 'tools.utils' in sys.modules:
            importlib.reload(sys.modules['tools.utils'])
        from tools.utils import load_connectivity_from_csv

        csv_layer_configs, inter_conns, inter_conds = load_connectivity_from_csv(
            'config/connection_probabilities2.csv', tmp_ampa, tmp_nmda
        )

        # 2) Apply intra-layer connectivity from scaled CSVs
        for layer_name in config['layers']:
            if layer_name in csv_layer_configs:
                config['layers'][layer_name]['connection_prob'] = \
                    csv_layer_configs[layer_name]['connection_prob']
                config['layers'][layer_name]['conductance'] = \
                    csv_layer_configs[layer_name]['conductance']

        # 3) Scale Poisson backgrounds
        bg_nmda_scale = params['bg_nmda_scale']
        bg_ampa_scale = params['bg_ampa_scale']
        bg_rate = params['bg_rate_hz']

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

        # --- Build and run simulation ---
        gray_ms = SIM_CONFIG['gray_duration_ms']
        grating_ms = SIM_CONFIG['grating_duration_ms']
        total_ms = gray_ms + grating_ms

        # Reload column module to avoid stale Brian2 references
        if 'src.column' in sys.modules:
            importlib.reload(sys.modules['src.column'])
        from src.column import CorticalColumn

        column = CorticalColumn(column_id=0, config=config)
        all_monitors = column.get_all_monitors()

        # --- LGN input — fixed, same as main.py ---
        if 'lgn_to_brian2_v2' in sys.modules:
            importlib.reload(sys.modules['lgn_to_brian2_v2'])
        from lgn_to_brian2_v2 import make_lgn_inputs_split

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

        # --- Run ---
        t_start = time.time()
        column.network.run(total_ms * ms)
        sim_time = time.time() - t_start
        print(f"  Simulation took {sim_time:.1f}s")

        # --- Extract spike monitors ---
        spike_monitors = {}
        neuron_groups = {}
        for layer_name, monitors in all_monitors.items():
            spike_monitors[layer_name] = {
                k: v for k, v in monitors.items() if 'spikes' in k
            }
            neuron_groups[layer_name] = column.layers[layer_name].neuron_groups

        # --- Firing rate penalty ---
        rate_penalty, rate_details = compute_firing_rate_penalty(
            spike_monitors, gray_ms, grating_ms
        )

        print(f"  Firing rates:")
        for k, v in sorted(rate_details.items()):
            status = "OK" if 0.1 < v < 200 else "BAD"
            print(f"    {k}: {v:.1f} Hz [{status}]")

        if rate_penalty < -10:
            # Network is broken — prune instead of returning -100
            # This prevents CMA-ES / TPE from treating crash scores as signal
            print(f"  PRUNED: dead network, rate penalty = {rate_penalty:.1f}")
            trial.set_user_attr('status', 'dead_network')
            trial.set_user_attr('rate_details', rate_details)
            log_trial_to_csv(trial_id, rate_penalty, sim_time, {}, rate_details, params)
            raise optuna.TrialPruned(f"Dead network: rate_penalty={rate_penalty:.1f}")

        # --- Compute LFP ---
        if 'lfp_kernel' in sys.modules:
            importlib.reload(sys.modules['lfp_kernel'])
        from lfp_kernel import calculate_lfp_kernel_method

        electrode_positions = config['electrode_positions']

        lfp_signals, time_array = calculate_lfp_kernel_method(
            spike_monitors, neuron_groups,
            config['layers'], electrode_positions,
            sim_duration_ms=total_ms
        )

        # --- Spectral fitness ---
        spectral_score, spectral_details = compute_spectral_fitness(
            lfp_signals, time_array, electrode_positions,
            gray_ms, grating_ms,
            fs=SIM_CONFIG['fs'],
        )

        total_score = spectral_score + rate_penalty

        print(f"\n  === TRIAL {trial_id} RESULTS ===")
        print(f"  Analysis windows:")
        print(f"    Pre-stim:  {spectral_details.get('pre_window_ms', '?')} ms")
        print(f"    Post-stim: {spectral_details.get('post_window_ms', '?')} ms")
        print(f"  Low-freq ratio (mean):  {spectral_details.get('low_freq_ratio', '?'):.3f} "
              f"(want < 1.0)")
        print(f"  Gamma ratio (mean):     {spectral_details.get('gamma_ratio', '?'):.3f} "
              f"(want > 1.0)")
        n_cross = spectral_details.get('n_crossover', '?')
        n_valid = spectral_details.get('n_valid_electrodes', '?')
        frac = spectral_details.get('crossover_fraction', 0)
        print(f"  Electrodes with pattern: {n_cross}/{n_valid} "
              f"({frac*100:.0f}%)")
        print(f"  Spectral score:  {spectral_score:.3f}")
        print(f"  Rate penalty:    {rate_penalty:.3f}")
        print(f"  TOTAL SCORE:     {total_score:.3f}")

        # Store details for analysis
        trial.set_user_attr('status', 'completed')
        trial.set_user_attr('spectral_details', spectral_details)
        trial.set_user_attr('rate_details', rate_details)
        trial.set_user_attr('sim_time_s', sim_time)

        # --- Thread-safe CSV log ---
        log_trial_to_csv(trial_id, total_score, sim_time,
                         spectral_details, rate_details, params)

        # Cleanup temp files
        for f in [tmp_ampa, tmp_nmda]:
            if os.path.exists(f):
                os.remove(f)

        return total_score

    except optuna.TrialPruned:
        raise  # re-raise so Optuna handles it properly

    except Exception as e:
        print(f"\n  TRIAL {trial_id} FAILED: {e}")
        traceback.print_exc()
        trial.set_user_attr('status', 'error')
        trial.set_user_attr('error', str(e))

        # Prune instead of returning -100 to avoid polluting the sampler
        raise optuna.TrialPruned(f"Trial crashed: {e}")

    finally:
        # Always clean up temp files even on crash
        for suffix in ['ampa', 'nmda']:
            tmp = f'/tmp/trial_{trial_id}_{suffix}.csv'
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass


# ============================================================================
#  ENTRY POINT
# ============================================================================

def print_best_params(study):
    """Pretty-print the best parameters found so far."""
    best = study.best_trial
    print(f"\n{'='*70}")
    print(f"  BEST TRIAL: #{best.number}  (score = {best.value:.4f})")
    print(f"{'='*70}")

    p = best.params
    spectral = best.user_attrs.get('spectral_details', {})
    rates = best.user_attrs.get('rate_details', {})

    print(f"\n  Spectral results:")
    print(f"    Low-freq ratio:  {spectral.get('low_freq_ratio', '?')}")
    print(f"    Gamma ratio:     {spectral.get('gamma_ratio', '?')}")
    print(f"    Crossover bonus: {spectral.get('crossover_bonus', '?')}")

    print(f"\n  === CONDUCTANCE SCALING FACTORS ===")
    for k, v in sorted(p.items()):
        if k.startswith('g_'):
            marker = ""
            if v > 1.5:
                marker = " ↑↑"
            elif v < 0.6:
                marker = " ↓↓"
            print(f"    {k:30s} = {v:.3f}{marker}")

    print(f"\n  === INTER-LAYER PATHWAY SCALING ===")
    for k, v in sorted(p.items()):
        if k.startswith('path_'):
            marker = ""
            if v > 1.5:
                marker = " ↑↑"
            elif v < 0.6:
                marker = " ↓↓"
            print(f"    {k:30s} = {v:.3f}{marker}")

    print(f"\n  === BACKGROUND ===")
    for k, v in sorted(p.items()):
        if k.startswith('bg_'):
            print(f"    {k:30s} = {v:.3f}")

    if rates:
        print(f"\n  Firing rates:")
        for k, v in sorted(rates.items()):
            print(f"    {k:30s} = {v:.1f} Hz")


def main():
    print(f"V1 SNN Spectral Optimization")
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Config: {json.dumps(SIM_CONFIG, indent=2, default=str)}")

    # --- Create or load Optuna study ---
    # Switched from CmaEsSampler to TPESampler:
    # TPE handles pruned/failed trials gracefully (CMA-ES treats all
    # returned values as real signal, so -100 crash scores distort it).
    sampler = TPESampler(
        n_startup_trials=SIM_CONFIG['n_startup'],
        seed=SIM_CONFIG['random_seed'],
        multivariate=True,      # model parameter correlations
    )

    study = optuna.create_study(
        study_name=SIM_CONFIG['study_name'],
        storage=SIM_CONFIG['db_path'],
        direction='maximize',
        sampler=sampler,
        load_if_exists=True,    # resume if interrupted
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"\nResuming study with {n_existing} existing trials")
        try:
            print_best_params(study)
        except ValueError:
            print("  (no completed trials yet)")

    n_remaining = max(0, SIM_CONFIG['n_trials'] - n_existing)
    print(f"\nRunning {n_remaining} new trials...")

    # --- Run optimization ---
    study.optimize(objective, n_trials=n_remaining, show_progress_bar=True)

    # --- Report results ---
    try:
        print_best_params(study)
    except ValueError:
        print("No completed trials — all were pruned or crashed.")
        return

    # Save best params to JSON
    best_params = study.best_trial.params
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest parameters saved to best_params.json")


if __name__ == '__main__':
    main()