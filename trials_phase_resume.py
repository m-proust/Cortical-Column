"""
Resume runner for results/trials_phase_07_05.

Skips any (seed, phase) trial whose .npz already exists. For each seed, reuses
the saved probe_seed_*.npz to recover the phase -> onset_ms mapping instead of
re-running the probe. If no probe file exists, runs a fresh probe.

Usage:
    ~/Desktop/venv/bin/python trials_phase_resume.py
"""

import os
import numpy as np

from config.config import CONFIG
from trials_phase import (
    run_probe,
    run_phase_locked_trial,
    save_config_snapshot,
    _find_phase_crossings,
    REFERENCE_LAYER,
    ALPHA_BAND,
)


SEEDS = [58910 + 100 * i for i in range(15)]
PHASES = (0, 45, 90, 135, 180, 225, 270, 315)
SAVE_DIR = "results/trials_phase_07_05"
T_MIN_MS = 2000
PROBE_MS = 2600
POST_STIM_MS = 2000


def _trial_path(save_dir, seed, ph_deg):
    return os.path.join(save_dir, f"trial_seed{seed}_phase{int(ph_deg):03d}.npz")


def _probe_path(save_dir, seed):
    return os.path.join(save_dir, f"probe_seed_{seed}.npz")


def _load_phase_to_onset(probe_file, target_phases_deg):
    """Return dict {phase_deg: onset_ms or None} from saved probe.

    Returns None if the saved probe doesn't have phase_to_onset_ms (e.g. it
    was a partial / older probe), in which case caller should re-run probe.
    """
    data = np.load(probe_file, allow_pickle=True)
    if 'phase_to_onset_ms' not in data.files or 'target_phases_deg' not in data.files:
        return None
    saved_phases = list(np.array(data['target_phases_deg']).tolist())
    saved_onsets = list(np.array(data['phase_to_onset_ms']).tolist())
    lookup = dict(zip(saved_phases, saved_onsets))
    out = {}
    for ph in target_phases_deg:
        v = lookup.get(int(ph))
        out[int(ph)] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
    return out


def _phase_to_onset_from_probe(config, seed, target_phases_deg, t_min_ms, probe_ms,
                               save_dir):
    """Get phase->onset mapping for a seed: load if saved probe is good, else
    rerun probe and save it."""
    probe_file = _probe_path(save_dir, seed)
    if os.path.exists(probe_file):
        loaded = _load_phase_to_onset(probe_file, target_phases_deg)
        if loaded is not None and any(v is not None for v in loaded.values()):
            print(f"  [probe-cache] loaded onsets from {probe_file}")
            return loaded
        reason = "missing phase_to_onset_ms" if loaded is None else "all onsets are NaN (old probe format)"
        print(f"  [probe-cache] {probe_file} {reason} — rerunning probe")

    probe = run_probe(config, seed, probe_ms=probe_ms, verbose=True)
    target_phases_rad = np.deg2rad(np.array(target_phases_deg))
    phase_to_t = _find_phase_crossings(
        probe['ref_phase'], probe['ref_t_ms'], target_phases_rad, t_min_ms,
    )
    onsets_arr = np.array(
        [phase_to_t[r] if phase_to_t[r] is not None else np.nan
         for r in target_phases_rad]
    )
    np.savez_compressed(
        probe_file,
        seed=seed,
        ref_t_ms=probe['ref_t_ms'].astype(np.float32),
        ref_signal=probe['ref_signal'].astype(np.float32),
        ref_phase=probe['ref_phase'].astype(np.float32),
        ref_filt=probe['ref_filt'].astype(np.float32),
        ref_channel_label=probe['ref_channel_label'],
        ref_channel_depth=probe['ref_channel_depth'],
        dt_ms=probe['dt_ms'],
        target_phases_deg=np.array(target_phases_deg),
        phase_to_onset_ms=onsets_arr,
        reference_layer=REFERENCE_LAYER,
        alpha_band_hz=np.array(ALPHA_BAND),
    )
    print(f"  [probe] saved fresh probe to {probe_file}")
    return {int(ph): (None if np.isnan(v) else float(v))
            for ph, v in zip(target_phases_deg, onsets_arr)}


def resume(config, seeds, target_phases_deg, t_min_ms, probe_ms, post_stim_ms,
           save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_config_snapshot(save_dir)

    # Plan: list missing trials up front so the user sees what's left.
    missing = []
    for seed in seeds:
        for ph in target_phases_deg:
            if not os.path.exists(_trial_path(save_dir, seed, ph)):
                missing.append((seed, ph))
    print(f"\nMissing trials: {len(missing)} of {len(seeds) * len(target_phases_deg)}")
    if not missing:
        print("Nothing to do.")
        return
    for seed, ph in missing:
        print(f"  - seed {seed} phase {ph:>3}")

    for seed in seeds:
        missing_for_seed = [ph for ph in target_phases_deg
                            if not os.path.exists(_trial_path(save_dir, seed, ph))]
        if not missing_for_seed:
            continue

        print(f"\n=== seed {seed} (missing phases: {missing_for_seed}) ===")
        phase_to_onset = _phase_to_onset_from_probe(
            config, seed, target_phases_deg, t_min_ms, probe_ms, save_dir,
        )

        for trial_idx, ph_deg in enumerate(target_phases_deg):
            trial_file = _trial_path(save_dir, seed, ph_deg)
            if os.path.exists(trial_file):
                continue
            t_on = phase_to_onset.get(int(ph_deg))
            if t_on is None:
                print(f"  skipping phase {ph_deg} deg (no crossing found in probe)")
                continue
            data = run_phase_locked_trial(
                config=config,
                trial_id=trial_idx,
                network_seed=seed,
                stim_onset_ms=t_on,
                target_phase_deg=ph_deg,
                post_stim_ms=post_stim_ms,
                verbose=True,
            )
            np.savez_compressed(trial_file, **data)
            print(f"    saved {trial_file}")


if __name__ == "__main__":
    resume(
        CONFIG,
        seeds=SEEDS,
        target_phases_deg=PHASES,
        t_min_ms=T_MIN_MS,
        probe_ms=PROBE_MS,
        post_stim_ms=POST_STIM_MS,
        save_dir=SAVE_DIR,
    )
