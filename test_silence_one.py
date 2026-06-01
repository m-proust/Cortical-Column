"""
Quick smoke test for silence_sweep.py: run one silenced trial of L4C PV
with the new gL+EL silencing, then report whether the population
actually went silent in the post-onset window.
"""

import os
import numpy as np
import silence_sweep as ss
from config.config import CONFIG

OUT_DIR = "results/silence_sweep_test"
os.makedirs(OUT_DIR, exist_ok=True)

condition = {
    "name": "L4C_PV_silence",
    "target": ("L4C", "PV"),
}

data = ss.run_single_trial(
    config=CONFIG,
    condition=condition,
    silenced=True,
    trial_id=0,
    network_seed=58910,
    baseline_ms=2000,
    stimuli_ms=2000,
    fs=10000,
    verbose=True,
)

fname = os.path.join(OUT_DIR, "L4C_PV_silenced_test.npz")
np.savez_compressed(fname, **data)
print(f"\nSaved {fname}")

# ----- diagnostics -----
sd = data['spike_data']
stim = float(data['stim_onset_ms'])
sp = sd['L4C']['PV_spikes']
t = np.asarray(sp['times_ms'])
i = np.asarray(sp['spike_indices'])
n_neurons = int(i.max()) + 1 if len(i) else 0

pre_lo, pre_hi = stim - 500, stim
post_lo, post_hi = stim + 200, stim + 700
n_pre = int(((t >= pre_lo) & (t < pre_hi)).sum())
n_post = int(((t >= post_lo) & (t < post_hi)).sum())

print("\n========= L4C PV silenced trial =========")
print(f"n_neurons in pop:        {n_neurons}")
print(f"silence_DeltaT_mV:       {float(data['silence_DeltaT_mV']):.2e}")
print(f"DeltaT_original_mV:      {float(data['DeltaT_original_mV']):.4f}")
print(f"silence_Vcut_mV:         {float(data['silence_Vcut_mV']):.2e}")
print(f"Vcut_original_mV:        {float(data['Vcut_original_mV']):.2f}")
print(f"silencing onset:         {stim:.1f} ms")
print(f"pre window  [{pre_lo:.0f}, {pre_hi:.0f}] ms:  "
      f"{n_pre:5d} spikes  "
      f"({n_pre / max(n_neurons,1) / 0.5:.2f} Hz/neuron)")
print(f"post window [{post_lo:.0f}, {post_hi:.0f}] ms: "
      f"{n_post:5d} spikes  "
      f"({n_post / max(n_neurons,1) / 0.5:.2f} Hz/neuron)")
print(f"any spikes after stim+10ms: {(t >= stim + 10).sum()}")

if n_post == 0:
    print("\n>>> SILENCING WORKS: population is mute after onset.")
elif n_post < n_pre * 0.05:
    print("\n>>> SILENCING MOSTLY WORKS: post << pre.")
else:
    print("\n>>> SILENCING FAILED: post-window rate is similar to pre.")
