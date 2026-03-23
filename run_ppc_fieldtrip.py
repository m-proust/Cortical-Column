import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from brian2 import second
from ppc_fieldtrip import *
from config.config2 import CONFIG

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Paired")

# ─────────────────────────────────────────────────────────────────
# Fake SpikeMonitor so ppc.py can call .spike_trains()
# ─────────────────────────────────────────────────────────────────
class FakeSpikeMonitor:
    """Wraps saved spike data (times_ms, spike_indices) to mimic
    Brian2 SpikeMonitor.spike_trains() interface."""

    def __init__(self, times_ms, indices):
        self._times_ms = np.asarray(times_ms)
        self._indices = np.asarray(indices)

    def spike_trains(self):
        trains = {}
        if len(self._indices) == 0:
            return trains
        for idx in np.unique(self._indices):
            mask = self._indices == idx
            trains[idx] = self._times_ms[mask] / 1000.0 * second
        return trains


# ─────────────────────────────────────────────────────────────────
# Load trial data
# ─────────────────────────────────────────────────────────────────
fname = "results/10s_with_PV_delay_PG/trial_000.npz"
data = np.load(fname, allow_pickle=True)
spike_data = data["spike_data"].item() if data["spike_data"].size == 1 else data["spike_data"]

lfp_signals = data["lfp_matrix"]
time_array = data["time_array_ms"]
electrode_positions = list(map(tuple, data["electrode_positions"]))

# ─────────────────────────────────────────────────────────────────
# Build spike_monitors dict
# ─────────────────────────────────────────────────────────────────
spike_monitors = {}
for layer_name, layer_mons in spike_data.items():
    spike_monitors[layer_name] = {}
    for mon_name, sd in layer_mons.items():
        spike_monitors[layer_name][mon_name] = FakeSpikeMonitor(
            sd["times_ms"], sd["spike_indices"]
        )

# ─────────────────────────────────────────────────────────────────
# PPC SPECTRUM — FieldTrip mtmconvol + PPC₀
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PPC SPECTRUM (FieldTrip-style: mtmconvol + PPC₀)")
print("=" * 60)

results = compute_all_ppc(
    spike_monitors=spike_monitors,
    lfp_signals=lfp_signals,
    lfp_time_array=time_array,
    layer_configs=CONFIG['layers'],
    electrode_positions=electrode_positions,
    t_discard=2.0,
    foi=np.arange(5, 101, 1.0),  # 5–100 Hz in 1 Hz steps
    n_cycles=5,                    # 5 cycles per frequency (FieldTrip default)
    min_spikes=20,
)

plot_ppc_spectra(results, save_path='ppc_spectrum_fieldtrip.png')
