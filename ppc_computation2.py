import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfiltfilt
from scipy.ndimage import gaussian_filter1d
from brian2 import second
from config.config2 import CONFIG


# ── Fake SpikeMonitor ──
class FakeSpikeMonitor:
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


# ── Bandpass using SOS (stable) ──
def bandpass(signal, fs, low, high, order=4):
    lo = max(low, 0.5)
    hi = min(high, fs / 2 - 1)
    sos = butter(order, [lo / (fs / 2), hi / (fs / 2)],
                 btype='band', output='sos')
    return sosfiltfilt(sos, signal)


# ── PPC for one neuron (Vinck et al. 2012) ──
def _ppc_one_neuron(spike_times_sec, phase_trimmed, time_trimmed,
                    t_discard, min_spikes=2):
    spikes = spike_times_sec[spike_times_sec >= t_discard]
    if len(spikes) < min_spikes:
        return np.nan

    spike_indices = np.searchsorted(time_trimmed, spikes)
    spike_indices = spike_indices[spike_indices < len(phase_trimmed)]

    if len(spike_indices) < min_spikes:
        return np.nan

    phases = phase_trimmed[spike_indices]
    dof = len(phases)
    if dof < 2:
        return np.nan

    cosSum = np.sum(np.cos(phases))
    sinSum = np.sum(np.sin(phases))
    ppc = (cosSum**2 + sinSum**2 - dof) / (dof * (dof - 1))
    return ppc


# ── PPC at one frequency band ──
def compute_ppc_at_band(spike_monitor, lfp, fs, dt_seconds,
                        band=(25, 45), t_discard=1.0, min_spikes=2):
    filtered = bandpass(lfp, fs, band[0], band[1])
    analytic = hilbert(filtered)
    phase = np.angle(analytic)

    n_samples = len(lfp)
    time_array = np.arange(n_samples) * dt_seconds
    n_skip = int(t_discard / dt_seconds)
    phase_trimmed = phase[n_skip:]
    time_trimmed = time_array[n_skip:]

    trains = spike_monitor.spike_trains()
    ppcs = []
    for idx in trains:
        st = np.array(trains[idx] / second)
        val = _ppc_one_neuron(st, phase_trimmed, time_trimmed,
                              t_discard, min_spikes)
        if not np.isnan(val) and np.isfinite(val):
            ppcs.append(val)

    mean_ppc = np.mean(ppcs) if ppcs else 0.0
    return mean_ppc, ppcs


# ── PPC spectrum ──
def compute_ppc_spectrum(spike_monitor, lfp, fs, dt_seconds,
                         t_discard=1.0, freq_range=(5, 100),
                         freq_step=2, bandwidth=10, min_spikes=2):
    center_freqs = np.arange(freq_range[0], freq_range[1] + freq_step,
                             freq_step)
    ppc_values = np.zeros(len(center_freqs))
    for fi, fc in enumerate(center_freqs):
        lo = fc - bandwidth / 2
        hi = fc + bandwidth / 2
        mean_ppc, _ = compute_ppc_at_band(
            spike_monitor, lfp, fs, dt_seconds,
            band=(lo, hi), t_discard=t_discard, min_spikes=min_spikes
        )
        ppc_values[fi] = mean_ppc
    return center_freqs, ppc_values


# ── Build E-spike LFP proxy (Tahvili et al.) ──
def build_e_spike_lfp(spike_data_layer, dt_ms, n_samples):
    e_key = [k for k in spike_data_layer.keys() if k.upper().startswith('E')][0]
    sd = spike_data_layer[e_key]
    times_ms = sd["times_ms"]
    indices = sd["spike_indices"]

    # Binned spike count (sum over all E neurons)
    spike_count = np.zeros(n_samples)
    bins = np.clip((times_ms / dt_ms).astype(int), 0, n_samples - 1)
    np.add.at(spike_count, bins, 1)

    # Normalize to population activity (Eq. 5 in paper)
    n_neurons = len(np.unique(indices))
    pop_activity = spike_count / (n_neurons * dt_ms / 1000.0)

    # Convolve with 1 ms Gaussian kernel
    sigma_samples = 1.0 / dt_ms
    lfp_proxy = gaussian_filter1d(pop_activity, sigma=sigma_samples)

    return lfp_proxy


# ── Cell type from key ──
def _cell_type_from_key(key):
    key_upper = key.upper()
    for ct in ['VIP', 'SOM', 'PV']:
        if ct in key_upper:
            return ct
    if 'E' in key.split('_')[0].upper() and len(key.split('_')[0]) <= 2:
        return 'E'
    return key.split('_')[0]


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

fname = "results/10s_with_PV_delay_PG/trial_000.npz"
data = np.load(fname, allow_pickle=True)
spike_data = data["spike_data"].item() if data["spike_data"].size == 1 else data["spike_data"]

dt_sec = float(CONFIG['simulation']['DT'] / second)
dt_ms = dt_sec * 1000.0
sim_duration_ms = float(data["time_array_ms"][-1])
n_samples = int(sim_duration_ms / dt_ms)
fs = 1.0 / dt_sec

print(f"dt = {dt_ms} ms, fs = {fs:.0f} Hz, duration = {sim_duration_ms:.0f} ms")

# Build spike monitors
spike_monitors = {}
for layer_name, layer_mons in spike_data.items():
    spike_monitors[layer_name] = {}
    for mon_name, sd in layer_mons.items():
        spike_monitors[layer_name][mon_name] = FakeSpikeMonitor(
            sd["times_ms"], sd["spike_indices"]
        )

# Compute PPC per layer
print("\n" + "=" * 60)
print("PPC SPECTRUM — E-spike LFP proxy (Tahvili et al. method)")
print("=" * 60)

results = {}
for layer_name in spike_monitors:
    results[layer_name] = {}
    lfp_proxy = build_e_spike_lfp(spike_data[layer_name], dt_ms, n_samples)

    print(f"\n{layer_name} (LFP proxy from {len(np.unique(spike_data[layer_name][[k for k in spike_data[layer_name] if k.upper().startswith('E')][0]]['spike_indices']))} E neurons):")

    for key, sm in spike_monitors[layer_name].items():
        ct = _cell_type_from_key(key)
        n_neurons = len(sm.spike_trains())
        print(f"  {ct} ({n_neurons} neurons)...", end=" ")

        freqs, spec = compute_ppc_spectrum(
            sm, lfp_proxy, fs, dt_sec,
            t_discard=2.0,
            freq_range=(5, 100),
            freq_step=2,
            bandwidth=10,
            min_spikes=2,
        )

        results[layer_name][ct] = {'ppc_spectrum': (freqs, spec)}

        if len(spec) > 0:
            peak_idx = np.argmax(spec)
            print(f"peak PPC = {spec[peak_idx]:.6f} at {freqs[peak_idx]:.1f} Hz")
        else:
            print("no valid PPC")

# ── Plot ──
colors = {'E': '#2ca02c', 'PV': '#d62728', 'SOM': '#1f77b4', 'VIP': '#ff7f0e'}
layers = list(results.keys())
n = len(layers)
fig, axes = plt.subplots(n, 1, figsize=(8, 3.5 * n), squeeze=False)

for i, layer in enumerate(layers):
    ax = axes[i, 0]
    for ct in ['PV', 'E', 'SOM', 'VIP']:
        if ct in results[layer] and 'ppc_spectrum' in results[layer][ct]:
            freqs, vals = results[layer][ct]['ppc_spectrum']
            if len(freqs) > 0:
                smooth = gaussian_filter1d(vals, sigma=1.5)
                ax.plot(freqs, smooth, color=colors[ct], label=ct, lw=2)
    ax.set_ylabel('PPC')
    ax.set_title(f'{layer} — E-spike LFP proxy')
    ax.legend()
    if i == n - 1:
        ax.set_xlabel('Frequency (Hz)')

plt.tight_layout()
plt.savefig('ppc_spectrum_e_proxy.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: ppc_spectrum_e_proxy.png")
plt.show()

# ── Summary ──
print("\n" + "=" * 60)
print("SUMMARY: Peak gamma PPC (30-50 Hz)")
print("=" * 60)
for layer_name in results:
    print(f"\n{layer_name}:")
    for ct in ['PV', 'E', 'SOM', 'VIP']:
        if ct in results[layer_name]:
            freqs, spec = results[layer_name][ct]['ppc_spectrum']
            gamma_mask = (freqs >= 30) & (freqs <= 50)
            if np.any(gamma_mask):
                print(f"  {ct}: mean={np.mean(spec[gamma_mask]):.6f}, "
                      f"peak={np.max(spec[gamma_mask]):.6f}")