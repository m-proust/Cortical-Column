import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfiltfilt
from brian2 import second
from config.config import CONFIG

T_DISCARD = 2.0
T_ANALYSIS = 10.0
MIN_SPIKES = 5
BW_HALF = 5.0
FREQ_RANGE = (5, 120)
FREQ_STEP = 2.0
MAX_E_NEURONS = 1000


def bandpass(signal, fs, low, high, order=3):
    nyq = fs / 2.0
    f_lo = max(low, 1.0)
    f_hi = min(high, nyq - 1.0)
    if f_hi <= f_lo:
        return np.zeros_like(signal)
    sos = butter(order, [f_lo / nyq, f_hi / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, signal)


def build_spike_trains(times_ms, spike_indices):
    trains = {}
    if len(spike_indices) == 0:
        return trains
    for idx in np.unique(spike_indices):
        mask = spike_indices == idx
        trains[idx] = times_ms[mask] / 1000.0
    return trains


def normalize_lfp(lfp, dt):
    lfp_dc_removed = lfp - np.mean(lfp)
    norm_factor = np.sqrt(np.sum(lfp_dc_removed**2) * dt)
    return lfp_dc_removed / norm_factor


def get_spike_phases_per_neuron(spike_trains, t_lfp, inst_phase,
                                t_discard, t_end):
    neuron_phases = {}
    for nid, st in spike_trains.items():
        mask = (st >= t_discard) & (st <= t_end)
        st_window = st[mask]
        if len(st_window) < 2:
            continue
        indices = np.searchsorted(t_lfp, st_window)
        indices = np.clip(indices, 0, len(inst_phase) - 1)
        neuron_phases[nid] = inst_phase[indices]
    return neuron_phases


def compute_ppc(phases):
    N = len(phases)
    if N < 2:
        return np.nan
    cs = np.sum(np.cos(phases))
    sn = np.sum(np.sin(phases))
    R_squared = (cs**2 + sn**2) / (N**2)
    return (N * R_squared - 1.0) / (N - 1.0)


def compute_ppc_spectrum(spike_trains, lfp_normalized, t_lfp, fs,
                         t_discard, t_end,
                         freq_range=(5, 90), freq_step=2.0, bw=5.0,
                         min_spikes=5, max_neurons=None):
    neuron_ids = list(spike_trains.keys())
    if max_neurons is not None and len(neuron_ids) > max_neurons:
        neuron_ids = list(np.random.choice(neuron_ids, max_neurons, replace=False))
        spike_trains_sub = {nid: spike_trains[nid] for nid in neuron_ids}
    else:
        spike_trains_sub = spike_trains

    freqs = np.arange(freq_range[0], freq_range[1] + freq_step, freq_step)
    ppc_mean = np.full(len(freqs), np.nan)
    ppc_sem = np.full(len(freqs), np.nan)

    for fi, fc in enumerate(freqs):
        f_lo = fc - bw
        f_hi = fc + bw

        lfp_bp = bandpass(lfp_normalized, fs, f_lo, f_hi)
        if np.all(lfp_bp == 0):
            continue
        phase_signal = np.angle(hilbert(lfp_bp))

        neuron_phases = get_spike_phases_per_neuron(
            spike_trains_sub, t_lfp, phase_signal, t_discard, t_end
        )

        ppc_vals = []
        for nid, ph in neuron_phases.items():
            if len(ph) >= min_spikes:
                val = compute_ppc(ph)
                if not np.isnan(val):
                    ppc_vals.append(val)

        if len(ppc_vals) > 0:
            ppc_mean[fi] = np.mean(ppc_vals)
            ppc_sem[fi] = np.std(ppc_vals) / np.sqrt(len(ppc_vals))

    return freqs, ppc_mean, ppc_sem


def circular_mean(phases):
    return np.angle(np.mean(np.exp(1j * phases)))


def circular_resultant(phases):
    return np.abs(np.mean(np.exp(1j * phases)))


def _cell_type_from_key(key):
    key_upper = key.upper()
    for ct in ['VIP', 'SOM', 'PV']:
        if ct in key_upper:
            return ct
    if 'E' in key.split('_')[0].upper() and len(key.split('_')[0]) <= 2:
        return 'E'
    return key.split('_')[0]


fname = "results/trials_24_04_12s/trial_000.npz"
data = np.load(fname, allow_pickle=True)
spike_data = (data["spike_data"].item()
              if data["spike_data"].size == 1
              else data["spike_data"])
lfp_full_data = (data["lfp_full"].item()
                 if data["lfp_full"].size == 1
                 else data["lfp_full"])
dt_sec = float(CONFIG['simulation']['DT'] / second)
dt_ms = dt_sec * 1000.0
sim_duration_ms = float(data["time_array_ms"][-1])
fs = 1.0 / dt_sec

print(f"dt = {dt_ms} ms, fs = {fs:.0f} Hz, duration = {sim_duration_ms:.0f} ms")

all_spike_trains = {}
for layer_name, layer_mons in spike_data.items():
    all_spike_trains[layer_name] = {}
    for mon_name, sd in layer_mons.items():
        all_spike_trains[layer_name][mon_name] = build_spike_trains(
            sd["times_ms"], sd["spike_indices"]
        )

layers = list(all_spike_trains.keys())
n_layers = len(layers)
print(f"Layers: {layers}")

lfp_cache = {}
for lfp_layer in layers:
    lfp_full = lfp_full_data[lfp_layer]
    t_full = np.arange(len(lfp_full)) * dt_sec
    t_end = T_DISCARD + T_ANALYSIS
    mask_analysis = (t_full >= T_DISCARD) & (t_full <= t_end)
    lfp = lfp_full[mask_analysis]
    t_lfp = t_full[mask_analysis]
    lfp_normalized = normalize_lfp(lfp, dt_sec)

    N_fft = len(lfp_normalized)
    freqs_fft = np.fft.rfftfreq(N_fft, d=dt_sec)
    fft_mag = np.abs(np.fft.rfft(lfp_normalized))
    gamma_mask_fft = (freqs_fft >= 30) & (freqs_fft <= 110)
    peak_idx_fft = np.argmax(fft_mag[gamma_mask_fft])
    f_peak = freqs_fft[gamma_mask_fft][peak_idx_fft]

    lfp_gamma = bandpass(lfp_normalized, fs, f_peak - BW_HALF, f_peak + BW_HALF)
    gamma_phase = np.angle(hilbert(lfp_gamma))

    lfp_cache[lfp_layer] = {
        'lfp_normalized': lfp_normalized,
        't_lfp': t_lfp,
        'f_peak': f_peak,
        'gamma_phase': gamma_phase,
    }
    print(f"  LFP {lfp_layer}: gamma peak = {f_peak:.1f} Hz")

t_end = T_DISCARD + T_ANALYSIS
results = {spk: {lfp: {} for lfp in layers} for spk in layers}
phase_results = {spk: {lfp: {} for lfp in layers} for spk in layers}

for spk_layer in layers:
    print(f"\nSpike layer {spk_layer}:")
    for lfp_layer in layers:
        cache = lfp_cache[lfp_layer]
        print(f"  vs LFP {lfp_layer} (peak {cache['f_peak']:.1f} Hz)...")

        for key, trains in all_spike_trains[spk_layer].items():
            ct = _cell_type_from_key(key)
            max_n = MAX_E_NEURONS if ct == 'E' else None

            freqs, ppc_mean, ppc_sem = compute_ppc_spectrum(
                trains, cache['lfp_normalized'], cache['t_lfp'], fs,
                t_discard=T_DISCARD, t_end=t_end,
                freq_range=FREQ_RANGE, freq_step=FREQ_STEP,
                bw=BW_HALF, min_spikes=MIN_SPIKES, max_neurons=max_n,
            )
            results[spk_layer][lfp_layer][ct] = (freqs, ppc_mean, ppc_sem)

            neuron_phases = get_spike_phases_per_neuron(
                trains, cache['t_lfp'], cache['gamma_phase'],
                T_DISCARD, t_end
            )
            all_phases = np.concatenate(
                [ph for ph in neuron_phases.values() if len(ph) >= 2]
            ) if neuron_phases else np.array([])

            if len(all_phases) > 0:
                phase_results[spk_layer][lfp_layer][ct] = {
                    'all_phases': all_phases,
                    'mean_phase': circular_mean(all_phases),
                    'resultant_length': circular_resultant(all_phases),
                    'n_spikes': len(all_phases),
                    'n_neurons': len(neuron_phases),
                }


colors = {'E': 'green', 'PV': 'red', 'SOM': 'blue', 'VIP': '#ff7f0e'}

fig, axes = plt.subplots(n_layers, n_layers,
                         figsize=(4 * n_layers, 3 * n_layers),
                         squeeze=False, sharex=True)

ymax = 0.0
for spk_layer in layers:
    for lfp_layer in layers:
        for ct, (freqs, ppc_mean, _) in results[spk_layer][lfp_layer].items():
            valid = ~np.isnan(ppc_mean)
            if np.any(valid):
                ymax = max(ymax, np.nanmax(ppc_mean[valid]))
ymax = ymax * 1.1 if ymax > 0 else 1.0

for i, spk_layer in enumerate(layers):
    for j, lfp_layer in enumerate(layers):
        ax = axes[i, j]
        for ct in ['E', 'PV', 'SOM', 'VIP']:
            if ct not in results[spk_layer][lfp_layer]:
                continue
            freqs, ppc_mean, ppc_sem = results[spk_layer][lfp_layer][ct]
            valid = ~np.isnan(ppc_mean)
            if not np.any(valid):
                continue
            ax.plot(freqs[valid], ppc_mean[valid],
                    color=colors[ct], linewidth=1.5, label=ct)
            ax.fill_between(freqs[valid],
                            ppc_mean[valid] - ppc_sem[valid],
                            ppc_mean[valid] + ppc_sem[valid],
                            color=colors[ct], alpha=0.15)

        ax.set_xlim(FREQ_RANGE)
        ax.set_ylim(0, ymax)
        if i == j:
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_color('black')
        if i == 0:
            ax.set_title(f'LFP: {lfp_layer}', fontsize=12, fontweight='bold')
        if j == 0:
            ax.set_ylabel(f'spikes: {spk_layer}\nPPC',
                          fontsize=12, fontweight='bold')
        if i == n_layers - 1:
            ax.set_xlabel('Frequency (Hz)', fontsize=11)
        if i == 0 and j == n_layers - 1:
            ax.legend(fontsize=9, loc='upper right')

fig.suptitle('Cross-layer spike–LFP phase locking (rows: spikes, cols: LFP)',
             fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig('cross_layer_ppc_spectrum.png', dpi=150, bbox_inches='tight')
plt.show()


fig_ph, axes_ph = plt.subplots(n_layers, n_layers,
                               figsize=(4 * n_layers, 3 * n_layers),
                               squeeze=False, sharex=True)

bins_phase = np.linspace(-np.pi, np.pi, 37)
bin_centers = 0.5 * (bins_phase[:-1] + bins_phase[1:])

for i, spk_layer in enumerate(layers):
    for j, lfp_layer in enumerate(layers):
        ax = axes_ph[i, j]
        f_peak = lfp_cache[lfp_layer]['f_peak']
        for ct in ['PV', 'SOM', 'E', 'VIP']:
            if ct not in phase_results[spk_layer][lfp_layer]:
                continue
            ph = phase_results[spk_layer][lfp_layer][ct]['all_phases']
            if len(ph) == 0:
                continue
            hist, _ = np.histogram(ph, bins=bins_phase, density=True)
            ax.plot(np.degrees(bin_centers), hist,
                    color=colors[ct], linewidth=1.5, label=ct)
            ax.fill_between(np.degrees(bin_centers), 0, hist,
                            color=colors[ct], alpha=0.2)

        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, 0, 180])
        if i == j:
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_color('black')
        if i == 0:
            ax.set_title(f'LFP: {lfp_layer}\n(peak {f_peak:.0f} Hz)',
                         fontsize=11, fontweight='bold')
        if j == 0:
            ax.set_ylabel(f'spikes: {spk_layer}\ndensity',
                          fontsize=12, fontweight='bold')
        if i == n_layers - 1:
            ax.set_xlabel('phase (deg)', fontsize=11)
            ax.set_xticklabels(['-180°', '0°', '180°'])
        if i == 0 and j == n_layers - 1:
            ax.legend(fontsize=9, loc='upper right')

fig_ph.suptitle('Cross-layer spike phase distributions at each LFP gamma peak',
                fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig('cross_layer_phase_distributions.png', dpi=150, bbox_inches='tight')
plt.show()


ALPHA_BAND = (8.0, 13.0)
GAMMA_BAND = (30.0, 50.0)

lfp_band_phases = {}
for lfp_layer in layers:
    lfp_norm = lfp_cache[lfp_layer]['lfp_normalized']
    alpha_bp = bandpass(lfp_norm, fs, ALPHA_BAND[0], ALPHA_BAND[1])
    gamma_bp = bandpass(lfp_norm, fs, GAMMA_BAND[0], GAMMA_BAND[1])
    lfp_band_phases[lfp_layer] = {
        'alpha_phase': np.angle(hilbert(alpha_bp)),
        'gamma_phase': np.angle(hilbert(gamma_bp)),
        'alpha_signal': alpha_bp,
        'gamma_signal': gamma_bp,
    }


def plv_and_mean_dphase(phase_a, phase_b):
    dphi = phase_a - phase_b
    z = np.mean(np.exp(1j * dphi))
    return np.abs(z), np.angle(z)


def compute_lfp_matrices(band_key):
    mean_dphase = np.full((n_layers, n_layers), np.nan)
    plv_matrix = np.full((n_layers, n_layers), np.nan)
    for i, la in enumerate(layers):
        for j, lb in enumerate(layers):
            pa = lfp_band_phases[la][band_key]
            pb = lfp_band_phases[lb][band_key]
            plv, dph = plv_and_mean_dphase(pa, pb)
            plv_matrix[i, j] = plv
            mean_dphase[i, j] = np.degrees(dph)
    return mean_dphase, plv_matrix


alpha_dphase, alpha_plv = compute_lfp_matrices('alpha_phase')
gamma_dphase, gamma_plv = compute_lfp_matrices('gamma_phase')


fig_lfp, axes_lfp = plt.subplots(2, 2, figsize=(12, 10))

im00 = axes_lfp[0, 0].imshow(alpha_dphase, cmap='twilight_shifted',
                              vmin=-180, vmax=180, aspect='auto')
axes_lfp[0, 0].set_title(f'Alpha ({ALPHA_BAND[0]:.0f}-{ALPHA_BAND[1]:.0f} Hz) '
                          f'mean phase difference (row − col, deg)',
                          fontsize=11)
plt.colorbar(im00, ax=axes_lfp[0, 0], label='degrees')

im01 = axes_lfp[0, 1].imshow(alpha_plv, cmap='viridis',
                              vmin=0, vmax=1, aspect='auto')
axes_lfp[0, 1].set_title(f'Alpha ({ALPHA_BAND[0]:.0f}-{ALPHA_BAND[1]:.0f} Hz) '
                          f'phase-locking value (PLV)', fontsize=11)
plt.colorbar(im01, ax=axes_lfp[0, 1], label='PLV')

im10 = axes_lfp[1, 0].imshow(gamma_dphase, cmap='twilight_shifted',
                              vmin=-180, vmax=180, aspect='auto')
axes_lfp[1, 0].set_title(f'Gamma ({GAMMA_BAND[0]:.0f}-{GAMMA_BAND[1]:.0f} Hz) '
                          f'mean phase difference (row − col, deg)',
                          fontsize=11)
plt.colorbar(im10, ax=axes_lfp[1, 0], label='degrees')

im11 = axes_lfp[1, 1].imshow(gamma_plv, cmap='viridis',
                              vmin=0, vmax=1, aspect='auto')
axes_lfp[1, 1].set_title(f'Gamma ({GAMMA_BAND[0]:.0f}-{GAMMA_BAND[1]:.0f} Hz) '
                          f'phase-locking value (PLV)', fontsize=11)
plt.colorbar(im11, ax=axes_lfp[1, 1], label='PLV')

for ax in axes_lfp.flat:
    ax.set_xticks(range(n_layers))
    ax.set_yticks(range(n_layers))
    ax.set_xticklabels(layers)
    ax.set_yticklabels(layers)
    ax.set_xlabel('LFP layer (col)')
    ax.set_ylabel('LFP layer (row)')
    for i in range(n_layers):
        for j in range(n_layers):
            val_src = ax.get_images()[0].get_array()
            v = val_src[i, j]
            ax.text(j, i, f'{v:.0f}' if abs(v) >= 1 else f'{v:.2f}',
                    ha='center', va='center',
                    color='white' if ax in [axes_lfp[0, 1], axes_lfp[1, 1]] and v < 0.5 else 'black',
                    fontsize=8)

fig_lfp.suptitle('LFP–LFP phase relationships across layers', fontsize=14)
plt.tight_layout()
plt.savefig('lfp_phase_relationships.png', dpi=150, bbox_inches='tight')
plt.show()


ref_layer = layers[0]
t_demo = lfp_cache[ref_layer]['t_lfp']
demo_mask = (t_demo - t_demo[0]) <= 1.0

fig_demo, axes_demo = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
for lfp_layer in layers:
    axes_demo[0].plot(t_demo[demo_mask],
                      lfp_band_phases[lfp_layer]['alpha_signal'][demo_mask],
                      label=lfp_layer, linewidth=1.2)
    axes_demo[1].plot(t_demo[demo_mask],
                      lfp_band_phases[lfp_layer]['gamma_signal'][demo_mask],
                      label=lfp_layer, linewidth=1.0)
axes_demo[0].set_title(f'Alpha-band LFP ({ALPHA_BAND[0]:.0f}-{ALPHA_BAND[1]:.0f} Hz), first 1 s')
axes_demo[1].set_title(f'Gamma-band LFP ({GAMMA_BAND[0]:.0f}-{GAMMA_BAND[1]:.0f} Hz), first 1 s')
axes_demo[1].set_xlabel('time (s)')
for ax in axes_demo:
    ax.set_ylabel('amplitude (a.u.)')
    ax.legend(fontsize=9, ncol=n_layers, loc='upper right')
plt.tight_layout()
plt.savefig('lfp_band_timeseries.png', dpi=150, bbox_inches='tight')
plt.show()


print("\n=== LFP–LFP phase relationships ===")
print(f"\nAlpha band ({ALPHA_BAND[0]:.0f}-{ALPHA_BAND[1]:.0f} Hz):")
print("  Mean phase difference (row − col, deg):")
print("       " + "  ".join(f'{l:>6}' for l in layers))
for i, la in enumerate(layers):
    row = "  ".join(f'{alpha_dphase[i, j]:6.1f}' for j in range(n_layers))
    print(f"  {la:>4} {row}")
print("  PLV:")
for i, la in enumerate(layers):
    row = "  ".join(f'{alpha_plv[i, j]:6.3f}' for j in range(n_layers))
    print(f"  {la:>4} {row}")

print(f"\nGamma band ({GAMMA_BAND[0]:.0f}-{GAMMA_BAND[1]:.0f} Hz):")
print("  Mean phase difference (row − col, deg):")
for i, la in enumerate(layers):
    row = "  ".join(f'{gamma_dphase[i, j]:6.1f}' for j in range(n_layers))
    print(f"  {la:>4} {row}")
print("  PLV:")
for i, la in enumerate(layers):
    row = "  ".join(f'{gamma_plv[i, j]:6.3f}' for j in range(n_layers))
    print(f"  {la:>4} {row}")


print("\n=== Gamma-band (30–50 Hz) PPC summary ===")
for spk_layer in layers:
    print(f"\nSpikes {spk_layer}:")
    for lfp_layer in layers:
        marker = " (within)" if spk_layer == lfp_layer else ""
        print(f"  vs LFP {lfp_layer}{marker}:")
        for ct in ['E', 'PV', 'SOM', 'VIP']:
            if ct not in results[spk_layer][lfp_layer]:
                continue
            freqs, ppc_mean, _ = results[spk_layer][lfp_layer][ct]
            gamma_mask = (freqs >= 30) & (freqs <= 50)
            vals = ppc_mean[gamma_mask]
            valid = ~np.isnan(vals)
            if np.any(valid):
                print(f"    {ct}: mean={np.nanmean(vals):.4f}, "
                      f"peak={np.nanmax(vals):.4f}")
