import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfiltfilt, freqz, sosfreqz, welch
from brian2 import second


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

from config.config2 import CONFIG

# --- Parameters matching Tahvili et al. ---
T_DISCARD = 2.0       # seconds of transient to discard
T_ANALYSIS = 10.0     # seconds of signal for analysis (paper: "10 s duration")
MIN_SPIKES = 5        # minimum spikes per neuron to include in PPC
BW_HALF = 5.0         # half-bandwidth for bandpass filter (Hz)
FREQ_RANGE = (5, 120)
FREQ_STEP = 2.0
MAX_E_NEURONS = 1000   # subsample E neurons for speed


# ══════════════════════════════════════════════════════════════════════════════
# Bandpass using 4th-order Butterworth IIR (matching Script 2)
# ══════════════════════════════════════════════════════════════════════════════

def bandpass(signal, fs, low, high, order=3):
    nyq = fs / 2.0
    f_lo = max(low, 1.0)
    f_hi = min(high, nyq - 1.0)
    if f_hi <= f_lo:
        return np.zeros_like(signal)
    sos = butter(order, [f_lo / nyq, f_hi / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, signal)


# ══════════════════════════════════════════════════════════════════════════════
# Build spike trains dict from raw arrays
# ══════════════════════════════════════════════════════════════════════════════

def build_spike_trains(times_ms, spike_indices):
    """
    Convert raw times_ms and spike_indices arrays into a dict:
        {neuron_id: spike_times_in_seconds}
    """
    trains = {}
    if len(spike_indices) == 0:
        return trains
    for idx in np.unique(spike_indices):
        mask = spike_indices == idx
        trains[idx] = times_ms[mask] / 1000.0
    return trains



# ══════════════════════════════════════════════════════════════════════════════
# Normalize LFP exactly as in the paper
# "We first removed its dc component, then normalize:
#  x_hat(t) = (x(t) - mean(x)) / sqrt(integral |x(t)-mean(x)|^2 dt)"
# ══════════════════════════════════════════════════════════════════════════════

def normalize_lfp(lfp, dt):
    lfp_dc_removed = lfp - np.mean(lfp)
    norm_factor = np.sqrt(np.sum(lfp_dc_removed**2) * dt)
    return lfp_dc_removed / norm_factor


# ══════════════════════════════════════════════════════════════════════════════
# Extract LFP phase at spike times for each neuron
# Uses np.searchsorted (matching Script 2)
# ══════════════════════════════════════════════════════════════════════════════

def get_spike_phases_per_neuron(spike_trains, t_lfp, inst_phase,
                                t_discard, t_end):
    """
    For each neuron, get array of LFP phases at its spike times.
    Only spikes in [t_discard, t_end] are used.

    Parameters
    ----------
    spike_trains : dict {neuron_id: spike_times_in_seconds}
    t_lfp : time array for the LFP (seconds)
    inst_phase : instantaneous phase array (same length as t_lfp)
    t_discard : start of analysis window (seconds)
    t_end : end of analysis window (seconds)

    Returns
    -------
    dict: neuron_id -> array of phases
    """
    neuron_phases = {}
    for nid, st in spike_trains.items():
        # Filter spikes to analysis window
        mask = (st >= t_discard) & (st <= t_end)
        st_window = st[mask]

        if len(st_window) < 2:
            continue

        # Map spike times to nearest LFP sample index
        indices = np.searchsorted(t_lfp, st_window)
        indices = np.clip(indices, 0, len(inst_phase) - 1)

        neuron_phases[nid] = inst_phase[indices]

    return neuron_phases


# ══════════════════════════════════════════════════════════════════════════════
# PPC computation (Vinck et al. 2010, 2012)
# PPC = (N * R^2 - 1) / (N - 1) where R = |mean(exp(i*theta))|
# ══════════════════════════════════════════════════════════════════════════════

def compute_ppc(phases):
    N = len(phases)
    if N < 2:
        return np.nan
    cs = np.sum(np.cos(phases))
    sn = np.sum(np.sin(phases))
    R_squared = (cs**2 + sn**2) / (N**2)
    return (N * R_squared - 1.0) / (N - 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# PPC spectrum — PPC as a function of frequency for a population
# ══════════════════════════════════════════════════════════════════════════════

def compute_ppc_spectrum(spike_trains, lfp_normalized, t_lfp, fs,
                         t_discard, t_end,
                         freq_range=(5, 90), freq_step=2.0, bw=5.0,
                         min_spikes=5, max_neurons=None):
    """
    PPC as a function of frequency for a population.

    Parameters
    ----------
    spike_trains : dict {neuron_id: spike_times_in_seconds}
    lfp_normalized : normalized LFP (dc-removed, unit-energy)
    t_lfp : corresponding time array (seconds)
    fs : sampling frequency (Hz)
    t_discard, t_end : analysis window boundaries (s)
    freq_range : (f_min, f_max)
    freq_step : frequency resolution (Hz)
    bw : half-bandwidth for bandpass filter (Hz)
    min_spikes : minimum number of spikes per neuron to include
    max_neurons : subsample if population too large (None = use all)

    Returns
    -------
    freqs, ppc_mean, ppc_sem
    """
    # Optionally subsample neurons
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

        # Band-pass filter
        lfp_bp = bandpass(lfp_normalized, fs, f_lo, f_hi)
        if np.all(lfp_bp == 0):
            continue

        # Instantaneous phase via Hilbert transform
        phase_signal = np.angle(hilbert(lfp_bp))

        # Get phases at spike times for each neuron
        neuron_phases = get_spike_phases_per_neuron(
            spike_trains_sub, t_lfp, phase_signal, t_discard, t_end
        )

        # PPC per neuron, then average
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


# ══════════════════════════════════════════════════════════════════════════════
# Circular statistics helpers
# ══════════════════════════════════════════════════════════════════════════════

def circular_mean(phases):
    """Circular mean angle."""
    return np.angle(np.mean(np.exp(1j * phases)))


def circular_resultant(phases):
    """Mean resultant length."""
    return np.abs(np.mean(np.exp(1j * phases)))


# ══════════════════════════════════════════════════════════════════════════════
# Cell type from key
# ══════════════════════════════════════════════════════════════════════════════

def _cell_type_from_key(key):
    key_upper = key.upper()
    for ct in ['VIP', 'SOM', 'PV']:
        if ct in key_upper:
            return ct
    if 'E' in key.split('_')[0].upper() and len(key.split('_')[0]) <= 2:
        return 'E'
    return key.split('_')[0]


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

fname = "results/trials_02_04_12s/trial_000.npz"
data = np.load(fname, allow_pickle=True)
spike_data = (data["spike_data"].item()
              if data["spike_data"].size == 1
              else data["spike_data"])
lfp_full_data = (data["lfp_full"].item()
                 if data["lfp_full"].size == 1
                 else data["lfp_full"])
bipolar_lfp = (data["bipolar_matrix"].item()
                 if data["bipolar_matrix"].size == 1
                 else data["bipolar_matrix"])
dt_sec = float(CONFIG['simulation']['DT'] / second)
dt_ms = dt_sec * 1000.0
sim_duration_ms = float(data["time_array_ms"][-1])
n_samples = int(sim_duration_ms / dt_ms)
fs = 1.0 / dt_sec

print(f"dt = {dt_ms} ms, fs = {fs:.0f} Hz, duration = {sim_duration_ms:.0f} ms")

# ── Build spike trains from raw arrays ──
all_spike_trains = {}
for layer_name, layer_mons in spike_data.items():
    all_spike_trains[layer_name] = {}
    for mon_name, sd in layer_mons.items():
        all_spike_trains[layer_name][mon_name] = build_spike_trains(
            sd["times_ms"], sd["spike_indices"]
        )

# ══════════════════════════════════════════════════════════════════════════════
# Per-layer analysis
# ══════════════════════════════════════════════════════════════════════════════

results = {}
electrodes = {'L23': 10, 'L4AB':8, 'L4C':6, 'L5':4, 'L6':3 }
for layer_name in all_spike_trains:
    results[layer_name] = {}
    electrode = electrodes[layer_name]

    # ── Step 1: Load pre-computed E smooth rate as LFP proxy ──
    lfp_full = lfp_full_data[layer_name]
    # lfp_full = bipolar_lfp[electrode]
    t_full = np.arange(len(lfp_full)) * dt_sec
    # t_full = data["time_array_ms"] / 1000.0





    # ── Step 2: Select analysis window [t_discard, t_discard + t_analysis] ──
    t_end = T_DISCARD + T_ANALYSIS
    mask_analysis = (t_full >= T_DISCARD) & (t_full <= t_end)
    lfp = lfp_full[mask_analysis]
    t_lfp = t_full[mask_analysis]

    print(f"\n{layer_name}: LFP analysis window {t_lfp[0]:.2f} to {t_lfp[-1]:.2f} s "
          f"({len(lfp)} samples, {len(lfp)*dt_sec:.2f} s)")

    # ── Step 3: Normalize LFP (DC removal + unit-energy) ──
    lfp_normalized = normalize_lfp(lfp, dt_sec)

    # ── Step 4: Find peak gamma frequency via FFT ──
    N_fft = len(lfp_normalized)
    freqs_fft = np.fft.rfftfreq(N_fft, d=dt_sec)
    fft_mag = np.abs(np.fft.rfft(lfp_normalized))

    gamma_mask_fft = (freqs_fft >= 30) & (freqs_fft <= 110)
    peak_idx_fft = np.argmax(fft_mag[gamma_mask_fft])
    f_peak = freqs_fft[gamma_mask_fft][peak_idx_fft]
    print(f"  Peak oscillation frequency: {f_peak:.1f} Hz")

    # ── Step 5: Compute PPC spectrum for each cell type ──
    e_key = [k for k in spike_data[layer_name] if k.upper().startswith('E')][0]
    n_e = len(np.unique(spike_data[layer_name][e_key]['spike_indices']))
    print(f"  LFP proxy from {n_e} E neurons")

    for key, trains in all_spike_trains[layer_name].items():
        ct = _cell_type_from_key(key)
        n_neurons = len(trains)

        # Subsample E neurons for speed, use all inhibitory neurons
        max_n = MAX_E_NEURONS if ct == 'E' else None

        print(f"  Computing PPC spectrum for {ct} ({n_neurons} neurons, "
              f"max_neurons={max_n})...", end=" ")

        freqs, ppc_mean, ppc_sem = compute_ppc_spectrum(
            trains, lfp_normalized, t_lfp, fs,
            t_discard=T_DISCARD, t_end=t_end,
            freq_range=FREQ_RANGE,
            freq_step=FREQ_STEP,
            bw=BW_HALF,
            min_spikes=MIN_SPIKES,
            max_neurons=max_n,
        )

        results[layer_name][ct] = {
            'ppc_spectrum': (freqs, ppc_mean, ppc_sem),
        }

        valid = ~np.isnan(ppc_mean)
        if np.any(valid):
            peak_idx = np.nanargmax(ppc_mean)
            print(f"peak PPC = {ppc_mean[peak_idx]:.6f} at {freqs[peak_idx]:.1f} Hz")
        else:
            print("no valid PPC")

    # ── Step 6: Phase distribution at gamma peak ──
    print(f"\n  Computing preferred spike phases at gamma peak ({f_peak:.1f} Hz)...")

    lfp_gamma = bandpass(lfp_normalized, fs, f_peak - BW_HALF, f_peak + BW_HALF)
    gamma_phase = np.angle(hilbert(lfp_gamma))

    phase_results = {}
    for key, trains in all_spike_trains[layer_name].items():
        ct = _cell_type_from_key(key)
        neuron_phases = get_spike_phases_per_neuron(
            trains, t_lfp, gamma_phase, T_DISCARD, t_end
        )

        all_phases = np.concatenate(
            [ph for ph in neuron_phases.values() if len(ph) >= 2]
        ) if neuron_phases else np.array([])

        if len(all_phases) > 0:
            mean_ph = circular_mean(all_phases)
            mrl = circular_resultant(all_phases)
            phase_results[ct] = {
                'all_phases': all_phases,
                'mean_phase': mean_ph,
                'resultant_length': mrl,
                'n_spikes': len(all_phases),
                'n_neurons': len(neuron_phases),
            }
            print(f"  {ct}: mean phase = {np.degrees(mean_ph):+.1f} deg, "
                  f"R = {mrl:.4f}, "
                  f"N_spikes = {len(all_phases)}, "
                  f"N_neurons = {len(neuron_phases)}")
        else:
            print(f"  {ct}: no valid phase data")

    results[layer_name]['phase_results'] = phase_results
    results[layer_name]['f_peak'] = f_peak
    results[layer_name]['freqs_fft'] = freqs_fft
    results[layer_name]['fft_mag'] = fft_mag

    # ── Phase delay analysis (e.g. PV -> SOM) ──
    if 'PV' in phase_results and 'SOM' in phase_results:
        phase_delay = np.angle(np.exp(1j * (
            phase_results['SOM']['mean_phase'] -
            phase_results['PV']['mean_phase']
        )))
        time_delay_ms = (phase_delay / (2 * np.pi)) * (1000.0 / f_peak)
        print(f"\n  Phase delay SOM - PV: {np.degrees(phase_delay):+.1f} deg "
              f"({time_delay_ms:+.1f} ms)")
        print(f"  (Paper reports ~80 deg, ~6-7 ms)")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: PPC spectrum per layer (Fig 1C left panel style)
# ══════════════════════════════════════════════════════════════════════════════

colors = {'E': 'green', 'PV': 'red', 'SOM': 'blue', 'VIP': '#ff7f0e'}
layers = list(results.keys())
n_layers = len(layers)

fig, axes = plt.subplots(n_layers, 1, figsize=(10, 5 * n_layers), squeeze=False)

for i, layer in enumerate(layers):
    ax = axes[i, 0]
    for ct in ['E', 'PV', 'SOM', 'VIP']:
        if ct in results[layer] and 'ppc_spectrum' in results[layer][ct]:
            freqs, ppc_mean, ppc_sem = results[layer][ct]['ppc_spectrum']
            valid = ~np.isnan(ppc_mean)
            if np.any(valid):
                ax.plot(freqs[valid], ppc_mean[valid],
                        color=colors[ct], linewidth=2, label=ct)
                ax.fill_between(
                    freqs[valid],
                    ppc_mean[valid] - ppc_sem[valid],
                    ppc_mean[valid] + ppc_sem[valid],
                    color=colors[ct], alpha=0.15
                )
    ax.set_ylabel('Spike-LFP locking (PPC)', fontsize=12)
    ax.set_xlim(FREQ_RANGE)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11)
    ax.set_title(f'{layer} — Computational model', fontsize=13, fontstyle='italic')
    if i == n_layers - 1:
        ax.set_xlabel('Frequency (Hz)', fontsize=12)

plt.tight_layout()
plt.savefig('ppc_spectrum_e_proxy.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: ppc_spectrum_e_proxy.png")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Phase distribution at gamma peak (Fig 1C right panel style)
# ══════════════════════════════════════════════════════════════════════════════

for layer in layers:
    phase_res = results[layer].get('phase_results', {})
    f_peak = results[layer].get('f_peak', None)
    if not phase_res or f_peak is None:
        continue

    fig_ph, ax_ph = plt.subplots(figsize=(7, 4))
    bins_phase = np.linspace(-np.pi, np.pi, 37)  # 10-deg bins
    bin_centers = 0.5 * (bins_phase[:-1] + bins_phase[1:])

    for ct in ['PV', 'SOM', 'E', 'VIP']:
        if ct in phase_res and len(phase_res[ct]['all_phases']) > 0:
            hist, _ = np.histogram(phase_res[ct]['all_phases'],
                                   bins=bins_phase, density=True)
            ax_ph.plot(np.degrees(bin_centers), hist,
                       color=colors[ct], linewidth=2, label=ct)
            ax_ph.fill_between(np.degrees(bin_centers), 0, hist,
                               color=colors[ct], alpha=0.2)

    ax_ph.set_xlabel('Preferred spike gamma-phase', fontsize=12)
    ax_ph.set_ylabel('Probability density', fontsize=12)
    ax_ph.set_xlim(-180, 180)
    ax_ph.set_xticks([-180, 0, 180])
    ax_ph.set_xticklabels(['-180°', '0°', '180°'])
    ax_ph.legend(fontsize=11)
    ax_ph.set_title(f'{layer} — Phase distribution at {f_peak:.0f} Hz',
                    fontsize=13, fontstyle='italic')
    plt.tight_layout()
    plt.savefig(f'phase_distribution_{layer}.png', dpi=150, bbox_inches='tight')
    print(f"Saved: phase_distribution_{layer}.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Normalized LFP power spectrum
# ══════════════════════════════════════════════════════════════════════════════

for layer in layers:
    freqs_fft = results[layer].get('freqs_fft', None)
    fft_mag = results[layer].get('fft_mag', None)
    f_peak = results[layer].get('f_peak', None)
    if freqs_fft is None:
        continue

    fig_sp, ax_sp = plt.subplots(figsize=(8, 4))
    freq_plot_mask = (freqs_fft >= 1) & (freqs_fft <= 100)
    ax_sp.plot(freqs_fft[freq_plot_mask], fft_mag[freq_plot_mask],
               'k-', linewidth=1.5)
    ax_sp.set_xlabel('Frequency (Hz)', fontsize=12)
    ax_sp.set_ylabel('|F[x̂(t)]|', fontsize=12)
    ax_sp.set_title(f'{layer} — Normalized LFP power spectrum', fontsize=13)
    ax_sp.set_xlim(1, 100)
    ax_sp.axvline(f_peak, color='red', linestyle='--', alpha=0.7,
                  label=f'Peak: {f_peak:.1f} Hz')
    ax_sp.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f'lfp_spectrum_{layer}.png', dpi=150, bbox_inches='tight')
    print(f"Saved: lfp_spectrum_{layer}.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Compute 25-45 Hz relative band power (as in their Fig. 3C)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("25-45 Hz relative band power")
print("=" * 60)

for layer in layers:
    freqs_fft = results[layer].get('freqs_fft', None)
    fft_mag = results[layer].get('fft_mag', None)
    if freqs_fft is None:
        continue

    df = freqs_fft[1] - freqs_fft[0]
    total_power = np.sum(fft_mag**2) * df
    band_mask = (freqs_fft >= 25) & (freqs_fft <= 45)
    gamma_band_power = np.sum(fft_mag[band_mask]**2) * df
    relative_gamma_power = gamma_band_power / total_power
    print(f"  {layer}: {relative_gamma_power:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SUMMARY: Peak gamma PPC (30-50 Hz)")
print("=" * 60)
for layer_name in results:
    print(f"\n{layer_name}:")
    for ct in ['PV', 'E', 'SOM', 'VIP']:
        if ct in results[layer_name] and 'ppc_spectrum' in results[layer_name][ct]:
            freqs, ppc_mean, ppc_sem = results[layer_name][ct]['ppc_spectrum']
            gamma_mask = (freqs >= 30) & (freqs <= 50)
            if np.any(gamma_mask):
                vals = ppc_mean[gamma_mask]
                valid = ~np.isnan(vals)
                if np.any(valid):
                    print(f"  {ct}: mean={np.nanmean(vals):.6f}, "
                          f"peak={np.nanmax(vals):.6f}")

print("\nPPC analysis complete.")


# ══════════════════════════════════════════════════════════════════════════════
# FILTER DIAGNOSTICS — check that bandpass is accurate and stable
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("FILTER DIAGNOSTICS (SOS vs b,a comparison)")
print("=" * 60)

nyq = fs / 2.0
test_centers = np.arange(FREQ_RANGE[0], FREQ_RANGE[1] + FREQ_STEP, FREQ_STEP)

# ── 1. SOS pole stability check ──
print(f"\nSampling rate: {fs:.0f} Hz, Nyquist: {nyq:.0f} Hz")
print(f"Filter: 4th-order Butterworth, BW_HALF = {BW_HALF} Hz")
print(f"\n--- SOS pole stability (per section, all must have |z| < 1) ---")
n_sos_unstable = 0
for fc in test_centers:
    f_lo = max(fc - BW_HALF, 1.0)
    f_hi = min(fc + BW_HALF, nyq - 1.0)
    if f_hi <= f_lo:
        continue
    sos = butter(4, [f_lo / nyq, f_hi / nyq], btype='band', output='sos')
    max_pole_sos = 0.0
    for section in sos:
        poles_sec = np.roots(section[3:])
        max_pole_sos = max(max_pole_sos, np.max(np.abs(poles_sec)))
    if max_pole_sos >= 1.0:
        print(f"  !! UNSTABLE at fc={fc:.0f} Hz: max|pole| = {max_pole_sos:.6f}")
        n_sos_unstable += 1
    elif max_pole_sos > 0.999:
        print(f"  WARNING near-unstable at fc={fc:.0f} Hz: max|pole| = {max_pole_sos:.6f}")
if n_sos_unstable == 0:
    print("  All SOS filters stable (all section poles inside unit circle)")

# ── 2. Check for NaN/Inf in filtered output ──
print(f"\n--- NaN/Inf check on actual LFP filtering (SOS) ---")
first_layer = list(results.keys())[0]
lfp_diag = lfp_full_data[first_layer]
t_diag_full = np.arange(len(lfp_diag)) * dt_sec
mask_diag = (t_diag_full >= T_DISCARD) & (t_diag_full <= T_DISCARD + T_ANALYSIS)
lfp_diag = normalize_lfp(lfp_diag[mask_diag], dt_sec)

nan_count = 0
inf_count = 0
for fc in test_centers:
    filtered = bandpass(lfp_diag, fs, fc - BW_HALF, fc + BW_HALF)
    n_nan = np.sum(np.isnan(filtered))
    n_inf = np.sum(np.isinf(filtered))
    if n_nan > 0:
        print(f"  !! NaN at fc={fc:.0f} Hz: {n_nan} samples")
        nan_count += n_nan
    if n_inf > 0:
        print(f"  !! Inf at fc={fc:.0f} Hz: {n_inf} samples")
        inf_count += n_inf
if nan_count == 0 and inf_count == 0:
    print("  No NaN or Inf in any filtered output — good")

# ── 3. Synthetic sine test ──
print(f"\n--- Synthetic sine test (SOS) ---")
t_synth = np.arange(0, 2.0, 1.0 / fs)
for f_true in [10, 25, 40, 60, 80]:
    sine_in = np.sin(2 * np.pi * f_true * t_synth)
    for fc in [10, 40, 80]:
        filtered = bandpass(sine_in, fs, fc - BW_HALF, fc + BW_HALF)
        core = filtered[500:-500]  # skip edges
        power_ratio = np.std(core) / np.std(sine_in)
        should_pass = abs(f_true - fc) <= BW_HALF
        status = "PASS" if should_pass else "REJECT"
        ok = (power_ratio > 0.5) if should_pass else (power_ratio < 0.1)
        flag = "" if ok else " !! UNEXPECTED"
        print(f"  Sine {f_true:2d} Hz -> BP [{fc-BW_HALF:.0f}-{fc+BW_HALF:.0f}] Hz: "
              f"amplitude ratio = {power_ratio:.4f} (expect {status}){flag}")

# ── 4. Diagnostic plots ──
fig_diag, axes_diag = plt.subplots(2, 3, figsize=(18, 10))

# 4a. Magnitude response: SOS (used) vs b,a (old) at 10 Hz — worst case
ax = axes_diag[0, 0]
for fc in [10, 20, 40, 60, 80]:
    f_lo = max(fc - BW_HALF, 1.0)
    f_hi = min(fc + BW_HALF, nyq - 1.0)
    sos = butter(4, [f_lo / nyq, f_hi / nyq], btype='band', output='sos')
    w, h = sosfreqz(sos, worN=4096, fs=fs)
    ax.plot(w, 20 * np.log10(np.abs(h) + 1e-12), label=f'fc={fc} Hz')
ax.set_xlim(0, 100)
ax.set_ylim(-60, 5)
ax.axhline(-3, color='gray', ls='--', alpha=0.5, label='-3 dB')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude (dB)')
ax.set_title('SOS magnitude response')
ax.legend(fontsize=8)

# 4b. Compare SOS vs b,a magnitude at fc=10 Hz (where b,a was worst)
ax = axes_diag[0, 1]
fc_cmp = 10
f_lo = max(fc_cmp - BW_HALF, 1.0)
f_hi = min(fc_cmp + BW_HALF, nyq - 1.0)
b, a = butter(4, [f_lo / nyq, f_hi / nyq], btype='band')
sos = butter(4, [f_lo / nyq, f_hi / nyq], btype='band', output='sos')
w_ba, h_ba = freqz(b, a, worN=4096, fs=fs)
w_sos, h_sos = sosfreqz(sos, worN=4096, fs=fs)
ax.plot(w_ba, 20 * np.log10(np.abs(h_ba) + 1e-12), 'r-', lw=2, alpha=0.7, label='b,a (old — unstable)')
ax.plot(w_sos, 20 * np.log10(np.abs(h_sos) + 1e-12), 'b--', lw=2, label='SOS (new — stable)')
ax.set_xlim(0, 50)
ax.set_ylim(-60, 5)
ax.axhline(-3, color='gray', ls='--', alpha=0.3)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude (dB)')
ax.set_title(f'SOS vs b,a at fc={fc_cmp} Hz (worst case)')
ax.legend(fontsize=9)

# 4c. Max SOS pole vs center frequency
ax = axes_diag[0, 2]
max_poles_sos = []
max_poles_ba = []
for fc in test_centers:
    f_lo = max(fc - BW_HALF, 1.0)
    f_hi = min(fc + BW_HALF, nyq - 1.0)
    if f_hi <= f_lo:
        max_poles_sos.append(np.nan)
        max_poles_ba.append(np.nan)
        continue
    # SOS poles
    sos = butter(4, [f_lo / nyq, f_hi / nyq], btype='band', output='sos')
    mp = 0.0
    for section in sos:
        poles_sec = np.roots(section[3:])
        mp = max(mp, np.max(np.abs(poles_sec)))
    max_poles_sos.append(mp)
    # b,a poles (for comparison)
    b, a = butter(4, [f_lo / nyq, f_hi / nyq], btype='band')
    poles_ba = np.roots(a)
    max_poles_ba.append(np.max(np.abs(poles_ba)))
ax.plot(test_centers, max_poles_ba, 'r.-', alpha=0.7, label='b,a (old)')
ax.plot(test_centers, max_poles_sos, 'b.-', label='SOS (new)')
ax.axhline(1.0, color='k', ls='--', lw=1.5, label='Stability limit')
ax.set_xlabel('Center frequency (Hz)')
ax.set_ylabel('Max |pole|')
ax.set_title('Pole stability: SOS vs b,a')
ax.legend()

# 4d. Synthetic sine through bandpass — time domain
ax = axes_diag[1, 0]
t_plot = t_synth[:3000]
sine_40 = np.sin(2 * np.pi * 40 * t_synth)
bp_match = bandpass(sine_40, fs, 35, 45)
bp_reject = bandpass(sine_40, fs, 15, 25)
ax.plot(t_plot * 1000, sine_40[:3000], 'b-', alpha=0.4, label='Original 40 Hz')
ax.plot(t_plot * 1000, bp_match[:3000], 'r-', lw=1.5, label='BP [35-45] (should pass)')
ax.plot(t_plot * 1000, bp_reject[:3000], 'g-', lw=1.5, label='BP [15-25] (should reject)')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Amplitude')
ax.set_title('Sanity check: 40 Hz sine (SOS)')
ax.legend(fontsize=8)
ax.set_xlim(50, 250)

# 4e. Hilbert phase quality at 10 Hz (previously NaN with b,a)
ax = axes_diag[1, 1]
lfp_bp_10 = bandpass(lfp_diag, fs, 10 - BW_HALF, 10 + BW_HALF)
phase_10 = np.angle(hilbert(lfp_bp_10))
t_lfp_diag = np.arange(len(lfp_diag)) * dt_sec
ax.plot(t_lfp_diag[:5000], phase_10[:5000], 'k-', lw=0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Phase (rad)')
ax.set_title('Hilbert phase at 10 Hz (was NaN with b,a)')
ax.set_ylim(-np.pi - 0.5, np.pi + 0.5)
ax.axhline(np.pi, color='gray', ls='--', alpha=0.3)
ax.axhline(-np.pi, color='gray', ls='--', alpha=0.3)

# 4f. Hilbert phase quality at 40 Hz
ax = axes_diag[1, 2]
lfp_bp_40 = bandpass(lfp_diag, fs, 40 - BW_HALF, 40 + BW_HALF)
phase_40 = np.angle(hilbert(lfp_bp_40))
ax.plot(t_lfp_diag[:5000], phase_40[:5000], 'k-', lw=0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Phase (rad)')
ax.set_title('Hilbert phase at 40 Hz (SOS)')
ax.set_ylim(-np.pi - 0.5, np.pi + 0.5)
ax.axhline(np.pi, color='gray', ls='--', alpha=0.3)
ax.axhline(-np.pi, color='gray', ls='--', alpha=0.3)

plt.suptitle('FILTER DIAGNOSTICS — SOS fix verification', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('filter_diagnostics.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: filter_diagnostics.png")
plt.show()

print("\nFilter diagnostics complete.")