import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfiltfilt
from brian2 import second

from config_farzin.config2 import CONFIG

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


fname = "results/model_with_stim_different_tau_E_without_delays/trial_000.npz"
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


all_spike_trains = {}
for layer_name, layer_mons in spike_data.items():
    all_spike_trains[layer_name] = {}
    for mon_name, sd in layer_mons.items():
        all_spike_trains[layer_name][mon_name] = build_spike_trains(
            sd["times_ms"], sd["spike_indices"]
        )

results = {}
electrodes = {'L23': 10, 'L4AB':8, 'L4C':6, 'L5':4, 'L6':3 }
for layer_name in all_spike_trains:
    results[layer_name] = {}
    electrode = electrodes[layer_name]

    lfp_full = lfp_full_data[layer_name]
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

    e_key = [k for k in spike_data[layer_name] if k.upper().startswith('E')][0]
    n_e = len(np.unique(spike_data[layer_name][e_key]['spike_indices']))

    for key, trains in all_spike_trains[layer_name].items():
        ct = _cell_type_from_key(key)
        n_neurons = len(trains)

        max_n = MAX_E_NEURONS if ct == 'E' else None



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

    results[layer_name]['phase_results'] = phase_results
    results[layer_name]['f_peak'] = f_peak
    results[layer_name]['freqs_fft'] = freqs_fft
    results[layer_name]['fft_mag'] = fft_mag

    if 'PV' in phase_results and 'SOM' in phase_results:
        phase_delay = np.angle(np.exp(1j * (
            phase_results['SOM']['mean_phase'] -
            phase_results['PV']['mean_phase']
        )))
        time_delay_ms = (phase_delay / (2 * np.pi)) * (1000.0 / f_peak)
   


colors = {'E': 'green', 'PV': 'red', 'SOM': 'blue', 'VIP': '#ff7f0e'}
layers = list(results.keys())
n_layers = len(layers)

n_cols = 3
n_rows = int(np.ceil(n_layers / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

for i, layer in enumerate(layers):
    ax = axes[i // n_cols, i % n_cols]
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
    ax.set_ylabel('PPC', fontsize=10)
    ax.set_xlim(FREQ_RANGE)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.set_title(f'{layer}', fontsize=11, fontstyle='italic')
    ax.set_xlabel('Frequency (Hz)', fontsize=10)

for j in range(n_layers, n_rows * n_cols):
    axes[j // n_cols, j % n_cols].set_visible(False)

plt.tight_layout()

plt.show()


for layer in layers:
    phase_res = results[layer].get('phase_results', {})
    f_peak = results[layer].get('f_peak', None)
    if not phase_res or f_peak is None:
        continue

    fig_ph, ax_ph = plt.subplots(figsize=(7, 4))
    bins_phase = np.linspace(-np.pi, np.pi, 37)
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
    plt.show()


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
    plt.show()

