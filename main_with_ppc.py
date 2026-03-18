import numpy as np
import brian2 as b2
from brian2 import *
from brian2tools import *
from config.config2 import CONFIG
from src.column import CorticalColumn
from src.visualization import *
from src.analysis import *
from scipy.signal import hilbert, butter, filtfilt, welch



COLORS = {'E': 'green', 'PV': 'red', 'SOM': 'blue', 'VIP': 'orange'}
MAX_NEURONS = {'E': 500, 'PV': None, 'SOM': None, 'VIP': None}


def normalize_lfp(lfp, dt_sim):
    lfp_dc = lfp - np.mean(lfp)
    norm = np.sqrt(np.sum(lfp_dc**2) * dt_sim)
    if norm == 0:
        return lfp_dc
    return lfp_dc / norm


def find_peak_frequency(lfp_normalized, dt_sim, freq_range=(5, 80)):
    N_fft = len(lfp_normalized)
    freqs = np.fft.rfftfreq(N_fft, d=dt_sim)
    mag = np.abs(np.fft.rfft(lfp_normalized))
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not np.any(mask) or np.all(mag[mask] == 0):
        return np.nan, freqs, mag
    peak_idx = np.argmax(mag[mask])
    return freqs[mask][peak_idx], freqs, mag


def get_spike_phases_per_neuron(spike_monitor, neuron_indices, t_lfp, inst_phase,
                                 t_discard, t_end):
    spike_times = np.array(spike_monitor.t / second)
    spike_ids = np.array(spike_monitor.i)
    window_mask = (spike_times >= t_discard) & (spike_times <= t_end)
    spike_times_w = spike_times[window_mask]
    spike_ids_w = spike_ids[window_mask]

    neuron_phases = {}
    for nid in neuron_indices:
        nid_mask = spike_ids_w == nid
        st = spike_times_w[nid_mask]
        if len(st) < 2:
            continue
        indices = np.searchsorted(t_lfp, st)
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


def compute_ppc_spectrum(spike_monitor, neuron_indices, lfp_signal, t_lfp, fs,
                          t_discard, t_end,
                          freq_range=(5, 90), freq_step=2.0, bw=5.0,
                          min_spikes=5, max_neurons=None):
    if max_neurons is not None and len(neuron_indices) > max_neurons:
        neuron_indices = np.random.choice(neuron_indices, max_neurons, replace=False)

    freqs = np.arange(freq_range[0], freq_range[1] + freq_step, freq_step)
    ppc_mean = np.full(len(freqs), np.nan)
    ppc_sem = np.full(len(freqs), np.nan)
    nyq = fs / 2.0

    for fi, fc in enumerate(freqs):
        f_lo = max(fc - bw, 1.0)
        f_hi = min(fc + bw, nyq - 1.0)
        if f_hi <= f_lo:
            continue
        b_bp, a_bp = butter(4, [f_lo / nyq, f_hi / nyq], btype='band')
        lfp_bp = filtfilt(b_bp, a_bp, lfp_signal)
        phase_signal = np.angle(hilbert(lfp_bp))

        neuron_phases = get_spike_phases_per_neuron(
            spike_monitor, neuron_indices, t_lfp, phase_signal, t_discard, t_end
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


################################################################################
# SIMULATION
################################################################################

def main():
    np.random.seed(CONFIG['simulation']['RANDOM_SEED'])
    b2.start_scope()
    b2.defaultclock.dt = CONFIG['simulation']['DT']

    # -------------------------------------------------------------------
    # Timing: need enough time for PPC analysis after discarding transient
    # Paper uses 2s transient + 10s analysis = 12s minimum
    # Adjust these as needed:
    # -------------------------------------------------------------------
    baseline_time = 12000   # ms — 2s transient + 10s analysis window
    stimuli_time = 0        # ms — set >0 if you want stimulus epoch too

    print("Creating cortical column...")
    column = CorticalColumn(column_id=0, config=CONFIG)

    all_monitors = column.get_all_monitors()

    w_ext_AMPA = CONFIG['synapses']['Q']['EXT_AMPA']

    # -------------------------------------------------------------------
    # Run baseline
    # -------------------------------------------------------------------
    column.network.run(baseline_time * ms)

    # -------------------------------------------------------------------
    # Optional: add stimuli and run stimulus epoch
    # -------------------------------------------------------------------
    if stimuli_time > 0:
        L4C = column.layers['L4C']
        cfg_L4C = CONFIG['layers']['L4C']

        L4C_E_grp = L4C.neuron_groups['E']
        L4C_E_stimAMPA = PoissonInput(L4C_E_grp, 'gE_AMPA',
                                       N=40, rate=40*Hz, weight=w_ext_AMPA)

        L4C_PV_grp = L4C.neuron_groups['PV']
        L4C_PV_stim = PoissonInput(L4C_PV_grp, 'gE_AMPA',
                                    N=30, rate=40*Hz, weight=w_ext_AMPA*2)

        L6 = column.layers['L6']
        cfg_L6 = CONFIG['layers']['L6']
        L6_PV_grp = L6.neuron_groups['PV']
        N_stim_L6_PV = int(cfg_L6['poisson_inputs']['PV']['N'])
        L6_PV_stim = PoissonInput(L6_PV_grp, 'gE_AMPA',
                                   N=N_stim_L6_PV, rate=15*Hz, weight=w_ext_AMPA*3)
        L6_E_grp = L6.neuron_groups['E']
        N_stim_L6_E = int(cfg_L6['poisson_inputs']['E']['N'])
        L6_E_stim = PoissonInput(L6_E_grp, 'gE_AMPA',
                                  N=N_stim_L6_E, rate=15*Hz, weight=w_ext_AMPA)

        column.network.add(L6_E_stim, L6_PV_stim)
        column.network.add(L4C_E_stimAMPA, L4C_PV_stim)

        column.network.run(stimuli_time * ms)

    print("Simulation complete")

    # -------------------------------------------------------------------
    # Extract monitors
    # -------------------------------------------------------------------
    spike_monitors = {}
    rate_monitors = {}
    neuron_groups = {}

    for layer_name, monitors in all_monitors.items():
        spike_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'spikes' in k
        }
        rate_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'rate' in k
        }
        neuron_groups[layer_name] = column.layers[layer_name].neuron_groups

    ############################################################################
    # PPC ANALYSIS — per layer, exactly following Tahvili et al. (2025)
    ############################################################################

    dt_sim = 0.1e-3          # 0.1 ms
    fs = 1.0 / dt_sim        # 10000 Hz
    t_discard = 2.0           # seconds of transient to discard
    total_sim_time = (baseline_time + stimuli_time) / 1000.0  # in seconds
    t_end = total_sim_time

    freq_range = (5, 90)
    freq_step = 2.0
    bw = 5.0
    peak_search_range = (5, 80)

    results = {}

    for layer_name in spike_monitors.keys():
        print(f"\n{'='*70}")
        print(f"  PPC ANALYSIS — {layer_name}")
        print(f"{'='*70}")

        layer_spike_mons = spike_monitors[layer_name]
        layer_rate_mons = rate_monitors[layer_name]
        layer_neuron_grps = neuron_groups[layer_name]

        # Cell types present
        cell_types = [k.replace('_spikes', '') for k in layer_spike_mons if '_spikes' in k]
        n_counts = CONFIG['layers'][layer_name].get('neuron_counts', {})
        print(f"  Cell types: {cell_types}")
        print(f"  Neuron counts: {n_counts}")

        if 'E_rate' not in layer_rate_mons:
            print(f"  WARNING: No E_rate monitor, skipping.")
            continue

        # ==================================================================
        # Step 1: LFP proxy from E population rate
        # ==================================================================
        lfp_full = np.array(layer_rate_mons['E_rate'].smooth_rate(
            window='gaussian', width=1*ms) / Hz)
        t_full = np.array(layer_rate_mons['E_rate'].t / second)

        mask = (t_full >= t_discard) & (t_full <= t_end)
        lfp = lfp_full[mask]
        t_lfp = t_full[mask]
        t_analysis = t_lfp[-1] - t_lfp[0]
        print(f"  LFP window: {t_lfp[0]:.3f}–{t_lfp[-1]:.3f} s "
              f"({t_analysis:.3f} s, {len(lfp)} samples)")

        # ==================================================================
        # Step 2: Normalize LFP
        # ==================================================================
        lfp_norm = normalize_lfp(lfp, dt_sim)

        # ==================================================================
        # Step 3: Find peak frequency
        # ==================================================================
        f_peak, freqs_fft, fft_mag = find_peak_frequency(
            lfp_norm, dt_sim, freq_range=peak_search_range)
        print(f"  Peak oscillation frequency: {f_peak:.1f} Hz")

        # ==================================================================
        # Step 4: PPC spectrum for each cell type
        # ==================================================================
        ppc_results = {}
        for ct in cell_types:
            spk_mon = layer_spike_mons[f'{ct}_spikes']
            n_neurons = n_counts.get(ct, 0)
            if n_neurons == 0 and ct in layer_neuron_grps:
                n_neurons = len(layer_neuron_grps[ct])
            max_n = MAX_NEURONS.get(ct, None)
            indices = np.arange(n_neurons)

            print(f"  Computing PPC for {ct} ({n_neurons} neurons, "
                  f"subsample={max_n})...")

            freqs_ppc, ppc_mean, ppc_sem = compute_ppc_spectrum(
                spk_mon, indices, lfp_norm, t_lfp, fs, t_discard, t_end,
                freq_range=freq_range, freq_step=freq_step, bw=bw,
                min_spikes=5, max_neurons=max_n)
            ppc_results[ct] = (ppc_mean, ppc_sem)

        # ==================================================================
        # Step 5: Phase analysis at peak frequency
        # ==================================================================
        phase_results = {}
        mean_phases = {}

        if not np.isnan(f_peak):
            print(f"  Computing spike phases at {f_peak:.1f} Hz...")
            bw_phase = 5.0
            f_lo = max(f_peak - bw_phase, 1.0)
            f_hi = min(f_peak + bw_phase, fs/2 - 1.0)
            nyq = fs / 2.0
            b_gam, a_gam = butter(4, [f_lo/nyq, f_hi/nyq], btype='band')
            lfp_filtered = filtfilt(b_gam, a_gam, lfp_norm)
            gamma_phase = np.angle(hilbert(lfp_filtered))

            for ct in cell_types:
                spk_mon = layer_spike_mons[f'{ct}_spikes']
                n_neurons = n_counts.get(ct, 0)
                if n_neurons == 0 and ct in layer_neuron_grps:
                    n_neurons = len(layer_neuron_grps[ct])

                neuron_ph = get_spike_phases_per_neuron(
                    spk_mon, np.arange(n_neurons), t_lfp, gamma_phase,
                    t_discard, t_end)

                all_phases = np.concatenate(
                    [ph for ph in neuron_ph.values() if len(ph) >= 2]
                ) if len(neuron_ph) > 0 else np.array([])

                phase_results[ct] = all_phases
                mean_phases[ct] = circular_mean(all_phases) if len(all_phases) > 0 else np.nan

            # Print phase results
            print(f"\n  --- Phase Results ({layer_name}) ---")
            for ct in cell_types:
                if not np.isnan(mean_phases.get(ct, np.nan)):
                    print(f"    {ct:>3s}: mean phase = "
                          f"{np.degrees(mean_phases[ct]):+7.1f} deg  "
                          f"(N_spikes = {len(phase_results[ct])})")

            if 'PV' in mean_phases and 'SOM' in mean_phases:
                if not np.isnan(mean_phases['PV']) and not np.isnan(mean_phases['SOM']):
                    delay_rad = np.angle(np.exp(1j*(mean_phases['SOM'] - mean_phases['PV'])))
                    delay_ms = (delay_rad / (2*np.pi)) * (1000.0 / f_peak)
                    print(f"    SOM-PV delay: {np.degrees(delay_rad):+.1f} deg "
                          f"({delay_ms:+.1f} ms)")
        else:
            print(f"  WARNING: No peak found, skipping phase analysis.")

        # ==================================================================
        # Step 6: 25–45 Hz band power
        # ==================================================================
        df_freq = freqs_fft[1] - freqs_fft[0] if len(freqs_fft) > 1 else 1.0
        total_power = np.sum(fft_mag**2) * df_freq
        band_mask = (freqs_fft >= 25) & (freqs_fft <= 45)
        gamma_power = np.sum(fft_mag[band_mask]**2) * df_freq
        rel_gamma = gamma_power / total_power if total_power > 0 else 0
        print(f"  25-45 Hz relative band power: {rel_gamma:.4f}")

        results[layer_name] = {
            'f_peak': f_peak,
            'freqs_ppc': freqs_ppc,
            'ppc': ppc_results,
            'phases': phase_results,
            'mean_phase': mean_phases,
            'cell_types': cell_types,
            'lfp_normalized': lfp_norm,
            't_lfp': t_lfp,
            'freqs_fft': freqs_fft,
            'fft_mag': fft_mag,
            'gamma_band_power': rel_gamma,
        }

    ############################################################################
    # PLOTS
    ############################################################################

    layer_names = list(results.keys())
    n_layers = len(layer_names)

    # --- Fig 1C per layer: PPC spectrum + phase distribution ---
    fig, axes = plt.subplots(n_layers, 2, figsize=(14, 5*n_layers), squeeze=False)

    for li, layer_name in enumerate(layer_names):
        res = results[layer_name]
        freqs_ppc = res['freqs_ppc']
        f_peak = res['f_peak']
        cell_types = res['cell_types']

        # Left: PPC spectrum
        ax1 = axes[li, 0]
        for ct in cell_types:
            if ct in res['ppc']:
                ppc_m, ppc_s = res['ppc'][ct]
                c = COLORS.get(ct, 'gray')
                ax1.plot(freqs_ppc, ppc_m, color=c, linewidth=2, label=ct)
                ax1.fill_between(freqs_ppc, ppc_m - ppc_s, ppc_m + ppc_s,
                                  color=c, alpha=0.15)
        ax1.set_xlabel('Frequency (Hz)', fontsize=12)
        ax1.set_ylabel('Spike-LFP locking (PPC)', fontsize=12)
        ax1.set_xlim(5, 90)
        ax1.set_ylim(bottom=0)
        ax1.legend(fontsize=10)
        ax1.set_title(f'{layer_name} — Spike-LFP locking', fontsize=13,
                       fontstyle='italic')

        # Right: Phase distribution
        ax2 = axes[li, 1]
        bins_phase = np.linspace(-np.pi, np.pi, 37)
        bin_centers = 0.5 * (bins_phase[:-1] + bins_phase[1:])

        for ct in ['PV', 'SOM', 'VIP']:
            if ct in res['phases'] and len(res['phases'][ct]) > 0:
                hist, _ = np.histogram(res['phases'][ct], bins=bins_phase, density=True)
                c = COLORS.get(ct, 'gray')
                ax2.plot(np.degrees(bin_centers), hist, color=c, linewidth=2, label=ct)
                ax2.fill_between(np.degrees(bin_centers), 0, hist, color=c, alpha=0.2)

        ax2.set_xlabel('Preferred spike gamma-phase', fontsize=12)
        ax2.set_ylabel('Probability density', fontsize=12)
        ax2.set_xlim(-180, 180)
        ax2.set_xticks([-180, 0, 180])
        ax2.set_xticklabels(['-180°', '0°', '180°'])
        ax2.legend(fontsize=10)
        peak_str = f'{f_peak:.0f}' if not np.isnan(f_peak) else '?'
        ax2.set_title(f'{layer_name} — Phase distribution at {peak_str} Hz',
                       fontsize=13, fontstyle='italic')

    plt.tight_layout()
    plt.savefig("ppc_analysis_all_layers.png", dpi=150, bbox_inches='tight')

    # --- LFP spectra per layer ---
    fig2, axes2 = plt.subplots(n_layers, 1, figsize=(10, 3.5*n_layers), squeeze=False)

    for li, layer_name in enumerate(layer_names):
        res = results[layer_name]
        freqs = res['freqs_fft']
        mag = res['fft_mag']
        f_peak = res['f_peak']

        fmask = (freqs >= 1) & (freqs <= 100)
        ax = axes2[li, 0]
        ax.plot(freqs[fmask], mag[fmask], 'k-', linewidth=1.5)
        ax.set_xlim(1, 100)
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('|F[x̂(t)]|', fontsize=12)
        if not np.isnan(f_peak):
            ax.axvline(f_peak, color='red', linestyle='--', alpha=0.7,
                        label=f'Peak: {f_peak:.1f} Hz')
            ax.legend(fontsize=10)
        ax.set_title(f'{layer_name} — Normalized LFP spectrum '
                      f'(γ-power: {res["gamma_band_power"]:.4f})', fontsize=13)

    plt.tight_layout()
    plt.savefig("lfp_spectra_all_layers.png", dpi=150, bbox_inches='tight')

    # --- Raster plot ---
    fig_raster = plot_raster(spike_monitors, baseline_time, stimuli_time, CONFIG['layers'])

    # --- Rate plot ---
    fig_rate = plot_rate(rate_monitors, CONFIG['layers'], baseline_time, stimuli_time,
                         smooth_window=15*ms, ylim_max=80, show_stats=True)

    plt.show()
    print("\nAll done.")


if __name__ == "__main__":
    main()
