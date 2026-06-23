"""Generate smoothed, split-band PPC (spike-LFP phase locking) plots per layer.

Two plots per layer:
  - low band:  0-20 Hz
  - high band: 40-100 Hz
with independent y-scales, smoothed curves and custom colors.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfiltfilt
from scipy.ndimage import gaussian_filter1d
from brian2 import second
from config.config import CONFIG

# ---- analysis params (match ppc_computation.py) ----
T_DISCARD = 2.0
T_ANALYSIS = 10.0
MIN_SPIKES = 5
BW_HALF = 5.0
FREQ_RANGE = (2, 100)   # compute over full range, slice for plotting
FREQ_STEP = 1.0         # finer step -> smoother curves
MAX_E_NEURONS = 1000

# ---- custom colors ----
COLORS = {
    'E':   '#2E8B57',
    'PV':  '#C0392B',
    'SOM': '#1F4E96',
    'VIP': '#D4A017',
}

SMOOTH_SIGMA = 1.5  # in units of FREQ_STEP samples


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


def get_spike_phases_per_neuron(spike_trains, t_lfp, inst_phase, t_discard, t_end):
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


def compute_ppc_spectrum(spike_trains, lfp_normalized, t_lfp, fs, t_discard, t_end,
                         freq_range, freq_step, bw, min_spikes, max_neurons=None):
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
        lfp_bp = bandpass(lfp_normalized, fs, fc - bw, fc + bw)
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


def _cell_type_from_key(key):
    key_upper = key.upper()
    for ct in ['VIP', 'SOM', 'PV']:
        if ct in key_upper:
            return ct
    if 'E' in key.split('_')[0].upper() and len(key.split('_')[0]) <= 2:
        return 'E'
    return key.split('_')[0]


def smooth_nan(y, sigma):
    """Gaussian smooth ignoring NaNs."""
    y = np.asarray(y, dtype=float)
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return y
    filled = np.interp(np.arange(len(y)), np.where(valid)[0], y[valid])
    sm = gaussian_filter1d(filled, sigma)
    out = sm.copy()
    out[~valid] = np.nan  # keep gaps where there was no data at all
    return out


# ============ load data ============
fname = "results/trial_12_06_12s_stimuli/trial_000.npz"
data = np.load(fname, allow_pickle=True)
spike_data = (data["spike_data"].item() if data["spike_data"].size == 1
              else data["spike_data"])
lfp_full_data = (data["lfp_full"].item() if data["lfp_full"].size == 1
                 else data["lfp_full"])

dt_sec = float(CONFIG['simulation']['DT'] / second)
sim_duration_ms = float(data["time_array_ms"][-1])
fs = 1.0 / dt_sec
print(f"dt = {dt_sec*1000} ms, fs = {fs:.0f} Hz, duration = {sim_duration_ms:.0f} ms")

all_spike_trains = {}
for layer_name, layer_mons in spike_data.items():
    all_spike_trains[layer_name] = {}
    for mon_name, sd in layer_mons.items():
        all_spike_trains[layer_name][mon_name] = build_spike_trains(
            sd["times_ms"], sd["spike_indices"]
        )

# ============ compute PPC spectra ============
results = {}
for layer_name in all_spike_trains:
    results[layer_name] = {}
    lfp_full = lfp_full_data[layer_name]
    t_full = np.arange(len(lfp_full)) * dt_sec
    t_end = T_DISCARD + T_ANALYSIS
    mask_analysis = (t_full >= T_DISCARD) & (t_full <= t_end)
    lfp = lfp_full[mask_analysis]
    t_lfp = t_full[mask_analysis]
    lfp_normalized = normalize_lfp(lfp, dt_sec)

    for key, trains in all_spike_trains[layer_name].items():
        ct = _cell_type_from_key(key)
        max_n = MAX_E_NEURONS if ct == 'E' else None
        freqs, ppc_mean, ppc_sem = compute_ppc_spectrum(
            trains, lfp_normalized, t_lfp, fs,
            t_discard=T_DISCARD, t_end=t_end,
            freq_range=FREQ_RANGE, freq_step=FREQ_STEP,
            bw=BW_HALF, min_spikes=MIN_SPIKES, max_neurons=max_n,
        )
        results[layer_name][ct] = (freqs, ppc_mean, ppc_sem)
    print(f"  done {layer_name}")

# ============ plotting ============
BANDS = {
    'low':  (0, 20),
    'high': (40, 100),
}
layers = list(results.keys())

for band_label, (flo, fhi) in BANDS.items():
    for layer in layers:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ymax = 0.0
        for ct in ['E', 'PV', 'SOM', 'VIP']:
            if ct not in results[layer]:
                continue
            freqs, ppc_mean, ppc_sem = results[layer][ct]
            band_mask = (freqs >= flo) & (freqs <= fhi)
            f_b = freqs[band_mask]
            m_b = smooth_nan(ppc_mean[band_mask], SMOOTH_SIGMA)
            s_b = smooth_nan(ppc_sem[band_mask], SMOOTH_SIGMA)
            valid = ~np.isnan(m_b)
            if not np.any(valid):
                continue
            ax.plot(f_b[valid], m_b[valid], color=COLORS[ct],
                    linewidth=2.2, label=ct)
            ax.fill_between(f_b[valid], m_b[valid] - s_b[valid],
                            m_b[valid] + s_b[valid],
                            color=COLORS[ct], alpha=0.15)
            ymax = max(ymax, np.nanmax(m_b[valid] + s_b[valid]))

        ax.set_xlim(flo, fhi)
        ax.set_ylim(0, ymax * 1.08 if ymax > 0 else 1)
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Spike-LFP phase locking (PPC)', fontsize=12)
        ax.set_title(f'{layer}  —  {flo}–{fhi} Hz', fontsize=13,
                     fontstyle='italic')
        ax.legend(fontsize=11, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        out = f'ppc_{layer}_{band_label}_{flo}-{fhi}Hz.png'
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved {out}")

print("done")
