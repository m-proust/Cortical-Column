"""
diagnose_lowfreq_v2.py — Deep diagnostic: what drives pathological
low-frequency power increase in bad trials?

Goes beyond firing rates to examine:
  - Population synchrony structure (Fano factor, pairwise correlations)
  - Cross-layer spike timing (lag correlations)
  - Inhibitory subtype balance (SOM/PV ratio)
  - Oscillatory coherence between layers
  - Pre-stimulus LFP phase at stimulus onset
  - Time-resolved divergence of low-freq power
  - Per-channel laminar profile of the effect
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, detrend, hilbert, coherence, butter, filtfilt
from scipy.stats import mannwhitneyu, pearsonr
from scipy.ndimage import gaussian_filter1d
import os
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
BASE_PATH = "results/trials3_06_04_3"
N_TRIALS = 100
FS = 10000

GOOD_TRIALS = [0,3,4,5,8,10,12,14,15,16,18,20,23,27,29,31,32,35,36,38,39,40,
               50,54,56,59,60,69,70,77,83,91,94,95,97,98]
BAD_TRIALS = [i for i in range(N_TRIALS) if i not in GOOD_TRIALS]

SAVE_DIR = "results/trial_diagnostics_v2"
os.makedirs(SAVE_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════════════════
all_data = []
for i in range(N_TRIALS):
    fname = os.path.join(BASE_PATH, f"trial_{i:03d}.npz")
    d = np.load(fname, allow_pickle=True)
    trial = {
        'trial_id': int(d['trial_id']),
        'stim_rates': d['stim_rates'].item() if d['stim_rates'].size == 1 else dict(d['stim_rates']),
        'time': d['time_array_ms'],
        'bipolar_lfp': d['bipolar_matrix'],
        'lfp_matrix': d['lfp_matrix'],
        'rate_data': d['rate_data'].item() if d['rate_data'].size == 1 else d['rate_data'],
        'spike_data': d['spike_data'].item() if d['spike_data'].size == 1 else d['spike_data'],
        'baseline_ms': float(d['baseline_ms']),
        'stim_onset_ms': float(d['stim_onset_ms']),
        'channel_depths': d['channel_depths'],
        'electrode_positions': d['electrode_positions'],
    }
    all_data.append(trial)

stim_onset = all_data[0]['stim_onset_ms']
time = all_data[0]['time']
n_channels = all_data[0]['bipolar_lfp'].shape[0]

print(f"Loaded {N_TRIALS} trials")
print(f"Good: {len(GOOD_TRIALS)}, Bad: {len(BAD_TRIALS)}")

# ═══════════════════════════════════════════════════════════════
# STEP 1: PROPERLY COMPUTE LOW-FREQ CHANGE PER TRIAL
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 1: PER-TRIAL LOW-FREQ POWER CHANGE")
print("="*70)

# ── Spectral analysis parameters ──
# Use 1000ms windows to get 1 Hz frequency resolution at 10 kHz.
# At 10 kHz, 1000ms = 10000 samples → freq resolution = 10000/10000 = 1 Hz.
# This gives ~19 bins in the 1-20 Hz range — plenty for reliable integration.
PRE_WINDOW = 1000   # ms (was 300 — too short for low-freq!)
POST_OFFSET = 200   # ms after stim onset
POST_WINDOW = 1000  # ms (was 300)
LOW_FREQ = (1, 20)
HIGH_FREQ = (20, 80)

# Use np.trapezoid if available (numpy >= 2.0), else fall back to np.trapz
try:
    _integrate = np.trapezoid
except AttributeError:
    _integrate = np.trapz

def compute_spectral_changes(trial, lfp_key='bipolar_lfp'):
    t = trial['time']
    stim = trial['stim_onset_ms']
    n_ch = trial[lfp_key].shape[0]

    pre_mask = (t >= stim - PRE_WINDOW) & (t < stim)
    post_mask = (t >= stim + POST_OFFSET) & (t < stim + POST_OFFSET + POST_WINDOW)

    low_changes = []
    high_changes = []

    for ch in range(n_ch):
        pre_seg = trial[lfp_key][ch][pre_mask].copy()
        post_seg = trial[lfp_key][ch][post_mask].copy()
        if len(pre_seg) < 1000 or len(post_seg) < 1000:
            continue

        # Remove mean (don't use scipy detrend — it can overflow with large arrays)
        pre_seg -= np.mean(pre_seg)
        post_seg -= np.mean(post_seg)

        # Use the FULL segment as nperseg for maximum frequency resolution.
        # At 10 kHz with 10000 samples, this gives 1 Hz resolution.
        nperseg_val = len(pre_seg)
        f, psd_pre = welch(pre_seg, fs=FS, nperseg=nperseg_val, window='hann')
        _, psd_post = welch(post_seg, fs=FS, nperseg=nperseg_val, window='hann')

        for band, out_list in [(LOW_FREQ, low_changes), (HIGH_FREQ, high_changes)]:
            m = (f >= band[0]) & (f <= band[1])
            if m.sum() < 2:  # need at least 2 points for integration
                continue
            pre_pow = _integrate(psd_pre[m], f[m])
            post_pow = _integrate(psd_post[m], f[m])
            out_list.append((post_pow - pre_pow) / (pre_pow + 1e-20) * 100)

    return {
        'low_change_per_ch': np.array(low_changes),
        'high_change_per_ch': np.array(high_changes),
        'mean_low_change': np.mean(low_changes) if low_changes else np.nan,
        'mean_high_change': np.mean(high_changes) if high_changes else np.nan,
    }

trial_spectral = [compute_spectral_changes(all_data[i]) for i in range(N_TRIALS)]
low_changes = np.array([ts['mean_low_change'] for ts in trial_spectral])
high_changes = np.array([ts['mean_high_change'] for ts in trial_spectral])

good_mask = np.array([i in GOOD_TRIALS for i in range(N_TRIALS)])
bad_mask = ~good_mask

print(f"\n  Low-freq change (<20 Hz):")
print(f"    Good: {low_changes[good_mask].mean():.1f}% +/- {low_changes[good_mask].std():.1f}%")
print(f"    Bad:  {low_changes[bad_mask].mean():.1f}% +/- {low_changes[bad_mask].std():.1f}%")
_, p = mannwhitneyu(low_changes[good_mask], low_changes[bad_mask])
print(f"    p = {p:.6f}")

n_dec_good = np.sum(low_changes[good_mask] < 0)
n_inc_good = np.sum(low_changes[good_mask] > 0)
n_dec_bad = np.sum(low_changes[bad_mask] < 0)
n_inc_bad = np.sum(low_changes[bad_mask] > 0)
print(f"\n  Low-freq DECREASE: Good={n_dec_good}/{len(GOOD_TRIALS)}  Bad={n_dec_bad}/{len(BAD_TRIALS)}")
print(f"  Low-freq INCREASE: Good={n_inc_good}/{len(GOOD_TRIALS)}  Bad={n_inc_bad}/{len(BAD_TRIALS)}")

# ═══════════════════════════════════════════════════════════════
# STEP 2: LOW-FREQ CHANGE HISTOGRAMS
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.hist(low_changes[good_mask], bins=20, alpha=0.6, color='steelblue', label='Good')
ax.hist(low_changes[bad_mask], bins=20, alpha=0.6, color='salmon', label='Bad')
ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('Low-freq power change (%)')
ax.set_title('Low-frequency (<20 Hz) change')
ax.legend()

ax = axes[0, 1]
ax.hist(high_changes[good_mask], bins=20, alpha=0.6, color='steelblue', label='Good')
ax.hist(high_changes[bad_mask], bins=20, alpha=0.6, color='salmon', label='Bad')
ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('High-freq power change (%)')
ax.set_title('High-frequency (20-80 Hz) change')
ax.legend()

ax = axes[1, 0]
ax.scatter(low_changes[good_mask], high_changes[good_mask], c='steelblue', alpha=0.6, s=30, label='Good')
ax.scatter(low_changes[bad_mask], high_changes[bad_mask], c='salmon', alpha=0.6, s=30, label='Bad')
ax.axvline(x=0, color='gray', linestyle='--'); ax.axhline(y=0, color='gray', linestyle='--')
ax.set_xlabel('Low-freq change (%)'); ax.set_ylabel('High-freq change (%)')
ax.set_title('Low vs High'); ax.legend()

ax = axes[1, 1]
colors = ['steelblue' if i in GOOD_TRIALS else 'salmon' for i in range(N_TRIALS)]
ax.bar(range(N_TRIALS), low_changes, color=colors, alpha=0.7)
ax.axhline(y=0, color='black', linewidth=2)
ax.set_xlabel('Trial ID'); ax.set_ylabel('Low-freq change (%)')
ax.set_title('Per-trial low-freq change')

plt.suptitle('Low-Frequency Power Change: The Key Discriminator', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "01_lowfreq_change.png"), dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════
# STEP 3: COLLECT ALL FEATURES (rates + beyond rates)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 3: COLLECTING FEATURES (rates + beyond rates)")
print("="*70)

features = {}

# ── A. Stimulus rates ──
for k in ['L4C_E', 'L4C_PV', 'L6_E', 'L6_PV']:
    features['stim_rate_' + k] = [all_data[i]['stim_rates'][k] for i in range(N_TRIALS)]

# ── B. Firing rates at various time windows ──
def get_firing_rates(spike_data, t_start, t_end):
    rates = {}
    for layer_name, layer_spikes in spike_data.items():
        for mon_name, mon in layer_spikes.items():
            times = mon['times_ms']
            indices = mon['spike_indices']
            mask = (times >= t_start) & (times < t_end)
            n_neurons = max(indices.max() + 1, 1) if len(indices) > 0 else 1
            rate = mask.sum() / (n_neurons * (t_end - t_start) / 1000.0)
            rates[layer_name + '_' + mon_name.replace('_spikes', '')] = rate
    return rates

for label, ts, te in [
    ('pre500',  stim_onset - 500, stim_onset),
    ('pre100',  stim_onset - 100, stim_onset),
    ('early',   stim_onset, stim_onset + 200),
    ('sust',    stim_onset + 200, stim_onset + 500),
]:
    for i in range(N_TRIALS):
        r = get_firing_rates(all_data[i]['spike_data'], ts, te)
        for k, v in r.items():
            features.setdefault(label + '_' + k, []).append(v)

# ── C. POPULATION SYNCHRONY (Fano factor per layer, per time window) ──
print("  Computing population synchrony (Fano factors)...")

def fano_factor(spike_data, t_start, t_end, layer_filter=None, pop_filter=None, bin_ms=5):
    """Fano factor of population spike count in bins. >1 = synchronous."""
    all_times = []
    for layer_name, layer_spikes in spike_data.items():
        if layer_filter and layer_name not in layer_filter:
            continue
        for mon_name, mon in layer_spikes.items():
            if pop_filter and pop_filter not in mon_name:
                continue
            times = mon['times_ms']
            mask = (times >= t_start) & (times < t_end)
            all_times.extend(times[mask])
    if len(all_times) < 20:
        return np.nan
    bins = np.arange(t_start, t_end, bin_ms)
    counts, _ = np.histogram(all_times, bins=bins)
    return counts.var() / counts.mean() if counts.mean() > 0 else np.nan

layers_list = ['L4C', 'L23', 'L5', 'L6', 'L4AB']

for layer in layers_list + [None]:
    lf = [layer] if layer else None
    layer_label = layer if layer else 'all'
    for win_label, ts, te in [
        ('pre', stim_onset - 500, stim_onset),
        ('early', stim_onset, stim_onset + 200),
        ('sust', stim_onset + 200, stim_onset + 500),
    ]:
        vals = []
        for i in range(N_TRIALS):
            vals.append(fano_factor(all_data[i]['spike_data'], ts, te, lf))
        features['fano_' + layer_label + '_' + win_label] = vals

# Also per population type
for pop_type in ['E', 'PV', 'SOM', 'VIP']:
    for win_label, ts, te in [
        ('pre', stim_onset - 500, stim_onset),
        ('sust', stim_onset + 200, stim_onset + 500),
    ]:
        vals = []
        for i in range(N_TRIALS):
            vals.append(fano_factor(all_data[i]['spike_data'], ts, te, pop_filter=pop_type))
        features['fano_' + pop_type + '_' + win_label] = vals

# ── D. SPIKE COUNT VARIABILITY (CV of per-neuron spike counts) ──
print("  Computing spike count variability...")

def spike_count_cv(spike_data, t_start, t_end, layer_filter=None):
    """CV of spike counts across neurons — measures heterogeneity of activity."""
    counts = []
    for layer_name, layer_spikes in spike_data.items():
        if layer_filter and layer_name not in layer_filter:
            continue
        for mon_name, mon in layer_spikes.items():
            times = mon['times_ms']
            indices = mon['spike_indices']
            mask = (times >= t_start) & (times < t_end)
            i_in = indices[mask]
            if len(i_in) == 0:
                continue
            n_neurons = indices.max() + 1
            neuron_counts = np.bincount(i_in, minlength=n_neurons)
            counts.extend(neuron_counts)
    counts = np.array(counts, dtype=float)
    if len(counts) < 5 or counts.mean() == 0:
        return np.nan
    return counts.std() / counts.mean()

for layer in layers_list + [None]:
    lf = [layer] if layer else None
    layer_label = layer if layer else 'all'
    for win_label, ts, te in [
        ('pre', stim_onset - 500, stim_onset),
        ('sust', stim_onset + 200, stim_onset + 500),
    ]:
        vals = [spike_count_cv(all_data[i]['spike_data'], ts, te, lf) for i in range(N_TRIALS)]
        features['cv_counts_' + layer_label + '_' + win_label] = vals


# ── E. INHIBITORY SUBTYPE BALANCE ──
print("  Computing inhibitory subtype ratios...")

def inh_subtype_ratio(spike_data, t_start, t_end, layer_filter=None):
    """Compute PV/SOM and PV/VIP ratios — captures interneuron balance."""
    pv_count, som_count, vip_count, e_count = 0, 0, 0, 0
    for layer_name, layer_spikes in spike_data.items():
        if layer_filter and layer_name not in layer_filter:
            continue
        for mon_name, mon in layer_spikes.items():
            times = mon['times_ms']
            mask = (times >= t_start) & (times < t_end)
            n = mask.sum()
            if 'PV' in mon_name:
                pv_count += n
            elif 'SOM' in mon_name:
                som_count += n
            elif 'VIP' in mon_name:
                vip_count += n
            elif 'E' in mon_name:
                e_count += n
    pv_som = pv_count / (som_count + 1)
    pv_total_inh = pv_count / (pv_count + som_count + vip_count + 1)
    e_i = e_count / (pv_count + som_count + vip_count + 1)
    return pv_som, pv_total_inh, e_i

for layer in layers_list + [None]:
    lf = [layer] if layer else None
    layer_label = layer if layer else 'all'
    for win_label, ts, te in [
        ('pre', stim_onset - 500, stim_onset),
        ('early', stim_onset, stim_onset + 200),
        ('sust', stim_onset + 200, stim_onset + 500),
    ]:
        pv_som_vals, pv_frac_vals, ei_vals = [], [], []
        for i in range(N_TRIALS):
            ps, pf, ei = inh_subtype_ratio(all_data[i]['spike_data'], ts, te, lf)
            pv_som_vals.append(ps)
            pv_frac_vals.append(pf)
            ei_vals.append(ei)
        features['PVoverSOM_' + layer_label + '_' + win_label] = pv_som_vals
        features['PVfrac_' + layer_label + '_' + win_label] = pv_frac_vals
        features['EI_' + layer_label + '_' + win_label] = ei_vals


# ── F. CROSS-LAYER SPIKE TIMING ──
print("  Computing cross-layer spike timing correlations...")

def cross_layer_correlation(spike_data, layer_a, layer_b, t_start, t_end, bin_ms=5):
    """Cross-correlation at zero lag between population spike trains of two layers."""
    def get_binned(layer, t_start, t_end, bin_ms):
        all_t = []
        if layer in spike_data:
            for mon_name, mon in spike_data[layer].items():
                if 'E' not in mon_name:
                    continue
                times = mon['times_ms']
                mask = (times >= t_start) & (times < t_end)
                all_t.extend(times[mask])
        bins = np.arange(t_start, t_end, bin_ms)
        counts, _ = np.histogram(all_t, bins=bins)
        return counts.astype(float)

    a = get_binned(layer_a, t_start, t_end, bin_ms)
    b = get_binned(layer_b, t_start, t_end, bin_ms)
    if len(a) < 5 or a.std() == 0 or b.std() == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]

layer_pairs = [('L4C', 'L23'), ('L4C', 'L5'), ('L23', 'L5'), ('L5', 'L6'), ('L4C', 'L6')]

for la, lb in layer_pairs:
    for win_label, ts, te in [
        ('pre', stim_onset - 500, stim_onset),
        ('sust', stim_onset + 200, stim_onset + 500),
    ]:
        vals = [cross_layer_correlation(all_data[i]['spike_data'], la, lb, ts, te) for i in range(N_TRIALS)]
        features['xcorr_' + la + '_' + lb + '_' + win_label] = vals


# ── G. LFP-BASED FEATURES (beyond band power) ──
print("  Computing LFP-based features...")

mid_ch = n_channels // 2

# G1. Pre-stimulus LFP phase at stimulus onset (in low-freq band)
def lowfreq_phase_at_onset(trial, ch, fs=FS):
    """Phase of the low-freq (1-20 Hz) LFP oscillation at the moment of stimulus onset."""
    t = trial['time']
    # Use a wider window to get good phase estimate
    mask = (t >= stim_onset - 500) & (t < stim_onset + 100)
    seg = trial['bipolar_lfp'][ch][mask]
    if len(seg) < 200:
        return np.nan

    # Bandpass 1-20 Hz
    nyq = fs / 2
    low, high = 1 / nyq, min(20 / nyq, 0.99)
    if low >= high:
        return np.nan
    b_filt, a_filt = butter(3, [low, high], btype='band')
    filtered = filtfilt(b_filt, a_filt, seg)

    analytic = hilbert(filtered)
    phase = np.angle(analytic)

    # Phase at stim onset (500 samples from start of mask window)
    onset_idx = int(500 * fs / 1000)
    if onset_idx >= len(phase):
        onset_idx = len(phase) - 1
    return phase[onset_idx]

for ch_idx in [0, mid_ch, n_channels - 1]:
    vals = [lowfreq_phase_at_onset(all_data[i], ch_idx) for i in range(N_TRIALS)]
    features['phase_at_onset_ch' + str(ch_idx)] = vals
    # Also use cos/sin for circular-linear correlation
    features['cos_phase_ch' + str(ch_idx)] = [np.cos(v) if not np.isnan(v) else np.nan for v in vals]
    features['sin_phase_ch' + str(ch_idx)] = [np.sin(v) if not np.isnan(v) else np.nan for v in vals]

# G2. Pre-stimulus LFP amplitude at onset
def lowfreq_amplitude_at_onset(trial, ch, fs=FS):
    t = trial['time']
    mask = (t >= stim_onset - 500) & (t < stim_onset + 100)
    seg = trial['bipolar_lfp'][ch][mask]
    if len(seg) < 200:
        return np.nan
    nyq = fs / 2
    low, high = 1 / nyq, min(20 / nyq, 0.99)
    if low >= high:
        return np.nan
    b_filt, a_filt = butter(3, [low, high], btype='band')
    filtered = filtfilt(b_filt, a_filt, seg)
    analytic = hilbert(filtered)
    amp = np.abs(analytic)
    onset_idx = int(500 * fs / 1000)
    if onset_idx >= len(amp):
        onset_idx = len(amp) - 1
    return amp[onset_idx]

for ch_idx in [0, mid_ch, n_channels - 1]:
    vals = [lowfreq_amplitude_at_onset(all_data[i], ch_idx) for i in range(N_TRIALS)]
    features['lowfreq_amp_at_onset_ch' + str(ch_idx)] = vals

# G3. Cross-channel LFP coherence in low-freq band during stimulus
def lfp_coherence(trial, ch_a, ch_b, t_start, t_end, band=(1, 20)):
    t = trial['time']
    mask = (t >= t_start) & (t < t_end)
    a = trial['bipolar_lfp'][ch_a][mask].copy()
    b = trial['bipolar_lfp'][ch_b][mask].copy()
    a -= np.mean(a); b -= np.mean(b)
    if len(a) < 2000:
        return np.nan
    # Use full segment for freq resolution
    nperseg_val = len(a)
    f, coh = coherence(a, b, fs=FS, nperseg=nperseg_val)
    m = (f >= band[0]) & (f <= band[1])
    return np.mean(coh[m]) if m.sum() > 1 else np.nan

# Deep-superficial coherence — use wider windows for low-freq
if n_channels >= 3:
    deep_ch = 0
    sup_ch = n_channels - 1
    for win_label, ts, te in [
        ('pre', stim_onset - 1000, stim_onset),
        ('sust', stim_onset + 200, stim_onset + 1200),
    ]:
        for band_name, band_range in [('low', (1, 20)), ('gamma', (30, 80))]:
            vals = [lfp_coherence(all_data[i], deep_ch, sup_ch, ts, te, band_range)
                    for i in range(N_TRIALS)]
            features['coh_deep_sup_' + band_name + '_' + win_label] = vals

# G4. LFP autocorrelation (proxy for oscillatory persistence)
def lfp_autocorr_time(trial, ch, t_start, t_end, max_lag_ms=50):
    """Autocorrelation decay time — longer = more oscillatory."""
    t = trial['time']
    mask = (t >= t_start) & (t < t_end)
    seg = trial['bipolar_lfp'][ch][mask].copy()
    seg -= np.mean(seg)
    if len(seg) < 500:
        return np.nan
    max_lag = int(max_lag_ms * FS / 1000)
    seg = seg - seg.mean()
    norm = np.sum(seg ** 2)
    if norm == 0:
        return np.nan
    autocorr = np.correlate(seg, seg, mode='full')
    autocorr = autocorr[len(seg) - 1:]  # positive lags only
    autocorr = autocorr[:max_lag + 1] / norm
    # Find first zero crossing
    zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
    if len(zero_crossings) > 0:
        return zero_crossings[0] * 1000.0 / FS  # in ms
    return max_lag_ms  # didn't decay

for ch_idx in [0, mid_ch, n_channels - 1]:
    for win_label, ts, te in [
        ('sust', stim_onset + 200, stim_onset + 500),
    ]:
        vals = [lfp_autocorr_time(all_data[i], ch_idx, ts, te) for i in range(N_TRIALS)]
        features['autocorr_time_ch' + str(ch_idx) + '_' + win_label] = vals


# ── H. ONSET TRANSIENT ──
print("  Computing onset transient features...")

for ch_idx in [0, mid_ch, n_channels - 1]:
    vals_peak = []
    vals_rms = []
    for i in range(N_TRIALS):
        t = all_data[i]['time']
        mask = (t >= stim_onset) & (t < stim_onset + 100)
        seg = all_data[i]['bipolar_lfp'][ch_idx][mask]
        vals_peak.append(np.max(np.abs(seg)) if len(seg) > 0 else np.nan)
        vals_rms.append(np.sqrt(np.mean(seg**2)) if len(seg) > 0 else np.nan)
    features['transient_peak_ch' + str(ch_idx)] = vals_peak
    features['transient_rms_ch' + str(ch_idx)] = vals_rms

# Mean across all channels
all_ch_transient = []
for i in range(N_TRIALS):
    t = all_data[i]['time']
    mask = (t >= stim_onset) & (t < stim_onset + 100)
    ch_peaks = [np.max(np.abs(all_data[i]['bipolar_lfp'][ch][mask]))
                for ch in range(n_channels)]
    all_ch_transient.append(np.mean(ch_peaks))
features['transient_peak_mean'] = all_ch_transient


# ── I. RATE-BASED FEATURES (transients, evoked) ──
print("  Computing rate-based transient/evoked features...")

for i in range(N_TRIALS):
    for layer_name, layer_rates in all_data[i]['rate_data'].items():
        for mon_name, mon in layer_rates.items():
            t_r = mon['t_ms']
            r = mon['rate_hz']

            # Peak rate in first 100ms (onset transient)
            trans_mask = (t_r >= stim_onset) & (t_r < stim_onset + 100)
            if trans_mask.sum() > 0:
                key = 'transient_peak_rate_' + layer_name + '_' + mon_name
                features.setdefault(key, []).append(np.max(r[trans_mask]))

            # Sustained rate
            sust_mask = (t_r >= stim_onset + 200) & (t_r < stim_onset + 500)
            if sust_mask.sum() > 0:
                key = 'sustained_rate_' + layer_name + '_' + mon_name
                features.setdefault(key, []).append(np.mean(r[sust_mask]))

            # Rate change (sustained - pre)
            pre_mask_r = (t_r >= stim_onset - 300) & (t_r < stim_onset)
            if sust_mask.sum() > 0 and pre_mask_r.sum() > 0:
                delta = np.mean(r[sust_mask]) - np.mean(r[pre_mask_r])
                key = 'rate_delta_' + layer_name + '_' + mon_name
                features.setdefault(key, []).append(delta)


# ── J. PER-CHANNEL PRE-STIM BAND POWER ──
print("  Computing per-channel pre-stim band powers...")

for ch_idx in range(n_channels):
    for band_name, band_range in [('low', (1, 20)), ('beta', (13, 30)), ('gamma', (30, 80))]:
        vals = []
        for i in range(N_TRIALS):
            t = all_data[i]['time']
            pre_mask = (t >= stim_onset - 1000) & (t < stim_onset)
            seg = all_data[i]['bipolar_lfp'][ch_idx][pre_mask].copy()
            seg -= np.mean(seg)
            if len(seg) < 2000:
                vals.append(np.nan)
                continue
            nperseg_val = len(seg)
            f, psd = welch(seg, fs=FS, nperseg=nperseg_val)
            m = (f >= band_range[0]) & (f <= band_range[1])
            vals.append(_integrate(psd[m], f[m]) if m.sum() > 1 else np.nan)
        features['pre_ch' + str(ch_idx) + '_' + band_name] = vals


# ═══════════════════════════════════════════════════════════════
# STEP 4: CORRELATE EVERYTHING WITH LOW-FREQ CHANGE
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 4: CORRELATIONS WITH LOW-FREQ CHANGE")
print("="*70)
print(f"  Total features: {len(features)}")

correlations = []
for feat_name, feat_vals in features.items():
    fv = np.array(feat_vals, dtype=float)
    if len(fv) != N_TRIALS:
        continue
    valid = ~np.isnan(fv) & ~np.isnan(low_changes)
    if valid.sum() < 20:
        continue
    r, p_val = pearsonr(fv[valid], low_changes[valid])
    if not np.isnan(r):
        correlations.append((feat_name, r, p_val))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"\n  TOP 50 correlates of low-freq change (neg change = biological):")
print(f"  {'Feature':<50} {'r':<10} {'p':<10}")
print("  " + "-"*72)
for feat_name, r, p_val in correlations[:50]:
    marker = ' ***' if p_val < 0.001 else (' **' if p_val < 0.01 else (' *' if p_val < 0.05 else ''))
    print(f"  {feat_name:<50} {r:>7.4f}   {p_val:>8.5f}{marker}")


# ═══════════════════════════════════════════════════════════════
# STEP 5: GROUP NON-RATE FEATURES BY CATEGORY
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 5: TOP FEATURES BY CATEGORY")
print("="*70)

categories = {
    'Synchrony (Fano)':     [c for c in correlations if c[0].startswith('fano_')],
    'Spike count CV':       [c for c in correlations if c[0].startswith('cv_counts_')],
    'Inh subtype (PV/SOM)': [c for c in correlations if c[0].startswith('PV')],
    'E/I ratio':            [c for c in correlations if c[0].startswith('EI_')],
    'Cross-layer corr':     [c for c in correlations if c[0].startswith('xcorr_')],
    'LFP phase at onset':   [c for c in correlations if 'phase' in c[0] or 'cos_' in c[0] or 'sin_' in c[0]],
    'LFP amp at onset':     [c for c in correlations if 'lowfreq_amp' in c[0]],
    'LFP coherence':        [c for c in correlations if c[0].startswith('coh_')],
    'LFP autocorrelation':  [c for c in correlations if c[0].startswith('autocorr_')],
    'Onset transient':      [c for c in correlations if 'transient' in c[0]],
    'Rate delta':           [c for c in correlations if c[0].startswith('rate_delta_')],
}

for cat_name, cat_corrs in categories.items():
    cat_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  {cat_name}:")
    for feat, r, p_val in cat_corrs[:5]:
        marker = ' ***' if p_val < 0.001 else (' **' if p_val < 0.01 else (' *' if p_val < 0.05 else ''))
        print(f"    {feat:<48} r={r:>7.4f}  p={p_val:.5f}{marker}")


# ═══════════════════════════════════════════════════════════════
# STEP 6: SCATTER PLOTS OF TOP NON-RATE PREDICTORS
# ═══════════════════════════════════════════════════════════════

# Pick top 2 from each category
top_feats = []
for cat_name, cat_corrs in categories.items():
    for feat, r, p_val in cat_corrs[:2]:
        if p_val < 0.1:  # only plot if somewhat significant
            top_feats.append((feat, r, p_val, cat_name))

top_feats.sort(key=lambda x: abs(x[1]), reverse=True)
top_feats = top_feats[:12]

if top_feats:
    n_cols = 4
    n_rows = int(np.ceil(len(top_feats) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes_flat = np.array(axes).flatten()

    for idx, (feat_name, r, p_val, cat) in enumerate(top_feats):
        ax = axes_flat[idx]
        fv = np.array(features[feat_name], dtype=float)
        ax.scatter(fv[good_mask], low_changes[good_mask], c='steelblue', alpha=0.5, s=20, label='Good')
        ax.scatter(fv[bad_mask], low_changes[bad_mask], c='salmon', alpha=0.5, s=20, label='Bad')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel(feat_name, fontsize=6)
        ax.set_ylabel('Low-freq change (%)')
        ax.set_title(f'{cat}\nr={r:.3f}, p={p_val:.4f}', fontsize=8)
        ax.legend(fontsize=6)

    for idx in range(len(top_feats), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle('Top Non-Rate Predictors of Low-Freq Change', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "02_nonrate_predictors.png"), dpi=150, bbox_inches='tight')
    plt.close()

# Also scatter top 12 overall
top_overall = correlations[:12]
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes_flat = np.array(axes).flatten()

for idx, (feat_name, r, p_val) in enumerate(top_overall):
    ax = axes_flat[idx]
    fv = np.array(features[feat_name], dtype=float)
    ax.scatter(fv[good_mask], low_changes[good_mask], c='steelblue', alpha=0.5, s=20, label='Good')
    ax.scatter(fv[bad_mask], low_changes[bad_mask], c='salmon', alpha=0.5, s=20, label='Bad')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel(feat_name, fontsize=6)
    ax.set_ylabel('Low-freq chg (%)')
    ax.set_title(f'r={r:.3f}, p={p_val:.4f}', fontsize=9)
    ax.legend(fontsize=6)

plt.suptitle('Top 12 Predictors of Low-Freq Power Change (all categories)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "03_top12_predictors.png"), dpi=150, bbox_inches='tight')
plt.close()


# ═══════════════════════════════════════════════════════════════
# STEP 7: PER-CHANNEL LOW-FREQ CHANGE PROFILE
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 7: PER-CHANNEL LAMINAR PROFILE")
print("="*70)

good_ch = np.zeros((len(GOOD_TRIALS), n_channels))
bad_ch = np.zeros((len(BAD_TRIALS), n_channels))
gi, bi = 0, 0
for i in range(N_TRIALS):
    ch_c = trial_spectral[i]['low_change_per_ch']
    if len(ch_c) < n_channels:
        ch_c = np.pad(ch_c, (0, n_channels - len(ch_c)), constant_values=np.nan)
    if i in GOOD_TRIALS:
        good_ch[gi] = ch_c[:n_channels]; gi += 1
    else:
        bad_ch[bi] = ch_c[:n_channels]; bi += 1

depths = all_data[0]['channel_depths']
bipolar_depths = (depths[:-1] + depths[1:]) / 2 if len(depths) > n_channels else depths[:n_channels]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, data_arr, title, color in [
    (axes[0], good_ch, 'Good trials', 'steelblue'),
    (axes[1], bad_ch, 'Bad trials', 'salmon'),
]:
    m = np.nanmean(data_arr, axis=0)
    s = np.nanstd(data_arr, axis=0) / np.sqrt(data_arr.shape[0])
    ax.barh(range(n_channels), m, xerr=s, alpha=0.7, color=color)
    ax.axvline(x=0, color='black', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Low-freq change (%)')
    ax.invert_yaxis()
    ax.set_yticks(range(n_channels))
    ax.set_yticklabels([f'{bipolar_depths[i]:.0f}um' if i < len(bipolar_depths) else ''
                        for i in range(n_channels)], fontsize=6)

diff = np.nanmean(bad_ch, axis=0) - np.nanmean(good_ch, axis=0)
axes[2].barh(range(n_channels), diff, alpha=0.7, color='purple')
axes[2].axvline(x=0, color='black', linewidth=2)
axes[2].set_title('Difference (bad - good)')
axes[2].set_xlabel('Difference (%)')
axes[2].invert_yaxis()
axes[2].set_yticks(range(n_channels))
axes[2].set_yticklabels([f'{bipolar_depths[i]:.0f}um' if i < len(bipolar_depths) else ''
                         for i in range(n_channels)], fontsize=6)

plt.suptitle('Per-Channel Low-Freq Power Change', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "04_per_channel.png"), dpi=150, bbox_inches='tight')
plt.close()


# ═══════════════════════════════════════════════════════════════
# STEP 8: TIME-RESOLVED LOW-FREQ POWER
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 8: TIME-RESOLVED LOW-FREQ POWER")
print("="*70)

def sliding_band_power(lfp, time_arr, fs, band, win_ms=500, step_ms=100):
    """Sliding window band power. win_ms=500 at 10kHz gives 2 Hz resolution."""
    win_samp = int(win_ms * fs / 1000)
    step_samp = int(step_ms * fs / 1000)
    centers, powers = [], []
    for start in range(0, len(lfp) - win_samp, step_samp):
        seg = lfp[start:start + win_samp].copy()
        seg -= np.mean(seg)
        # Use full window as nperseg
        f, psd = welch(seg, fs=fs, nperseg=len(seg), window='hann')
        m = (f >= band[0]) & (f <= band[1])
        if m.sum() > 1:
            powers.append(_integrate(psd[m], f[m]))
            centers.append(time_arr[start + win_samp // 2])
    return np.array(centers), np.array(powers)

good_traces, bad_traces = [], []
common_centers = None

for i in range(N_TRIALS):
    centers, powers = sliding_band_power(
        all_data[i]['bipolar_lfp'][mid_ch], all_data[i]['time'], FS, LOW_FREQ)
    if common_centers is None:
        common_centers = centers
    if len(centers) == len(common_centers):
        (good_traces if i in GOOD_TRIALS else bad_traces).append(powers)

if good_traces and bad_traces:
    fig, ax = plt.subplots(figsize=(14, 5))
    for traces, color, label in [(good_traces, 'steelblue', 'Good'), (bad_traces, 'salmon', 'Bad')]:
        m = np.mean(traces, axis=0)
        s = np.std(traces, axis=0) / np.sqrt(len(traces))
        ax.plot(common_centers, m, color=color, linewidth=2, label=label)
        ax.fill_between(common_centers, m - s, m + s, alpha=0.2, color=color)
    ax.axvline(x=stim_onset, color='black', linestyle='--', linewidth=2, label='Stim onset')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Low-freq power (1-20 Hz)')
    ax.set_title(f'Time-resolved Low-Frequency Power (ch {mid_ch})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "05_lowfreq_timecourse.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved time course plot")


# ═══════════════════════════════════════════════════════════════
# STEP 9: WORST vs BEST TRIALS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 9: WORST vs BEST TRIALS")
print("="*70)

sorted_idx = np.argsort(low_changes)
best_5 = sorted_idx[:5]
worst_5 = sorted_idx[-5:]

print("  Best 5 trials (largest low-freq decrease):")
for idx in best_5:
    sr = all_data[idx]['stim_rates']
    print(f"    Trial {idx}: low_change={low_changes[idx]:.1f}%  "
          f"L4C_E={sr['L4C_E']:.3f} L4C_PV={sr['L4C_PV']:.3f}")

print("\n  Worst 5 trials (largest low-freq increase):")
for idx in worst_5:
    sr = all_data[idx]['stim_rates']
    print(f"    Trial {idx}: low_change={low_changes[idx]:.1f}%  "
          f"L4C_E={sr['L4C_E']:.3f} L4C_PV={sr['L4C_PV']:.3f}")

print("\n  Feature comparison (top 15):")
print(f"  {'Feature':<50} {'Best 5':<12} {'Worst 5':<12}")
print("  " + "-"*75)
for feat_name, r, p_val in correlations[:15]:
    fv = np.array(features[feat_name], dtype=float)
    print(f"  {feat_name:<50} {np.mean(fv[best_5]):<12.4f} {np.mean(fv[worst_5]):<12.4f}")


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

# Count significant features by category
print("\n  Significant features (p < 0.05) by category:")
for cat_name, cat_corrs in categories.items():
    n_sig = sum(1 for _, _, p in cat_corrs if p < 0.05)
    n_tot = len(cat_corrs)
    if n_sig > 0:
        top_r = max(abs(c[1]) for c in cat_corrs) if cat_corrs else 0
        print(f"    {cat_name:<30} {n_sig}/{n_tot} significant  (max |r|={top_r:.3f})")

print(f"\n  Top 10 overall predictors:")
for i, (feat, r, p_val) in enumerate(correlations[:10]):
    direction = "more -> low-freq UP (bad)" if r > 0 else "more -> low-freq DOWN (good)"
    print(f"    {i+1}. {feat}  r={r:.3f}  p={p_val:.5f}  [{direction}]")

print(f"\n  Plots saved to: {SAVE_DIR}/")
print("DONE")