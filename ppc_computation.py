import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Paired")

fname = "results/farzin_trial/trial_000.npz"
fs = 10000                 
dt = 1.0 / fs     
sigma_ms = 1.0  
sigma_samples = sigma_ms * fs / 1000.0
transient_ms = 1000      
freqs_of_interest = np.arange(1, 101) 
bw = 2.0        
bp_order = 4       

layers = ['L1']
pops = ['E', 'PV', 'SOM']
pop_colors = {'E': "#1d970d", 'PV': '#d62728', 'SOM': "#3629c8", 'VIP': "#d22dac"}

data = np.load(fname, allow_pickle=True)
spike_data = data["spike_data"].item() if data["spike_data"].size == 1 else data["spike_data"]
time_array_ms = data["time_array_ms"]
n_samples = len(time_array_ms)
t_start_sample = int(transient_ms * fs / 1000)



def make_lfp_proxy(spike_times_ms, n_samples, fs, sigma_samples):
    spike_train = np.zeros(n_samples)
    spike_samples = np.round(spike_times_ms * fs / 1000).astype(int)
    spike_samples = spike_samples[(spike_samples >= 0) & (spike_samples < n_samples)]
    np.add.at(spike_train, spike_samples, 1.0)
    lfp_proxy = gaussian_filter1d(spike_train, sigma_samples)
    return lfp_proxy


def bandpass(sig, f_center, bw, fs, order=4):
    low = max(f_center - bw, 0.5)
    high = f_center + bw
    nyq = fs / 2.0
    if high >= nyq:
        high = nyq - 1
    if low >= high:
        return np.zeros_like(sig)
    sos = signal.butter(order, [low / nyq, high / nyq], btype='band', output='sos')
    return signal.sosfiltfilt(sos, sig)


def compute_ppc_single_neuron(spike_times_ms, lfp_phase, fs, t_start_sample, min_spikes=5):
    spike_samples = np.round(spike_times_ms * fs / 1000).astype(int)
    valid = (spike_samples >= t_start_sample) & (spike_samples < len(lfp_phase))
    spike_samples = spike_samples[valid]
    if len(spike_samples) < min_spikes:
        return np.nan
    phases = lfp_phase[spike_samples]
    z = np.exp(1j * phases)
    n = len(z)
    vec_sum = np.sum(z)
    ppc = (np.abs(vec_sum)**2 - n) / (n * (n - 1))
    return ppc


ppc_results = {}

for layer in layers:
    print(f"processing {layer}...")

    e_spk = spike_data[layer]['E_spikes']
    lfp_proxy = make_lfp_proxy(e_spk['times_ms'], n_samples, fs, sigma_samples)

    lfp_phases = {}
    for freq in freqs_of_interest:
        lfp_bp = bandpass(lfp_proxy, freq, bw, fs, bp_order)
        analytic = signal.hilbert(lfp_bp)
        lfp_phases[freq] = np.angle(analytic)

    for pop in pops:
        spk = spike_data[layer][pop + '_spikes']
        times = spk['times_ms']
        indices = spk['spike_indices']
        neurons = np.unique(indices)

        neuron_ppcs = []
        for neuron_id in neurons:
            mask = indices == neuron_id
            neuron_times = times[mask]

            ppc_per_freq = np.empty(len(freqs_of_interest))
            for fi, freq in enumerate(freqs_of_interest):
                ppc_per_freq[fi] = compute_ppc_single_neuron(
                    neuron_times, lfp_phases[freq], fs, t_start_sample)
            if np.any(~np.isnan(ppc_per_freq)):
                neuron_ppcs.append(ppc_per_freq)

        ppc_results[(layer, pop)] = neuron_ppcs
        print(f"  {pop}: {len(neuron_ppcs)} neurons with enough spikes")

ppc_mean = {}
ppc_sem = {}

for layer in layers:
    for pop in pops:
        key = (layer, pop)
        neuron_list = ppc_results[key]
        if len(neuron_list) > 0:
            stack = np.array(neuron_list)  
            ppc_mean[key] = np.nanmean(stack, axis=0)
            ppc_sem[key] = np.nanstd(stack, axis=0) / np.sqrt(np.sum(~np.isnan(stack), axis=0))
            peak_idx = np.nanargmax(ppc_mean[key])
            print(f"{layer} {pop}: {len(neuron_list)} neurons, "
                  f"peak PPC = {ppc_mean[key][peak_idx]:.4f} "
                  f"at {freqs_of_interest[peak_idx]} Hz")


fig, axes = plt.subplots(len(layers), 1, figsize=(8, 3 * len(layers)), sharex=True)

for ax, layer in zip(axes, layers):
    has_data = False
    for pop in pops:
        key = (layer, pop)
        if key in ppc_mean:
            mean = ppc_mean[key]
            sem = ppc_sem[key]
            ax.plot(freqs_of_interest, mean,
                    label=f"{pop} (n={len(ppc_results[key])})",
                    color=pop_colors[pop], linewidth=1.5)
            ax.fill_between(freqs_of_interest, mean - sem, mean + sem,
                            color=pop_colors[pop], alpha=0.2)
            has_data = True
    ax.set_ylabel('PPC')
    ax.set_title(layer)
    if has_data:
        ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel('Frequency (Hz)')
fig.suptitle('Pairwise Phase Consistency', fontsize=14, y=1.01)
fig.tight_layout()
plt.savefig("results/essai_PPC/ppc_by_layer.png", dpi=200, bbox_inches='tight')
plt.show()
