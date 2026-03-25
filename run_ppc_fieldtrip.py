import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert, butter, sosfiltfilt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Paired")

fname = "results/10s_with_stim/trial_000.npz"
data = np.load(fname, allow_pickle=True)

bipolar_lfp = data["lfp_matrix"]
spike_data = data["spike_data"].item() if data["spike_data"].size == 1 else data["spike_data"]

fs = 10000 
T_skip = 1.0  
T_skip_samples = int(T_skip * fs)

layers = ['L23', 'L4AB', 'L4C', 'L5', 'L6']
bipolar_lfp_indices = {'L23': 11, 'L4AB': 8, 'L4C': 6, 'L5': 5, 'L6': 3}
pops = ['E', 'PV', 'SOM']
freqs = np.arange(5, 105, 5) 
bandwidth = 5  

ppc_results = {}

for layer in layers:
    channel_idx = bipolar_lfp_indices[layer]
    lfp_signal = bipolar_lfp[channel_idx]

    inst_phases = np.zeros((len(freqs), len(lfp_signal)))
    for f_idx, f_center in enumerate(freqs):
        f_low = max(f_center - bandwidth, 1)
        f_high = f_center + bandwidth
        sos = butter(4, [f_low / (fs/2), f_high / (fs/2)], btype='band', output='sos')
        lfp_filtered = sosfiltfilt(sos, lfp_signal)
        inst_phases[f_idx] = np.angle(hilbert(lfp_filtered))
    print(f"  {layer}: LFP phases computed for {len(freqs)} frequencies")

    for pop in pops:
        spike_info = spike_data[layer][pop + '_spikes']
        times_ms = spike_info["times_ms"]
        indices = spike_info["spike_indices"]

        unique_neurons = np.unique(indices)

        for neuron_id in unique_neurons:
            mask = (indices == neuron_id) & (times_ms > T_skip * 1000)
            neuron_spike_times_ms = times_ms[mask]

            if len(neuron_spike_times_ms) < 2:
                continue

            spike_samples = (neuron_spike_times_ms * fs / 1000).astype(int)
            spike_samples = spike_samples[(spike_samples >= 0) & (spike_samples < len(lfp_signal))]

            if len(spike_samples) < 2:
                continue

            N = len(spike_samples)
            ppc_neuron = np.zeros(len(freqs))

            for f_idx in range(len(freqs)):
                spike_phases = inst_phases[f_idx, spike_samples]
                vec_sum = np.sum(np.exp(1j * spike_phases))
                ppc_neuron[f_idx] = (np.abs(vec_sum) ** 2 - N) / (N * (N - 1))

            if (layer, pop) not in ppc_results:
                ppc_results[(layer, pop)] = []
            ppc_results[(layer, pop)].append(ppc_neuron)

        print(f"    {pop}: done ({len(ppc_results.get((layer, pop), []))} neurons)")


for layer in layers:
    for pop in pops:
        key = (layer, pop)
        if key in ppc_results and len(ppc_results[key]) > 0:
            mean_ppc = np.mean(ppc_results[key], axis=0)
            peak_idx = np.argmax(mean_ppc)
            print(f"{layer} {pop}: {len(ppc_results[key])} neurons, "
                  f"peak PPC = {mean_ppc[peak_idx]:.4f} at {freqs[peak_idx]} Hz")

pop_colors = {'E': "#1d970d", 'PV': '#d62728', 'SOM': "#3629c8", 'VIP': "#d22dac"}

fig, axes = plt.subplots(len(layers), 1, figsize=(8, 3 * len(layers)), sharex=True)

for ax, layer in zip(axes, layers):
    has_data = False
    for pop in pops:
        key = (layer, pop)
        if key in ppc_results and len(ppc_results[key]) > 0:
            ppc_array = np.array(ppc_results[key])
            mean_ppc = np.mean(ppc_array, axis=0)
            sem_ppc = np.std(ppc_array, axis=0) / np.sqrt(len(ppc_array))
            ax.plot(freqs, mean_ppc, label=pop, color=pop_colors[pop], linewidth=1.5)
            ax.fill_between(freqs,
                            mean_ppc - sem_ppc,
                            mean_ppc + sem_ppc,
                            color=pop_colors[pop], alpha=0.2)
            has_data = True
    ax.set_ylabel('PPC')
    ax.set_title(layer)
    if has_data:
        ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel('Frequency (Hz)')
fig.suptitle('Pairwise Phase Consistency by Layer (Hilbert method)', fontsize=14, y=1.01)
fig.tight_layout()
plt.savefig("results/10s_with_stim/ppc_hilbert_by_layer.png", dpi=200, bbox_inches='tight')
plt.show()