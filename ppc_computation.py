import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import zscore
import seaborn as sns
from matplotlib.gridspec import GridSpec

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Paired")

fname="results/10s_with_PV_delay/trial_000.npz"
data=np.load(fname, allow_pickle=True)

lfp_matrix=data["lfp_matrix"]
spike_data=data["spike_data"].item() if data["spike_data"].size == 1 else data["spike_data"]

fs=10000  

layers=['L23', 'L4AB', 'L4C', 'L5', 'L6']
lfp_indices={'L23': 11, 'L4AB': 8, 'L4C': 6, 'L5': 5, 'L6': 3}
pops=['E', 'PV', 'SOM', 'VIP']
half_win_ms=75 #hann window
half_win_samples=int(half_win_ms * fs / 1000)
win_samples=2 * half_win_samples
hann_window=signal.windows.hann(win_samples)
freqs=np.fft.rfftfreq(win_samples, d=1.0 / fs)

ppc_results={}

for layer in layers:
    for pop in pops:
        spike_info=spike_data[layer][pop+'_spikes']
        times=spike_info["times_ms"]
        indices=spike_info["spike_indices"]
        channel_idx=lfp_indices[layer]

        neurons=np.unique(indices)

        for neuron_id in neurons:
            mask=(indices == neuron_id) & (times > 1000) 
            neuron_spike_times=times[mask]
            phases_per_freq=[[] for _ in range(len(freqs))]

            for t_spike in neuron_spike_times:
                t_sample=int(t_spike * fs / 1000)

                if t_sample - half_win_samples < 0 or t_sample + half_win_samples > lfp_matrix.shape[1]:
                    continue

                lfp_segment=lfp_matrix[channel_idx, t_sample - half_win_samples: t_sample + half_win_samples]

                lfp_fft=np.fft.rfft(lfp_segment * hann_window)

                phases=np.angle(lfp_fft)
                for f_idx in range(len(freqs)):
                    phases_per_freq[f_idx].append(phases[f_idx])

            N=len(phases_per_freq[0])
            if N < 2:
                continue

            ppc_neuron=np.zeros(len(freqs))

            for f_idx in range(len(freqs)):
                theta=np.array(phases_per_freq[f_idx])
                vec_sum=np.sum(np.exp(1j*theta))
                ppc_neuron[f_idx]=(np.abs(vec_sum)**2-N)/(N*(N-1))

            if (layer, pop) not in ppc_results:
                ppc_results[(layer, pop)]=[]
            ppc_results[(layer, pop)].append(ppc_neuron)


for layer in layers:
    for pop in pops:
        key=(layer, pop)
        if key in ppc_results and len(ppc_results[key]) > 0:
            mean_ppc=np.mean(ppc_results[key], axis=0)
            print(f"{layer} {pop}: {len(ppc_results[key])} neurons, "
                  f"peak PPC={np.max(mean_ppc):.4f} at {freqs[np.argmax(mean_ppc)]:.1f} Hz")

max_freq=100  
freq_mask=freqs <= max_freq
pop_colors={'E': "#1d970d", 'PV': '#d62728', 'SOM': "#3629c8", 'VIP': "#d22dac"}

fig, axes=plt.subplots(len(layers), 1, figsize=(8, 3 * len(layers)), sharex=True)

for ax, layer in zip(axes, layers):
    has_data=False
    for pop in pops:
        key=(layer, pop)
        if key in ppc_results and len(ppc_results[key]) > 0:
            mean_ppc=np.mean(ppc_results[key], axis=0)
            sem_ppc=np.std(ppc_results[key], axis=0) / np.sqrt(len(ppc_results[key]))
            ax.plot(freqs[freq_mask], mean_ppc[freq_mask],
                    label=pop, color=pop_colors[pop], linewidth=1.5)
            ax.fill_between(freqs[freq_mask],
                            mean_ppc[freq_mask] - sem_ppc[freq_mask],
                            mean_ppc[freq_mask] + sem_ppc[freq_mask],
                            color=pop_colors[pop], alpha=0.2)
            has_data=True
    ax.set_ylabel('PPC')
    ax.set_title(layer)
    if has_data:
        ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel('Frequency (Hz)')
fig.suptitle('Pairwise Phase Consistency by Layer', fontsize=14, y=1.01)
fig.tight_layout()
plt.savefig("results/10s_without_stim/ppc_by_layer.png", dpi=200, bbox_inches='tight')
plt.show()
