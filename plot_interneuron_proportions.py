"""to do :maybe merge this with the connectiivty plots. that way it's a script to plot the config, like what's the proportions and connectiivty."""
import numpy as np
import matplotlib.pyplot as plt
from config.config import CONFIG

POP_COLORS = {
    'E':   '#2E8B57',
    'PV':  '#C0392B',
    'SOM': '#1F4E96',
    'VIP': '#D4A017',
}

inh_pops = ['PV', 'SOM', 'VIP']
layer_names = [n for n in CONFIG['layers'].keys() if 'neuron_counts' in CONFIG['layers'][n]]

proportions = {p: [] for p in inh_pops}
totals = []
for layer in layer_names:
    counts = CONFIG['layers'][layer]['neuron_counts']
    total_inh = sum(counts.get(p, 0) for p in inh_pops)
    totals.append(total_inh)
    for p in inh_pops:
        proportions[p].append(counts.get(p, 0) / total_inh if total_inh > 0 else 0.0)

header = f"{'Layer':<8}" + "".join(f"{p:>14}" for p in inh_pops) + f"{'Total inh':>14}"
print(header)
print("-" * len(header))
for i, layer in enumerate(layer_names):
    counts = CONFIG['layers'][layer]['neuron_counts']
    row = f"{layer:<8}"
    for p in inh_pops:
        row += f"{counts.get(p, 0):>5} ({proportions[p][i]*100:5.1f}%)"
    row += f"{totals[i]:>14}"
    print(row)
print()

fig, ax = plt.subplots(figsize=(8, 5))

y = np.arange(len(layer_names))
left = np.zeros(len(layer_names))

for p in inh_pops:
    vals = np.array(proportions[p])
    ax.barh(y, vals, left=left, color=POP_COLORS[p],
            edgecolor='white', linewidth=1.2,
            label=p, alpha=0.95, height=0.7)
    for yi, v, l in zip(y, vals, left):
        if v > 0.05:
            ax.text(l + v / 2, yi, f'{v*100:.0f}%',
                    ha='center', va='center',
                    color='white', fontsize=10, fontweight='bold')
    left += vals

ax.set_yticks(y)
ax.set_yticklabels(layer_names, fontsize=12)
ax.invert_yaxis()
ax.set_xlabel('Relative proportion', fontsize=13)
ax.set_ylabel('Layer', fontsize=13)
ax.set_title('Interneuron subtype composition per layer',
             fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)
ax.set_xticks(np.linspace(0, 1, 6))
ax.set_xticklabels([f'{int(t*100)}%' for t in np.linspace(0, 1, 6)])
ax.legend(title='Interneuron', loc='center left',
          bbox_to_anchor=(1.02, 0.5), frameon=False,
          fontsize=11, title_fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('interneuron_proportions.png', dpi=200, bbox_inches='tight')
plt.show()
