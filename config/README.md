# config/

All parameters and equations of the model are regrouped here. You can change most parameters easily, including the neuron equations.
The only exception are the synaptic equations, which are defined in src/layer.py and src/column.py.

| File | Content |
| --- | --- |
| `config.py` | neuron parameters, synapse time constants, background poisson inputs |
| `connection_probabilities.csv` | the connection probabilities of each neuron population|
| `conductances_AMPA_GABA.csv` | AMPA (excitatory) and GABA (inhibitory) conductances |
| `conductances_NMDA.csv` | NMDA (slow excitatory) synapse strengths |

The CSV files are tables (rows = source population, columns = target population). For more readability, you can install Rainbow CSV extension in VS Code.
