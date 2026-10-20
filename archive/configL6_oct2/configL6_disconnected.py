"""configL6 with every inter-layer connection silenced, and Poisson N re-tuned so
every population keeps its CONNECTED configL6 rate.

The inter-layer synapses are still built (then silenced), so the wiring and the
heterogeneity are the same network as configL6; only the Poisson N differ.
Input rate stays at configL6's 4 Hz in every layer.

Tuned 2026-09-30 against configL6 as it was that day (seed 58925, rates over
500-2000 ms, as in the bipolar-spectrum window). Every population within 5.5%.
Disconnected / connected rate (Hz):
L23    E 1.46/1.45  PV 3.36/3.38  SOM 4.71/4.74  VIP 2.01/1.94
L4AB   E 1.98/2.05  PV 2.82/2.89  SOM 5.02/5.08  VIP 4.64/4.87
L4C    E 1.28/1.29  PV 4.70/4.86  SOM 6.06/6.05  VIP 1.83/1.94
L5     E 1.71/1.69  PV 5.41/5.36  SOM 3.42/3.46  VIP 1.60/1.55
L6     E 1.66/1.70  PV 6.56/6.48  SOM 4.65/4.62  VIP 2.86/2.95

Re-tune after changing configL6 -- these N are only valid for its current local
wiring and connected rates.
"""
import copy
from config.configL6 import CONFIG as _CONNECTED

DISCONNECTED_N = {
    'L23':  {'E': 61, 'PV': 264, 'SOM': 95, 'VIP': 24},
    'L4AB': {'E': 66, 'PV': 167, 'SOM': 95, 'VIP': 38},
    'L4C':  {'E': 75, 'PV': 238, 'SOM': 131, 'VIP': 28},
    'L5':   {'E': 53, 'PV': 187, 'SOM': 60, 'VIP': 37},
    'L6':   {'E': 61, 'PV': 119, 'SOM': 40, 'VIP': 27},
}

CONFIG = copy.deepcopy(_CONNECTED)
CONFIG['lesioned_inter_layer'] = sorted(CONFIG['inter_layer_connections'].keys())
for _L, _ns in DISCONNECTED_N.items():
    for _p, _n in _ns.items():
        CONFIG['layers'][_L]['poisson_inputs'][_p]['N'] = _n
