"""Minimal config: ONE isolated layer, defined by three 4x4 CSVs + layer.json.

Everything that is not connectivity or cell counts (neuron model, intrinsic
params, heterogeneity, synaptic time constants, delays, Poisson input rate,
geometry, electrodes) comes from config/config.py or is fixed below, and is
therefore identical in every experiment. So two experiments can differ only in
    - the three CSVs      (connection probabilities, AMPA/GABA, NMDA)
    - layer.json          (neuron counts, Poisson N per population)
Poisson N is not a free choice: set it with match_rates.py so every population
fires at target_rates_hz (rate matching).

An experiment is a named condition in experiments/conditions.py (a change to
a base layer's circuit, built in memory). Select it with an env var:
    MINIMAL_CONDITION=L4C+L6_VIP-SOM python main.py
with main.py importing `from minimal_config.config import CONFIG`.
Unset, this directory itself (intact L6) is used.
"""
import copy
import json
import os

import pandas as pd
from brian2 import Hz, ms

from config.config import CONFIG as _BASE

HERE = os.path.dirname(os.path.abspath(__file__))
CELL_TYPES = ['E', 'PV', 'SOM', 'VIP']

# Fixed for every experiment (L6's values), so they can never differ between runs.
INPUT_RATE = 5 * Hz
COORDINATES = {'x': (-0.15, 0.15), 'y': (-0.15, 0.15), 'z': (-0.62, -0.34)}


def tables_to_dicts(prob, ampa, nmda):
    """Same rules as tools.utils.load_connectivity_from_csv, for one 4x4 block.

    A pathway exists when its probability is > 0. NOTE: layer.py then falls back
    to 0.01 nS (E) / 0.02 nS (inhibitory) if the conductance is 0, so remove a
    pathway by setting its PROBABILITY to 0, not its conductance.
    """
    connection_prob, conductance = {}, {}
    for src in CELL_TYPES:
        for tgt in CELL_TYPES:
            if prob.loc[src, tgt] > 0:
                connection_prob[f'{src}_{tgt}'] = float(prob.loc[src, tgt])
            if src == 'E':
                if ampa.loc[src, tgt] > 0:
                    conductance[f'{src}_{tgt}_AMPA'] = float(ampa.loc[src, tgt])
                if nmda.loc[src, tgt] > 0:
                    conductance[f'{src}_{tgt}_NMDA'] = float(nmda.loc[src, tgt])
            elif ampa.loc[src, tgt] > 0:
                conductance[f'{src}_{tgt}'] = float(ampa.loc[src, tgt])
    return connection_prob, conductance


def load(exp=None):
    """(tables, spec) for a condition name from experiments/conditions.py, or a folder
    holding the three CSVs + layer.json. Default: MINIMAL_CONDITION env var, else intact L6."""
    exp = exp or os.environ.get('MINIMAL_CONDITION') or HERE
    if os.path.isdir(exp):
        tables = {f: pd.read_csv(os.path.join(exp, f'{f}.csv'), index_col=0)
                  for f in ['connection_probabilities', 'conductances_AMPA_GABA', 'conductances_NMDA']}
        return tables, json.load(open(os.path.join(exp, 'layer.json')))
    from minimal_config.experiments.conditions import build
    return build(exp)


def make_config(exp=None, poisson_N=None):
    """Full CONFIG for a condition name or folder. poisson_N overrides the stored N (match_rates)."""
    tables, spec = load(exp)
    connection_prob, conductance = tables_to_dicts(tables['connection_probabilities'],
                                                   tables['conductances_AMPA_GABA'],
                                                   tables['conductances_NMDA'])
    n_in = dict(spec['poisson_N'], **(poisson_N or {}))

    cfg = copy.deepcopy(_BASE)
    cfg['layers'] = {spec['name']: {
        'connection_prob': connection_prob,
        'conductance': conductance,
        'poisson_inputs': {p: {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': int(n_in[p])}
                           for p in spec['neuron_counts']},
        'input_rate': INPUT_RATE,
        'neuron_counts': {p: int(n) for p, n in spec['neuron_counts'].items()},
        'coordinates': COORDINATES,
    }}
    cfg['inter_layer_connections'] = {}
    cfg['inter_layer_conductances'] = {}
    cfg['lesioned_inter_layer'] = []
    cfg['minimal'] = {'exp': exp, 'target_rates_hz': spec['target_rates_hz']}
    if 'tau_i_SOM_ms' in spec:                  # SOM synaptic decay (all SOM synapses share it)
        cfg['models']['common_namespace']['tau_i_SOM'] = spec['tau_i_SOM_ms'] * ms
        cfg['time_constants']['I_SOM'] = spec['tau_i_SOM_ms'] * ms
    if 'tau_i_SOM_E_ms' in spec:                # SOM->E decay only: E cells' gSOM gets its own constant
        import re
        eq, n = re.subn(r'dgSOM/dt\s*=\s*-gSOM/tau_i_SOM\b', 'dgSOM/dt = -gSOM/tau_i_SOM_E',
                        cfg['models']['equations']['E'])
        if n != 1:
            raise ValueError('expected one SOM decay line in the E equation')
        cfg['models']['equations']['E'] = eq
        cfg['models']['common_namespace']['tau_i_SOM_E'] = spec['tau_i_SOM_E_ms'] * ms
    for pop, params in spec.get('intrinsic', {}).items():   # {pop: {'a': nS, 'b': pA, 'tauw': ms}}
        from brian2 import nS, pA
        units = {'a': nS, 'b': pA, 'tauw': ms, 'gL': nS}
        for k, v in params.items():                          # read by add_heterogeneity_to_layer
            cfg['intrinsic_params'][pop][k] = v * units[k]
    cfg['minimal']['tonic_inh'] = spec.get('tonic_inh', {})
    cfg['minimal']['w_fixed_pA'] = spec.get('w_fixed_pA', {})
    if 'intra_layer_delays' in spec:            # {(pre, post): (mean_ms, std_ms)}, read by src/layer.py
        cfg['intra_layer_delays'] = dict(spec['intra_layer_delays'])
    return cfg


def add_tonic_inhibition(col, cfg):
    """Non-recurrent inhibition: {pop: (N, rate_hz, g_nS)} from the condition's 'tonic_inh' becomes an
    independent Poisson input onto that population's gSOM (the control for an added SOM->SOM)."""
    from brian2 import PoissonInput, nS
    (layer,) = col.layers.values()
    for pop, w in cfg['minimal'].get('w_fixed_pA', {}).items():   # frozen adaptation current (after heterogeneity)
        from brian2 import pA
        layer.neuron_groups[pop].w = w * pA
    for pop, (n, rate, g) in cfg['minimal']['tonic_inh'].items():
        col.network.add(PoissonInput(layer.neuron_groups[pop], 'gSOM', N=int(n), rate=rate * Hz, weight=g * nS))


CONFIG = make_config()
