"""Every swap experiment, defined as a change to a base layer's circuit.

A condition is built in memory (no folders): take the BASE layer's 4x4 tables and
cell counts, then move in parts of the DONOR layer:
    rows    whole source rows (all outputs of that population), all three CSVs
    cells   single (pre, post) connections, all three CSVs
    conv    scale moved conductances by N_pre(donor) / N_pre(base), so each target
            cell gets the donor's total input (N_pre * p * g) -- convergence-matched
    counts  use the donor's neuron counts
    scale   {(pre, post): factor} multiplies that connection's conductances (AMPA/GABA
            and NMDA), probability unchanged
    remove  [(pre, post), ...] sets the probability to 0 (knockout)
    add     {(pre, post): (p, g_nS)} sets a connection's probability and AMPA/GABA conductance
            (for connections no layer has, e.g. SOM->SOM)
    pops    only build these populations (others do not exist; their connections are skipped)
Targets are intact firing rates of the named layer. Poisson N comes from
tuned_N.json (written by match_rates.py).

L6's tables are minimal_config/*.csv; other layers are read from config/*.csv.
"""
import itertools
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MINI = os.path.dirname(HERE)
ROOT = os.path.dirname(MINI)
CSVS = ['connection_probabilities', 'conductances_AMPA_GABA', 'conductances_NMDA']
TYPES = ['E', 'PV', 'SOM', 'VIP']
TUNED_N = os.path.join(HERE, 'tuned_N.json')

# Intact column, baseline, seed 58925, 500-3000 ms (measured 2026-09-29).
INTACT_RATES_HZ = {
    'L23':  {'E': 1.45, 'PV': 6.86, 'SOM': 3.50, 'VIP': 1.59},
    'L4AB': {'E': 2.18, 'PV': 5.71, 'SOM': 4.20, 'VIP': 3.75},
    'L4C':  {'E': 1.48, 'PV': 6.69, 'SOM': 4.12, 'VIP': 4.12},
    'L5':   {'E': 1.49, 'PV': 5.46, 'SOM': 2.36, 'VIP': 1.30},
    'L6':   {'E': 1.60, 'PV': 6.05, 'SOM': 3.80, 'VIP': 2.16},
}
# Poisson N that rate-matched each layer when disconnected; starting point for new conditions.
START_N = {
    'L23':  {'E': 66, 'PV': 415, 'SOM': 75, 'VIP': 19},
    'L4AB': {'E': 76, 'PV': 309, 'SOM': 71, 'VIP': 25},
    'L4C':  {'E': 54, 'PV': 160, 'SOM': 70, 'VIP': 32},
    'L5':   {'E': 49, 'PV': 190, 'SOM': 60, 'VIP': 44},
    'L6':   {'E': 45, 'PV': 44,  'SOM': 28, 'VIP': 16},
}
START_N['L6new'] = START_N['L6']

CONDITIONS = {
    # Forward: L4C circuit + pieces of L6, at L6 rates
    'L4C':             dict(base='L4C', targets='L6'),
    'L4C+L6_E+counts': dict(base='L4C', donor='L6', rows=['E'], counts=True, targets='L6'),
    'L4C+L6_PV':       dict(base='L4C', donor='L6', rows=['PV'], targets='L6'),
    'L4C+L6_SOM':      dict(base='L4C', donor='L6', rows=['SOM'], targets='L6'),
    'L4C+L6_VIP':      dict(base='L4C', donor='L6', rows=['VIP'], targets='L6'),
    'L4C+L6_counts':   dict(base='L4C', donor='L6', counts=True, targets='L6'),
    'L4C+L6_VIP-SOM':  dict(base='L4C', donor='L6', cells=[('VIP', 'SOM')], targets='L6'),
    'L4C+L6_PV-VIP':   dict(base='L4C', donor='L6', cells=[('PV', 'VIP')], targets='L6'),
    # (L4C + L6 E row alone cannot be rate-matched: SOM fires at 7 Hz with no Poisson input.)

    # Reverse: L6 circuit + pieces of L4C, at L6 rates (c = convergence-matched)
    'L6':                     dict(base='L6'),
    'L6+L4C_VIP-SOM':         dict(base='L6', donor='L4C', cells=[('VIP', 'SOM')]),
    'L6+L4C_VIP-SOMc':        dict(base='L6', donor='L4C', cells=[('VIP', 'SOM')], conv=True),
    'L6+L4C_PV-VIPc':         dict(base='L6', donor='L4C', cells=[('PV', 'VIP')], conv=True),
    'L6+L4C_E-SOMc':          dict(base='L6', donor='L4C', cells=[('E', 'SOM')], conv=True),
    'L6+L4C_VIP-SOMc+PV-VIPc': dict(base='L6', donor='L4C', conv=True,
                                    cells=[('VIP', 'SOM'), ('PV', 'VIP')]),
    'L6+L4C_VIP-SOMc+E-SOMc': dict(base='L6', donor='L4C', conv=True,
                                   cells=[('VIP', 'SOM'), ('E', 'SOM')]),
    'L6+L4C_VIP-SOMc+PV-VIPc+E-SOMc': dict(base='L6', donor='L4C', conv=True,
                                           cells=[('VIP', 'SOM'), ('PV', 'VIP'), ('E', 'SOM')]),

    # Other layers + pieces of L6, at their own rates
    # (L5's VIP->SOM already equals L6's, so it has no VIP variants.)
    'L5':                     dict(base='L5'),
    'L5+L6_E-SOMc':           dict(base='L5', donor='L6', cells=[('E', 'SOM')], conv=True),
    'L23':                    dict(base='L23'),
    'L23+L6_E-SOMc':          dict(base='L23', donor='L6', cells=[('E', 'SOM')], conv=True),
    'L23+L6_VIP-SOMc':        dict(base='L23', donor='L6', cells=[('VIP', 'SOM')], conv=True),
    'L23+L6_E-SOMc+VIP-SOMc': dict(base='L23', donor='L6', conv=True,
                                   cells=[('E', 'SOM'), ('VIP', 'SOM')]),
    'L4AB':                   dict(base='L4AB'),
    'L4C@own':                dict(base='L4C'),   # L4C at its own intact rates

    # Reduced L6 circuits: only the listed populations exist, at L6 rates
    'L6_E+SOM':               dict(base='L6', pops=['E', 'SOM']),
    'L6_E+SOM+PV':            dict(base='L6', pops=['E', 'PV', 'SOM']),
    'L6_E+SOM+VIP':           dict(base='L6', pops=['E', 'SOM', 'VIP']),
    'L6_E+PV':                dict(base='L6', pops=['E', 'PV']),
}

# E->SOM x VIP->SOM map on the L6 circuit, at L6 rates: E->SOM conductances (AMPA and
# NMDA) scaled by SWEEP_E_SCALE; VIP->SOM conductance scaled to give SWEEP_VIP_CONV
# (L6's own is 70 * 0.11 * 0.2015 = 1.55). Names: 'sweep_E0.65_V7.0'.
SWEEP_E_SCALE = [0.35, 0.5, 0.65, 0.8, 1.0, 1.15]
SWEEP_VIP_CONV = [0.5, 1.55, 4.0, 7.0, 10.0, 13.0, 16.0]
L6_VIP_SOM_CONV = 70 * 0.11 * 0.2015
for _e in SWEEP_E_SCALE:
    for _v in SWEEP_VIP_CONV:
        CONDITIONS[f'sweep_E{_e}_V{_v}'] = dict(
            base='L6', scale={('E', 'SOM'): _e, ('VIP', 'SOM'): _v / L6_VIP_SOM_CONV})

# Knockout screen: remove one existing connection at a time from the L6 and L4C
# circuits, both at L6 rates. Names: 'L6-ko_VIP-SOM', 'L4C-ko_E-PV'.
KO_TYPES = ['E', 'PV', 'SOM', 'VIP']


# configL6's L6 circuit ('L6new': L6 block of the "copy" CSVs that configL6 loads) with
# single old (config.py) L6 connections put back, at L6 rates. Asks which of the changed
# L6 connections removed its alpha. Names: 'L6new+old_SOM-E'.
CONDITIONS['L6new'] = dict(base='L6new', targets='L6')
for _name, _cells in {'SOM-E': [('SOM', 'E')], 'PV-E': [('PV', 'E')], 'SOM-PV': [('SOM', 'PV')],
                      'SOM-VIP': [('SOM', 'VIP')],
                      'VIProw': [('VIP', 'E'), ('VIP', 'PV'), ('VIP', 'VIP')],
                      'SOM-E+PV-E': [('SOM', 'E'), ('PV', 'E')],
                      'SOM-E+PV-E+SOM-PV': [('SOM', 'E'), ('PV', 'E'), ('SOM', 'PV')]}.items():
    CONDITIONS[f'L6new+old_{_name}'] = dict(base='L6new', donor='L6', cells=_cells, targets='L6')

# Cosyne Fig 1 factorial: every subset of four components moved from the donor layer into
# the base layer, convergence-matched receptor by receptor (per-cell K*g of AMPA, NMDA and
# GABA each equal the donor's), at the base layer's own rates.
#   L23 <- L6 values:  'fx_L23+ES+VS' ...      L6 <- L23 values: 'fx_L6+ES+VS' ...
# The empty subset is the plain base condition ('L23' / 'L6').
FX_COMPONENTS = {'ES': ('E', 'SOM'), 'SE': ('SOM', 'E'), 'VS': ('VIP', 'SOM'), 'PE': ('PV', 'E'),
                 'SP': ('SOM', 'PV'), 'EP': ('E', 'PV')}


def fx_name(base, comps):
    return base if not comps else 'fx_' + base + ''.join('+' + c for c in comps)


for _base, _donor in [('L23', 'L6'), ('L6', 'L23')]:
    for _k in range(1, len(FX_COMPONENTS) + 1):
        for _comps in itertools.combinations(FX_COMPONENTS, _k):
            CONDITIONS[fx_name(_base, _comps)] = dict(
                base=_base, donor=_donor, conv=True, start_from=_base,
                cells=[FX_COMPONENTS[c] for c in _comps])

# Finite-size test: isolated L6 with every population x k and every probability / k, so each
# cell keeps the same in-degree, conductances and Poisson drive (same input statistics); only N
# changes. Noise-driven resonance predicts population alpha power ~ 1/N. Names: 'L6_Nx2'.
NSCALE = [0.5, 1, 2, 4, 8]
for _k in NSCALE:
    CONDITIONS[f'L6_Nx{_k:g}'] = dict(base='L6', nscale=_k, start_from='L6')

# PV->E normalisation check (reviewer: L6's weaker per-cell PV->E is partly because L6 is
# thinner, 280 vs 650 um, with one p across the layer). L6/L23 PV->E K*g = 0.572 =
# count 0.750 x p 0.917 x g 0.832. Corrections applied to L6's own PV->E conductance:
#   x1.333  remove the PV-count factor only (L6 as if it had L23's PV count)
#   x1.75   per PV/E ratio (L6 = L23 per ratio, 0.987) -- same K*g as fx_L6+PE
#   x2.33   per PV density (L6 PV/mm3 is 1.74x L23's)
#   x0.684  data anchor: Billeh et al. compilation, per-cell PV->E L6/L23 ~0.39 (model 0.572)
for _f in [0.684, 1.333, 1.75, 2.33]:
    CONDITIONS[f'L6_PEx{_f:g}'] = dict(base='L6', scale={('PV', 'E'): _f}, start_from='L6')

# Pre-registered disinhibition test (minimal_config/experiments/disinh/PREREGISTRATION.md):
# D = PV->E x SOM->PV per-cell K*g, changed through one factor at a time. Names 'pd_L23_SPx0.5'.
for _base, _pre, _post, _fs in [('L23', 'SOM', 'PV', [0.5, 0.25, 0.125, 0.0625]),
                                ('L23', 'PV', 'E', [0.5, 0.25]),
                                ('L6', 'SOM', 'PV', [0.25, 0.5, 2, 4])]:
    _tag = 'SP' if _pre == 'SOM' else 'PE'
    for _f in _fs:
        CONDITIONS[f'pd_{_base}_{_tag}x{_f:g}'] = dict(base=_base, scale={(_pre, _post): _f}, start_from=_base)

# '+PD': PV's external drive also moved, i.e. PV's Poisson N fixed at the donor's tuned
# value (same input rate and weight, so the same external drive per PV cell). PV can then
# NOT be rate-matched (its rate is free); only E, SOM and VIP are matched.
PD_SETS = [(), ('PE', 'SP', 'EP'), tuple(FX_COMPONENTS)]
FX_PV_N = {'L23': 363, 'L6': 44}     # tuned_N.json PV N of the isolated layers
for _base, _donor in [('L23', 'L6'), ('L6', 'L23')]:
    for _comps in PD_SETS:
        CONDITIONS[fx_name(_base, _comps + ('PD',))] = dict(
            base=_base, donor=_donor, conv=True, start_from=_base,
            cells=[FX_COMPONENTS[c] for c in _comps], fixed_N={'PV': FX_PV_N[_donor]})


# Frequency figure: SOM synaptic decay tau_i_SOM (one constant shared by every SOM synapse: onto
# E, PV and VIP) set to <t> ms, with every SOM output conductance x 20/<t> so the charge per spike
# is unchanged, at L6 rates. tau 20 ms is the plain 'L6' condition. Names: 'L6_tauSOM10'.
TAU_SOM_MS = [10, 15, 30, 40]
for _t in TAU_SOM_MS:
    CONDITIONS[f'L6_tauSOM{_t:g}'] = dict(base='L6', start_from='L6', tau_som_ms=_t,
                                          scale={('SOM', _post): 20.0 / _t for _post in ['E', 'PV', 'VIP']})

# E-SOM loop delay: extra mean delay <d> ms on BOTH E->SOM and SOM->E (default 1.5 +- 0.5 ms each,
# src/layer.py), so the loop gains 2<d> ms; at L6 rates. d = 0 is 'L6'. Names: 'L6_loopdelay+5'.
LOOP_EXTRA_DELAY_MS = [2.5, 5, 10]
for _d in LOOP_EXTRA_DELAY_MS:
    CONDITIONS[f'L6_loopdelay+{_d:g}'] = dict(base='L6', start_from='L6',
                                              delays={('E', 'SOM'): (1.5 + _d, 0.5), ('SOM', 'E'): (1.5 + _d, 0.5)})


# CSID frequency figure (Sweep A): ONLY the SOM->E decay (the gSOM decay in E cells) set to <t> ms,
# SOM->E conductance x 20/<t> (constant charge); SOM->PV and SOM->VIP keep 20 ms. As in the handoff's
# intact-L6 timing test. tau 20 = 'L6'. Names: 'L6_tauSOME10'.
for _t in [10, 15, 30, 40]:
    CONDITIONS[f'L6_tauSOME{_t:g}'] = dict(base='L6', start_from='L6', tau_som_e_ms=_t,
                                           scale={('SOM', 'E'): 20.0 / _t})

# CSID frequency figure (Sweep B): E->SOM and SOM->E delays (mean AND sd; default 1.5 +- 0.5 ms each,
# src/layer.py) both multiplied by <s>. x1 = 'L6'. Names: 'L6_loopdelayx0.5'.
for _s in [0.5, 1.5, 2]:
    CONDITIONS[f'L6_loopdelayx{_s:g}'] = dict(base='L6', start_from='L6',
                                              delays={('E', 'SOM'): (1.5 * _s, 0.5 * _s), ('SOM', 'E'): (1.5 * _s, 0.5 * _s)})

# CSID per-layer figure: L4AB knockouts at L4AB's own rates (the L6/L4C knockout screen ran at L6 rates).
for _pre, _post in [('VIP', 'SOM'), ('E', 'VIP')]:
    CONDITIONS[f'L4AB@own-ko_{_pre}-{_post}'] = dict(base='L4AB', remove=[(_pre, _post)], start_from='L4AB')

# SOM->SOM test (12 Oct 2026): does the alpha need SOM cells NOT to inhibit each other? No layer has
# SOM->SOM, so it is added: p = 0.3 (L6's SOM->PV probability, 33 SOM inputs per SOM cell), GABA
# conductance <g> nS (0.44 = L6 SOM->E, 1.95 = L6 PV->PV), decay tau_i_SOM, delay 1.5 +- 0.5 ms.
# In the E+SOM circuit and in the whole L6 circuit, at L6 rates. Names: 'L6_E+SOM_SS0.44', 'L6_SS0.44'.
SOM_SOM_G = [0.2, 0.44, 1.0, 1.95]
for _g in SOM_SOM_G:
    CONDITIONS[f'L6_E+SOM_SS{_g:g}'] = dict(base='L6', pops=['E', 'SOM'], start_from='L6_E+SOM', add={('SOM', 'SOM'): (0.3, _g)})
    # control: the same mean inhibition (33 inputs at SOM's 3.8 Hz target, <g> nS) but from independent
    # Poisson sources, so SOM cells are inhibited without inhibiting EACH OTHER. Names: 'L6_E+SOM_tonic0.44'.
    CONDITIONS[f'L6_E+SOM_tonic{_g:g}'] = dict(base='L6', pops=['E', 'SOM'], start_from=f'L6_E+SOM_SS{_g:g}',
                                               tonic_inh={'SOM': (33, 3.8, _g)})
    CONDITIONS[f'L6_SS{_g:g}'] = dict(base='L6', start_from='L6', add={('SOM', 'SOM'): (0.3, _g)})

# Slow (~5 Hz) rhythm test (12 Oct 2026): which populations of L4C / L4AB are needed for it? Every
# subset of populations that contains E, at the layer's own rates; plus E+SOM+VIP with E->VIP removed
# (VIP still inhibits SOM but is driven by Poisson input only, so the E->VIP-|SOM-|E loop is open).
# The full circuits are 'L4C@own' and 'L4AB'. Names: 'L4C@own_E+SOM+VIP', 'L4AB_E+SOM+VIP-ko_E-VIP'.
for _base, _tag in [('L4C', 'L4C@own'), ('L4AB', 'L4AB')]:
    for _pops in [['E'], ['E', 'PV'], ['E', 'SOM'], ['E', 'VIP'], ['E', 'PV', 'SOM'], ['E', 'PV', 'VIP'],
                  ['E', 'SOM', 'VIP']]:
        CONDITIONS[f'{_tag}_' + '+'.join(_pops)] = dict(base=_base, pops=_pops, start_from=_tag)
    CONDITIONS[f'{_tag}_E+SOM+VIP-ko_E-VIP'] = dict(base=_base, pops=['E', 'SOM', 'VIP'], start_from=_tag,
                                                   remove=[('E', 'VIP')])


# Adaptation test (Cosyne reviewer, 15 Oct 2026): is spike-frequency adaptation (tauw E 100, SOM 180,
# VIP 150 ms; PV has none) the hidden slow element? 'noadapt' sets a = b = 0 (no adaptation current);
# 'fastadapt' keeps a and b but sets tauw = 10 ms. Applied by minimal_config/config.py before
# heterogeneity, at L6 rates. Names: 'L6_noadaptE', 'L6_noadapt', 'L6_E+SOM_noadapt'.
_NOADAPT = {'a': 0, 'b': 0}
for _tag, _intr in {'noadaptE': {'E': _NOADAPT}, 'noadaptSOM': {'SOM': _NOADAPT},
                    'noadapt': {p: _NOADAPT for p in ['E', 'SOM', 'VIP']},
                    'fastadapt': {p: {'tauw': 10} for p in ['E', 'SOM', 'VIP']}}.items():
    CONDITIONS[f'L6_{_tag}'] = dict(base='L6', start_from='L6', intrinsic=_intr)
# 'meanw': adaptation DYNAMICS removed but its mean kept -- a = b = 0, tauw = 1e12 ms (w constant) and
# every cell's w fixed at its population's mean baseline adaptation current (measured 15 Oct, isolated L6,
# 3 seeds, 1-4 s: E 41.1, SOM 56.8, VIP 22.3 pA; E+SOM circuit: E 39.4, SOM 49.8). Needed because with
# a = b = 0 SOM fires at 17-23 Hz even with Poisson N = 1, so the plain 'noadapt' cannot be rate-matched.
_FROZEN = {'a': 0, 'b': 0, 'tauw': 1e12}
_W_L6 = {'E': 41.1, 'SOM': 56.8, 'VIP': 22.3}
for _tag, _pops in {'E': ['E'], 'SOM': ['SOM'], '': ['E', 'SOM', 'VIP']}.items():
    CONDITIONS[f'L6_meanw{_tag}'] = dict(base='L6', start_from='L6', intrinsic={p: _FROZEN for p in _pops},
                                         w_fixed_pA={p: _W_L6[p] for p in _pops})
CONDITIONS['L6_E+SOM_meanw'] = dict(base='L6', pops=['E', 'SOM'], start_from='L6_E+SOM',
                                    intrinsic={p: _FROZEN for p in ['E', 'SOM']},
                                    w_fixed_pA={'E': 39.4, 'SOM': 49.8})
# 'fastadapt_b': tauw = 10 ms with a = 0 (spike-triggered part only). With tauw = 10 ms the subthreshold
# term a(v - EL) acts as an extra ~4 nS leak (gL is 4.2-4.3 nS), so 'fastadapt' also halves C/gL.
CONDITIONS['L6_fastadapt_b'] = dict(base='L6', start_from='L6_fastadapt',
                                    intrinsic={p: {'tauw': 10, 'a': 0} for p in ['E', 'SOM', 'VIP']})
# Leak control for 'fastadapt': adaptation frozen at its mean ('meanw') plus gL raised by a (E, SOM +4 nS,
# VIP +2 nS) -- the extra leak a fast subthreshold adaptation adds, with no adaptation dynamics at all.
CONDITIONS['L6_meanw_leak'] = dict(base='L6', start_from='L6_meanw', w_fixed_pA=dict(_W_L6),
                                   intrinsic={'E': dict(_FROZEN, gL=8.2), 'SOM': dict(_FROZEN, gL=8.3),
                                              'VIP': dict(_FROZEN, gL=5.3)})
CONDITIONS['L6_E+SOM_noadapt'] = dict(base='L6', pops=['E', 'SOM'], start_from='L6_E+SOM',
                                      intrinsic={p: _NOADAPT for p in ['E', 'SOM']})

# Sensitivity map (Cosyne reviewer): per-cell PV->E and SOM->PV input (K*g, nS) on a grid spanning the L6
# and L23 values, at VIP->SOM of L6 or L23, built on BOTH the L6 and the L23 circuit, each at its own
# rates. Set through the conductance (p unchanged). Names: 'map_L6_PE19.6_SP11.6_VS1.55'.
MAP_KG = {'PE': [9.8, 19.6, 26.0, 34.2, 48.0],          # L6 19.56, L23 34.20
          'SP': [5.8, 11.6, 21.0, 37.2, 58.0],          # L6 11.58, L23 37.15
          'VS': [1.55, 12.9]}                          # L6 1.55,  L23 12.90
_MAP_PAIR = {'PE': ('PV', 'E'), 'SP': ('SOM', 'PV'), 'VS': ('VIP', 'SOM')}


def _add_map():
    for base in ['L6', 'L23']:
        tab, cnt = layer_tables(base), layer_counts(base)
        own = {k: cnt[a] * tab['connection_probabilities'].loc[a, b] * tab['conductances_AMPA_GABA'].loc[a, b]
               for k, (a, b) in _MAP_PAIR.items()}
        for pe, sp, vs in itertools.product(MAP_KG['PE'], MAP_KG['SP'], MAP_KG['VS']):
            CONDITIONS[f'map_{base}_PE{pe:g}_SP{sp:g}_VS{vs:g}'] = dict(
                base=base, start_from=base,
                scale={_MAP_PAIR[k]: v / own[k] for k, v in zip(['PE', 'SP', 'VS'], [pe, sp, vs])})


def _add_knockouts():
    for base in ['L6', 'L4C']:
        prob = layer_tables(base)['connection_probabilities']
        for pre in KO_TYPES:
            for post in KO_TYPES:
                if prob.loc[pre, post] > 0:
                    CONDITIONS[f'{base}-ko_{pre}-{post}'] = dict(
                        base=base, remove=[(pre, post)], targets='L6',
                        start_from='L4C' if base == 'L4C' else 'L6')


def layer_tables(layer):
    """{csv name: 4x4 DataFrame (E/PV/SOM/VIP)} for one layer."""
    if layer == 'L6':
        return {f: pd.read_csv(os.path.join(MINI, f'{f}.csv'), index_col=0) for f in CSVS}
    src = layer
    if layer == 'L6new':                            # configL6's CSVs (no NMDA copy exists)
        src = 'L6'
    out = {}
    for f in CSVS:
        idx = [f'{t}_{src}' for t in TYPES]
        fname = f'{f} copy.csv' if layer == 'L6new' and f != 'conductances_NMDA' else f'{f}.csv'
        df = pd.read_csv(os.path.join(ROOT, 'config', fname), index_col=0).loc[idx, idx].copy()
        df.index, df.columns = TYPES, TYPES
        df.index.name = 'Source'
        out[f] = df
    return out


def layer_counts(layer):
    if layer in ('L6', 'L6new'):
        return dict(json.load(open(os.path.join(MINI, 'layer.json')))['neuron_counts'])
    import sys
    sys.path.insert(0, ROOT)
    from config.config import CONFIG
    return dict(CONFIG['layers'][layer]['neuron_counts'])


def build(name):
    """(tables, spec) for a condition; spec has neuron_counts, poisson_N, target_rates_hz."""
    c = CONDITIONS[name]
    base, donor = c['base'], c.get('donor')
    tables = {f: df.copy() for f, df in layer_tables(base).items()}
    counts = layer_counts(base)
    if donor:
        d_tables, d_counts = layer_tables(donor), layer_counts(donor)
        for f in CSVS:
            for r in c.get('rows', []):
                tables[f].loc[r] = d_tables[f].loc[r]
            for pre, post in c.get('cells', []):
                v = d_tables[f].loc[pre, post]
                if c.get('conv') and f != 'connection_probabilities':
                    v = round(float(v * d_counts[pre] / counts[pre]), 4)
                tables[f].loc[pre, post] = v
        if c.get('counts'):
            counts = d_counts
    for pre, post in c.get('remove', []):
        tables['connection_probabilities'].loc[pre, post] = 0.0
    for (pre, post), (p, g) in c.get('add', {}).items():
        tables['connection_probabilities'].loc[pre, post] = p
        tables['conductances_AMPA_GABA'].loc[pre, post] = g
    for (pre, post), factor in c.get('scale', {}).items():
        for f in CSVS[1:]:
            tables[f].loc[pre, post] = round(float(tables[f].loc[pre, post] * factor), 5)
    start = START_N[donor if c.get('counts') else base]
    tuned_all = json.load(open(TUNED_N)) if os.path.exists(TUNED_N) else {}
    tuned = tuned_all.get(name, {})
    if not tuned and c.get('start_from') in tuned_all:          # start from the tuned base
        start = tuned_all[c['start_from']]['poisson_N']
    spec = {'name': 'L6', 'neuron_counts': counts,
            'poisson_N': tuned.get('poisson_N', start),
            'target_rates_hz': INTACT_RATES_HZ[c.get('targets', base)]}
    if 'nscale' in c:                                    # N x k, p / k: same in-degree per cell
        k = c['nscale']
        counts = {pop: int(round(n * k)) for pop, n in counts.items()}
        spec['neuron_counts'] = counts
        tables['connection_probabilities'] = tables['connection_probabilities'] / k
    if 'tau_som_ms' in c:                                # applied by minimal_config/config.py
        spec['tau_i_SOM_ms'] = c['tau_som_ms']
    if 'delays' in c:
        spec['intra_layer_delays'] = c['delays']
    if 'tau_som_e_ms' in c:
        spec['tau_i_SOM_E_ms'] = c['tau_som_e_ms']
    if 'fixed_N' in c:                                   # not tuned by match_rates
        spec['poisson_N'] = dict(spec['poisson_N'], **c['fixed_N'])
        spec['fixed_pops'] = sorted(c['fixed_N'])
    if 'pops' in c:
        spec = {k: ({p: v[p] for p in c['pops']} if isinstance(v, dict) else v) for k, v in spec.items()}
    if 'tonic_inh' in c:
        spec['tonic_inh'] = c['tonic_inh']
    if 'intrinsic' in c:
        spec['intrinsic'] = c['intrinsic']
    if 'w_fixed_pA' in c:
        spec['w_fixed_pA'] = c['w_fixed_pA']
    return tables, spec


def save_tuned(name, poisson_N, rates):
    import fcntl
    with open(TUNED_N + '.lock', 'w') as lock:      # several match_rates may run in parallel
        fcntl.flock(lock, fcntl.LOCK_EX)
        tuned = json.load(open(TUNED_N)) if os.path.exists(TUNED_N) else {}
        tuned[name] = {'poisson_N': poisson_N,
                       'matched_rates_hz': {p: round(r, 3) for p, r in rates.items()}}
        json.dump(dict(sorted(tuned.items())), open(TUNED_N, 'w'), indent=1)


_add_knockouts()
_add_map()
