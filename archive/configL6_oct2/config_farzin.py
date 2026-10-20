"""E-PV-SOM network of code_E_PV_SOM_Fig1.ipynb, as a config for this framework.

Same neuron counts, equations, intrinsic parameters, synaptic time constants,
connection probabilities, conductances, external Poisson drive, zero delays and
no heterogeneity as the notebook. Use it from main.py with
    from config.config_farzin import CONFIG
The whole network is one "layer" called 'net' (no inter-layer connections).

Notebook -> framework mapping:
  - one inhibitory conductance gi per cell  ->  gPV + gSOM with the SAME decay
    (the sum behaves identically; it just lets the framework route PV and SOM)
  - ge                                     ->  gE_AMPA (no NMDA anywhere)
  - Delay = 0 ms                           ->  'intra_layer_delays' all (0, 0)
  - no heterogeneity                       ->  'heterogeneity_scale': 0
  - PV->SOM and SOM->SOM have weight 0 and the 1-cell SOM Poisson group fires
    at 0 Hz in the notebook, so they are left out (no effect on dynamics).
  - SOM gets no external input (notebook rate 0 Hz).
Not identical: the random draws (connectivity, initial v, Poisson spikes) come in a
different order, so runs match the notebook statistically, not spike-for-spike.
"""
from brian2 import *

# ---------------------------------------------------------------- notebook values
N_E, N_PV = 8000, 800
N_SOM = int(600 * (1 - 0.002))          # NI2*(1-rr) = 598
pp = 0.017
Q_e, Q_i = 1.25, 5                        # nS
tau_i, tau_e = 5*ms, 5*ms                 # onto E (RS)
tau_i2 = 5*ms                             # inhibition onto PV and SOM
tau_e2, tau_e3 = 1*ms, 2*ms               # excitation onto PV, onto SOM
tauE_r, tauI_r = 5*ms, 5*ms               # reference time scales for weight rescaling
N_ext, f_ext = int(pp * N_E), 2*Hz        # 136 Poisson inputs at 4 Hz onto E and PV

# conductances (nS), each exactly as written in the notebook's on_pre
G = {
    'E_E':    0.5 * Q_e,                                  # Gee
    'E_PV':   0.5 * 1.2 * Q_e * float(tauE_r / tau_e2),   # 3.75
    'E_SOM':  0.8 * Q_e * float(tauE_r / tau_e3),         # 2.5
    'PV_E':   0.7 * Q_i,                                  # 1.2*Gie = 6
    'PV_PV':  1.2 * Q_i * float(tauI_r / tau_i2),         # 6
    'SOM_E':  Q_i,                                        # Gii = 5
    'SOM_PV': Q_i,                                        # Gii = 5
}
P = {'E_E': pp, 'E_PV': 2 * pp, 'E_SOM': pp,
     'PV_E': pp, 'PV_PV': pp, 'SOM_E': pp, 'SOM_PV': pp}


def _eqs(tau_exc, tau_inh):
    return f"""
        dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT)
                 + gE_AMPA*(Ee - v) - (gPV + gSOM)*(v - Ei) - w + I)/C : volt (unless refractory)
        IsynE = gE_AMPA*(Ee - v) : amp
        IsynI = (gPV + gSOM)*(Ei - v) : amp
        dgE_AMPA/dt = -gE_AMPA/{tau_exc} : siemens
        dgPV/dt     = -gPV/{tau_inh} : siemens
        dgSOM/dt    = -gSOM/{tau_inh} : siemens
        dw/dt = (a*(v - EL) - w)/tauw : amp
        I : amp
        a : siemens
        b : amp
        DeltaT : volt
        Vcut : volt
        EL : volt
        C : farad
        gL : siemens
        tauw : second
    """


_IC = {'v': '-65*mV + rand()*15*mV',               # uniform(-65, -50) mV
       'gE_AMPA': 0*nS, 'gPV': 0*nS, 'gSOM': 0*nS,
       'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5}   # Vcut = VT + 5*DeltaT

CONFIG = {
    'simulation': {'SIMULATION_TIME': 4000*ms, 'DT': 0.1*ms, 'RANDOM_SEED': 57},

    'models': {
        'equations': {
            'E':   _eqs('tau_e', 'tau_i'),         # RS:  eqs
            'PV':  _eqs('tau_e2', 'tau_i2'),       # FS:  eqs2
            'SOM': _eqs('tau_e3', 'tau_i2'),       # SST: eqs3
        },
        'threshold': 'v>Vcut',
        'reset': 'v=V_reset; w+=b',
        'common_namespace': {
            'tau_e': tau_e, 'tau_i': tau_i, 'tau_e2': tau_e2, 'tau_e3': tau_e3, 'tau_i2': tau_i2,
            'VT': -50*mV, 'V_reset': -65*mV, 'Ee': 0*mV, 'Ei': -80*mV,
        },
    },

    # all populations use the notebook's global tauw = 500 ms
    'intrinsic_params': {
        'E':   {'a': 4*nS, 'b': 130*pA, 'DeltaT': 2*mV,   'C': 200*pF, 'gL': 10*nS,
                'tauw': 500*ms, 'EL': -60*mV},
        'PV':  {'a': 0*nS, 'b': 0*pA,   'DeltaT': 0.5*mV, 'C': 200*pF, 'gL': 10*nS,
                'tauw': 500*ms, 'EL': -60*mV},
        'SOM': {'a': 4*nS, 'b': 25*pA,  'DeltaT': 1.5*mV, 'C': 200*pF, 'gL': 10*nS,
                'tauw': 500*ms, 'EL': -55*mV},
    },
    'heterogeneity_scale': 0,                       # identical neurons, as in the notebook

    'neurons': {'V_RESET': -65*mV, 'VT': -50*mV, 'EE': 0*mV, 'EI': -80*mV,
                'T_REF': {'E': 5*ms, 'PV': 5*ms, 'SOM': 5*ms}},

    'initial_conditions': {'DEFAULT': _IC, 'E': _IC, 'PV': _IC, 'SOM': _IC},

    'synapses': {'Q': {'EXT_AMPA': Q_e*nS}},        # Ge_extE

    'intra_layer_delays': {(a, b): (0.0, 0.0)       # Delay = 0 ms
                           for a in ['E', 'PV', 'SOM'] for b in ['E', 'PV', 'SOM']},

    'layers': {
        'net': {
            'connection_prob': P,
            # framework convention: excitatory keys end in _AMPA, inhibitory keys are pre_post
            'conductance': {('E_' + k[2:] + '_AMPA' if k.startswith('E_') else k): g
                            for k, g in G.items()},
            'poisson_inputs': {
                'E':  {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': N_ext},
                'PV': {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': N_ext},
                # SOM: none (notebook rate 0 Hz)
            },
            'input_rate': f_ext,
            'neuron_counts': {'E': N_E,  'SOM' : N_SOM, 'PV': N_PV},
            'coordinates': {'x': (-0.15, 0.15), 'y': (-0.15, 0.15), 'z': (-0.62, -0.34)},
        },
    },

    'inter_layer_connections': {},
    'inter_layer_conductances': {},
    'lesioned_inter_layer': [],
    'inter_layer_scaling': 1.0,

    'electrode_positions': [(0, 0, z) for z in
                            [-0.94, -0.79, -0.64, -0.49, -0.34, -0.19, -0.04, 0.10,
                             0.26, 0.40, 0.56, 0.70, 0.86, 1.00, 1.16, 1.30]],
}
