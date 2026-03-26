from tools.utils import *
from brian2 import *
import numpy as np
import pandas as pd
from collections import defaultdict

EXT_AMPA_WEIGHT = 1.25*nS
tau_e_AMPA = 5*ms
tau_i_PV   = 5*ms
tau_i_SOM  = 5*ms
v_reset = -65.*mV
vt      = -50.*mV
ee      = 0.*mV
ei      = -80.*mV

t_ref = {
    'E':   5*ms,
    'PV':  5*ms,
    'SOM': 5*ms,
}


csv_layer_configs, _INTER_LAYER_CONNECTIONS, _INTER_LAYER_CONDUCTANCES = load_connectivity_from_csv(
    'config_farzin/connection_probabilities2.csv',
    'config_farzin/conductances_AMPA2_alpha_v2.csv',
    'config_farzin/conductances_NMDA2_alpha_v2.csv',
    layers=['L1'],
    cell_types=['E', 'PV', 'SOM']  )


_LAYER_CONFIGS = {
    'L1': {
        'connection_prob': csv_layer_configs['L1']['connection_prob'],
        'conductance': csv_layer_configs['L1']['conductance'],
        'poisson_inputs': {
            'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 136},
            'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 136},
            # SOM gets 0 Hz Poisson in Farzin code, so no input
        },
        'input_rate': 4*Hz,
        'neuron_counts': {'E': 8000, 'PV': 800, 'SOM': 598},
        'coordinates': {
            'x': (-0.15, 0.15),
            'y': (-0.15, 0.15),
            'z': (-0.62, 1.1),
        },
    },
}


CONFIG = {
    'simulation': {
        'SIMULATION_TIME': 4000*ms,
        'DT': 0.1*ms,
        'RANDOM_SEED': 57,
    },

    'models': {
        'equations': {
            'E': """
        dv/dt = (
            gL*(EL - v)
          + gL*DeltaT*exp((v - VT)/DeltaT)
          + gE_AMPA*(Ee - v)
          - (gPV + gSOM)*(v - Ei)
          - w + I
        )/C : volt (unless refractory)

        IsynE_AMPA = gE_AMPA*(Ee - v) : amp
        IsynE      = IsynE_AMPA : amp

        IsynIPV    = gPV*(Ei - v) : amp
        IsynISOM   = gSOM*(Ei - v) : amp

        gI   = gPV + gSOM : siemens
        gE   = gE_AMPA : siemens
        IsynI = gI*(v - Ei) : amp

        dgE_AMPA/dt = -gE_AMPA/tau_e_AMPA : siemens
        dgPV/dt     = -gPV/tau_i_PV       : siemens
        dgSOM/dt    = -gSOM/tau_i_SOM     : siemens

        dw/dt   = (a*(v - EL) - w)/tauw : amp
        taum    = C/gL : second
        I       : amp
        a       : siemens
        b       : amp
        DeltaT  : volt
        Vcut    : volt
        EL      : volt
        C       : farad
        gL      : siemens
        tauw    : second
            """,

            'PV': """
        dv/dt = (
            gL*(EL - v)
          + gL*DeltaT*exp((v - VT)/DeltaT)
          + gE_AMPA*(Ee - v)
          - (gPV + gSOM)*(v - Ei)
          - w + I
        )/C : volt (unless refractory)

        IsynE_AMPA = gE_AMPA*(Ee - v) : amp
        IsynE      = IsynE_AMPA : amp

        IsynIPV    = gPV*(Ei - v) : amp
        IsynISOM   = gSOM*(Ei - v) : amp

        gI   = gPV + gSOM : siemens
        gE   = gE_AMPA : siemens
        IsynI = gI*(v - Ei) : amp

        dgE_AMPA/dt = -gE_AMPA/tau_e_AMPA : siemens
        dgPV/dt     = -gPV/tau_i_PV       : siemens
        dgSOM/dt    = -gSOM/tau_i_SOM     : siemens

        dw/dt   = (a*(v - EL) - w)/tauw : amp
        taum    = C/gL : second
        I       : amp
        a       : siemens
        b       : amp
        DeltaT  : volt
        Vcut    : volt
        EL      : volt
        C       : farad
        gL      : siemens
        tauw    : second
            """,

            'SOM': """
        dv/dt = (
            gL*(EL - v)
          + gL*DeltaT*exp((v - VT)/DeltaT)
          + gE_AMPA*(Ee - v)
          - (gPV + gSOM)*(v - Ei)
          - w + I
        )/C : volt (unless refractory)

        IsynE_AMPA = gE_AMPA*(Ee - v) : amp
        IsynE      = IsynE_AMPA : amp

        IsynIPV    = gPV*(Ei - v) : amp
        IsynISOM   = gSOM*(Ei - v) : amp

        gI   = gPV + gSOM : siemens
        gE   = gE_AMPA : siemens
        IsynI = gI*(v - Ei) : amp

        dgE_AMPA/dt = -gE_AMPA/tau_e_AMPA : siemens
        dgPV/dt     = -gPV/tau_i_PV       : siemens
        dgSOM/dt    = -gSOM/tau_i_SOM     : siemens

        dw/dt   = (a*(v - EL) - w)/tauw : amp
        taum    = C/gL : second
        I       : amp
        a       : siemens
        b       : amp
        DeltaT  : volt
        Vcut    : volt
        EL      : volt
        C       : farad
        gL      : siemens
        tauw    : second
            """,
        },

        'threshold': 'v>Vcut',
        'reset': 'v=V_reset; w+=b',

        'common_namespace': {
            'tau_e_AMPA': tau_e_AMPA,
            'tau_i_PV':   tau_i_PV,
            'tau_i_SOM':  tau_i_SOM,
            'VT':         vt,
            'V_reset':    v_reset,
            'Ee':         ee,
            'Ei':         ei,
        },
    },

    'intrinsic_params': {
        # RS (Regular Spiking) - Excitatory
        'E':   {'a': 4*nS, 'b': 130*pA, 'DeltaT': 2*mV,
                'C': 200*pF, 'gL': 10*nS, 'tauw': 500*ms, 'EL': -60*mV},
        # FS (Fast Spiking) - PV
        'PV':  {'a': 0*nS, 'b': 0*pA, 'DeltaT': 0.5*mV,
                'C': 200*pF, 'gL': 10*nS, 'tauw': 500*ms, 'EL': -60*mV},
        # SST (Somatostatin)
        'SOM': {'a': 4*nS, 'b': 25*pA, 'DeltaT': 1.5*mV,
                'C': 200*pF, 'gL': 10*nS, 'tauw': 500*ms, 'EL': -55*mV},
    },

    'neurons': {
        'V_RESET': v_reset,
        'VT':      vt,
        'EE':      ee,
        'EI':      ei,
        'T_REF':   t_ref,
        'INITIAL_VOLTAGE': -60*mV,
    },

    'initial_conditions': {
        'DEFAULT': {'v': '-65*mV + rand()*15*mV',
                    'gE_AMPA': 0*nS,
                    'gPV': 0*nS, 'gSOM': 0*nS,
                    'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'E':   {'v': '-65*mV + rand()*15*mV',
                'gE_AMPA': 0*nS,
                'gPV': 0*nS, 'gSOM': 0*nS,
                'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'PV':  {'v': '-65*mV + rand()*15*mV',
                'gE_AMPA': 0*nS,
                'gPV': 0*nS, 'gSOM': 0*nS,
                'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'SOM': {'v': '-65*mV + rand()*15*mV',
                'gE_AMPA': 0*nS,
                'gPV': 0*nS, 'gSOM': 0*nS,
                'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
    },

    'time_constants': {
        'E_AMPA': tau_e_AMPA,
        'I_PV':   tau_i_PV,
        'I_SOM':  tau_i_SOM,
    },

    'synapses': {
        'Q': {
            'EXT_AMPA': EXT_AMPA_WEIGHT,
        },
    },

    'layers': _LAYER_CONFIGS,

    'inter_layer_connections': _INTER_LAYER_CONNECTIONS,
    'inter_layer_conductances': _INTER_LAYER_CONDUCTANCES,
    'inter_layer_scaling': 1.0,
    'electrode_positions' : [
        (0, 0, -0.94),
        (0, 0, -0.79),
        (0, 0, -0.64),
        (0, 0, -0.49),
        (0, 0, -0.34),
        (0, 0, -0.19),
        (0, 0, -0.04),
        (0, 0, 0.10),
        (0, 0, 0.26),
        (0, 0, 0.40),
        (0, 0, 0.56),
        (0, 0, 0.70),
        (0, 0, 0.86),
        (0, 0, 1.00),
        (0, 0, 1.16),
        (0, 0, 1.30),
    ],
}