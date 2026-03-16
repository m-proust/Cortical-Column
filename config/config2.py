from tools.utils import *
from brian2 import *
import numpy as np
import pandas as pd
from collections import defaultdict
      

EXT_AMPA_WEIGHT = 1.25*nS
EXT_NMDA_WEIGHT = 0.45*nS 

tau_e_AMPA = 3*ms
tau_e_NMDA = 100*ms
tau_i_PV   =  6*ms  
tau_i_SOM  = 20*ms
tau_i_VIP  = 8*ms

v_reset = -65.*mV
vt      = -50.*mV
ee      = 0.*mV
ei      = -80.*mV
t_ref = {
    'E':   5*ms,
    'PV':  1*ms,
    'SOM': 5*ms,
    'VIP': 5*ms,
}



def g_NMDA(v_mV):
    return 1.0 / (1.0 + 0.28 * np.exp(-0.062 * v_mV))

csv_layer_configs, _INTER_LAYER_CONNECTIONS, _INTER_LAYER_CONDUCTANCES = load_connectivity_from_csv(
    'config/connection_probabilities2.csv',
    'config/conductances_AMPA2_alpha_v2.csv',
    'config/conductances_NMDA2_alpha_v2.csv'  )



# Temporarily disable inter-layer connections to test alpha generation
# _INTER_LAYER_CONNECTIONS = {}
# _INTER_LAYER_CONDUCTANCES = {}

_LAYER_CONFIGS = {
    'L23': {
        'connection_prob': csv_layer_configs['L23']['connection_prob'],
        'conductance': csv_layer_configs['L23']['conductance'],
        # 'intrinsic_params': {
        #     'E': {'b': 80*pA, 'tauw': 150*ms}, 
        # },
        'poisson_inputs': {
            'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 55},
            # 'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 10},
            'SOM':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 10},
            'VIP':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 15},
            'E_NMDA':   {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 15},
            'SOM_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 10},
            'VIP_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 20},

        },
        'input_rate': 5*Hz,
        'neuron_counts': {'E': 3520, 'PV': 260, 'SOM': 410, 'VIP': 263},
        'coordinates' : {
            'x': (-0.15,0.15),
            'y': (-0.15,0.15),
            'z': (0.45, 1.1),
        },
    },

    'L4AB': {
        'connection_prob': csv_layer_configs['L4AB']['connection_prob'],
        'conductance': csv_layer_configs['L4AB']['conductance'],
        'poisson_inputs': {
            'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 55},
            'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 15},
            'SOM':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 15},
            'VIP':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 15},
            'E_NMDA':   {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 15},
            'SOM_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 15},
            'VIP_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 15},

        },
        'input_rate': 5*Hz,
        'neuron_counts': {'E': 2720, 'PV': 340, 'SOM': 210, 'VIP': 130},
        'coordinates' : {
            'x': (-0.15,0.15),
            'y': (-0.15,0.15),
            'z': (0.14, 0.45),
        }
    },

    'L4C': {
        'connection_prob': csv_layer_configs['L4C']['connection_prob'],
        'conductance': csv_layer_configs['L4C']['conductance'],
        'poisson_inputs': {
            'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 65},
            'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 15},
            'SOM':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 15},
            'VIP':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 10},
            'E_NMDA':   {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 15},
            'SOM_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 15},
            'VIP_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 20},

        },
        'input_rate': 5*Hz,
        'neuron_counts': {'E': 3192, 'PV': 320, 'SOM': 200, 'VIP': 88},
        'coordinates' : {
            'x': (-0.15,0.15),
            'y': (-0.15,0.15),
            'z': (-0.14, 0.14),
        }
    },

    'L5': {
        'connection_prob': csv_layer_configs['L5']['connection_prob'],
        'conductance': csv_layer_configs['L5']['conductance'],
        'poisson_inputs': {
          'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 35},
            'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 5},
            'SOM':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 5},
            'VIP':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 5},
            'E_NMDA':   {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 25},
            # 'PV_NMDA':  {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 10},
            'SOM_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 35},
            'VIP_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 35},

        },
        'input_rate': 4*Hz,
        'neuron_counts': {'E': 1600, 'PV': 220, 'SOM': 110, 'VIP': 70},
        'coordinates' : {
            'x': (-0.15,0.15),
            'y': (-0.15,0.15),
            'z': (-0.34, -0.14),
        }
    },

    'L6': {
        'connection_prob': csv_layer_configs['L6']['connection_prob'],
        'conductance': csv_layer_configs['L6']['conductance'],
        'poisson_inputs': {
            'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 40},
            'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 10},
            'SOM':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 10},
            'VIP':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 10},
            'E_NMDA':   {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 55},
            'PV_NMDA':  {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 10},
            'SOM_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 25},
            'VIP_NMDA': {'target': 'gE_NMDA', 'weight': 'EXT_NMDA', 'N': 15},

        },
        'input_rate': 5*Hz,
        'neuron_counts': {'E': 2040, 'PV': 195, 'SOM': 110, 'VIP': 70},
        'coordinates' : {
            'x': (-0.15,0.15),
            'y': (-0.15,0.15),
            'z': (-0.62, -0.34),
        }
    },
}


CONFIG = {
    'simulation': {
        'SIMULATION_TIME': 2000*ms,
        'DT': 0.1*ms,
        'RANDOM_SEED': 58879,
    },

    'models': {
        'equations': {
            'E': """
        dv/dt = (
            gL*(EL - v)
          + gL*DeltaT*exp((v - VT)/DeltaT)
          + gE_AMPA*(Ee - v)
          + gE_NMDA*(Ee - v)/(1 + 0.28*exp(-0.062*v/mV))
          - (gPV + gSOM + gVIP)*(v - Ei)
          - w + I
        )/C : volt (unless refractory)

        IsynE_AMPA = gE_AMPA*(Ee - v) : amp
        IsynE_NMDA = gE_NMDA*(Ee - v)/(1 + 0.28*exp(-0.062*v/mV)) : amp
        IsynE      = IsynE_AMPA + IsynE_NMDA : amp

        IsynIPV    = gPV*(Ei - v) : amp
        IsynISOM   = gSOM*(Ei - v) : amp
        IsynIVIP   = gVIP*(Ei - v) : amp

        gI   = gPV + gSOM + gVIP : siemens
        gE   = gE_AMPA : siemens
        IsynI = gI*(v - Ei) : amp

        dgE_AMPA/dt = -gE_AMPA/tau_e_AMPA : siemens
        dgE_NMDA/dt = -gE_NMDA/tau_e_NMDA : siemens
        dgPV/dt     = -gPV/tau_i_PV       : siemens
        dgSOM/dt    = -gSOM/tau_i_SOM     : siemens
        dgVIP/dt    = -gVIP/tau_i_VIP     : siemens

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
          + gE_NMDA*(Ee - v)/(1 + 0.28*exp(-0.062*v/mV))
          - (gPV + gSOM + gVIP)*(v - Ei)
          - w + I
        )/C : volt (unless refractory)

        IsynE_AMPA = gE_AMPA*(Ee - v) : amp
        IsynE_NMDA = gE_NMDA*(Ee - v)/(1 + 0.28*exp(-0.062*v/mV)) : amp
        IsynE      = IsynE_AMPA + IsynE_NMDA : amp

        IsynIPV    = gPV*(Ei - v) : amp
        IsynISOM   = gSOM*(Ei - v) : amp
        IsynIVIP   = gVIP*(Ei - v) : amp

        gI   = gPV + gSOM + gVIP : siemens
        gE   = gE_AMPA : siemens
        IsynI = gI*(v - Ei) : amp

        dgE_AMPA/dt = -gE_AMPA/tau_e_AMPA : siemens
        dgE_NMDA/dt = -gE_NMDA/tau_e_NMDA : siemens
        dgPV/dt     = -gPV/tau_i_PV       : siemens
        dgSOM/dt    = -gSOM/tau_i_SOM     : siemens
        dgVIP/dt    = -gVIP/tau_i_VIP     : siemens

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
          + gE_NMDA*(Ee - v)/(1 + 0.28*exp(-0.062*v/mV))
          - (gPV + gSOM + gVIP)*(v - Ei)
          - w + I
        )/C : volt (unless refractory)

        IsynE_AMPA = gE_AMPA*(Ee - v) : amp
        IsynE_NMDA = gE_NMDA*(Ee - v)/(1 + 0.28*exp(-0.062*v/mV)) : amp
        IsynE      = IsynE_AMPA + IsynE_NMDA : amp

        IsynIPV    = gPV*(Ei - v) : amp
        IsynISOM   = gSOM*(Ei - v) : amp
        IsynIVIP   = gVIP*(Ei - v) : amp

        gI   = gPV + gSOM + gVIP : siemens
        gE   = gE_AMPA : siemens
        IsynI = gI*(v - Ei) : amp

        dgE_AMPA/dt = -gE_AMPA/tau_e_AMPA : siemens
        dgE_NMDA/dt = -gE_NMDA/tau_e_NMDA : siemens
        dgPV/dt     = -gPV/tau_i_PV       : siemens
        dgSOM/dt    = -gSOM/tau_i_SOM     : siemens
        dgVIP/dt    = -gVIP/tau_i_VIP     : siemens

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

            'VIP': """
        dv/dt = (
            gL*(EL - v)
          + gL*DeltaT*exp((v - VT)/DeltaT)
          + gE_AMPA*(Ee - v)
          + gE_NMDA*(Ee - v)/(1 + 0.28*exp(-0.062*v/mV))
          - (gPV + gSOM + gVIP)*(v - Ei)
          - w + I
        )/C : volt (unless refractory)

        IsynE_AMPA = gE_AMPA*(Ee - v) : amp
        IsynE_NMDA = gE_NMDA*(Ee - v)/(1 + 0.28*exp(-0.062*v/mV)) : amp
        IsynE      = IsynE_AMPA + IsynE_NMDA : amp

        IsynIPV    = gPV*(Ei - v) : amp
        IsynISOM   = gSOM*(Ei - v) : amp
        IsynIVIP   = gVIP*(Ei - v) : amp

        gI   = gPV + gSOM + gVIP : siemens
        gE   = gE_AMPA : siemens
        IsynI = gI*(v - Ei) : amp

        dgE_AMPA/dt = -gE_AMPA/tau_e_AMPA : siemens
        dgE_NMDA/dt = -gE_NMDA/tau_e_NMDA : siemens
        dgPV/dt     = -gPV/tau_i_PV       : siemens
        dgSOM/dt    = -gSOM/tau_i_SOM     : siemens
        dgVIP/dt    = -gVIP/tau_i_VIP     : siemens

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
            'tau_e_NMDA': tau_e_NMDA,
            'tau_i_PV':   tau_i_PV,
            'tau_i_SOM':  tau_i_SOM,
            'tau_i_VIP':  tau_i_VIP,
            'VT':         vt,
            'V_reset':    v_reset,
            'Ee':         ee,
            'Ei':         ei,
        },
    },

    'intrinsic_params': {
        'E':   {'a': 4*nS, 'b': 100*pA, 'DeltaT': 2*mV,
                'C': 97*pF, 'gL': 4.2*nS, 'tauw': 100*ms, 'EL': -65*mV},
        'PV':  {'a': 0*nS, 'b': 0*pA, 'DeltaT': 0.5*mV,
                'C': 50*pF, 'gL': 4.8*nS, 'tauw': 50*ms,  'EL': -68*mV},
        'SOM': {'a': 4*nS, 'b': 25*pA, 'DeltaT': 1.5*mV,
                'C': 80*pF, 'gL': 4.3*nS, 'tauw': 200*ms, 'EL': -65*mV},
        'VIP': {'a': 2*nS, 'b': 20*pA, 'DeltaT': 2*mV,
                'C': 50*pF, 'gL': 4*nS, 'tauw': 150*ms, 'EL': -61*mV},
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
        'DEFAULT': {'v': '-60*mV + rand()*15*mV',
                    'gE_AMPA': 0*nS, 'gE_NMDA': 0*nS,
                    'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS,
                    'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'E':   {'v': '-60*mV + rand()*15*mV',
                'gE_AMPA': 0*nS, 'gE_NMDA': 0*nS,
                'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS,
                'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'PV':  {'v': '-60*mV + rand()*15*mV',
                'gE_AMPA': 0*nS, 'gE_NMDA': 0*nS,
                'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS,
                'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'SOM': {'v': '-60*mV + rand()*15*mV',
                'gE_AMPA': 0*nS, 'gE_NMDA': 0*nS,
                'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS,
                'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
        'VIP': {'v': '-60*mV + rand()*15*mV',
                'gE_AMPA': 0*nS, 'gE_NMDA': 0*nS,
                'gPV': 0*nS, 'gSOM': 0*nS, 'gVIP': 0*nS,
                'w': 0*pA, 'I': 0*pA, 'Vcut_offset_factor': 5},
    },

    'time_constants': {
        'E_AMPA': tau_e_AMPA,
        'E_NMDA': tau_e_NMDA,
        'I_PV':   tau_i_PV,
        'I_SOM':  tau_i_SOM,
        'I_VIP':  tau_i_VIP,
    },

    'synapses': {
        'Q': {
            'EXT_AMPA': EXT_AMPA_WEIGHT,
            'EXT_NMDA': EXT_NMDA_WEIGHT,
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


