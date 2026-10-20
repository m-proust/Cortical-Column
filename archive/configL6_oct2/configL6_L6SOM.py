from tools.utils import *                           #                                   >                                 
from brian2 import *                                #                                   8  k                              
import numpy as np                                  #                                  .i>?.                              
import pandas as pd                                 #                   .'            .nk                                 
from collections import defaultdict                 #             r~.    :           .l[                                  
                                                    #              .JI;. ;          ^II                                   
EXT_NMDA_WEIGHT = 0.45*nS                           #                `iI_;|          ;I                                   
EXT_AMPA_WEIGHT = 1.25*nS                           #                  'I;!          zIlq..`h!p.      i.                                                
                                                    #    .k  i.         .YIU         l;I;la..foc:     \    .b|k           
tau_e_NMDA = 100*ms                                 #     .#` M.    tZ}?]I;Ik.        ;;!.          :.{.  kIpo'           
tau_i_PV   =  6*ms                                  #      .YIIh.         `f:a.     '*;Iw           #;!!Il0'              
tau_i_SOM  = 20*ms                                  # .       1>}.          Z;;Z:.`d;;;;_       ..;x;lC`                  
tau_i_VIP  = 8*ms                                   #  o\.     `|h'        (I;;;;;I;;;;;;Iw' 'hI;;_+'    OcZ              
v_reset = -65.*mV                                   #  .\,L     .(lM'   'O;;;;;;;;;;;;;;;;;;;;;lq'   .dm; "f{             
vt      = -50.*mV                                   # .  n;IX:^ .pI;;;;I;;;;;;;I;w][[]]{;;;;;;;x    uI"        .^:o/|l    
ee      = 0.*mV                                     #     p;;u*c}C;;;;;;;;;;;;;I?[]]]]]]n;;;;;;II;:I;;Ih-.  `h!I;\|llv^   
ei      = -80.*mV                                   #    .&     .mII;lll;;;;;;;/[]]]]]][[I;;;;;;/   ,~;I\*w#Y;i.          
                                                    #           nl;;Z   .)I;;;;l?]]]]]]]c;;;;;;u      .;IO'   .]p..       
t_ref = {                                           #    ')    !IJ/[.    .l;;;;;x]]]][[QI;;;;;;         w;I.   .M{;io.    
    'E':   5*ms,                                    #      .&lI;1 p.      );;;;;;IlII;I;;;;;;:.          Z''       O*     
    'PV':  2*ms,                                    #         .Xn        ';;;;;;;;;;;;;;;;;It                       "-    
    'SOM': 5*ms,                                    #         .ho       mI;I{qI;;;;;;;;;;;;J ."bh.                    
    'VIP': 5*ms,                                    #         ..      .I;Io   .;;;;;;;;I:+}i;;lo,"Umzf                   
}                                                   #    ....       .m;;;.     {;;;;lz.      bU?qC"  .v'                  
                                                    #    ..XIIl;;:Il*lI;      q{mlI;I                                     
def g_NMDA(v_mV):                                   #      .~I'  oL. .II     <;:::;i.                               
    return 1.0 / (1.0 + 0.28*np.exp(-0.062*v_mV))   #      .k    "  .a;I|`  ,:!::::;.                                     
                                                    #              C:Ic ^/  M:::::lm                                      
                                                    #               'u      lI:::::.                                      
                                                    #               [       :l:::;:.                                      
tau_e_AMPA_E   = 5*ms                               #                       *n}I;Ii                                       
tau_e_AMPA_PV  = 1*ms                               #                       ::::,Id                                       
tau_e_AMPA_SOM = 2*ms                               #                      .I:::::;.                                      
tau_e_AMPA_VIP = 2*ms                               #                       :;::::;w                                          
                                                    #                      .M;:::::".                                     
                                                    #                        l;::::!m                                     
                                                    #                       'L::::;;Z                                     
                                                    #                         Q>I>l,:p                                    
                                                    #                         ';:::::i|                                   
                                                    #                          ^i::::,l^                                  
                                                    #                           nI::::,l.                                 
                                                    #                            h!;:::"+.                                
                                                    #                              k-!;-0{'                               
                                                    #                              mI::::::^                              
                                                    #                               MI::::::.                             
                                                    #                                q!::::;<'                            
                                                    #                                 il::::lh                            
                                                    #                                 ',;::>h'                            
                                                    #                                   d:;::I,'                          
                                                    #                                   .!::::,,C                         
                                                    #                                    `;:::::lc                        
                                                    #                                     nI:::::,I.                      
                                                    #                                      *;;;|0l;Iw.                    
                                                    #                                       *I;;;;;;;;;/'.                
                                                    #                                       *I;I;;;;IYh,;I..              
                                                    #                                      .I;k' I;;I  .'+;lh'     .|1    
                                                    #                                     .l;;j  d;;l}    -;;;II;cv<kj"   
                                                    #                                  ' Cw b;o  aII;?    .i;:Y;_         
                                                    #                                 ^I;-  (;' .-I.<I`    ]I> .:_h.      
                                                    #                                .LI;  .;l  ';1 .Lt     MI.   ..ovO(. 
                                                    #                                     ";Z   a:   'l(     Ul '     u:( 
                                                    #                                    YlIC  x;;`   I;Q    ";:p         
                                                    #                                      .    '`     .                  
                                                                                                                        



# Group scaling factors on connections INTO L23/L4AB/L4C, applied to the CSVs below
# (tools/tuning.py; CSVs are not edited). Groups, per target layer T (CSV columns *_T):
#   EE E_T->E_T | EI(_PV/_SOM/_VIP) E_T->interneurons | IE interneurons_T->E_T
#   II(_PV/_SOM/_VIP) interneurons_T->interneurons | XE / XI all other layers -> T
#   SOM_E SOM_T->E_T | SOM_ALL SOM_T->all cell types of T
# Excitatory factors scale AMPA and NMDA alike. 'bg' scales N x rate of the background
# Poisson input per population.
# ONE knob: inter-layer E -> PV conductances (AMPA + NMDA) into L23, L4AB and L4C are all
# multiplied by this (1 = CSV values). Connected, PV gets 95-98% of its excitation from
# E spikes, mostly from other layers, so you may need more PV Poisson N to keep PV rates
# (at x0.5 roughly: L23 N ~ 3 -> ~65, L4AB 4 -> ~40, L4C 10 -> ~53, baseline-rate estimate).
INTERLAYER_PV_SCALE = 1
TUNING = {L: {'XE_PV': INTERLAYER_PV_SCALE} for L in ['L23', 'L4AB', 'L4C']}
# Rejected Sep 30 2026 (connected; 5 Hz VIP-SOM rhythm, 14-21 Hz beta instead of alpha):
# TUNING = {
#     'L4C':  {'II': 1.5,  'EI_SOM': 1.25, 'bg': {'E': 0.8, 'PV': 2, 'SOM': 2, 'VIP': 2}},
#     'L4AB': {'II': 1.25, 'EI_SOM': 1.25, 'bg': {'PV': 2, 'VIP': 1.5}},
#     'L23':  {'EI_SOM': 1.25,             'bg': {'PV': 2, 'SOM': 3, 'VIP': 3}},
# }
from tools.tuning import load_tuned_connectivity, apply_background
# configL6_L6SOM (Oct 1 2026): configL6 with two changes, CSVs in config/L6SOM_x3_localII/:
#   1. SOM_L6 -> E_L4C connection probability 0.045 -> 0.27 (x3 the Sep 14 value of 0.09)
#   2. local inhibitory -> inhibitory rows (PV/SOM/VIP -> PV/SOM/VIP within each layer,
#      except VIP -> PV) back to their 14092026TEST values
# Restores the stimulus-induced 1-10 Hz power drop while keeping E_L5 -> E_L23 p at 0.06.
# TUNING above is a no-op (XE_PV = 1, no 'bg'), so the CSVs are loaded directly.
from tools.utils import load_connectivity_from_csv
csv_layer_configs, _INTER_LAYER_CONNECTIONS, _INTER_LAYER_CONDUCTANCES = \
    load_connectivity_from_csv('config/L6SOM_x3_localII/connection_probabilities.csv',
                               'config/L6SOM_x3_localII/conductances_AMPA_GABA.csv',
                               'config/L6SOM_x3_localII/conductances_NMDA.csv')



import os as _os
_env_lesion = _os.environ.get('LESION_PAIR')
if _env_lesion is None:
    _SUPPRESSED_INTER_LAYER = [
        # ('L6', 'L23'),
    ]
elif _env_lesion.upper() == 'NONE':
    _SUPPRESSED_INTER_LAYER = []
else:
    _SUPPRESSED_INTER_LAYER = [tuple(_env_lesion.split(','))]
_SUPPRESS_ALL_INTER_LAYER = False   # layers CONNECTED (set True to disconnect; use the "disconnected" N and input_rate in the comments)

# Lesioned pairs are NOT removed from the dicts: the column still builds their
# Synapses and then silences them. Building them keeps the RNG stream identical,
# so every other pathway's wiring and the heterogeneity match the intact network.
# (Popping them skipped their connect()/delay draws and re-randomised everything
# built afterwards, so each lesion was also a different network.)
if _SUPPRESS_ALL_INTER_LAYER:
    _LESIONED_INTER_LAYER = sorted(_INTER_LAYER_CONNECTIONS.keys())
else:
    for _pair in _SUPPRESSED_INTER_LAYER:
        if _pair not in _INTER_LAYER_CONNECTIONS:
            raise KeyError(
                f"Cannot suppress inter-layer pair {_pair}: not present in "
                f"_INTER_LAYER_CONNECTIONS. Available pairs: "
                f"{sorted(_INTER_LAYER_CONNECTIONS.keys())}"
            )
    _LESIONED_INTER_LAYER = list(_SUPPRESSED_INTER_LAYER)

_LAYER_CONFIGS = {
    'L23': {
        'connection_prob': csv_layer_configs['L23']['connection_prob'],
        'conductance': csv_layer_configs['L23']['conductance'],
        'poisson_inputs': {
            'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 34},   # connected, tuned 2026-09-30 | disconnected: 64
            'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 15},   # connected, tuned 2026-09-30 | disconnected: 312
            'VIP':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 15},   # connected, tuned 2026-09-30 | disconnected: 15
            'SOM':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 15},   # connected, tuned 2026-09-30 | disconnected: 65

        },
        'input_rate': 4*Hz,
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
            'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 30},   # connected, tuned 2026-09-30 | disconnected: 76
            'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 10},   # connected, tuned 2026-09-30 | disconnected: 322
            'VIP':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 14},   # connected, tuned 2026-09-30 | disconnected: 27
            'SOM':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 22},   # connected, tuned 2026-09-30 | disconnected: 68

        },
        'input_rate': 4*Hz,
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
            'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 45},   # connected, tuned 2026-09-30 | disconnected: 54
            'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 50},   # connected, tuned 2026-09-30 | disconnected: 160
            'VIP':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 15},   # connected, tuned 2026-09-30 | disconnected: 22
            'SOM':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 20},   # connected, tuned 2026-09-30 | disconnected: 60

        },
        'input_rate': 4*Hz,   # connected | disconnected: 5 Hz
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
          'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 35},   # connected, tuned 2026-09-30 | disconnected: 52
            'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 6},   # connected, tuned 2026-09-30 | disconnected: 147
            'VIP':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 9},   # connected, tuned 2026-09-30 | disconnected: 24
            'SOM':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 18},   # connected, tuned 2026-09-30 | disconnected: 51


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
            'E':        {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 45},   # connected, tuned 2026-09-30 | disconnected: 49
            'PV':       {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 38},   # connected, tuned 2026-09-30 | disconnected: 95
             'VIP':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 19},   # connected, tuned 2026-09-30 | disconnected: 21
            'SOM':      {'target': 'gE_AMPA', 'weight': 'EXT_AMPA', 'N': 13},   # connected, tuned 2026-09-30 | disconnected: 32

        },
        'input_rate': 4*Hz,   # connected | disconnected: 5 Hz
        'neuron_counts': {'E': 2040, 'PV': 195, 'SOM': 110, 'VIP': 70},
        'coordinates' : {
            'x': (-0.15,0.15),
            'y': (-0.15,0.15),
            'z': (-0.62, -0.34),
        }
    },
}

# Only silence pairs whose two layers are both present, so layers can be
# commented out above (e.g. L6 alone) without silence_lesioned raising.
_LESIONED_INTER_LAYER = [p for p in _LESIONED_INTER_LAYER
                         if p[0] in _LAYER_CONFIGS and p[1] in _LAYER_CONFIGS]

# L6_SOM_ROW_SCALE block removed Sep 30 2026 (the L6 SOM row is already doubled in the
# copy CSV; the block also scaled the L4C SOM row x4).
apply_background(_LAYER_CONFIGS, TUNING)

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

        dgE_AMPA/dt = -gE_AMPA/tau_e_AMPA_E : siemens
        dgE_NMDA/dt = -gE_NMDA/tau_e_NMDA   : siemens
        dgPV/dt     = -gPV/tau_i_PV          : siemens
        dgSOM/dt    = -gSOM/tau_i_SOM        : siemens
        dgVIP/dt    = -gVIP/tau_i_VIP        : siemens

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

        dgE_AMPA/dt = -gE_AMPA/tau_e_AMPA_PV : siemens
        dgE_NMDA/dt = -gE_NMDA/tau_e_NMDA    : siemens
        dgPV/dt     = -gPV/tau_i_PV           : siemens
        dgSOM/dt    = -gSOM/tau_i_SOM         : siemens
        dgVIP/dt    = -gVIP/tau_i_VIP         : siemens

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

        dgE_AMPA/dt = -gE_AMPA/tau_e_AMPA_SOM : siemens
        dgE_NMDA/dt = -gE_NMDA/tau_e_NMDA     : siemens
        dgPV/dt     = -gPV/tau_i_PV            : siemens
        dgSOM/dt    = -gSOM/tau_i_SOM          : siemens
        dgVIP/dt    = -gVIP/tau_i_VIP          : siemens

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

        dgE_AMPA/dt = -gE_AMPA/tau_e_AMPA_VIP : siemens
        dgE_NMDA/dt = -gE_NMDA/tau_e_NMDA     : siemens
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
            'tau_e_AMPA_E':   tau_e_AMPA_E,
            'tau_e_AMPA_PV':  tau_e_AMPA_PV,
            'tau_e_AMPA_SOM': tau_e_AMPA_SOM,
            'tau_e_AMPA_VIP': tau_e_AMPA_VIP,
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
                'C': 38*pF, 'gL': 3.8*nS, 'tauw': 50*ms,  'EL': -68*mV},
        'SOM': {'a': 4*nS, 'b': 50*pA, 'DeltaT': 1.5*mV,
                'C': 92*pF, 'gL': 4.3*nS, 'tauw': 180*ms, 'EL': -63*mV},
        'VIP': {'a': 2*nS, 'b': 40*pA, 'DeltaT': 2*mV,
                'C': 70*pF, 'gL': 3.3*nS, 'tauw': 150*ms, 'EL': -61*mV},
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
        'E_AMPA_E':   tau_e_AMPA_E,
        'E_AMPA_PV':  tau_e_AMPA_PV,
        'E_AMPA_SOM': tau_e_AMPA_SOM,
        'E_AMPA_VIP': tau_e_AMPA_VIP,
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
    'lesioned_inter_layer': _LESIONED_INTER_LAYER,   # built, then silenced
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


