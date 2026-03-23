import numpy as np
import brian2 as b2
from brian2 import *
from brian2tools import *
from config.config2 import CONFIG
from src.column import CorticalColumn
from src.visualization import *
from src.analysis import *
from ppc import compute_all_ppc, plot_ppc_spectra


def main():
    np.random.seed(CONFIG['simulation']['RANDOM_SEED'])
    b2.start_scope()
    b2.defaultclock.dt = CONFIG['simulation']['DT']

    total_time = 10000

    print(" Creating cortical column...")
    column = CorticalColumn(column_id=0, config=CONFIG)

    all_monitors = column.get_all_monitors()
    w_ext_AMPA = CONFIG['synapses']['Q']['EXT_AMPA']

    # --- L4C stimulation ---
    L4C = column.layers['L4C']
    cfg_L4C = CONFIG['layers']['L4C']
    L4C_E_grp = L4C.neuron_groups['E']
    N_stim_E = 40
    stim_rate_E = 80*Hz
    L4C_E_stimAMPA = PoissonInput(L4C_E_grp, 'gE_AMPA',
                                   N=N_stim_E,
                                   rate=stim_rate_E,
                                   weight=w_ext_AMPA)

    L4C_PV_grp = L4C.neuron_groups['PV']
    N_stim_PV = 30
    stim_rate_PV = 40*Hz
    L4C_PV_stim = PoissonInput(L4C_PV_grp, 'gE_AMPA',
                                N=N_stim_PV,
                                rate=stim_rate_PV,
                                weight=w_ext_AMPA)

    # --- L6 stimulation ---
    L6 = column.layers['L6']
    cfg_L6 = CONFIG['layers']['L6']
    L6_PV_grp = L6.neuron_groups['PV']
    N_stim_L6_PV = 30
    stim_rate_L6_PV = 15*Hz
    L6_PV_stim = PoissonInput(L6_PV_grp, 'gE_AMPA',
                               N=N_stim_L6_PV,
                               rate=stim_rate_L6_PV,
                               weight=w_ext_AMPA)

    L6_E_grp = L6.neuron_groups['E']
    N_stim_L6_E = int(cfg_L6['poisson_inputs']['E']['N'])
    stim_rate_L6_E = 15*Hz
    L6_E_stim = PoissonInput(L6_E_grp, 'gE_AMPA',
                              N=N_stim_L6_E,
                              rate=stim_rate_L6_E,
                              weight=w_ext_AMPA)

    column.network.add(L6_E_stim, L6_PV_stim)
    column.network.add(L4C_E_stimAMPA, L4C_PV_stim)

    column.network.run(total_time * ms)
    print("Simulation complete")

    # ─────────────────────────────────────────────────────────────────
    # Organize monitors
    # ─────────────────────────────────────────────────────────────────
    spike_monitors = {}
    state_monitors = {}
    rate_monitors = {}
    neuron_groups = {}

    for layer_name, monitors in all_monitors.items():
        spike_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'spikes' in k
        }
        state_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'state' in k
        }
        rate_monitors[layer_name] = {
            k: v for k, v in monitors.items() if 'rate' in k
        }
        neuron_groups[layer_name] = column.layers[layer_name].neuron_groups

    # ─────────────────────────────────────────────────────────────────
    # LFP (your kernel method)
    # ─────────────────────────────────────────────────────────────────
    electrode_positions = CONFIG['electrode_positions']
    print("Computing LFP using kernel method...")
    
    lfp_signals, time_array = calculate_lfp_kernel_method(
        spike_monitors,
        neuron_groups,
        CONFIG['layers'],
        electrode_positions,
        sim_duration_ms=total_time
    )

    # ─────────────────────────────────────────────────────────────────
    # PPC SPECTRUM using YOUR LFP, with per-layer electrode selection
    # ─────────────────────────────────────────────────────────────────
    dt_sec = float(CONFIG['simulation']['DT'] / second)

    print("\n" + "="*60)
    print("PPC SPECTRUM (PPC vs Frequency, per-layer LFP)")
    print("="*60)

    results = compute_all_ppc(
        spike_monitors=spike_monitors,
        lfp_signals=lfp_signals,
        lfp_time_array=time_array,
        layer_configs=CONFIG['layers'],
        electrode_positions=electrode_positions,
        dt_seconds=dt_sec,
        t_discard=2.0,
        freq_range=(5, 100),
        freq_step=2,
        bandwidth=10,
    )

    plot_ppc_spectra(results, save_path='ppc_spectrum.png')


if __name__ == '__main__':
    main()