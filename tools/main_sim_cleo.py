import numpy as np
import brian2 as b2
from brian2 import *
from brian2tools import *
from config.config import CONFIG
from src.column import CorticalColumn
import cleo
from cleo import ephys
from src.analysis import *
from src.cleo_plots import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def _existing_syn(layer, pre, post):

    key = f"{pre}_{post}"
    s = layer.synapses.get(key)
    if s is None:
        return None
    try:
        n = len(s)
    except Exception:
        n = getattr(s, "N", None)
        if n is None:
            try:
                n = len(s.i[:])
            except Exception:
                n = 0
    return s if (n and n > 0) else None


def main():

    cell_colors = {
        'E':   '#2ca02c',  
        'PV':  '#d62728', 
        'SOM': '#1f77b4',
        'VIP': '#e377c2', 
    }

    np.random.seed(CONFIG['simulation']['RANDOM_SEED'])
    b2.start_scope()
    b2.defaultclock.dt = CONFIG['simulation']['DT']

    print(" Creating cortical column...")
    column = CorticalColumn(column_id=0, config=CONFIG)

    all_monitors = column.get_all_monitors()

    layer_names = ['L23', 'L4AB', 'L4C', 'L5', 'L6']
    cell_types = ['E', 'PV', 'SOM', 'VIP']

    groups = []
    colors = []
    for ln in layer_names:
        for ct in cell_types:
            groups.append(column.layers[ln].neuron_groups[ct])
            colors.append(cell_colors[ct])

    fig, ax = cleo.viz.plot(
        *groups,
        devices=[column.electrode],
        scatterargs={"alpha": 0.6},
        colors=colors,
        figsize=(10, 10),
        dpi=600,
        invert_z=False,
    )

    ax.tick_params(axis='x', labelsize=6, pad=1)
    ax.tick_params(axis='y', labelsize=6, pad=1)
    ax.tick_params(axis='z', labelsize=6, pad=15)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cell_colors[ct],
               markersize=8, label=ct) for ct in cell_types
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9)

    fig.savefig("cleo_plot.png", dpi=600, bbox_inches="tight")

    sim = cleo.CLSimulator(column.network)
    
    
    
    rwslfp = ephys.RWSLFPSignalFromPSCs()
    mua = ephys.MultiUnitActivity(threshold_sigma=4.5)
    ss = ephys.SortedSpiking()
    probe = column.electrode
    sim.set_io_processor(cleo.ioproc.RecordOnlyProcessor(sample_period=1 * b2.ms))
    probe.add_signals(mua, ss, rwslfp)

    for layer_name, layer in column.layers.items():
        if layer_name != 'L1':
            group = layer.neuron_groups['E']
            sim.inject(
                probe, group,
                Iampa_var_names=['IsynE'],
                Igaba_var_names=['IsynI'] 
            )
            # sim.inject(
            #     probe,  
            #     group, 
            #     tklfp_type="exc",
            #     ampa_syns=[layer.synapses['E_E']], 
            #     gaba_syns=[
            #         layer.synapses['PV_E'],
            #         layer.synapses['SOM_E'],
            #         layer.synapses['VIP_E']
            #     ]
            # )












    sim.run(500 * b2.ms)
    w_ext_AMPA = CONFIG['synapses']['Q']['EXT_AMPA']
    w_ext_NMDA = CONFIG['synapses']['Q']['EXT_NMDA']
    

     

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

    sim.run(500 * b2.ms)


    spike_mon = all_monitors['L4C']['E_spikes']
    fig, axs = plt.subplots(3, 1, sharex=True, layout="constrained", figsize=(6, 6))
    

    axs[0].plot(spike_mon.t / b2.ms, spike_mon.i, ".", rasterized=True, ms=2)
    axs[0].set(ylabel="NeuronGroup index", title="L4 E — ground-truth spikes")

    axs[1].plot(ss.t / b2.ms, ss.i, ".", rasterized=True)
    axs[1].set(title="L4C E — sorted spikes", ylabel="sorted unit index")

    axs[2].plot(mua.t / b2.ms, mua.i, ".", rasterized=True)
    axs[2].set(
        title="multi-unit activity",
        ylabel="channel index",
        xlabel="time (ms)",
        ylim=[-0.5, column.electrode.n - 0.5],
    )


    fig, ax = plt.subplots(1, 1, figsize=(6, 7), sharey=False, layout="constrained")

    lfp = rwslfp.lfp 
    channel_offsets = -np.abs(np.quantile(lfp, 0.9)) * np.arange(probe.n)
    lfp2plot = lfp + channel_offsets
    ax.plot(lfp2plot, color="white", lw=1)
    ax.set(
        yticks=channel_offsets,
        xlabel="t (ms)",
        title="TKLFP",
    )

    extent = (0, 250, lfp2plot.min(), lfp2plot.max())
    
    im = ax.imshow(
        lfp.T,
        aspect="auto",
        extent=extent,
        vmin=-np.max(np.abs(lfp)),
        vmax=np.max(np.abs(lfp)),
    )

    fig.colorbar(im, aspect=40, label="LFP (a.u.)", ticks=[])

    ax.set(
        ylabel="channel index",
        yticklabels=range(1, 17),
    )

    

    # fig_laminar, axes_laminar, psth, lfp_bipolar = analyze_and_plot_laminar_recording(
    #     sim, column, probe, tklfp, ss,
    #     stim_onset_time=500*b2.ms  
    # )

        
    results = analyze_and_plot_laminar_recording_mua(
        sim, column, probe, rwslfp, mua,
        stim_onset_time=500*b2.ms, plot=plot
    )
    fig_laminar, axes_laminar, psth, lfp = results['fig'], results['axes'], results['psth'], results['lfp']

    


    fig_lfp_layers_mono = results['fig_lfp_layers_mono']
    fig_lfp_layers_bip  = results['fig_lfp_layers_bip']

    
    plt.show()


if __name__ == "__main__":
    main()