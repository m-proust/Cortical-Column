
import numpy as np
from brian2 import *


DEFAULT_TARGET_DRIVE = {
    'L4C': {
        'E':  2000.0, 
        'PV': 3000.0,  
    },
    'L6': {
        'E':  375.0,  
        'PV': 375.0, 
    },
    'L23': {
        'E':  500.0,   
        'PV': 750.0,
    },
    'L4AB': {
        'E':  1500.0,  
        'PV': 2250.0,
    },
    'L5': {
        'E':  500.0,
        'PV': 750.0,
    },
}

TC_NMDA_AMPA_RATIO = 0



def load_lgn_spikes_npz(npz_path):
    data = np.load(npz_path)
    on_t, on_ids   = data['on_times'], data['on_ids']
    off_t, off_ids = data['off_times'], data['off_ids']
    n_on, n_off    = int(data['n_on']), int(data['n_off'])
    print(f"  Loaded from {npz_path}")
    print(f"  tON:  {len(on_t)} spikes, {len(np.unique(on_ids))}/{n_on} cells")
    print(f"  tOFF: {len(off_t)} spikes, {len(np.unique(off_ids))}/{n_off} cells")
    return on_t, on_ids, off_t, off_ids, n_on, n_off


def trim_spikes(times, ids, t_start_ms, t_stop_ms):
    mask = (times >= t_start_ms) & (times < t_stop_ms)
    return times[mask] - t_start_ms, ids[mask]


def make_spike_generator(times_ms, ids, n_cells, name):
    if len(times_ms) == 0:
        return SpikeGeneratorGroup(n_cells, [], [] * ms, name=name)
    return SpikeGeneratorGroup(
        n_cells, indices=ids, times=times_ms * ms,
        sorted=True, name=name
    )


def estimate_lgn_rate_per_neuron(spike_times, spike_ids, n_cells, n_target,
                                 p_connect, duration_ms):

    if duration_ms <= 0 or len(spike_times) == 0:
        return 0.0

    total_spikes = len(spike_times)
    pool_rate = total_spikes / (duration_ms / 1000.0)  
    per_cell_rate = pool_rate / max(n_cells, 1)
    n_connected = n_cells * p_connect
    rate_per_neuron = per_cell_rate * n_connected

    return rate_per_neuron


def compute_weight_from_target(target_drive_nS_per_s, rate_per_neuron_hz):

    if rate_per_neuron_hz <= 0:
        return 0.0
    return target_drive_nS_per_s / rate_per_neuron_hz



def make_lgn_synapses(lgn_group, target_group,
                      weight_ampa, weight_nmda,
                      p_connect, name):
    syn = Synapses(
        lgn_group, target_group,
        on_pre='gE_AMPA += w_ampa ; gE_NMDA += w_nmda',
        namespace={'w_ampa': weight_ampa, 'w_nmda': weight_nmda},
        name=name
    )
    syn.connect(p=p_connect)

    n_per_neuron = len(syn.i) / max(target_group.N, 1)
    print(f"    {name}: {len(syn.i)} connections "
          f"(~{n_per_neuron:.0f}/neuron, "
          f"wA={weight_ampa/nS:.2f}nS, wN={weight_nmda/nS:.2f}nS)")
    return syn

def make_lgn_inputs(column, CONFIG,
                    npz_path='lgn_spikes.npz',
                    baseline_time_ms=2000,
                    stimuli_time_ms=1500,
                    lgn_t_start=1000.0,
                    layers_to_connect=None,
                    target_drive=None,
                    nmda_ampa_ratio=TC_NMDA_AMPA_RATIO,
                    drive_scale=1.0):
 
    if layers_to_connect is None:
        layers_to_connect = ['L4C', 'L6']
    if target_drive is None:
        target_drive = DEFAULT_TARGET_DRIVE

    print("\n" + "=" * 65)
    print("  LGN -> V1 integration (auto-calibrated)")
    print("=" * 65)
    on_t, on_ids, off_t, off_ids, n_on, n_off = load_lgn_spikes_npz(npz_path)

    lgn_t_stop = lgn_t_start + stimuli_time_ms
    print(f"\n  Trimming to [{lgn_t_start}, {lgn_t_stop}] ms")
    on_t,  on_ids  = trim_spikes(on_t,  on_ids,  lgn_t_start, lgn_t_stop)
    off_t, off_ids = trim_spikes(off_t, off_ids, lgn_t_start, lgn_t_stop)

    on_t  += baseline_time_ms
    off_t += baseline_time_ms

    print(f"  After trim + offset: tON={len(on_t)} spikes, tOFF={len(off_t)} spikes")
    if len(on_t) > 0:
        print(f"  Time range: {on_t.min():.0f} - {max(on_t.max(), off_t.max() if len(off_t) else 0):.0f} ms")

    lgn_on  = make_spike_generator(on_t,  on_ids,  n_on,  'lgn_on')
    lgn_off = make_spike_generator(off_t, off_ids, n_off, 'lgn_off')

    all_groups   = [lgn_on, lgn_off]
    all_synapses = []

    for layer_name in layers_to_connect:
        if layer_name not in column.layers:
            print(f"\n  WARNING: '{layer_name}' not in column, skipping")
            continue
        if layer_name not in target_drive:
            print(f"\n  WARNING: no target drive for '{layer_name}', skipping")
            continue

        layer = column.layers[layer_name]
        print(f"\n  ── {layer_name} {'─' * (50 - len(layer_name))}")

        for cell_type in ['E', 'PV']:
            if cell_type not in layer.neuron_groups:
                print(f"    {cell_type} not in {layer_name}, skipping")
                continue
            if cell_type not in target_drive[layer_name]:
                print(f"    No target drive for {layer_name} {cell_type}, skipping")
                continue

            tgt = layer.neuron_groups[cell_type]
            target_nS_s = target_drive[layer_name][cell_type]

        
            p_on  = min(1.0, 20.0 / n_on)  
            p_off = min(1.0, 20.0 / n_off)

      
            rate_on = estimate_lgn_rate_per_neuron(
                on_t, on_ids, n_on, tgt.N, p_on, stimuli_time_ms
            )
            rate_off = estimate_lgn_rate_per_neuron(
                off_t, off_ids, n_off, tgt.N, p_off, stimuli_time_ms
            )
            total_rate = rate_on + rate_off

            print(f"    {cell_type}: target={target_nS_s:.0f} nS/s, "
                  f"LGN rate/neuron={total_rate:.0f} Hz "
                  f"(ON:{rate_on:.0f} + OFF:{rate_off:.0f})")

            if total_rate <= 0:
                print(f"    WARNING: zero LGN rate, skipping {layer_name} {cell_type}")
                continue

            w_nS = compute_weight_from_target(target_nS_s, total_rate)
            w_nS *= drive_scale

            weight_ampa = w_nS * nS
            weight_nmda = w_nS * nmda_ampa_ratio * nS

            print(f"    -> weight: {w_nS:.2f} nS AMPA, "
                  f"{w_nS * nmda_ampa_ratio:.2f} nS NMDA")


            all_synapses.append(make_lgn_synapses(
                lgn_on, tgt, weight_ampa, weight_nmda, p_on,
                f'syn_on_{layer_name}_{cell_type}'
            ))
 
            all_synapses.append(make_lgn_synapses(
                lgn_off, tgt, weight_ampa, weight_nmda, p_off,
                f'syn_off_{layer_name}_{cell_type}'
            ))

    total_conn = sum(len(s.i) for s in all_synapses)
    print(f"\n{'=' * 65}")
    print(f"  Summary:")
    print(f"    Layers:            {layers_to_connect}")
    print(f"    Synapse objects:   {len(all_synapses)}")
    print(f"    Total connections: {total_conn:,}")
    print(f"    NMDA/AMPA ratio:   {nmda_ampa_ratio}")
    print(f"    Drive scale:       {drive_scale}")
    print(f"    SOM/VIP:           no LGN input")
    if drive_scale != 1.0:
        print(f"    NOTE: drive_scale={drive_scale} applied globally")
    print(f"{'=' * 65}\n")

    return {'groups': all_groups, 'synapses': all_synapses}