
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
        'E':  100.0,
        'PV': 200.0,
    },
    'L5': {
        'E':  500.0,
        'PV': 750.0,
    },
}

TC_NMDA_AMPA_RATIO = 0.1



def load_lgn_spikes_npz(npz_path):
    data = np.load(npz_path)

    # Support both old 2-type and new 4-type .npz formats
    if 'tON_times' in data:
        # New 4-type format
        lgn = {}
        for typ in ['tON', 'tOFF', 'sON', 'sOFF']:
            t_key, id_key, n_key = f'{typ}_times', f'{typ}_ids', f'n_{typ}'
            if t_key in data:
                lgn[typ] = (data[t_key], data[id_key], int(data[n_key]))
                print(f"  {typ}: {len(data[t_key])} spikes, "
                      f"{len(np.unique(data[id_key]))}/{int(data[n_key])} cells")
        print(f"  Loaded 4-type format from {npz_path}")
        return lgn
    else:
        # Old 2-type format — wrap into same dict structure
        on_t, on_ids = data['on_times'], data['on_ids']
        off_t, off_ids = data['off_times'], data['off_ids']
        n_on, n_off = int(data['n_on']), int(data['n_off'])
        print(f"  Loaded legacy 2-type format from {npz_path}")
        print(f"  tON:  {len(on_t)} spikes, {len(np.unique(on_ids))}/{n_on} cells")
        print(f"  tOFF: {len(off_t)} spikes, {len(np.unique(off_ids))}/{n_off} cells")
        return {'tON': (on_t, on_ids, n_on), 'tOFF': (off_t, off_ids, n_off)}


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
                    total_lgn_duration_ms=4000,
                    layers_to_connect=None,
                    target_drive=None,
                    nmda_ampa_ratio=TC_NMDA_AMPA_RATIO,
                    drive_scale=1.0):
    """
    Create LGN → V1 connections with single drive_scale for all spikes.
    Kept for backwards compatibility — prefer make_lgn_inputs_split.
    """
    return make_lgn_inputs_split(
        column, CONFIG,
        npz_path=npz_path,
        total_lgn_duration_ms=total_lgn_duration_ms,
        layers_to_connect=layers_to_connect,
        target_drive=target_drive,
        nmda_ampa_ratio=nmda_ampa_ratio,
        gray_drive_scale=drive_scale,
        grating_drive_scale=drive_scale,
        gray_duration_ms=None,  # single-regime: no split
    )


def make_lgn_inputs_split(column, CONFIG,
                          npz_path='lgn_spikes.npz',
                          total_lgn_duration_ms=4000,
                          layers_to_connect=None,
                          target_drive=None,
                          nmda_ampa_ratio=TC_NMDA_AMPA_RATIO,
                          gray_drive_scale=0.6,
                          grating_drive_scale=1.3,
                          gray_duration_ms=2000):
    """
    Create LGN → V1 connections with separate drive scales for gray screen
    and grating periods.

    Splits LGN spikes into two epochs, creates separate SpikeGeneratorGroups
    and Synapses for each, with weights calibrated against each period's
    actual firing rates and then scaled by the respective drive_scale.

    Parameters
    ----------
    gray_duration_ms : float or None
        Duration of gray screen period (ms). Grating runs from gray_duration_ms
        to total_lgn_duration_ms. If None, no split is performed (single regime
        using grating_drive_scale for everything).
    gray_drive_scale : float
        Weight multiplier for gray screen period.
    grating_drive_scale : float
        Weight multiplier for grating period.
    """

    if layers_to_connect is None:
        layers_to_connect = ['L4C', 'L6']
    if target_drive is None:
        target_drive = DEFAULT_TARGET_DRIVE

    do_split = gray_duration_ms is not None
    print("\n" + "=" * 65)
    if do_split:
        print(f"  LGN -> V1 (split: gray 0–{gray_duration_ms}ms "
              f"scale={gray_drive_scale}, grating {gray_duration_ms}–"
              f"{total_lgn_duration_ms}ms scale={grating_drive_scale})")
    else:
        print(f"  LGN -> V1 (single regime, scale={grating_drive_scale})")
    print("=" * 65)

    lgn_types = load_lgn_spikes_npz(npz_path)

    # Print spike info
    for typ, (times, ids, n_cells) in lgn_types.items():
        if len(times) > 0:
            print(f"  {typ}: {len(times)} spikes, time range "
                  f"{times.min():.0f}–{times.max():.0f} ms")
        else:
            print(f"  {typ}: 0 spikes")

    # --- Define epochs ---
    if do_split:
        epochs = [
            ('gray', 0, gray_duration_ms, gray_drive_scale),
            ('grating', gray_duration_ms, total_lgn_duration_ms, grating_drive_scale),
        ]
    else:
        epochs = [
            ('full', 0, total_lgn_duration_ms, grating_drive_scale),
        ]

    all_groups = []
    all_synapses = []

    for epoch_name, t_start, t_stop, scale in epochs:
        epoch_dur = t_stop - t_start
        print(f"\n  ╔═ Epoch: {epoch_name} ({t_start}–{t_stop} ms, "
              f"scale={scale}) ═╗")

        # Split spikes for this epoch (keep original times — no offset)
        epoch_lgn = {}
        for typ, (times, ids, n_cells) in lgn_types.items():
            mask = (times >= t_start) & (times < t_stop)
            epoch_lgn[typ] = (times[mask], ids[mask], n_cells)
            n_spk = mask.sum()
            if n_spk > 0:
                rate = n_spk / (epoch_dur / 1000.0) / n_cells
                print(f"    {typ}: {n_spk} spikes ({rate:.1f} Hz/cell)")

        # Build SpikeGeneratorGroups for this epoch
        # Spikes keep their original timestamps so they fire at the right sim time
        epoch_groups = {}
        for typ, (times, ids, n_cells) in epoch_lgn.items():
            name = f'lgn_{typ}_{epoch_name}'
            epoch_groups[typ] = make_spike_generator(times, ids, n_cells, name)
        all_groups.extend(epoch_groups.values())

        # Connect to cortical targets
        for layer_name in layers_to_connect:
            if layer_name not in column.layers:
                print(f"\n    WARNING: '{layer_name}' not in column, skipping")
                continue
            if layer_name not in target_drive:
                print(f"\n    WARNING: no target drive for '{layer_name}', skipping")
                continue

            layer = column.layers[layer_name]
            print(f"\n    ── {layer_name} {'─' * (46 - len(layer_name))}")

            for cell_type in ['E', 'PV']:
                if cell_type not in layer.neuron_groups:
                    continue
                if cell_type not in target_drive[layer_name]:
                    continue

                tgt = layer.neuron_groups[cell_type]
                target_nS_s = target_drive[layer_name][cell_type]

                # Rate estimation for THIS epoch only
                total_rate = 0.0
                type_rates = {}
                for typ, (times, ids, n_cells) in epoch_lgn.items():
                    p_conn = min(1.0, 20.0 / n_cells)
                    rate = estimate_lgn_rate_per_neuron(
                        times, ids, n_cells, tgt.N, p_conn, epoch_dur
                    )
                    type_rates[typ] = (rate, p_conn)
                    total_rate += rate

                rate_str = " + ".join(
                    f"{typ}:{r:.0f}" for typ, (r, _) in type_rates.items()
                )
                print(f"      {cell_type}: target={target_nS_s:.0f} nS/s, "
                      f"rate/neuron={total_rate:.0f} Hz ({rate_str})")

                if total_rate <= 0:
                    print(f"      WARNING: zero rate, skipping")
                    continue

                w_nS = compute_weight_from_target(target_nS_s, total_rate)
                w_nS *= scale

                weight_ampa = w_nS * nS
                weight_nmda = w_nS * nmda_ampa_ratio * nS

                print(f"      -> weight: {w_nS:.2f} nS AMPA, "
                      f"{w_nS * nmda_ampa_ratio:.2f} nS NMDA")

                for typ in epoch_groups:
                    _, p_conn = type_rates[typ]
                    syn_name = f'syn_{typ}_{layer_name}_{cell_type}_{epoch_name}'
                    all_synapses.append(make_lgn_synapses(
                        epoch_groups[typ], tgt,
                        weight_ampa, weight_nmda, p_conn, syn_name
                    ))

    total_conn = sum(len(s.i) for s in all_synapses)
    print(f"\n{'=' * 65}")
    print(f"  Summary:")
    print(f"    Epochs:            {[e[0] for e in epochs]}")
    print(f"    Layers:            {layers_to_connect}")
    print(f"    Synapse objects:   {len(all_synapses)}")
    print(f"    Total connections: {total_conn:,}")
    print(f"    NMDA/AMPA ratio:   {nmda_ampa_ratio}")
    if do_split:
        print(f"    Gray drive scale:  {gray_drive_scale}")
        print(f"    Grating drive scale: {grating_drive_scale}")
    else:
        print(f"    Drive scale:       {grating_drive_scale}")
    print(f"    SOM/VIP:           no LGN input")
    print(f"{'=' * 65}\n")

    return {'groups': all_groups, 'synapses': all_synapses}



def make_lgn_inputs_split_optimized(column, CONFIG,
                                     npz_path='lgn_spikes.npz',
                                     total_lgn_duration_ms=4000,
                                     layers_to_connect=None,
                                     target_drive=None,
                                     gray_drive_scale=None,
                                     grating_drive_scale=None,
                                     gray_duration_ms=2000,
                                     tc_nmda_ratio=None):
    """
    Extended version of make_lgn_inputs_split that supports:
      - Per-cell-type drive scales: gray_drive_scale={'E': 0.5, 'PV': 0.5}
      - Per-cell-type NMDA ratios:  tc_nmda_ratio={'E': 0.15, 'PV': 0.0}

    Drop-in replacement — if you pass scalar values, they work as before.
    """
    from brian2 import ms, nS, SpikeGeneratorGroup, Synapses
    import numpy as np

    if layers_to_connect is None:
        layers_to_connect = ['L4C', 'L6']
    if target_drive is None:
        target_drive = DEFAULT_TARGET_DRIVE
    if tc_nmda_ratio is None:
        tc_nmda_ratio = {'E': TC_NMDA_AMPA_RATIO, 'PV': TC_NMDA_AMPA_RATIO}
    if isinstance(tc_nmda_ratio, (int, float)):
        tc_nmda_ratio = {'E': tc_nmda_ratio, 'PV': tc_nmda_ratio}

    def _as_dict(val, default=1.0):
        if val is None:
            return {'E': default, 'PV': default}
        if isinstance(val, (int, float)):
            return {'E': val, 'PV': val}
        return val

    gray_scales = _as_dict(gray_drive_scale, 0.5)
    grating_scales = _as_dict(grating_drive_scale, 1.2)

    do_split = gray_duration_ms is not None

    print("\n" + "=" * 65)
    if do_split:
        print(f"  LGN -> V1 OPTIMIZED (split: gray 0-{gray_duration_ms}ms, "
              f"grating {gray_duration_ms}-{total_lgn_duration_ms}ms)")
        print(f"    Gray scales:    E={gray_scales.get('E','-')}, "
              f"PV={gray_scales.get('PV','-')}")
        print(f"    Grating scales: E={grating_scales.get('E','-')}, "
              f"PV={grating_scales.get('PV','-')}")
        print(f"    TC NMDA ratio:  E={tc_nmda_ratio.get('E','-')}, "
              f"PV={tc_nmda_ratio.get('PV','-')}")
    print("=" * 65)

    lgn_types = load_lgn_spikes_npz(npz_path)

    if do_split:
        epochs = [
            ('gray', 0, gray_duration_ms, gray_scales),
            ('grating', gray_duration_ms, total_lgn_duration_ms, grating_scales),
        ]
    else:
        epochs = [('full', 0, total_lgn_duration_ms, grating_scales)]

    all_groups = []
    all_synapses = []

    for epoch_name, t_start, t_stop, scale_dict in epochs:
        epoch_dur = t_stop - t_start
        print(f"\n  -- Epoch: {epoch_name} ({t_start}-{t_stop} ms) --")

        epoch_lgn = {}
        for typ, (times, ids, n_cells) in lgn_types.items():
            mask = (times >= t_start) & (times < t_stop)
            epoch_lgn[typ] = (times[mask], ids[mask], n_cells)
            n_spk = mask.sum()
            if n_spk > 0:
                rate = n_spk / (epoch_dur / 1000.0) / n_cells
                print(f"    {typ}: {n_spk} spikes ({rate:.1f} Hz/cell)")

        epoch_groups = {}
        for typ, (times, ids, n_cells) in epoch_lgn.items():
            name = f'lgn_{typ}_{epoch_name}'
            epoch_groups[typ] = make_spike_generator(times, ids, n_cells, name)
        all_groups.extend(epoch_groups.values())

        for layer_name in layers_to_connect:
            if layer_name not in column.layers:
                continue
            if layer_name not in target_drive:
                continue

            layer = column.layers[layer_name]

            for cell_type in ['E', 'PV']:
                if cell_type not in layer.neuron_groups:
                    continue
                if cell_type not in target_drive[layer_name]:
                    continue

                tgt = layer.neuron_groups[cell_type]
                target_nS_s = target_drive[layer_name][cell_type]

                total_rate = 0.0
                type_rates = {}
                for typ, (times, ids, n_cells) in epoch_lgn.items():
                    p_conn = min(1.0, 20.0 / n_cells)
                    rate = estimate_lgn_rate_per_neuron(
                        times, ids, n_cells, tgt.N, p_conn, epoch_dur
                    )
                    type_rates[typ] = (rate, p_conn)
                    total_rate += rate

                if total_rate <= 0:
                    print(f"      {layer_name}.{cell_type}: zero rate, skipping")
                    continue

                w_nS = compute_weight_from_target(target_nS_s, total_rate)

                # Per-cell-type scaling
                cell_scale = scale_dict.get(cell_type, 1.0)
                w_nS *= cell_scale

                # Per-cell-type NMDA ratio
                nmda_ratio = tc_nmda_ratio.get(cell_type, 0.0)

                weight_ampa = w_nS * nS
                weight_nmda = w_nS * nmda_ratio * nS

                print(f"      {layer_name}.{cell_type}: "
                      f"wA={weight_ampa/nS:.2f}nS, wN={weight_nmda/nS:.2f}nS "
                      f"(scale={cell_scale:.2f}, nmda_r={nmda_ratio:.2f})")

                for typ in epoch_groups:
                    _, p_conn = type_rates[typ]
                    syn_name = f'syn_{typ}_{layer_name}_{cell_type}_{epoch_name}'
                    all_synapses.append(make_lgn_synapses(
                        epoch_groups[typ], tgt,
                        weight_ampa, weight_nmda, p_conn, syn_name
                    ))

    total_conn = sum(len(s.i) for s in all_synapses)
    print(f"\n  Total LGN connections: {total_conn:,}")
    print(f"{'='*65}\n")

    return {'groups': all_groups, 'synapses': all_synapses}