import numpy as np
from brian2 import mm, ms, amp, Hz



R_E_OHM_M = 2.30
MIN_DIST_M = 10e-6   # 10 um


def _gather_positions(neuron_groups, layer_configs):
 
    all_xyz = []
    index_map = {}
    cursor = 0
    for layer_name, pops in neuron_groups.items():
        if layer_name not in layer_configs:
            continue
        index_map[layer_name] = {}
        for pop_name, grp in pops.items():
            x = np.asarray(grp.x / mm) * 1e-3
            y = np.asarray(grp.y / mm) * 1e-3
            z = np.asarray(grp.z / mm) * 1e-3
            n = len(x)
            index_map[layer_name][pop_name] = (cursor, cursor + n)
            all_xyz.append(np.stack([x, y, z], axis=1))
            cursor += n
    positions_m = np.vstack(all_xyz) if all_xyz else np.zeros((0, 3))
    return positions_m, index_map


def calculate_lfp_current_method(state_monitors,
                                 neuron_groups,
                                 layer_configs,
                                 electrode_positions,
                                 dt_ms=0.1,
                                 sim_duration_ms=4000,
                                 include_excitatory=True,
                                 include_inhibitory=True,
                                 use_absolute_distance=True):
    
    positions_m, index_map = _gather_positions(neuron_groups, layer_configs)
    N_total = positions_m.shape[0]
    if N_total == 0:
        raise ValueError("No neurons found in neuron_groups.")

    n_samples = int(round(sim_duration_ms / dt_ms))
    time_array = np.arange(n_samples) * dt_ms

    elec_m = np.asarray(electrode_positions) * 1e-3
    n_elec = elec_m.shape[0]

    lfp = np.zeros((n_elec, n_samples), dtype=np.float64)

    for layer_name, pop_monitors in state_monitors.items():
        if layer_name not in index_map:
            continue
        for mon_key, mon in pop_monitors.items():
            if not mon_key.endswith('_state'):
                continue
            pop_name = mon_key.replace('_state', '')
            if pop_name not in index_map[layer_name]:
                continue
            start, stop = index_map[layer_name][pop_name]

            parts = []
            if include_excitatory and 'IsynE' in mon.record_variables:
                parts.append(np.asarray(mon.IsynE / amp))
            if include_inhibitory and 'IsynI' in mon.record_variables:
                parts.append(np.asarray(mon.IsynI / amp))
            if not parts:
                continue
            I_total = sum(parts)   # (n_pop, n_t), still in amperes

            n_t_rec = I_total.shape[1]
            if n_t_rec != n_samples:
                t_rec_ms = np.asarray(mon.t / ms)
                I_resamp = np.empty((I_total.shape[0], n_samples),
                                    dtype=np.float64)
                for k in range(I_total.shape[0]):
                    I_resamp[k] = np.interp(time_array, t_rec_ms,
                                            I_total[k])
                I_total = I_resamp

            xyz_pop = positions_m[start:stop]   
            diff = xyz_pop[None, :, :] - elec_m[:, None, :]
            if use_absolute_distance:
                d = np.sqrt(np.sum(diff ** 2, axis=2))
            else:
                d = np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)
            np.maximum(d, MIN_DIST_M, out=d)  

            inv_d = 1.0 / d   

            contrib = (R_E_OHM_M / (4.0 * np.pi)) * (inv_d @ I_total)
            lfp += contrib

    lfp_uV = lfp * 1e6
    return lfp_uV, time_array
