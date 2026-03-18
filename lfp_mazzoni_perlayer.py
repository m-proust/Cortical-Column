"""
LFP proxy from synaptic currents — Mazzoni et al. (2015) RWS method.

Reference:
    Mazzoni, Linden, Cuntz, Lansner, Panzeri & Einevoll (2015)
    "Computing the Local Field Potential (LFP) from Integrate-and-Fire
     Network Models"
    PLoS Comput Biol, 11(12): e1004584

The LFP is approximated as a weighted sum of the absolute values of
excitatory and inhibitory synaptic currents received by pyramidal (E)
neurons:

    LFP(t) = Sigma_over_E_neurons [ |I_AMPA(t)| + |I_NMDA(t)|
                                + alpha*(|I_PV(t)| + |I_SOM(t)| + |I_VIP(t)|) ]

where alpha = 1.65 (Reference Weighted Sum coefficient from the paper).

Only currents onto E neurons are used because pyramidal cells dominate
LFP generation due to their open-field dendritic geometry. Interneuron
dendrites approximate a closed-field configuration and contribute
minimally. The absolute values are taken because apical excitatory
synapses and perisomatic inhibitory synapses generate dipoles that
sum with the same sign.

One LFP signal is produced per layer (= per local E population).
"""

import numpy as np
from scipy.signal import butter, filtfilt


ALPHA_RWS = 1.65


def calculate_lfp_mazzoni_perlayer(
    state_monitors,
    layer_configs,
    alpha=ALPHA_RWS,
    fs=10000,
    lowpass_cutoff=300.0,
    dt_ms=0.1,
):
    """
    Compute one LFP proxy signal per layer from E-neuron synaptic currents.

    Parameters
    ----------
    state_monitors : dict
        state_monitors[layer_name]['E_state'] -> Brian2 StateMonitor
        Must record: IsynE (or IsynE_AMPA + IsynE_NMDA)
        and IsynIPV, IsynISOM, IsynIVIP (or IsynI).
    layer_configs : dict
        CONFIG['layers'] -- used only for layer ordering.
    alpha : float
        Inhibitory weighting coefficient. Default 1.65.
    fs : float
        Output sampling rate (Hz).
    lowpass_cutoff : float
        LFP low-pass cutoff (Hz). Standard is 300.
    dt_ms : float
        Simulation timestep (ms).

    Returns
    -------
    lfp_signals : dict  {layer_name: 1D np.ndarray}
        One LFP proxy time series per layer (in pA, arbitrary scale).
    time_array : np.ndarray
        Time vector in ms at the output sampling rate.
    """
    sim_fs = 1.0 / (dt_ms * 1e-3)
    downsample_factor = max(1, int(round(sim_fs / fs)))
    actual_fs = sim_fs / downsample_factor

    lfp_signals = {}

    for layer_name in layer_configs:
        if layer_name not in state_monitors:
            print(f"  [WARN] No monitors for {layer_name}, skipping.")
            continue

        layer_mons = state_monitors[layer_name]
        if 'E_state' not in layer_mons:
            print(f"  [WARN] No E_state monitor for {layer_name}, skipping.")
            continue

        e_mon = layer_mons['E_state']

        # ---- excitatory current onto E neurons ----
        try:
            I_exc = (np.array(e_mon.IsynE_AMPA / 1e-12)
                     + np.array(e_mon.IsynE_NMDA / 1e-12))
        except AttributeError:
            try:
                I_exc = np.array(e_mon.IsynE / 1e-12)
            except AttributeError:
                print(f"  [ERR] No excitatory current vars for {layer_name}.")
                continue

        # ---- inhibitory current onto E neurons ----
        try:
            I_inh = (np.array(e_mon.IsynIPV / 1e-12)
                     + np.array(e_mon.IsynISOM / 1e-12)
                     + np.array(e_mon.IsynIVIP / 1e-12))
        except AttributeError:
            try:
                # IsynI = gI*(v - Ei) is positive; flip sign for convention
                I_inh = -np.array(e_mon.IsynI / 1e-12)
            except AttributeError:
                print(f"  [ERR] No inhibitory current vars for {layer_name}.")
                continue

        # ---- RWS proxy: sum |I_exc| + alpha*|I_inh| over all E neurons ----
        proxy = np.sum(np.abs(I_exc) + alpha * np.abs(I_inh), axis=0)

        # low-pass filter then downsample
        proxy = _lowpass(proxy, lowpass_cutoff, sim_fs)
        lfp_signals[layer_name] = proxy[::downsample_factor]

    if not lfp_signals:
        raise RuntimeError("Could not compute LFP for any layer.")

    n_out = len(next(iter(lfp_signals.values())))
    time_array = np.arange(n_out) * (1000.0 / actual_fs)

    return lfp_signals, time_array


def calculate_lfp_mazzoni_perlayer_decomposed(
    state_monitors,
    layer_configs,
    alpha=ALPHA_RWS,
    fs=10000,
    lowpass_cutoff=300.0,
    dt_ms=0.1,
):
    """
    Same as calculate_lfp_mazzoni_perlayer but returns excitatory and inhibitory
    contributions separately, useful for E/I balance analysis.

    Returns
    -------
    lfp_exc   : dict {layer: 1D array}  -- excitatory component
    lfp_inh   : dict {layer: 1D array}  -- inhibitory component (x alpha)
    lfp_total : dict {layer: 1D array}  -- sum
    time_array : np.ndarray (ms)
    """
    sim_fs = 1.0 / (dt_ms * 1e-3)
    downsample_factor = max(1, int(round(sim_fs / fs)))
    actual_fs = sim_fs / downsample_factor

    lfp_exc, lfp_inh, lfp_total = {}, {}, {}

    for layer_name in layer_configs:
        if layer_name not in state_monitors:
            continue
        layer_mons = state_monitors[layer_name]
        if 'E_state' not in layer_mons:
            continue

        e_mon = layer_mons['E_state']

        try:
            I_exc = (np.array(e_mon.IsynE_AMPA / 1e-12)
                     + np.array(e_mon.IsynE_NMDA / 1e-12))
        except AttributeError:
            try:
                I_exc = np.array(e_mon.IsynE / 1e-12)
            except AttributeError:
                continue

        try:
            I_inh = (np.array(e_mon.IsynIPV / 1e-12)
                     + np.array(e_mon.IsynISOM / 1e-12)
                     + np.array(e_mon.IsynIVIP / 1e-12))
        except AttributeError:
            try:
                I_inh = -np.array(e_mon.IsynI / 1e-12)
            except AttributeError:
                continue

        exc_comp = np.sum(np.abs(I_exc), axis=0)
        inh_comp = alpha * np.sum(np.abs(I_inh), axis=0)
        total    = exc_comp + inh_comp

        lfp_exc[layer_name]   = _lowpass(exc_comp, lowpass_cutoff, sim_fs)[::downsample_factor]
        lfp_inh[layer_name]   = _lowpass(inh_comp, lowpass_cutoff, sim_fs)[::downsample_factor]
        lfp_total[layer_name] = _lowpass(total,    lowpass_cutoff, sim_fs)[::downsample_factor]

    n_out = len(next(iter(lfp_total.values())))
    time_array = np.arange(n_out) * (1000.0 / actual_fs)
    return lfp_exc, lfp_inh, lfp_total, time_array


# -------------------------------------------------------------------
def _lowpass(sig, cutoff, fs, order=4):
    nyq = fs / 2.0
    if cutoff >= nyq:
        return sig
    b, a = butter(order, cutoff / nyq, btype='low')
    pad = min(3 * max(len(b), len(a)), len(sig) - 1)
    if pad < 1:
        return sig
    return filtfilt(b, a, sig, padlen=pad)
