"""Stimulus input builders. Each takes (column, w_ext_AMPA, w_ext_NMDA) and
returns a list of PoissonInput objects.
"""
from brian2 import PoissonInput, Hz


def feedforward_input(column, w_ext_AMPA, w_ext_NMDA):
    """Feedforward thalamocortical AMPA drive onto L4C and L6 (E + PV).

    PV gets a stronger drive than E in vivo; do not reduce PV weights below E.
    """
    L4C = column.layers['L4C']
    L6 = column.layers['L6']

    L4C_E = L4C.neuron_groups['E']
    L4C_PV = L4C.neuron_groups['PV']
    L6_E = L6.neuron_groups['E']
    L6_PV = L6.neuron_groups['PV']

    return [
        PoissonInput(L4C_E,  'gE_AMPA', N=30, rate=5 * Hz, weight=w_ext_AMPA),
        PoissonInput(L4C_PV, 'gE_AMPA', N=40, rate=7 * Hz, weight=w_ext_AMPA * 2.5),
        PoissonInput(L6_E,   'gE_AMPA', N=10, rate=5 * Hz, weight=w_ext_AMPA * 1.5),
        PoissonInput(L6_PV,  'gE_AMPA', N=10, rate=6 * Hz, weight=w_ext_AMPA * 1.5),
    ]


def feedback_input(column, w_ext_AMPA, w_ext_NMDA):
    """Cortico-cortical feedback: AMPA+NMDA onto L23 and L5 SOM cells.

    Martinotti / SOM-mediated dendritic feedback. LM->V1 feedback in mouse
    preferentially targets SOM interneurons. This is the input used for the
    trials saved under results/trials_06_05-fb.
    """
    L23 = column.layers['L23']
    L5 = column.layers['L5']

    L23_SOM = L23.neuron_groups['SOM']
    L5_SOM = L5.neuron_groups['SOM']

    return [
        PoissonInput(L23_SOM, 'gE_NMDA', N=20, rate=7 * Hz, weight=w_ext_NMDA * 1.5),
        PoissonInput(L23_SOM, 'gE_AMPA', N=10, rate=7 * Hz, weight=w_ext_AMPA),
        PoissonInput(L5_SOM,  'gE_NMDA', N=20, rate=7 * Hz, weight=w_ext_NMDA * 1.5),
        PoissonInput(L5_SOM,  'gE_AMPA', N=10, rate=7 * Hz, weight=w_ext_AMPA),
    ]


def make_profile(baseline=None, stim=None):
    return {"baseline": baseline, "stim": stim}


STIM_PROFILES = {
    "feedforward": make_profile(baseline=None, stim=feedforward_input),
    "feedback":    make_profile(baseline=feedforward_input, stim=feedback_input),
}

EPOCHS = ("baseline", "stim")


def build_epoch(profile, epoch, column, w_ext_AMPA, w_ext_NMDA):
    if profile not in STIM_PROFILES:
        raise ValueError(
            f"Unknown STIM_PROFILE {profile!r}. "
            f"Choose one of {sorted(STIM_PROFILES)}.")
    if epoch not in EPOCHS:
        raise ValueError(
            f"Unknown epoch {epoch!r}. Choose one of {EPOCHS}.")

    builder = STIM_PROFILES[profile][epoch]
    if builder is None:
        return []
    return builder(column, w_ext_AMPA, w_ext_NMDA)


def build_stimulus(profile, column, w_ext_AMPA, w_ext_NMDA):
    return build_epoch(profile, "stim", column, w_ext_AMPA, w_ext_NMDA)
