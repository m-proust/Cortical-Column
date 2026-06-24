"""Stimulus input builders. Here we have feedforward and feedback inputs but you 
can define any other input here and use it in main.py or trials.py
"""
from brian2 import PoissonInput, Hz

# in those functions you can define any kind of particular stimulus, specifying the neuron targets, n° of sources and rate of the poisson inputs.
# you can also specify the target receptors (AMPA, GABA or NMDA)

def feedforward_input(column, w_ext_AMPA, w_ext_NMDA):

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

# and here is the actual stimulus definition, where you specify which neurons to target during baseline and stimulus time (can be None)

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
