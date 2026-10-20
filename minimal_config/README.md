# minimal_config: one isolated layer

A single layer with no other layers, used to find what in L6's circuit makes it
produce alpha when the other layers do not.

Everything except **connectivity** and **cell counts** is shared with
`config/config.py` or fixed in `config.py`, so it is the same in every experiment:
neuron model, intrinsic params, heterogeneity, synaptic time constants,
intra-layer delays, Poisson input rate (5 Hz), geometry (L6's z-range) and
electrodes. Poisson N is re-tuned for every experiment so all populations fire
at their target rates (rate matching).

## Files

| file | what it is |
|---|---|
| `connection_probabilities.csv`, `conductances_AMPA_GABA.csv`, `conductances_NMDA.csv` | intact L6, 4x4 (rows = source, cols = target; E rows AMPA/NMDA, others GABA) |
| `layer.json` | intact L6 cell counts, Poisson N, target rates |
| `config.py` | builds `CONFIG` for a condition |
| `match_rates.py` | tunes Poisson N to the target rates |
| `run_spectrum.py` | 10 s baseline run -> rate and bipolar LFP spectra (.npz) |
| `experiments/conditions.py` | **every experiment**, each defined as a change to a base layer |
| `experiments/tuned_N.json` | tuned Poisson N per condition (written by `match_rates.py`) |
| (`experiments/plot_block_swaps.py`, the block-swap figures, is in `archive/minimal_config/experiments/`) | |

## Adding an experiment

1. Add a line to `CONDITIONS` in `experiments/conditions.py`, e.g.

       'L6+L4C_SOM-E': dict(base='L6', donor='L4C', cells=[('SOM', 'E')], conv=True),

   (`rows` = whole output rows, `cells` = single connections, `conv=True` =
   convergence-matched, `counts=True` = donor's cell counts, `targets` = whose rates.)
2. Rate-match: `~/Desktop/venv/bin/python minimal_config/match_rates.py L6+L4C_SOM-E`
3. Spectra: `~/Desktop/venv/bin/python minimal_config/run_spectrum.py L6+L4C_SOM-E minimal_config/experiments/blockswap/_spectra/L6+L4C_SOM-E_s58925.npz --seed 58925` (repeat for seeds 1, 2)
4. Or run it in `main.py`: import `from minimal_config.config import CONFIG`, then
   `MINIMAL_CONDITION=L6+L4C_SOM-E ~/Desktop/venv/bin/python main.py`

## Gotchas

- **To remove a pathway, set its probability to 0, not its conductance.**
  `src/layer.py` builds a synapse whenever p > 0 and falls back to 0.01 nS (E) /
  0.02 nS (inhibitory) when the conductance is 0.
- **Compare over the whole baseline**, not main.py's 1.5 s window (too short to
  resolve shifts of a few Hz).
- **With one layer, every electrode sees the same sources**; channels are not laminar here.
- Non-L6 layers are read from the **working copy** of `config/*.csv`, including
  uncommitted edits.
- Rate matching replaces inter-layer input with **independent** Poisson input. For
  layers whose interneurons get most of their drive from other layers (L23: PV
  needs N~350), this weakens every local loop and can remove rhythms entirely.
