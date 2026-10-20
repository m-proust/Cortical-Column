# studies/

One folder per study. Each holds both the simulations and the analysis of that study. Run everything from the
project root with `~/Desktop/venv/bin/python`. Simulations save into `saved_trials/` and figures go into
`figures/` (both gitignored). Isolated-layer simulations are built with [`minimal_config/`](../minimal_config/README.md).

| Folder | Question |
| --- | --- |
| [`resonance/`](#resonance) | What sets the alpha peak frequency? Linear-response prediction from the neurons' and synapses' filters |
| [`phase_gating/`](#phase_gating) | Does the alpha phase gate the response to a weak input, and more so when alpha is strong? |
| [`feedback/`](#feedback) | Feedback on vs off: change in alpha power and in phase gating |
| [`cosyne_abstract/`](#cosyne_abstract) | The remaining model panels of the Cosyne 2027 abstract, and the tables and plots of all of them |

Earlier prototypes that are not used by any of these are in [`../archive/`](../archive/README.md).

---

## resonance

Rates of the populations, for small fluctuations: `r = M(f) r + noise`, with `M[i,k] = sum_X N[i,X](f) G[i,k,X](f)`.
N (rate change per nS of AMPA / NMDA / GABA conductance) is measured by spike replay. G (mean conductance per Hz
of presynaptic rate: in-degree x jump x tau/(1 + i w tau) x delay factor) is exact from the parameters. The
predicted peak is the peak of the E-rate spectrum `sum_k |(I - M)^-1|^2_Ek q_k`, with Poisson noise
`q_k = rate_k / N_k`. Nothing is fitted. The method is derived in Appendix B of the first-year summary.

| File | What it does |
| --- | --- |
| `replay_susceptibility.py` | Simulation. Runs isolated L6 closed-loop, then replays its spikes open-loop with small multisine probe conductances added: N(f) at the intact operating state |
| `run_replay_susceptibility.sh` | The 5 networks used in the abstract → `saved_trials/cosyne/replay_susceptibility/` |
| `freqmech.py` | Theory library: N(f) from a replay run, G(f), M(f), (I - M)^-1, spectrum, peak |
| `analyse.py` | Prediction vs measured peak, held-out timing sweeps, phase budget, counterfactuals → `saved_trials/resonance/<cond>_results.json`, `figures/frequency_mechanism/` |
| `measured.py` | Measured peak frequencies of the spiking runs (`minimal_config/experiments/fig1/_runs/`) |
| `checks.py` | Checks of the estimator → `saved_trials/resonance/L6_checks.json` |
| `eif_theory.py`, `renewal.py`, `run_theory.py` | N(f) from the parameters alone (EIF Fokker–Planck + renewal adaptation), compared with the measured N |
| `export_abstract_data.py` | `freq_theory.dat` for the abstract (`cosyne_abstract/abstract_data.py` also writes it) |
| `figure_abstract.py`, `figure_amplification.py`, `figure_poster.py` | Abstract and poster figures → `figures/frequency_mechanism/` |

Data: `saved_trials/resonance/` holds the original replay runs (`L6_s*.npz`), the held-out conditions, the
results JSONs and `analytic/`. The abstract's 5 longer runs (100 s, 5 pA) are in
`saved_trials/cosyne/replay_susceptibility/`.

## phase_gating

| File | What it does |
| --- | --- |
| `run_pulses.py` | Simulation. Intact column. At an onset every 25 ms the state is stored, and pulses of each dose are given from that same state (0 pA sham, 10, 25, 50, 100 pA for 10 ms into L4C E; or into L6 E; or an LGN volley). The alpha phase is measured afterwards. The docstring has the exact command for the abstract runs |
| `analyse_gating.py` | One dataset (`EXP5C_LAYER=L4C` or `L6`; add `EXP5C_DIR=exp5c_restore_pulse_LGN` for the LGN volley): pulse − sham response vs phase, with a cosine fit per dose |
| `plot_key_figure.py` | One-panel summary: L4C response vs local alpha phase, weak vs strong vs thalamic input. `abstract_data.py` uses its smoother for Fig. 3B |
| `phase_x_amplitude.py` | The same, split into thirds of ongoing alpha amplitude: does gating deepen with alpha? `abstract_data.py` uses its loader for Fig. 3B |

Data: `saved_trials/exp5c_restore_pulse_{L4C,L6,LGN}/`; the abstract's 10 networks are in
`saved_trials/cosyne/pulse_gating_L4C/`. Figures go to `figures/exp5_phase/`. 

## feedback

Intact column with feed-forward input from 0 ms, plus feedback input to L2/3 and L5 SOM from 2000 ms in the
"on" arm. 6 paired networks (seeds 920101–920106): a 4 s spectral run, then a probe run with 96 sham / 25 pA L4C /
25 pA L2/3 pulses from 4000 ms. Run by Codex on 10 Oct 2026. The pre-registered plan is
`saved_trials/cosyne/feedback_protocol/EXPERIMENT_PLAN.md`.

| File | What it does |
| --- | --- |
| `run_queue.py` | Runs every pair (via `run_feedback.py`, which uses helpers from `run_pilot.py`), then `analyze.py` |
| `model/` | **Frozen** copy of `config/` and `src/` as of 8 Oct 2026, so later edits to the model do not affect it |
| `stim_profiles_snapshot.py`, `lfp_kernel_snapshot.py`, `laminar_power_change_snapshot.py` | Frozen copies of the project's stimulus, LFP and multitaper code |
| `validate.py` | Checks the smoke run, and that the snapshots still match the hashes in `source_manifest.json` |
| `analyze.py` | Abstract 3C, 3D step 1: multitaper power change pre vs post feedback, per layer, and a cosine fit of pulse − sham vs phase → `metrics.json`, `RESULTS.md` |
| `plot_phase_feedback.py` | Abstract 3D step 2: mean phase curves per condition with bootstrap intervals → `phase_curves/` |

Data: `saved_trials/cosyne/feedback_protocol/`. 

## cosyne_abstract

The abstract is `~/Desktop/cosyne_abstract/cosyne_abstract.tex`. Its pgfplots figures read plain-text tables
built from the saved runs in `saved_trials/cosyne/`.

```bash
~/Desktop/venv/bin/python studies/cosyne_abstract/abstract_data.py --check         # all tables -> figures/cosyne_abstract/data/, compared with the abstract's copy
~/Desktop/venv/bin/python studies/cosyne_abstract/plot_abstract_panels.py           # -> figures/cosyne_abstract/fig2_model.png, fig3_model.png
~/Desktop/venv/bin/python studies/cosyne_abstract/abstract_data.py --out ~/Desktop/cosyne_abstract/figures/data   # update the abstract itself
```

| Panel | Simulation | Saved runs (`saved_trials/cosyne/…`) |
| --- | --- | --- |
| 2A, 2C, 3A "no input" | `run_isolated_layers.py --set layers`: isolated, rate-matched L2/3, L4AB, L4C, L5, L6, and the L2/3 transplant; 20 networks | `isolated_layers/` |
| 2B | `run_reduced_L6.sh`: isolated L6 with only E+SOM, E+SOM+PV or E+PV | `reduced_L6/` |
| 2D | `run_finite_size.sh`: isolated L6 at 0.5–8 x its size, independent or partly shared drive | `finite_size/` |
| 2E network | `run_isolated_layers.py --set sweeps`: SOM→E decay and loop-delay sweeps | `isolated_layers/` |
| 2E theory | `../resonance/run_replay_susceptibility.sh` | `replay_susceptibility/` |
| 3A | `record_rest.py` (resting spikes of the intact column), then `run_layer_replay.sh` (uses `run_replay.py`): isolated L4C / L2/3 driven by the recorded spikes vs rate-matched Poisson | `rest_sources/`, `layer_replay/` |
| 3B | `../phase_gating/run_pulses.py` | `pulse_gating_L4C/` |
| 3C, 3D | `../feedback/run_queue.py` | `feedback_protocol/` |

| Other file | What it does |
| --- | --- |
| `metrics.py` | `residual_db` (power in dB above a 1/f fit) and `alpha_metrics`, used by the scripts here |
| `run_reviewer.py`, `reviewer_checks.py` | Reviewer checks (adaptation, sensitivity map); runs in `minimal_config/experiments/reviewer/` |

How the panels are computed:
- **2A–C, 3A.** Mean over networks of the PSD in dB above a 1/f fit, from 1.5 Hz up. 2A uses the in-layer bipolar
  LFP channel (index 3); the others use the E population rate.
- **2D.** Per-cell E-rate PSD (population PSD × 10⁶ / N_E²), averaged over 8–13 Hz and over networks.
- **2E.** Network: the mean E-rate PSD, normalised to its peak. Theory: `resonance/` with only the SOM→E decay
  changed, at constant charge.
- **3B.** Pulse (25 pA) minus sham L4C E rate, 0–20 ms after onset, against the local 7–16 Hz alpha phase, split into
  thirds of ongoing alpha amplitude ranked within each network. Kernel smoother (SD 25°), 95 % bootstrap over networks.

