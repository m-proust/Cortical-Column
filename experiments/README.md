# experiments/

Scripts that run simulations and save the results. Run them from the
project root, e.g. `python experiments/run_lesion_experiment.py`.

| Script | Content |
| --- | --- |
| `run_lesion_experiment.py` | runs trials performing 'lesions' on the model, e.g control trials and trials with a specific connection removed (for example L23->L5) |
| `trials_resampled_network.py` | runs trials, reseeding the network connections and parameters each time |
| `interlayer_p_sweep.py` | gradually scales all the connections between layers from 0 to the full actual connectivity of the column|
| `cleo/main_sim_cleo.py` | one simulated electrode recording (uses the [Cleo](https://cleosim.readthedocs.io/) package) |
| `cleo/trials_cleo.py` | runs [Cleo](https://github.com/siplab-gt/cleo) electrode-recording trials |

Each script has a settings block at the top (number of trials, save folder, etc) that you can edit. Results are saved into `saved_trials/` by default. Plot them
with the scripts in [../analysis/](../analysis/).
