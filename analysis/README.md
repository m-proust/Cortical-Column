# analysis/

Scripts that perform analysis on trials that have been ran and saved (see trials.py and the /experiments subfolder). Run from the project root, e.g. `python analysis/spectral/view_trial.py`.

Each script has a settings block at the top in which you can set the trial folder and output paths,
then run it. Figures are saved into `figures/` by default.

| Subfolder | Content |
| --- | --- |
| `spectral/` | power spectrum changes with stimulus |
| `lesions/` | plots of individual lesion experiment + the global connectivity p sweep experiment |
| `coupling/` | cross-frequency and spike-LFP coupling |
| `laminar/` | depth-resolved CSD / LFP aligned to oscillation troughs |
| `lfp/` | comparison of the two LFP estimation methods (kernel and synaptic currents) |
| `visualisations/` | visualisation of connectivity matrices and interneuron proportions in the config and single trial visualisation|
| `cleo/` | plots of saved [Cleo](https://github.com/siplab-gt/cleo) electrode-recording trials |
