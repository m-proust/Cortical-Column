# Cortical Column Simulation

A [Brian2](https://brian2.readthedocs.io/en/stable/) spiking-network model of a primate V1
**cortical column** (layers L2/3, L4A/B, L4C, L5, L6). Each layer has **excitatory (E)**
neurons and three inhibitory types — **PV**, **SOM**, **VIP** — built from AdEx neurons.
The model rests at an alpha (~10 Hz) rhythm and switches to gamma (~40+ Hz) when a
feedforward stimulus is applied.

---

## Getting started

You need Python 3. From a terminal:

```bash
git clone https://github.com/mathilde-sbri/Cortical-Column.git
cd Cortical-Column

python -m venv venv                 # create a virtual environment
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt     # install the packages it needs
```

The virtual environment keeps this project's packages separate from the rest of your
computer. Activate it (`source venv/bin/activate`) every time you open a new terminal to
work on the project.

> **Always run scripts from this folder** (the project root). Each subfolder has its own
> short README explaining what is inside.

---

## Repository structure

```
Cortical-Column/
├── main.py            # run ONE simulation and show figures
├── trials.py          # run MANY trials and save them to disk
├── stim_profiles.py   # the stimulus input profiles (feedforward / feedback)
│
├── config/            # the model's settings (parameters + CSV tables)
├── src/               # the model itself (the cortical column code)
├── tools/             # shared helpers (LFP estimators, CSV loading)
├── experiments/       # ready-made scripts that RUN simulations
├── analysis/          # scripts that PLOT already-saved trials
│
├── saved_trials/      # where new runs are saved
└── figures/           # where analysis scripts save figures
```

---

## Run one simulation

```bash
python main.py
```

This runs a baseline period then a stimulus period and shows rasters, firing rates and
power spectra. To change the input, edit the line near the top of the file:

```python
STIM_PROFILE = "feedforward"   # or "feedback"
```

---

## Run several trials

```bash
python trials.py --save-dir saved_trials/my_run --n-trials 20
```

This runs 20 simulations and saves each one to `saved_trials/my_run/`. You can then plot
them with the scripts in [analysis/](analysis/).
