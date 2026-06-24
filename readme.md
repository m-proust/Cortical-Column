# Cortical Column Simulation

A [Brian2](https://brian2.readthedocs.io/en/stable/) spiking-network laminar model of a primate V1
cortical column. Each layer has excitatory (E)
neurons and interneurons subtypes (PV, SOM, VIP) modeled by the Adex equations.

---

## Getting started

In your terminal,

```bash
git clone https://github.com/mathilde-sbri/Cortical-Column.git
cd Cortical-Column

python -m venv venv           
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt    
```

---

## Repository structure

```
Cortical-Column/
├── main.py            # run a simulation
├── trials.py          # run trials and save them
├── stim_profiles.py   # the stimulus input profiles (feedforward / feedback) that will be added to the column during simulation.
│
├── config/            # the model's settings (parameters + CSV tables)
├── src/               # scripts that build the model 
├── tools/             # helpers 
├── experiments/       # lesions, etc
├── analysis/          # scripts for plotting saved trials
```

---

## To run a simulation

```bash
python main.py
```


---

## Run several trials

```bash
python trials.py --save-dir saved_trials/my_run --n-trials 20
```

This runs 20 simulations and saves each one to `saved_trials/my_run/`. 
