# Cortical Column Simulation 

This repository contains a modular simulation of a **cortical column** using the [Brian2](https://brian2.readthedocs.io/en/stable/) spiking neural network simulator. It models **multiple layers** of the cortex with  populations of **excitatory** and **inhibitory neurons**, including **PV**, **SOM**, and **VIP** subtypes.


---

## Structure

- `main.py`  
  Main script to launch the simulation.

- `config/`
  - `config.py`  
    Configuration file with simulation params, neuron/synapse constants, per-layer settings, inter-layer connectivity, initial conditions, and inputs.
    - `config_veit.py`  
    Configuration file reproducing Veit et al 2022.



- `src/`
  - `column.py`  
    Full cortical column structure, with multiple layers.
  - `layer.py`  
    Code for an individual cortical layer


## Getting Started

To run the simulation locally:

### 1. Clone the repository

```bash
git clone https://github.com/mathilde-sbri/Cortical-Column.git
cd Cortical-Column
```

### 2. Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies


```bash
pip install -r requirements.txt
```

### 4. Run the simulation


```bash
python main.py
```
