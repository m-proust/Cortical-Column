# configL6 snapshot (archived 20 Oct 2026)

Copy of `config/` taken on 2 Oct 2026, previously the folder `config copy/`. Not used by the Cosyne abstract.

- `config.py` and the three main CSVs match `config/` as of 20 Oct 2026 (one blank line differs in `config.py`).
- Only copy of the configL6 model: `configL6.py`, which loads `conductances_AMPA_GABA copy.csv` and
  `connection_probabilities copy.csv`; plus `configL6_L6SOM.py`, `configL6_disconnected.py`, `config_farzin.py`.
- `AB/`, `A_restore_L4C_PV_to_L23/`, `L6SOM_x3_localII/`, `O3_plus_L6toL4C_halved/`: CSV variants from the
  configL6 tuning (Sep 30 - Oct 1).

To run configL6 again, copy `configL6.py` and the two `* copy.csv` files back into `config/`.
The `L6new` conditions in `minimal_config/experiments/conditions.py` also need those two CSVs in `config/`.
