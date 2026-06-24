"""Run one trials.py batch per inter-layer lesion plus a control, with matched
per-trial seeds (LESION_PAIR env var drops the connection in config.py).

Run:
    path/to/your/venv/bin/python experiments/run_lesion_experiment.py
Then plot with analysis/lesions/plot_lesion_results.py.
"""
import os
import subprocess
import sys
from datetime import date

# parameters
PYTHON = os.path.expanduser("~/Desktop/venv/bin/python")
N_TRIALS = 20
LAYERS = ["L23", "L4AB", "L4C", "L5", "L6"]

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_ROOT = os.path.join(PROJECT_DIR, "saved_trials", f"lesions_{date.today():%Y-%m-%d}")


def run_one(label, lesion_env):
    """Run one trials.py batch with LESION_PAIR set. Returns True on success."""
    out_dir = os.path.join(OUT_ROOT, label)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")
    print(f"START {label}  (LESION_PAIR={lesion_env})")

    env = dict(os.environ, LESION_PAIR=lesion_env)
    cmd = [PYTHON, "trials.py",
           "--save-dir", out_dir,
           "--n-trials", str(N_TRIALS),
           "--quiet"]
    with open(log_path, "w") as log:
        rc = subprocess.call(cmd, cwd=PROJECT_DIR, env=env,
                             stdout=log, stderr=subprocess.STDOUT)
    if rc == 0:
        print(f"OK    {label}")
        return True
    # absent pairs make config.py raise; we just skip them
    print(f"FAIL  {label} (exit {rc}) -- see {log_path}")
    return False


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f"Writing trials to {OUT_ROOT}")

    run_one("control", "NONE")
    for src in LAYERS:
        for tgt in LAYERS:
            if src == tgt:
                continue
            run_one(f"{src}_{tgt}", f"{src},{tgt}")

    print(f"ALL DONE -- results under {OUT_ROOT}")


if __name__ == "__main__":
    main()
