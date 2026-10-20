#!/bin/bash
# Cosyne abstract Fig. 2B: isolated L6 reduced to E+SOM, E+SOM+PV and E+PV.
# Each reduced layer is rate-matched once (minimal_config/match_rates.py, stored in
# minimal_config/experiments/tuned_N.json), then minimal_config/run_spectrum.py runs one 10.5 s
# network per seed. Resumable: existing outputs are skipped.
#   bash studies/cosyne_abstract/run_reduced_L6.sh
cd "$(dirname "$0")/../.."
PY=~/Desktop/venv/bin/python
OUT=saved_trials/cosyne/reduced_L6
mkdir -p $OUT/logs
for c in 'L6_E+SOM' 'L6_E+SOM+PV' 'L6_E+PV'; do
  grep -q "\"$c\"" minimal_config/experiments/tuned_N.json || $PY minimal_config/match_rates.py "$c" >> "$OUT/logs/$c.log" 2>&1
  for s in 58925 1 2; do
    out="$OUT/${c}_s$s.npz"
    [ -f "$out" ] || $PY minimal_config/run_spectrum.py "$c" "$out" --seed $s >> "$OUT/logs/$c.log" 2>&1
  done
  echo "$c done"
done
