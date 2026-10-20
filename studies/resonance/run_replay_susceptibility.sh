#!/bin/bash
# Cosyne abstract Fig. 2E (theory) and Appendix B of the first-year summary: measured population
# susceptibilities of isolated L6 by spike replay (studies/resonance/replay_susceptibility.py).
# Five networks; each runs closed, then replayed 15 times (exact check, 3 probe types x +/-, and the same
# with adaptation frozen), 2 s settle + 100 s each, probe 5 pA rms. About 40 min per network (seed 1 took 37 min).
#   bash studies/resonance/run_replay_susceptibility.sh
cd "$(dirname "$0")/../.."
PY=~/Desktop/venv/bin/python
OUT=saved_trials/cosyne/replay_susceptibility
mkdir -p $OUT/logs
for s in 1 2 3 4 5; do
  out=$OUT/L6_s$s.npz
  [ -f $out ] || $PY studies/resonance/replay_susceptibility.py L6 $out --seed $s --T 100 --amp 5 > $OUT/logs/L6_s$s.log 2>&1
  echo "seed $s done"
done
