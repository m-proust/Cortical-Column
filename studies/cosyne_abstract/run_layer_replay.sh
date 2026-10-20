#!/bin/bash
# Cosyne abstract Fig. 3A: isolated L4C driven by L6's recorded resting spikes, and isolated L2/3 driven
# by L4C's, against rate-matched Poisson trains with the same per-cell rates (timing removed).
# Sources: the 20 resting recordings of record_rest.py (segment k = recording rest_s<101+k>).
# The target's background Poisson N is matched once with the Poisson source and stored in
# minimal_config/experiments/tuned_N.json as 'replay:<src>-><cond>:rest'; both arms use it.
# 20 seeds (58925, 1..19). Resumable; pass --rematch to redo the rate matching.
#   bash studies/cosyne_abstract/run_layer_replay.sh [--rematch]
cd "$(dirname "$0")/../.."
export REPLAY_SOURCES=rest
PY=~/Desktop/venv/bin/python
D=saved_trials/cosyne/layer_replay
mkdir -p $D/logs
SEEDS=(58925 $(seq 1 19))
one_pair() {
  tgt=$1; src=$2; tag="${src}_to_${tgt}"
  if [ "$REMATCH" = 1 ] || ! grep -q "replay:$src->$tgt:rest" minimal_config/experiments/tuned_N.json; then
    $PY studies/cosyne_abstract/run_replay.py "$tgt" "$src" match >> "$D/logs/$tag.log" 2>&1 || { echo "$tag UNMATCHED"; return; }
  fi
  for k in $(seq 0 19); do
    for mode in replay poisson; do
      out="$D/${tag}_${mode}_s${SEEDS[$k]}.npz"
      [ -f "$out" ] || $PY studies/cosyne_abstract/run_replay.py "$tgt" "$src" $mode "$out" --seed ${SEEDS[$k]} --segment $k >> "$D/logs/$tag.log" 2>&1
    done
  done
  echo "$tag done"
}
[ "$1" = --rematch ] && REMATCH=1
one_pair 'L4C@own' L6 &
one_pair L23 L4C &
wait
