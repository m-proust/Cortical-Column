#!/usr/bin/env bash
# Run trials.py once per top-ranked seed from a trials3.py run. Each batch
# fixes the network seed; trials.py reseeds the Poisson inputs per trial.
#
# Usage:
#   ./run_seed_batches.sh                     # defaults: top 5 from trials_19_05_2
#   ./run_seed_batches.sh results/trials_19_05_2 5 10
#                                             # base_path  top_k  n_trials

set -euo pipefail

PYTHON="/Users/mathildeproust/Desktop/venv/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_PATH="${1:-results/trials_19_05_2}"
TOP_K="${2:-5}"
N_TRIALS="${3:-10}"
BASELINE_MS=2000
STIMULI_MS=2000

echo "Ranking trials in $BASE_PATH and selecting top $TOP_K seeds..."
SEEDS=()
while IFS= read -r line; do
  [ -n "$line" ] && SEEDS+=("$line")
done < <(
  "$PYTHON" rank_trials_19_05.py \
    --base-path "$BASE_PATH" \
    --top-k "$TOP_K" \
    --no-plot \
  | awk '/network_seed=/ {
      for (i=1; i<=NF; i++) {
        if ($i ~ /^network_seed=/) {
          split($i, a, "=");
          print a[2];
        }
      }
    }'
)

if [ "${#SEEDS[@]}" -eq 0 ]; then
  echo "No seeds parsed from rank_trials_19_05.py output. Aborting." >&2
  exit 1
fi

echo "Will run ${#SEEDS[@]} batches with seeds: ${SEEDS[*]}"

TS="$(date +%Y%m%d_%H%M%S)"

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  save_dir="results/trials_fixed_rank$(printf '%02d' "$i")_seed${seed}_${TS}"

  echo "=========================================================="
  echo "Batch $((i+1))/${#SEEDS[@]} | rank $i | network_seed=${seed}"
  echo "Saving to ${save_dir}"
  echo "=========================================================="

  "$PYTHON" trials.py \
    --network-seed "$seed" \
    --n-trials "$N_TRIALS" \
    --baseline-ms "$BASELINE_MS" \
    --stimuli-ms "$STIMULI_MS" \
    --save-dir "$save_dir"
done

echo "All ${#SEEDS[@]} batches finished. Results under results/trials_fixed_rank*_${TS}/"
