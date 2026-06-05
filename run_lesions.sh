#!/usr/bin/env bash
# Run trials.py once per inter-layer lesion + one no-lesion control.
# Each run writes to results/lesions_<date>/<src>_<tgt>/ with a run.log.
# Pairs absent from connection_probabilities.csv will fail fast (config.py
# raises KeyError); that lesion is logged and the loop continues.

set -u

PYTHON="$HOME/Desktop/venv/bin/python"
PROJECT_DIR="$HOME/Desktop/Cortical-Column"
DATE="$(date +%Y-%m-%d)"
PARENT="$PROJECT_DIR/results/lesions_$DATE"
N_TRIALS=20

LAYERS=(L23 L4AB L4C L5 L6)

mkdir -p "$PARENT"
SUMMARY="$PARENT/_summary.log"
: > "$SUMMARY"

run_one() {
    local label="$1"      # subdir name, e.g. "L6_L23" or "control"
    local lesion_env="$2" # value for LESION_PAIR
    local out_dir="$PARENT/$label"
    mkdir -p "$out_dir"
    local log="$out_dir/run.log"

    echo "[$(date '+%H:%M:%S')] START $label  (LESION_PAIR=$lesion_env)" | tee -a "$SUMMARY"

    cd "$PROJECT_DIR"
    LESION_PAIR="$lesion_env" "$PYTHON" trials.py \
        --save-dir "$out_dir" \
        --n-trials "$N_TRIALS" \
        --quiet \
        > "$log" 2>&1
    local rc=$?

    if [ $rc -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] OK    $label" | tee -a "$SUMMARY"
    else
        echo "[$(date '+%H:%M:%S')] FAIL  $label (exit $rc) -- see $log" | tee -a "$SUMMARY"
    fi
}

# No-lesion control
run_one "control" "NONE"

# All ordered (src, tgt) layer pairs
for src in "${LAYERS[@]}"; do
    for tgt in "${LAYERS[@]}"; do
        if [ "$src" = "$tgt" ]; then
            continue
        fi
        run_one "${src}_${tgt}" "${src},${tgt}"
    done
done

echo "[$(date '+%H:%M:%S')] ALL DONE -- summary at $SUMMARY" | tee -a "$SUMMARY"
