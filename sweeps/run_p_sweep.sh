#!/bin/bash

set -e

LOG_DIR="results/sweep_logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SAVE_DIR="results/p_sweeps"
mkdir -p "$SAVE_DIR"

BATCH_LOG="$LOG_DIR/p_sweep_${TIMESTAMP}.log"
RUN_LOG="$LOG_DIR/p_sweep_run_${TIMESTAMP}.log"

echo "======================================" | tee -a "$BATCH_LOG"
echo "Starting inter-layer p sweep at $(date)" | tee -a "$BATCH_LOG"
echo "p range: 0.0 to 2.0 (40 values)" | tee -a "$BATCH_LOG"
echo "Save dir: $SAVE_DIR" | tee -a "$BATCH_LOG"
echo "======================================" | tee -a "$BATCH_LOG"
echo "" | tee -a "$BATCH_LOG"

if python main_p.py \
    --p-min 0.0 \
    --p-max 2.0 \
    --n-p 40 \
    --sim-ms 2500 \
    --analysis-start-ms 1000 \
    --save-dir "$SAVE_DIR" \
    2>&1 | tee "$RUN_LOG"; then
    echo "✓ Sweep completed at $(date)" | tee -a "$BATCH_LOG"
else
    echo "✗ Sweep FAILED at $(date) — see $RUN_LOG" | tee -a "$BATCH_LOG"
fi

echo ""
echo "Summary log: $BATCH_LOG"
echo "Run log:     $RUN_LOG"
echo "Results in:  $SAVE_DIR"
