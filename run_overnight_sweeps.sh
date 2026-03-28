#!/bin/bash



set -e

LOG_DIR="results/sweep_logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_LOG="$LOG_DIR/batch_run_${TIMESTAMP}.log"

echo "======================================" | tee -a "$BATCH_LOG"
echo "Starting batch run at $(date)" | tee -a "$BATCH_LOG"
echo "======================================" | tee -a "$BATCH_LOG"
echo "" | tee -a "$BATCH_LOG"

run_sweep() {
    local layer=$1
    local pop=$2
    local input_type=$3
    local weight=$4
    
    echo "--------------------------------------" | tee -a "$BATCH_LOG"
    echo "Running: Layer=$layer, Pop=$pop, Input=$input_type, Weight=$weight" | tee -a "$BATCH_LOG"
    echo "Started at: $(date)" | tee -a "$BATCH_LOG"
    
    local run_log="$LOG_DIR/${layer}_${pop}_${input_type}_w${weight}_${TIMESTAMP}.log"
    

    if python input_sweep.py \
        --layer "$layer" \
        --pop "$pop" \
        --input-type "$input_type" \
        --weight-scale "$weight" \
        --rate-min 0 \
        --rate-max 15 \
        --rate-step 1 \
        --baseline-ms 1000 \
        --stim-ms 1500 \
        --save-dir "results/input_sweeps2" \
        2>&1 | tee "$run_log"; then
        
        echo "✓ Completed successfully at: $(date)" | tee -a "$BATCH_LOG"
    else
        echo "✗ FAILED at: $(date)" | tee -a "$BATCH_LOG"
        echo "  Check log: $run_log" | tee -a "$BATCH_LOG"
    fi
    
    echo "" | tee -a "$BATCH_LOG"
}



run_sweep "L4C"   "VIP" "AMPA" 1.0
run_sweep "L4C"   "E" "AMPA" 1.0
run_sweep "L4C"   "PV" "AMPA" 1.0
run_sweep "L4C"   "SOM" "AMPA" 1.0


run_sweep "L4AB"   "VIP" "AMPA" 1.0
run_sweep "L4AB"   "E" "AMPA" 1.0
run_sweep "L4AB"   "PV" "AMPA" 1.0
run_sweep "L4AB"   "SOM" "AMPA" 1.0


run_sweep "L23"   "VIP" "AMPA" 1.0
run_sweep "L23"   "E" "AMPA" 1.0
run_sweep "L23"   "PV" "AMPA" 1.0
run_sweep "L23"   "SOM" "AMPA" 1.0


run_sweep "L5"   "VIP" "AMPA" 1.0
run_sweep "L5"   "E" "AMPA" 1.0
run_sweep "L5"   "PV" "AMPA" 1.0
run_sweep "L5"   "SOM" "AMPA" 1.0


run_sweep "L6"   "VIP" "AMPA" 1.0
run_sweep "L6"   "E" "AMPA" 1.0
run_sweep "L6"   "PV" "AMPA" 1.0
run_sweep "L6"   "SOM" "AMPA" 1.0




echo "======================================" | tee -a "$BATCH_LOG"
echo "Batch run completed at $(date)" | tee -a "$BATCH_LOG"
echo "======================================" | tee -a "$BATCH_LOG"
echo "" | tee -a "$BATCH_LOG"
echo "Summary log saved to: $BATCH_LOG"
echo ""
echo "Individual run logs available in: $LOG_DIR"
echo ""
