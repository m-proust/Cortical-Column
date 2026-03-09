
set -e

LOG_DIR="results/sweep_logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_LOG="$LOG_DIR/batch_run_${TIMESTAMP}.log"
SAVE_DIR="results/input_sweeps/19_02_all_layers_AMPA"

echo "======================================" | tee -a "$BATCH_LOG"
echo "Starting batch run at $(date)" | tee -a "$BATCH_LOG"
echo "Save dir: $SAVE_DIR" | tee -a "$BATCH_LOG"
echo "======================================" | tee -a "$BATCH_LOG"
echo "" | tee -a "$BATCH_LOG"


run_sweep() {
    local layer=$1
    local pop=$2
    local input_type=$3
    local weight=$4

    local label="${layer}_${pop}_${input_type}_w${weight}"
    echo "--------------------------------------" | tee -a "$BATCH_LOG"
    echo "Running: $label" | tee -a "$BATCH_LOG"
    echo "Started at: $(date)" | tee -a "$BATCH_LOG"

    local run_log="$LOG_DIR/${label}_${TIMESTAMP}.log"

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
        --save-dir "$SAVE_DIR" \
        2>&1 | tee "$run_log"; then
        echo "✓ Completed: $label at $(date)" | tee -a "$BATCH_LOG"
    else
        echo "✗ FAILED: $label at $(date)" | tee -a "$BATCH_LOG"
        echo "  Log: $run_log" | tee -a "$BATCH_LOG"
    fi
    echo "" | tee -a "$BATCH_LOG"
}


run_multi() {
    local targets_args=""
    local label=""
    for spec in "$@"; do
        targets_args="$targets_args --target $spec"
        local part
        part=$(echo "$spec" | tr ':' '_')
        label="${label}+${part}"
    done
    label="${label:1}"  # strip leading +

    echo "--------------------------------------" | tee -a "$BATCH_LOG"
    echo "Running multi: $label" | tee -a "$BATCH_LOG"
    echo "Started at: $(date)" | tee -a "$BATCH_LOG"

    local run_log="$LOG_DIR/${label}_${TIMESTAMP}.log"

    if python input_sweep.py \
        $targets_args \
        --rate-min 0 \
        --rate-max 15 \
        --rate-step 1 \
        --baseline-ms 1000 \
        --stim-ms 1500 \
        --save-dir "$SAVE_DIR" \
        2>&1 | tee "$run_log"; then
        echo "✓ Completed: $label at $(date)" | tee -a "$BATCH_LOG"
    else
        echo "✗ FAILED: $label at $(date)" | tee -a "$BATCH_LOG"
        echo "  Log: $run_log" | tee -a "$BATCH_LOG"
    fi
    echo "" | tee -a "$BATCH_LOG"
}



for layer in L23 L4AB L4C L5 L6; do
    echo "" | tee -a "$BATCH_LOG"
    echo "### ${layer} single-population AMPA sweeps ###" | tee -a "$BATCH_LOG"
    for pop in PV E SOM VIP; do
        run_sweep "$layer" "$pop" "AMPA" 1.0
    done
done


echo "" | tee -a "$BATCH_LOG"
echo "======================================" | tee -a "$BATCH_LOG"
echo "Batch run completed at $(date)" | tee -a "$BATCH_LOG"
echo "======================================" | tee -a "$BATCH_LOG"
echo "" | tee -a "$BATCH_LOG"
echo "Summary log: $BATCH_LOG"
echo "Results in:  $SAVE_DIR"
echo "Run logs in: $LOG_DIR"
