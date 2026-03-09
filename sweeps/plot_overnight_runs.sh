
PLOTS_DIR="plots_20_02"
mkdir -p "$PLOTS_DIR"

INPUT_DIR="results/input_sweeps/19_02_all_layers_AMPA"


EXCLUDE_LAYERS=""

echo "Plotting overnight sweep results..."
echo "Output directory: $PLOTS_DIR"
if [ -n "$EXCLUDE_LAYERS" ]; then
    echo "Excluding layers: $EXCLUDE_LAYERS"
fi
echo ""

EXCLUDE_FLAG=""
if [ -n "$EXCLUDE_LAYERS" ]; then
    EXCLUDE_FLAG="--exclude $EXCLUDE_LAYERS"
fi

for npz_file in "$INPUT_DIR"/*.npz; do
    filename=$(basename "$npz_file" .npz)

    echo "Plotting: $filename"


    python plot_input_sweep.py "$npz_file" --diff $EXCLUDE_FLAG --save "$PLOTS_DIR/${filename}.png"
done

echo ""
echo "Done! Plots saved to: $PLOTS_DIR/"
