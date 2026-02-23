#!/bin/bash
# Quick Ablation Study Runner
# Usage: ./run_ablation.sh [study_type] [optional: config_path]

set -e

STUDY=${1:-loss}
CONFIG=${2:-config.yml}
OUTPUT_DIR="ablation_results"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Ablation Study Runner for NeuroGS                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Study Type: $STUDY"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "❌ Error: Config file not found: $CONFIG"
    exit 1
fi

# Check if Python script exists
if [ ! -f "ablation_studies.py" ]; then
    echo "❌ Error: ablation_studies.py not found"
    echo "   Please run from the hisnegs/src/ directory"
    exit 1
fi

# Run ablation study
echo "▶ Running ablation study..."
echo ""

python3 ablation_studies.py \
    --study "$STUDY" \
    --config "$CONFIG" \
    --output "$OUTPUT_DIR"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Ablation study completed successfully!"
    echo ""
    
    # Find latest results directory
    LATEST_DIR=$(ls -td "$OUTPUT_DIR"/*_"$STUDY"* 2>/dev/null | head -1)
    
    if [ -n "$LATEST_DIR" ]; then
        echo "Results saved to: $LATEST_DIR"
        echo ""
        echo "▶ Running analysis..."
        echo ""
        
        python3 analyze_ablation.py "$LATEST_DIR"
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Analysis complete!"
            echo ""
            echo "View results:"
            echo "  📊 cat $LATEST_DIR/comparison_table.md"
            echo "  📈 open $LATEST_DIR/learning_curves.png"
            echo "  📉 open $LATEST_DIR/psnr_comparison.png"
        else
            echo "⚠ Analysis failed, but training results are saved"
        fi
    fi
else
    echo ""
    echo "❌ Ablation study failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "Done!"
