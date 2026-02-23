#!/bin/bash
#
# Render 3D volumes for all ablation experiments
# This generates the volumes needed for SSIM/LPIPS computation
#

set -e  # Exit on error

RESOLUTION=128  # Adjust if needed (higher = better quality but slower)
BASE_DIR="/workspace/hisnegs/src/ablation_results/loss_components"

# List of ablations
ABLATIONS=(
    "baseline"
    "no_grad_loss"
    "no_tube_reg"
    "no_cross_reg"
    "no_scale_reg"
    "no_regularizers"
    "reconstruction_only"
)

echo "========================================="
echo "Rendering Volumes for Ablation Study"
echo "========================================="
echo "Resolution: ${RESOLUTION}³"
echo "Ablations: ${#ABLATIONS[@]}"
echo "========================================="
echo ""

# Render each ablation
for ABLATION in "${ABLATIONS[@]}"; do
    CHECKPOINT="${BASE_DIR}/${ABLATION}/checkpoints/model_step20000.pt"
    OUTPUT="${BASE_DIR}/${ABLATION}/rendered_volume.pt"
    
    echo ">>> Rendering: ${ABLATION}"
    echo "    Checkpoint: ${CHECKPOINT}"
    echo "    Output: ${OUTPUT}"
    
    if [ ! -f "${CHECKPOINT}" ]; then
        echo "    ⚠ ERROR: Checkpoint not found, skipping..."
        continue
    fi
    
    if [ -f "${OUTPUT}" ]; then
        echo "    ℹ Volume already exists, skipping... (delete to re-render)"
        continue
    fi
    
    # Run rendering
    python render_volume_from_checkpoint.py \
        --checkpoint "${CHECKPOINT}" \
        --output "${OUTPUT}" \
        --resolution ${RESOLUTION}
    
    echo "    ✓ Done"
    echo ""
done

echo "========================================="
echo "✓ All volumes rendered!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Open ablation_comprehensive_analysis.ipynb"
echo "  2. Re-run Cell 4 to compute SSIM/LPIPS from volumes"
echo "  3. Re-run Cell 5 to see populated metrics in summary table"
echo ""
