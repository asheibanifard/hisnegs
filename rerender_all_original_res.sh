#!/bin/bash
# Re-render all ablation volumes at original GT resolution (100x647x813)
set -e

BASE=/workspace/hisnegs/src/ablation_results/loss_components
SCRIPT=/workspace/hisnegs/render_volume_from_checkpoint.py
RES=813  # longest axis of GT (100,647,813)
AR="100,647,813"

ABLATIONS="baseline no_grad_loss no_tube_reg no_cross_reg no_scale_reg no_regularizers reconstruction_only"

for abl in $ABLATIONS; do
    echo "============================================================"
    echo "Rendering: $abl  (resolution=${RES}, aspect=${AR})"
    echo "============================================================"
    python "$SCRIPT" \
        --checkpoint "$BASE/$abl/checkpoints/model_step20000.pt" \
        --output "$BASE/$abl/rendered_volume.pt" \
        --resolution "$RES" \
        --aspect-ratio "$AR" \
        --device cuda
    echo ""
done

echo "✓ All 7 ablation volumes re-rendered at original resolution"
