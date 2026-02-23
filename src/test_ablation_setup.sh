#!/bin/bash
# Quick test of ablation framework
cd /workspace/hisnegs/src

echo "Testing ablation framework..."
echo ""

# Test 1: Check if base config exists
if [ ! -f config.yml ]; then
    echo "❌ Error: config.yml not found"
    exit 1
fi
echo "✓ Base config found: config.yml"

# Test 2: Check Python scripts
for script in ablation_studies.py analyze_ablation.py validate_ablation.py; do
    if [ ! -f "$script" ]; then
        echo "❌ Error: $script not found"
        exit 1
    fi
    echo "✓ Script found: $script"
done

echo ""
echo "All files present! You can now run:"
echo ""
echo "  1. Validate framework:"
echo "     python3 validate_ablation.py"
echo ""
echo "  2. Run specific ablation study:"
echo "     python3 ablation_studies.py --study loss"
echo "     python3 ablation_studies.py --study densify"
echo "     python3 ablation_studies.py --study sampling"
echo "     python3 ablation_studies.py --study init"
echo "     python3 ablation_studies.py --study hyperparams"
echo ""
echo "  3. Run all ablation studies:"
echo "     python3 ablation_studies.py --study all"
echo ""
echo "  4. Or use the convenience script:"
echo "     ./run_ablation.sh loss"
echo ""
