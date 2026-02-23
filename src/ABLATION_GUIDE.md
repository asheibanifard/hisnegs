# Ablation Studies Guide

## Overview

This directory contains comprehensive ablation study tools for the Gaussian Mixture Field model. The ablation studies systematically evaluate the contribution of each component in the model and training pipeline.

## Quick Start

### 1. Run All Ablation Studies

```bash
cd /workspace/hisnegs/src
python3 ablation_studies.py --study all --config config.yml
```

### 2. Run Specific Ablation Study

```bash
# Test loss components
python3 ablation_studies.py --study loss

# Test densification strategies
python3 ablation_studies.py --study densify

# Test sampling strategies
python3 ablation_studies.py --study sampling

# Test initialization methods
python3 ablation_studies.py --study init

# Test hyperparameters
python3 ablation_studies.py --study hyperparams
```

### 3. Analyze Results

After running ablation studies:

```bash
python3 analyze_ablation.py ablation_results/TIMESTAMP_STUDYNAME/
```

This will generate:
- `comparison_table.md` - Markdown table comparing all experiments
- `learning_curves.png` - PSNR and loss curves for all experiments
- `psnr_comparison.png` - Bar chart of best PSNR values
- `detailed_analysis.json` - Detailed metrics in JSON format

## Ablation Study Categories

### 1. Loss Components (`--study loss`)

Tests the contribution of each loss component:

- **baseline** - Full model with all losses
- **no_grad_loss** - Remove gradient supervision
- **no_tube_reg** - Remove tube regularization
- **no_cross_reg** - Remove cross regularization
- **no_scale_reg** - Remove scale regularization
- **no_regularizers** - Remove all regularizers
- **reconstruction_only** - Only reconstruction loss

**Expected Insights:**
- Importance of gradient supervision for quality
- Effect of regularizers on model compactness
- Trade-offs between reconstruction and regularization

### 2. Densification (`--study densify`)

Evaluates adaptive Gaussian management:

- **no_densify** - Fixed number of Gaussians
- **with_densify** - Adaptive densification
- **densify_early_stop** - Stop densification early
- **densify_aggressive** - More aggressive densification

**Expected Insights:**
- Importance of adaptive capacity
- Optimal densification schedule
- Trade-off between model size and quality

### 3. Sampling Strategy (`--study sampling`)

Tests different point sampling methods:

- **uniform_sampling** - Uniform spatial sampling
- **intensity_weighted** - Sample based on intensity
- **more_volume_samples** - 2×more samples per step
- **fewer_volume_samples** - 0.5× samples per step

**Expected Insights:**
- Effect of sampling bias on training
- Sample efficiency
- Computational trade-offs

### 4. Initialization (`--study init`)

Compares initialization strategies:

- **init_swc** - Initialize from neuron morphology (SWC file)
- **init_random** - Random initialization
- **init_scale_small** - Smaller initial Gaussian size
- **init_scale_large** - Larger initial Gaussian size
- **fewer_gaussians** - 5K Gaussians
- **more_gaussians** - 15K Gaussians

**Expected Insights:**
- Value of structure-aware initialization
- Optimal initial scale
- Effect of initial model capacity

### 5. Hyperparameters (`--study hyperparams`)

Tests key training hyperparameters:

- **lr_low** - Lower learning rate (1e-3)
- **lr_high** - Higher learning rate (5e-3)
- **no_early_stopping** - Train full duration
- **no_mixed_precision** - FP32 training
- **no_grad_clip** - No gradient clipping

**Expected Insights:**
- Learning rate sensitivity
- Benefit of training optimizations
- Stability vs. speed trade-offs

## Custom Ablations

You can easily create custom ablation studies by adding new functions to `ablation_studies.py`:

```python
def ablation_custom(base_config: dict) -> list:
    """Your custom ablation study."""
    ablations = []
    
    ablations.append(('experiment_name', {
        'training.parameter': value,
        'model.parameter': value,
    }))
    
    return ablations
```

Then register it in the studies dictionary:

```python
studies = {
    'custom': ('custom_study', ablation_custom),
    # ... other studies
}
```

## Output Structure

```
ablation_results/
├── TIMESTAMP_loss_components/
│   ├── configs/
│   │   ├── baseline.yml
│   │   ├── no_grad_loss.yml
│   │   └── ...
│   ├── results.json
│   ├── SUMMARY.md
│   ├── comparison_table.md
│   ├── learning_curves.png
│   └── psnr_comparison.png
├── TIMESTAMP_training_modes/
│   └── ...
└── ...
```

## Interpreting Results

### PSNR (Peak Signal-to-Noise Ratio)
- **Higher is better**
- Typical range: 25-40 dB for this task
- >35 dB indicates high-quality reconstruction

### Loss Values
- **Lower is better**
- Track convergence and stability
- Compare final and best values

### Number of Gaussians
- Shows model complexity
- Balance between quality and efficiency
- With densification: typically grows during training

### Training Time
- ms/step indicates computational cost
- Compare efficiency of different methods

## Tips for Running Ablations

1. **Start Small**: Run a single study first to validate the pipeline
2. **Use GPU**: Set `CUDA_VISIBLE_DEVICES` for multi-GPU systems
3. **Monitor Progress**: Check logs in real-time:
   ```bash
   tail -f logs/ablation_EXPERIMENT/training.log
   ```
4. **Adjust Steps**: For faster iteration, reduce training steps in config
5. **Parallel Runs**: Run independent studies on different GPUs

## Example Workflow

```bash
# 1. Run loss component ablations
python3 ablation_studies.py --study loss --output results/loss_study

# 2. Monitor progress
watch -n 5 "ls -lh results/loss_study/configs/"

# 3. After completion, analyze results
python3 analyze_ablation.py results/loss_study/TIMESTAMP_loss_components/

# 4. View results
cat results/loss_study/TIMESTAMP_loss_components/comparison_table.md
```

## Troubleshooting

### Out of Memory
- Reduce `vol_points_per_step` or `mip_pixels_per_step`
- Disable mixed precision temporarily
- Use smaller `max_gaussians`

### Training Instability
- Lower learning rate
- Enable gradient clipping
- Check initialization (output should be non-zero)

### No Convergence
- Check that losses are being applied
- Verify data is loaded correctly
- Ensure Gaussians have reasonable scale

## Advanced: Batch Processing

For running multiple ablation studies in sequence:

```bash
#!/bin/bash
# run_all_ablations.sh

STUDIES=("loss" "densify" "sampling" "init" "hyperparams")

for study in "${STUDIES[@]}"; do
    echo "Running $study ablation..."
    python3 ablation_studies.py --study $study --output results/batch_$study
    python3 analyze_ablation.py results/batch_$study/*/
done

echo "All ablations complete!"
```

## Citation

If you use these ablation studies in your research, please cite the original NeuroGS paper and note the specific components tested.
