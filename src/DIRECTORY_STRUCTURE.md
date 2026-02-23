# Ablation Study Directory Structure

## Overview

Each ablation experiment now has its own organized directory structure with separate folders for configurations, checkpoints, and logs.

## Directory Layout

```
ablation_results/
├── TIMESTAMP_loss_components/           # Central coordination directory
│   ├── configs/                         # Generated config files
│   │   ├── baseline.yml
│   │   ├── no_grad_loss.yml
│   │   └── ...
│   ├── results.json                     # Aggregated results
│   ├── SUMMARY.md                       # Auto-generated summary
│   ├── comparison_table.md              # Analysis results
│   ├── learning_curves.png              # Plots
│   └── psnr_comparison.png
│
├── loss_components/                     # Study-specific results
│   ├── baseline/                        # Individual experiment
│   │   ├── checkpoints/
│   │   │   ├── model.pt                # Final checkpoint
│   │   │   ├── model_best.pt           # Best checkpoint (early stopping)
│   │   │   ├── model_step2000.pt       # Intermediate checkpoints
│   │   │   ├── model_step4000.pt
│   │   │   └── ...
│   │   └── logs/
│   │       └── training_TIMESTAMP.log
│   │
│   ├── no_grad_loss/
│   │   ├── checkpoints/
│   │   │   ├── model.pt
│   │   │   ├── model_best.pt
│   │   │   └── ...
│   │   └── logs/
│   │       └── training_TIMESTAMP.log
│   │
│   └── ...                              # Other experiments
│
├── training_modes/                      # Another study
│   ├── mode_volume/
│   │   ├── checkpoints/
│   │   └── logs/
│   ├── mode_mip/
│   │   ├── checkpoints/
│   │   └── logs/
│   └── ...
│
└── ...                                  # Other studies

# For parallel runs (different configs)
hisnegs/src/
├── config.yml                           # Base configuration
├── ablation_studies.py                  # Framework
├── analyze_ablation.py                  # Analysis tool
└── run_ablation.sh                      # Quick runner
```

## File Organization

### Per Experiment Files

Each experiment (`baseline`, `no_grad_loss`, etc.) has:

1. **Checkpoints Directory** (`ablation_results/STUDY/EXPERIMENT/checkpoints/`)
   - `model.pt` - Final model state
   - `model_best.pt` - Best model (if early stopping is enabled)
   - `model_stepXXXX.pt` - Intermediate checkpoints

2. **Logs Directory** (`ablation_results/STUDY/EXPERIMENT/logs/`)
   - `training_TIMESTAMP.log` - Complete training log
   - Contains: PSNR evaluations, loss values, timing info, etc.

3. **Configuration** (in central `TIMESTAMP_STUDY/configs/`)
   - `EXPERIMENT.yml` - Configuration used for this run
   - Includes all hyperparameters and modified settings

### Shared Study Files

Each study has a central coordination directory with:

- **configs/** - All experiment configurations
- **results.json** - Raw results from all experiments
- **SUMMARY.md** - Auto-generated summary
- **comparison_table.md** - Detailed comparison (after analysis)
- **learning_curves.png** - PSNR/loss plots (after analysis)
- **psnr_comparison.png** - Bar chart (after analysis)

## Benefits of This Structure

✅ **Isolated**: Each experiment has completely separate files
✅ **Organized**: Clear hierarchy by study type
✅ **Safe**: No risk of overwriting between experiments
✅ **Traceable**: Easy to find specific experiment results
✅ **Resumable**: Can resume individual experiments
✅ **Comparable**: All configs in one place for comparison

## Finding Results

### For a Specific Experiment

```bash
# Checkpoints
ls ablation_results/loss_components/baseline/checkpoints/

# Logs
cat ablation_results/loss_components/baseline/logs/training_*.log

# Configuration
cat ablation_results/TIMESTAMP_loss_components/configs/baseline.yml
```

### For All Experiments in a Study

```bash
# List all experiment directories
ls -d ablation_results/loss_components/*/

# Compare all checkpoints
du -sh ablation_results/loss_components/*/checkpoints/

# View summary
cat ablation_results/TIMESTAMP_loss_components/comparison_table.md
```

## Checkpoint Management

### Checkpoint Interval

From `config.yml`:
```yaml
training:
  checkpoint_interval: 2000  # Save every 2000 steps
```

Creates: `model_step2000.pt`, `model_step4000.pt`, etc.

### Early Stopping Checkpoints

If `early_stopping: true`:
```yaml
training:
  early_stopping: true
  early_stopping_patience: 20
```

Creates: `model_best.pt` - saved when PSNR improves

### Final Checkpoint

Always saved at the end: `model.pt`

## Loading Checkpoints

```python
import torch
from model import GaussianMixtureField

# Load a specific experiment's checkpoint
checkpoint_path = "ablation_results/loss_components/baseline/checkpoints/model_best.pt"
state_dict = torch.load(checkpoint_path)

# Create model and load
field = GaussianMixtureField(num_gaussians=10000)
field.load_state_dict(state_dict)
```

## Disk Usage Estimation

Per experiment (with 100K training steps):
- **Checkpoints**: ~500MB (depends on checkpoint_interval)
  - Each checkpoint: ~10-20MB
  - 50 intervals × 15MB ≈ 750MB worst case
- **Logs**: ~5-10MB (text logs)
- **Total**: ~500-800MB per experiment

For 30 experiments:
- **Total**: ~15-25GB

## Cleanup

### Remove Intermediate Checkpoints

Keep only best and final:
```bash
find ablation_results -name "model_step*.pt" -delete
```

### Remove a Specific Experiment

```bash
rm -rf ablation_results/loss_components/no_grad_loss/
```

### Remove Entire Study

```bash
rm -rf ablation_results/loss_components/
rm -rf ablation_results/TIMESTAMP_loss_components/
```

## Configuration Example

When you run an ablation, the generated config contains:

```yaml
training:
  save_path: "ablation_results/loss_components/baseline/checkpoints/model.pt"
  log_dir: "ablation_results/loss_components/baseline/logs"
  checkpoint_interval: 2000
  # ... other settings
```

This ensures all files go to the right place automatically.

## Validation

To verify the directory structure is set up correctly:

```bash
cd /workspace/hisnegs/src
python3 validate_ablation.py
```

This will:
1. Generate all configurations
2. Verify directory paths are correct
3. Show expected checkpoint and log locations
4. Save test configs to `ablation_validation/`

## Notes

- **Timestamped coordination directories**: Each run gets a unique timestamp to prevent mixing results
- **Named experiment directories**: Results are stored by experiment name for easy access
- **Automatic creation**: Directories are created automatically when training starts
- **Parallel-safe**: Different studies can run simultaneously without conflicts
