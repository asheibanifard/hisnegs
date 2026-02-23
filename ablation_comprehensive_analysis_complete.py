# # Comprehensive Ablation Study Analysis
# # Analysis of Loss Component Contributions
# 
# This notebook performs all critical analyses outlined in `analysis_gaps.md`:
# 
# 1. **Quantitative Metrics** - PSNR, MSE, MAE, SSIM, LPIPS across all ablations
# 2. **Structural Quality** - Tubular constraint violations, smoothness metrics
# 3. **Training Dynamics** - Convergence comparison across ablation variants
# 4. **Visual Comparisons** - Qualitative rendering differences
# 5. **Downstream Tasks** - Morphological feature extraction
# 6. **Statistical Significance** - Hypothesis testing and confidence intervals
# 
# **Goal**: Demonstrate that regularizers improve structural quality without sacrificing reconstruction accuracy.

# ## Setup and Configuration

import os
import sys
import re
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, '/workspace/hisnegs/src')

# Configuration
BASE_DIR = Path('/workspace/hisnegs')
RESULTS_DIR = BASE_DIR / 'src' / 'ablation_results' / 'loss_components'
OUTPUT_DIR = BASE_DIR / 'analysis_output'
OUTPUT_DIR.mkdir(exist_ok=True)

# Ablation variants
ABLATIONS = [
    'baseline',
    'no_grad_loss',
    'no_tube_reg',
    'no_cross_reg',
    'no_scale_reg',
    'no_regularizers',
    'reconstruction_only'
]

# Display settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 4)

print(f"✓ Setup complete")
print(f"  Results directory: {RESULTS_DIR}")
print(f"  Output directory: {OUTPUT_DIR}")
print(f"  Ablations: {len(ABLATIONS)}")

# ## 1. Load Training Logs and Extract Metrics
# 
# Parse all training logs to extract PSNR, MSE, MAE, and regularizer values at key checkpoints.

def parse_training_log(log_path: Path) -> pd.DataFrame:
    """Extract metrics from training log file."""
    data = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Parse PSNR evaluations
            if 'PSNR@' in line:
                match = re.search(r'PSNR@(\d+):\s+([\d.]+)\s+dB.*MSE=([\d.]+).*MAE=([\d.]+)', line)
                if match:
                    step = int(match.group(1))
                    psnr = float(match.group(2))
                    mse = float(match.group(3))
                    mae = float(match.group(4))
                    data.append({
                        'step': step,
                        'psnr': psnr,
                        'mse': mse,
                        'mae': mae
                    })
            
            # Parse final step regularizer values
            if 'Step 20000:' in line and "'v_grad'" in line:
                # Extract regularizer values
                v_rec = re.search(r"'v_rec':\s+([\d.e-]+)", line)
                v_grad = re.search(r"'v_grad':\s+([\d.e-]+)", line)
                v_tube = re.search(r"'v_tube':\s+([\d.e-]+)", line)
                v_csym = re.search(r"'v_csym':\s+([\d.e-]+)", line)
                v_scale = re.search(r"'v_scale':\s+([\d.e-]+)", line)
                
                if data:  # Add to last entry
                    data[-1].update({
                        'v_rec': float(v_rec.group(1)) if v_rec else 0,
                        'v_grad': float(v_grad.group(1)) if v_grad else 0,
                        'v_tube': float(v_tube.group(1)) if v_tube else 0,
                        'v_csym': float(v_csym.group(1)) if v_csym else 0,
                        'v_scale': float(v_scale.group(1)) if v_scale else 0
                    })
    
    return pd.DataFrame(data)


# Load logs for all ablations
all_metrics = {}

for ablation in tqdm(ABLATIONS, desc="Loading logs"):
    log_dir = RESULTS_DIR / ablation / 'logs'
    
    # Find the latest log file
    log_files = sorted(log_dir.glob('training_*.log'))
    if log_files:
        latest_log = log_files[-1]
        df = parse_training_log(latest_log)
        all_metrics[ablation] = df
        print(f"  {ablation}: {len(df)} checkpoints")
    else:
        print(f"  ⚠️  {ablation}: No log files found")

print(f"\n✓ Loaded metrics for {len(all_metrics)} ablations")

# NOTE: Continue with remaining analysis sections...
# This is a template for the complete notebook structure
