#!/usr/bin/env python3
"""
Ablation Studies for Gaussian Mixture Field Model

This script runs systematic ablation studies to evaluate the contribution
of each component in the model and training pipeline.
"""

import os
import json
import yaml
import shutil
import itertools
from pathlib import Path
from datetime import datetime
import subprocess
from tqdm import tqdm


def load_base_config(config_path: str = "config.yml") -> dict:
    """Load the base configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: dict, path: str):
    """Save configuration to a YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def create_ablation_config(base_config: dict, name: str, modifications: dict, study_name: str = "") -> dict:
    """Create an ablation configuration by modifying the base config."""
    import copy
    config = copy.deepcopy(base_config)
    
    # Apply modifications
    for key_path, value in modifications.items():
        keys = key_path.split('.')
        target = config
        for k in keys[:-1]:
            target = target[k]
        target[keys[-1]] = value
    
    # Create organized directory structure
    # Format: ablation_results/STUDY_NAME/EXPERIMENT_NAME/
    if study_name:
        base_dir = f"ablation_results/{study_name}/{name}"
    else:
        base_dir = f"ablation_results/{name}"
    
    # Update save paths and log directories with unique folders per experiment
    config['training']['save_path'] = f"{base_dir}/checkpoints/model.pt"
    config['training']['log_dir'] = f"{base_dir}/logs"
    
    # Ensure checkpoint interval saves are in the same directory
    # (train.py will automatically use the base from save_path)
    
    # Reduce training steps for ablation studies
    config['training']['steps'] = 20000  # Faster ablation runs
    
    return config


def run_training(config_path: str, experiment_name: str) -> dict:
    """
    Run training with the given configuration.
    Returns a dictionary with results.
    """
    # Use tqdm.write for output that doesn't conflict with progress bars
    tqdm.write(f"\n{'='*80}")
    tqdm.write(f"Experiment: {experiment_name}")
    tqdm.write(f"Config: {config_path}")
    tqdm.write(f"{'='*80}")
    
    # Run training using run.py with streaming output
    # This allows us to see progress in real-time
    try:
        process = subprocess.Popen(
            ['python3', 'run.py', '--config', config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output line by line
        output_lines = []
        for line in process.stdout:
            # Show training progress without overwhelming the progress bar
            line = line.rstrip()
            if line:
                # Only show important lines to avoid clutter
                if any(keyword in line.lower() for keyword in ['psnr', 'step', 'error', 'completed', 'checkpoint']):
                    tqdm.write(f"  {line}")
                output_lines.append(line)
        
        process.wait()
        success = process.returncode == 0
        
        # Get last 100 lines for results
        recent_output = '\n'.join(output_lines[-100:]) if len(output_lines) > 100 else '\n'.join(output_lines)
        
    except Exception as e:
        tqdm.write(f"✗ Training failed: {e}")
        success = False
        recent_output = str(e)
    
    results = {
        'name': experiment_name,
        'config': config_path,
        'success': success,
        'stdout': recent_output,
        'stderr': '',
    }
    
    return results


# ============================================================================
# Ablation Study Definitions
# ============================================================================

def ablation_loss_components(base_config: dict) -> list:
    """Ablate individual loss components."""
    ablations = []
    
    # Baseline
    ablations.append(('baseline', {}))
    
    # Remove gradient loss
    ablations.append(('no_grad_loss', {
        'training.use_grad_loss': False,
        'training.lambda_grad': 0.0,
    }))
    
    # Remove tube regularization
    ablations.append(('no_tube_reg', {
        'training.lambda_tube': 0.0,
    }))
    
    # Remove cross regularization
    ablations.append(('no_cross_reg', {
        'training.lambda_cross': 0.0,
    }))
    
    # Remove scale regularization
    ablations.append(('no_scale_reg', {
        'training.lambda_scale': 0.0,
    }))
    
    # Remove all regularizers
    ablations.append(('no_regularizers', {
        'training.lambda_tube': 0.0,
        'training.lambda_cross': 0.0,
        'training.lambda_scale': 0.0,
    }))
    
    # Remove all losses except reconstruction
    ablations.append(('reconstruction_only', {
        'training.use_grad_loss': False,
        'training.lambda_grad': 0.0,
        'training.lambda_tube': 0.0,
        'training.lambda_cross': 0.0,
        'training.lambda_scale': 0.0,
    }))
    
    return ablations


# Training modes ablation removed - modes are always 'volume' in this setup


def ablation_densification(base_config: dict) -> list:
    """Ablate densification strategy."""
    ablations = []
    
    # Without densification
    ablations.append(('no_densify', {
        'training.densify_enabled': False,
    }))
    
    # With densification (baseline)
    ablations.append(('with_densify', {
        'training.densify_enabled': True,
    }))
    
    # Early densification stop
    ablations.append(('densify_early_stop', {
        'training.densify_enabled': True,
        'training.densify_until_iter': 15000,
    }))
    
    # More aggressive densification
    ablations.append(('densify_aggressive', {
        'training.densify_enabled': True,
        'training.densify_grad_threshold': 1.0e-4,  # Lower threshold
        'training.densify_interval': 100,  # More frequent
        'training.max_gaussians': 20000,  # Higher cap
    }))
    
    return ablations


def ablation_sampling_strategy(base_config: dict) -> list:
    """Ablate sampling strategies."""
    ablations = []
    
    # Uniform sampling (no intensity weighting)
    ablations.append(('uniform_sampling', {
        'training.vol_intensity_weighted': False,
    }))
    
    # Intensity-weighted sampling (baseline)
    ablations.append(('intensity_weighted', {
        'training.vol_intensity_weighted': True,
    }))
    
    # Different sample counts
    ablations.append(('more_volume_samples', {
        'training.vol_points_per_step': 16384,
    }))
    
    ablations.append(('fewer_volume_samples', {
        'training.vol_points_per_step': 4096,
    }))
    
    return ablations


def ablation_initialization(base_config: dict) -> list:
    """Ablate initialization strategies."""
    ablations = []
    
    # SWC-based initialization (baseline)
    ablations.append(('init_swc', {}))
    
    # Random initialization
    ablations.append(('init_random', {
        'data.swc_path': None,
    }))
    
    # Different initial scales
    ablations.append(('init_scale_small', {
        'model.init_scale': 0.05,
        'training.log_scale_max': -3.0,  # Adjust clamp range
    }))
    
    ablations.append(('init_scale_large', {
        'model.init_scale': 0.15,
        'training.log_scale_max': -1.9,  # Adjust clamp range
    }))
    
    # Different number of Gaussians
    ablations.append(('fewer_gaussians', {
        'model.num_gaussians': 5000,
    }))
    
    ablations.append(('more_gaussians', {
        'model.num_gaussians': 15000,
    }))
    
    return ablations


def ablation_hyperparameters(base_config: dict) -> list:
    """Ablate key hyperparameters."""
    ablations = []
    
    # Learning rate variations
    ablations.append(('lr_low', {
        'training.learning_rate': 1.0e-3,
    }))
    
    ablations.append(('lr_high', {
        'training.learning_rate': 5.0e-3,
    }))
    
    # Without early stopping
    ablations.append(('no_early_stopping', {
        'training.early_stopping': False,
    }))
    
    # Without mixed precision
    ablations.append(('no_mixed_precision', {
        'training.mixed_precision': False,
    }))
    
    # Without gradient clipping
    ablations.append(('no_grad_clip', {
        'training.grad_clip_norm': 0.0,
    }))
    
    return ablations


# ============================================================================
# Main Ablation Runner
# ============================================================================

def run_ablation_suite(
    study_name: str,
    ablation_fn,
    base_config_path: str = "config.yml",
    output_dir: str = "ablation_results"
):
    """
    Run a suite of ablation studies.
    
    Args:
        study_name: Name of the ablation study (e.g., 'loss_components')
        ablation_fn: Function that returns list of (name, modifications) tuples
        base_config_path: Path to base configuration file
        output_dir: Directory to save results
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = Path(output_dir) / f"{timestamp}_{study_name}"
    study_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base configuration
    base_config = load_base_config(base_config_path)
    
    # Get ablations
    ablations = ablation_fn(base_config)
    
    print(f"\n{'='*80}")
    print(f"ABLATION STUDY: {study_name}")
    print(f"Number of experiments: {len(ablations)}")
    print(f"Output directory: {study_dir}")
    print(f"{'='*80}\n")
    
    # Run each ablation with progress bar
    all_results = []
    pbar = tqdm(ablations, desc=f"Running {study_name}", unit="exp")
    
    for i, (name, modifications) in enumerate(pbar, 1):
        pbar.set_description(f"{study_name} [{i}/{len(ablations)}]: {name}")
        pbar.set_postfix({"status": "configuring"})
        
        # Create configuration with study_name for organized directories
        config = create_ablation_config(base_config, name, modifications, study_name)
        config_path = study_dir / "configs" / f"{name}.yml"
        save_config(config, str(config_path))
        
        # Log the directory structure for this experiment
        exp_dir = Path("ablation_results") / study_name / name
        tqdm.write(f"\n[{i}/{len(ablations)}] {name}")
        tqdm.write(f"  Checkpoints: {exp_dir}/checkpoints/")
        tqdm.write(f"  Logs: {exp_dir}/logs/")
        tqdm.write(f"  Config: {config_path}")
        
        # Run training
        pbar.set_postfix({"status": "training"})
        result = run_training(str(config_path), name)
        all_results.append(result)
        
        # Update progress bar with result
        status = "✓" if result['success'] else "✗"
        pbar.set_postfix({"status": status, "completed": i})
        
        # Save intermediate results
        results_path = study_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    pbar.close()
    
    # Generate summary report
    generate_summary_report(study_name, all_results, study_dir)
    
    print(f"\n{'='*80}")
    print(f"ABLATION STUDY COMPLETE: {study_name}")
    print(f"Results saved to: {study_dir}")
    print(f"{'='*80}\n")
    
    return all_results


def generate_summary_report(study_name: str, results: list, output_dir: Path):
    """Generate a markdown summary report of ablation results."""
    report_path = output_dir / "SUMMARY.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# Ablation Study: {study_name}\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Experiments**: {len(results)}\n\n")
        
        f.write("## Results Overview\n\n")
        f.write("| Experiment | Success | Notes |\n")
        f.write("|------------|---------|-------|\n")
        
        for result in results:
            status = "✅" if result['success'] else "❌"
            f.write(f"| {result['name']} | {status} | - |\n")
        
        f.write("\n## Detailed Results\n\n")
        
        for result in results:
            f.write(f"### {result['name']}\n\n")
            f.write(f"- **Config**: `{result['config']}`\n")
            f.write(f"- **Success**: {result['success']}\n\n")
            
            if not result['success']:
                f.write("**Error Output**:\n```\n")
                f.write(result['stderr'][-1000:])  # Last 1000 chars
                f.write("\n```\n\n")
    
    print(f"Summary report saved to: {report_path}")


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument(
        '--study',
        choices=['loss', 'densify', 'sampling', 'init', 'hyperparams', 'all'],
        default='all',
        help='Which ablation study to run'
    )
    parser.add_argument(
        '--config',
        default='config.yml',
        help='Base configuration file'
    )
    parser.add_argument(
        '--output',
        default='ablation_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Define study mapping
    studies = {
        'loss': ('loss_components', ablation_loss_components),
        'densify': ('densification', ablation_densification),
        'sampling': ('sampling_strategy', ablation_sampling_strategy),
        'init': ('initialization', ablation_initialization),
        'hyperparams': ('hyperparameters', ablation_hyperparameters),
    }
    
    if args.study == 'all':
        # Run all studies with progress tracking
        print(f"\n{'='*80}")
        print(f"Running ALL ablation studies")
        print(f"Total studies: {len(studies)}")
        print(f"{'='*80}\n")
        
        study_pbar = tqdm(studies.items(), desc="Overall Progress", unit="study", position=0)
        for study_key, (name, fn) in study_pbar:
            study_pbar.set_description(f"Study: {name}")
            run_ablation_suite(name, fn, args.config, args.output)
        study_pbar.close()
        
        print(f"\n{'='*80}")
        print(f"ALL ABLATION STUDIES COMPLETE!")
        print(f"Results saved to: {args.output}")
        print(f"{'='*80}\n")
    else:
        # Run specific study
        name, fn = studies[args.study]
        run_ablation_suite(name, fn, args.config, args.output)
