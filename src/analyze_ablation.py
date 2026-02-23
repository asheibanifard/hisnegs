#!/usr/bin/env python3
"""
Analyze Ablation Study Results

Parse logs and checkpoints from ablation studies to generate comparative metrics.
"""

import os
import re
import json
import yaml
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def parse_log_file(log_path: str) -> dict:
    """Parse training log file to extract metrics."""
    metrics = {
        'psnr_history': [],
        'loss_history': [],
        'final_psnr': None,
        'best_psnr': None,
        'final_k': None,
        'training_time': None,
    }
    
    if not os.path.exists(log_path):
        return metrics
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract PSNR values
    psnr_pattern = r'PSNR@(\d+): ([\d.]+) dB'
    for match in re.finditer(psnr_pattern, content):
        step = int(match.group(1))
        psnr = float(match.group(2))
        metrics['psnr_history'].append((step, psnr))
    
    if metrics['psnr_history']:
        metrics['best_psnr'] = max(psnr for _, psnr in metrics['psnr_history'])
        metrics['final_psnr'] = metrics['psnr_history'][-1][1]
    
    # Extract loss values
    loss_pattern = r"Step (\d+): K=(\d+)\s+best_total=([\d.e+-]+)"
    for match in re.finditer(loss_pattern, content):
        step = int(match.group(1))
        k = int(match.group(2))
        loss = float(match.group(3))
        metrics['loss_history'].append((step, loss))
        metrics['final_k'] = k
    
    # Extract timing info
    timing_pattern = r'TOTAL\s*:\s*([\d.]+)\s*ms/step'
    timing_match = re.search(timing_pattern, content)
    if timing_match:
        metrics['training_time'] = float(timing_match.group(1))
    
    return metrics


def find_log_file(experiment_name: str, results_dir: Path, study_name: str = None) -> str | None:
    """Find the log file for an experiment."""
    # Try new structure: ablation_results/STUDY_NAME/EXPERIMENT_NAME/logs/
    if study_name:
        log_dir = Path("ablation_results") / study_name / experiment_name / "logs"
    else:
        # Try to infer from results_dir
        # results_dir format: ablation_results/TIMESTAMP_STUDYNAME/
        study_name_match = results_dir.name.split('_', 1)
        if len(study_name_match) > 1:
            study_name_part = '_'.join(study_name_match[1:])  # Remove timestamp
            log_dir = Path("ablation_results") / study_name_part / experiment_name / "logs"
        else:
            # Fallback to old structure
            log_dir = results_dir.parent.parent / "logs" / experiment_name
    
    if not log_dir.exists():
        # Try alternative: check if logs are in the experiment directory itself
        log_dir = Path("ablation_results") / experiment_name / "logs"
        if not log_dir.exists():
            return None
    
    # Find most recent log file
    log_files = list(log_dir.glob("*.log"))
    if log_files:
        return str(sorted(log_files)[-1])
    
    return None


def analyze_ablation_results(results_dir: str) -> dict:
    """Analyze results from an ablation study directory."""
    results_path = Path(results_dir)
    
    # Extract study name from directory (format: TIMESTAMP_STUDYNAME)
    dir_name = results_path.name
    study_name = None
    if '_' in dir_name:
        parts = dir_name.split('_', 1)
        if len(parts) > 1:
            study_name = parts[1]
    
    # Load results.json if it exists
    results_json_path = results_path / "results.json"
    if results_json_path.exists():
        with open(results_json_path, 'r') as f:
            results_data = json.load(f)
    else:
        results_data = []
    
    # Parse logs for each experiment
    configs_dir = results_path / "configs"
    if not configs_dir.exists():
        print(f"Warning: Configs directory not found: {configs_dir}")
        return {}
    
    analysis = {}
    
    config_files = list(configs_dir.glob("*.yml"))
    
    for config_file in tqdm(config_files, desc="Analyzing experiments", unit="exp"):
        exp_name = config_file.stem
        
        # Load config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Find and parse log using new structure
        log_path = find_log_file(exp_name, results_path, study_name)
        metrics = parse_log_file(log_path) if log_path else {}
        
        # Find checkpoint directory
        if study_name:
            ckpt_dir = Path("ablation_results") / study_name / exp_name / "checkpoints"
        else:
            ckpt_dir = Path("ablation_results") / exp_name / "checkpoints"
        
        analysis[exp_name] = {
            'config': config,
            'metrics': metrics,
            'log_path': log_path,
            'checkpoint_dir': str(ckpt_dir) if ckpt_dir.exists() else None,
        }
    
    return analysis


def generate_comparison_table(analysis: dict, output_path: str):
    """Generate a markdown comparison table."""
    experiments = sorted(analysis.keys())
    
    with open(output_path, 'w') as f:
        f.write("# Ablation Study Comparison\n\n")
        
        # Directory structure
        f.write("## Experiment Directories\n\n")
        f.write("Each experiment has its own organized directory:\n\n")
        for exp in experiments[:3]:  # Show first 3 as examples
            ckpt_dir = analysis[exp].get('checkpoint_dir')
            log_path = analysis[exp].get('log_path')
            if ckpt_dir or log_path:
                f.write(f"**{exp}**:\n")
                if ckpt_dir:
                    f.write(f"- Checkpoints: `{ckpt_dir}`\n")
                if log_path:
                    log_dir = str(Path(log_path).parent)
                    f.write(f"- Logs: `{log_dir}`\n")
                f.write("\n")
        if len(experiments) > 3:
            f.write(f"*(... and {len(experiments) - 3} more experiments)*\n\n")
        
        # Summary table
        f.write("## Performance Summary\n\n")
        f.write("| Experiment | Final PSNR (dB) | Best PSNR (dB) | Final #Gaussians | Training Time (ms/step) |\n")
        f.write("|------------|-----------------|----------------|------------------|-------------------------|\n")
        
        for exp in experiments:
            metrics = analysis[exp]['metrics']
            final_psnr = f"{metrics['final_psnr']:.2f}" if metrics['final_psnr'] else "N/A"
            best_psnr = f"{metrics['best_psnr']:.2f}" if metrics['best_psnr'] else "N/A"
            final_k = f"{metrics['final_k']}" if metrics['final_k'] else "N/A"
            train_time = f"{metrics['training_time']:.2f}" if metrics['training_time'] else "N/A"
            
            f.write(f"| {exp} | {final_psnr} | {best_psnr} | {final_k} | {train_time} |\n")
        
        f.write("\n## PSNR Ranking\n\n")
        
        # Sort by best PSNR
        ranked = [(exp, analysis[exp]['metrics']['best_psnr']) 
                  for exp in experiments 
                  if analysis[exp]['metrics']['best_psnr'] is not None]
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (exp, psnr) in enumerate(ranked, 1):
            f.write(f"{rank}. **{exp}**: {psnr:.2f} dB\n")
        
        f.write("\n## Configuration Differences\n\n")
        
        # Extract key configuration differences
        for exp in experiments:
            config = analysis[exp]['config']
            f.write(f"### {exp}\n\n")
            
            # Show training config
            tc = config.get('training', {})
            f.write("**Key Settings:**\n")
            f.write(f"- Mode: `{tc.get('mode', 'N/A')}`\n")
            f.write(f"- Learning Rate: `{tc.get('learning_rate', 'N/A')}`\n")
            f.write(f"- Gradient Loss: `{tc.get('use_grad_loss', 'N/A')}` (λ={tc.get('lambda_grad', 'N/A')})\n")
            f.write(f"- Tube Reg: λ={tc.get('lambda_tube', 'N/A')}\n")
            f.write(f"- Cross Reg: λ={tc.get('lambda_cross', 'N/A')}\n")
            f.write(f"- Scale Reg: λ={tc.get('lambda_scale', 'N/A')}\n")
            f.write(f"- Densification: `{tc.get('densify_enabled', 'N/A')}`\n")
            f.write(f"- Intensity Weighted: `{tc.get('vol_intensity_weighted', 'N/A')}`\n")
            f.write("\n")


def plot_learning_curves(analysis: dict, output_dir: str):
    """Plot learning curves for all experiments."""
    print("Generating learning curves...")
    plt.figure(figsize=(12, 6))
    
    # Plot PSNR curves
    plt.subplot(1, 2, 1)
    for exp_name, data in tqdm(analysis.items(), desc="Plotting PSNR curves", leave=False):
        psnr_history = data['metrics']['psnr_history']
        if psnr_history:
            steps, psnrs = zip(*psnr_history)
            plt.plot(steps, psnrs, label=exp_name, alpha=0.7)
    
    plt.xlabel('Training Step')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Learning Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot loss curves
    plt.subplot(1, 2, 2)
    for exp_name, data in tqdm(analysis.items(), desc="Plotting loss curves", leave=False):
        loss_history = data['metrics']['loss_history']
        if loss_history:
            steps, losses = zip(*loss_history)
            plt.plot(steps, losses, label=exp_name, alpha=0.7)
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=150, bbox_inches='tight')
    print(f"Learning curves saved to: {output_dir}/learning_curves.png")


def plot_psnr_comparison(analysis: dict, output_dir: str):
    """Create bar plot comparing final PSNR values."""
    experiments = []
    psnrs = []
    
    for exp_name, data in analysis.items():
        if data['metrics']['best_psnr'] is not None:
            experiments.append(exp_name)
            psnrs.append(data['metrics']['best_psnr'])
    
    # Sort by PSNR
    sorted_pairs = sorted(zip(experiments, psnrs), key=lambda x: x[1], reverse=True)
    experiments, psnrs = zip(*sorted_pairs)
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(range(len(experiments)), psnrs)
    
    # Color bars by performance
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(psnrs)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.yticks(range(len(experiments)), experiments)
    plt.xlabel('Best PSNR (dB)')
    plt.title('Ablation Study: Best PSNR Comparison')
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'psnr_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"PSNR comparison saved to: {output_dir}/psnr_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation study results")
    parser.add_argument('results_dir', help='Path to ablation results directory')
    parser.add_argument('--output', default=None, help='Output directory (defaults to results_dir)')
    
    args = parser.parse_args()
    
    output_dir = args.output or args.results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Analyzing ablation study results")
    print(f"Source: {args.results_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    # Analyze results
    print("Step 1/4: Parsing logs and extracting metrics...")
    analysis = analyze_ablation_results(args.results_dir)
    
    if not analysis:
        print("❌ No results found!")
        return
    
    print(f"✓ Found {len(analysis)} experiments\n")
    
    # Create progress bar for remaining steps
    tasks = [
        ("Generating comparison table", lambda: generate_comparison_table(analysis, os.path.join(output_dir, 'comparison_table.md'))),
        ("Saving detailed analysis", lambda: save_detailed_analysis(analysis, output_dir)),
        ("Plotting learning curves", lambda: plot_learning_curves(analysis, output_dir)),
        ("Plotting PSNR comparison", lambda: plot_psnr_comparison(analysis, output_dir)),
    ]
    
    with tqdm(tasks, desc="Analysis tasks", unit="task") as pbar:
        for task_name, task_fn in pbar:
            pbar.set_description(f"Step: {task_name}")
            try:
                task_fn()
                pbar.set_postfix({"status": "✓"})
            except Exception as e:
                pbar.set_postfix({"status": "✗"})
                tqdm.write(f"⚠ Warning: {task_name} failed: {e}")
    
    print(f"\n{'='*80}")
    print(f"✓ Analysis complete!")
    print(f"{'='*80}\n")
    print(f"Results saved to: {output_dir}")
    print(f"  - comparison_table.md")
    print(f"  - detailed_analysis.json")
    print(f"  - learning_curves.png")
    print(f"  - psnr_comparison.png")
    print()


def save_detailed_analysis(analysis: dict, output_dir: str):
    """Save detailed analysis as JSON."""
    analysis_path = os.path.join(output_dir, 'detailed_analysis.json')
    # Remove config objects for JSON serialization
    serializable_analysis = {}
    for exp_name, data in analysis.items():
        serializable_analysis[exp_name] = {
            'metrics': data['metrics'],
            'log_path': data['log_path'],
        }
    
    with open(analysis_path, 'w') as f:
        json.dump(serializable_analysis, f, indent=2)


if __name__ == "__main__":
    main()
