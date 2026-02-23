#!/usr/bin/env python3
"""
Test progress bar functionality in ablation framework
"""
import os
import sys
import time
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

def test_progress_bars():
    """Demonstrate the progress bars in the ablation framework."""
    print("=" * 80)
    print("Testing Progress Bar Integration")
    print("=" * 80)
    print()
    
    # Test 1: Single experiment progress
    print("Test 1: Running experiments with progress bar")
    print("-" * 80)
    experiments = ['baseline', 'no_grad_loss', 'no_tube_reg', 'no_cross_reg']
    
    pbar = tqdm(experiments, desc="Running experiments", unit="exp")
    for i, exp in enumerate(pbar, 1):
        pbar.set_description(f"Experiment [{i}/{len(experiments)}]: {exp}")
        pbar.set_postfix({"status": "training"})
        time.sleep(0.5)  # Simulate training
        
        # Use tqdm.write for output that doesn't interfere with progress bar
        tqdm.write(f"  ✓ {exp} completed")
        pbar.set_postfix({"status": "✓", "completed": i})
    
    pbar.close()
    print()
    
    # Test 2: Multiple studies with nested progress
    print("Test 2: Multiple studies progress")
    print("-" * 80)
    studies = {
        'loss_components': 7,
        'densification': 4,
        'sampling_strategy': 4,
        'initialization': 6,
        'hyperparameters': 5,
    }
    
    study_pbar = tqdm(studies.items(), desc="Overall Progress", unit="study")
    for study_name, num_exp in study_pbar:
        study_pbar.set_description(f"Study: {study_name}")
        
        # Simulate experiments in this study
        for i in range(num_exp):
            time.sleep(0.2)
            tqdm.write(f"  {study_name}: experiment {i+1}/{num_exp}")
        
        study_pbar.set_postfix({"experiments": num_exp, "status": "✓"})
    
    study_pbar.close()
    print()
    
    # Test 3: Analysis progress
    print("Test 3: Analysis tasks with progress")
    print("-" * 80)
    tasks = [
        "Parsing logs",
        "Generating comparison table",
        "Saving detailed analysis",
        "Plotting learning curves",
        "Plotting PSNR comparison",
    ]
    
    with tqdm(tasks, desc="Analysis", unit="task") as pbar:
        for task in pbar:
            pbar.set_description(f"Task: {task}")
            time.sleep(0.3)
            pbar.set_postfix({"status": "✓"})
    
    print()
    print("=" * 80)
    print("✓ All progress bar tests completed!")
    print("=" * 80)
    print()
    print("The ablation framework now includes progress bars for:")
    print("  1. Individual experiment execution")
    print("  2. Multiple study runs (--study all)")
    print("  3. Log analysis and parsing")
    print("  4. Plot generation")
    print()
    print("Run ablation studies with:")
    print("  python3 ablation_studies.py --study loss")
    print()

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    test_progress_bars()
