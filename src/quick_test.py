#!/usr/bin/env python3
"""
Quick runner for ablation studies - tests one experiment from each category
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from ablation_studies import (
    load_base_config,
    ablation_loss_components,
    ablation_densification,
    ablation_sampling_strategy,
    ablation_initialization,
    ablation_hyperparameters,
    create_ablation_config,
    save_config,
)

def main():
    print("=" * 80)
    print("ABLATION FRAMEWORK TEST")
    print("=" * 80)
    print()
    
    # Load base config
    config_path = "config.yml"
    if not os.path.exists(config_path):
        print(f"❌ Error: Config file not found: {config_path}")
        return 1
    
    base_config = load_base_config(config_path)
    print(f"✓ Loaded base config: {config_path}")
    print(f"  - Gaussians: {base_config['model']['num_gaussians']}")
    print(f"  - Training steps: {base_config['training']['steps']}")
    print(f"  - Data: {base_config['data']['tif_path']}")
    print()
    
    # Test each ablation suite by generating one config
    test_studies = [
        ('loss_components', ablation_loss_components, 'baseline'),
        ('densification', ablation_densification, 'with_densify'),
        ('sampling_strategy', ablation_sampling_strategy, 'intensity_weighted'),
        ('initialization', ablation_initialization, 'init_swc'),
        ('hyperparameters', ablation_hyperparameters, 'lr_low'),
    ]
    
    test_dir = "ablation_test"
    os.makedirs(test_dir, exist_ok=True)
    
    print("Testing configuration generation:")
    print("-" * 80)
    
    for study_name, ablation_fn, test_exp in test_studies:
        try:
            # Get ablations
            ablations = ablation_fn(base_config)
            
            # Find the test experiment
            test_config = None
            for name, mods in ablations:
                if name == test_exp:
                    test_config = create_ablation_config(base_config, name, mods, study_name)
                    break
            
            if test_config:
                # Save test config
                config_path = os.path.join(test_dir, f"{study_name}_{test_exp}.yml")
                save_config(test_config, config_path)
                
                print(f"✓ {study_name:25s} - Generated {len(ablations)} experiments")
                print(f"  Test config: {config_path}")
                print(f"  Checkpoints: {test_config['training']['save_path']}")
                print(f"  Logs: {test_config['training']['log_dir']}")
            else:
                print(f"❌ {study_name:25s} - Test experiment not found")
                return 1
                
        except Exception as e:
            print(f"❌ {study_name:25s} - Error: {e}")
            return 1
    
    print()
    print("=" * 80)
    print("TEST COMPLETE - Framework is working!")
    print("=" * 80)
    print()
    print("Test configs saved to:", test_dir)
    print()
    print("To run ablation studies:")
    print()
    print("  # Single study (recommended to start)")
    print("  python3 ablation_studies.py --study loss")
    print()
    print("  # All studies (takes many hours)")
    print("  python3 ablation_studies.py --study all")
    print()
    print("  # Or use the bash script")
    print("  ./run_ablation.sh loss")
    print()
    
    return 0

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    sys.exit(main())
