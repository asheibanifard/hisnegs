#!/usr/bin/env python3
"""
Dry-run validation for ablation study framework.
Tests configuration generation without running training.
"""

import os
import yaml
from ablation_studies import (
    load_base_config,
    ablation_loss_components,
    ablation_training_modes,
    ablation_densification,
    ablation_sampling_strategy,
    ablation_initialization,
    ablation_hyperparameters,
    create_ablation_config,
    save_config,
)


def validate_ablation_suite():
    """Validate all ablation configurations without running training."""
    
    print("=" * 80)
    print("ABLATION FRAMEWORK VALIDATION")
    print("=" * 80)
    print()
    
    # Load base config
    config_path = "config.yml"
    if not os.path.exists(config_path):
        print(f"❌ Error: Config file not found: {config_path}")
        print("   Please run from the hisnegs/src/ directory")
        return False
    
    print(f"✓ Loading base config: {config_path}")
    base_config = load_base_config(config_path)
    print(f"  Model: {base_config['model']['num_gaussians']} Gaussians")
    print(f"  Training steps: {base_config['training']['steps']}")
    print()
    
    # Define all ablation functions
    ablation_suites = {
        'Loss Components': ablation_loss_components,
        'Densification': ablation_densification,
        'Sampling Strategy': ablation_sampling_strategy,
        'Initialization': ablation_initialization,
        'Hyperparameters': ablation_hyperparameters,
    }
    
    total_experiments = 0
    validation_dir = "ablation_validation"
    
    # Test each ablation suite
    for suite_name, ablation_fn in ablation_suites.items():
        print(f"Testing: {suite_name}")
        print("-" * 80)
        
        try:
            ablations = ablation_fn(base_config)
            print(f"  ✓ Generated {len(ablations)} ablation configs")
            
            # Validate each config
            for name, modifications in ablations:
                try:
                    # Use suite_name as study_name for proper directory organization
                    study_name_clean = suite_name.lower().replace(' ', '_')
                    config = create_ablation_config(base_config, name, modifications, study_name_clean)
                    
                    # Save to validation directory
                    config_path = os.path.join(validation_dir, study_name_clean, f"{name}.yml")
                    save_config(config, config_path)
                    
                    # Basic validation
                    assert 'model' in config, "Missing 'model' section"
                    assert 'training' in config, "Missing 'training' section"
                    assert 'data' in config, "Missing 'data' section"
                    
                    # Verify paths are properly set
                    expected_base = f"ablation_results/{study_name_clean}/{name}"
                    assert config['training']['save_path'].startswith(expected_base), \
                        f"Checkpoint path incorrect: {config['training']['save_path']}"
                    assert config['training']['log_dir'].startswith(expected_base), \
                        f"Log dir incorrect: {config['training']['log_dir']}"
                    
                    print(f"    ✓ {name:30s} - Valid")
                    print(f"      Checkpoints: {config['training']['save_path']}")
                    print(f"      Logs: {config['training']['log_dir']}")
                    total_experiments += 1
                    
                except Exception as e:
                    print(f"    ❌ {name:30s} - Error: {e}")
                    return False
            
            print()
            
        except Exception as e:
            print(f"  ❌ Failed to generate ablations: {e}")
            return False
    
    print("=" * 80)
    print(f"VALIDATION COMPLETE")
    print("=" * 80)
    print(f"✅ All {total_experiments} experiment configs generated successfully")
    print(f"✅ Configs saved to: {validation_dir}/")
    print()
    print("Next steps:")
    print("  1. Review generated configs in the validation directory")
    print("  2. Run ablation studies: ./run_ablation.sh loss")
    print("  3. Check ABLATION_SUITE_README.md for full documentation")
    print()
    
    return True


if __name__ == "__main__":
    import sys
    
    success = validate_ablation_suite()
    sys.exit(0 if success else 1)
