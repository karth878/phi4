#!/usr/bin/env python3
"""
Quick config validation script for Axolotl
"""
import yaml
import sys
import os

def validate_config(config_path):
    """Validate Axolotl config for common issues"""
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Failed to parse YAML: {e}")
        return False
    
    print(f"✅ Config file loaded: {config_path}")
    
    # Check for conflicting parameters
    conflicts = []
    
    # Evaluation conflicts
    if config.get('eval_steps') and config.get('evals_per_epoch'):
        conflicts.append('eval_steps and evals_per_epoch are mutually exclusive')
    
    # Warmup conflicts
    if config.get('warmup_steps') and config.get('warmup_ratio'):
        conflicts.append('warmup_steps and warmup_ratio are mutually exclusive')
    
    # Saving conflicts
    if config.get('save_steps') and config.get('saves_per_epoch'):
        conflicts.append('save_steps and saves_per_epoch are mutually exclusive')
    
    # Check dataset path
    dataset_path = None
    if 'datasets' in config and config['datasets']:
        dataset_path = config['datasets'][0].get('path')
        if dataset_path and not os.path.exists(dataset_path):
            conflicts.append(f'Dataset path does not exist: {dataset_path}')
        else:
            print(f"✅ Dataset path exists: {dataset_path}")
    
    # Report results
    if conflicts:
        print("❌ Config validation failed:")
        for conflict in conflicts:
            print(f"   - {conflict}")
        return False
    else:
        print("✅ No conflicts found!")
        
        # Print key settings
        print(f"\n📊 Key Settings:")
        print(f"   Model: {config.get('base_model', 'Not specified')}")
        print(f"   Dataset: {dataset_path or 'Not specified'}")
        print(f"   Sequence Length: {config.get('sequence_len', 'Not specified')}")
        print(f"   Micro Batch Size: {config.get('micro_batch_size', 'Not specified')}")
        print(f"   Gradient Accumulation: {config.get('gradient_accumulation_steps', 'Not specified')}")
        print(f"   Learning Rate: {config.get('learning_rate', 'Not specified')}")
        print(f"   Warmup Steps: {config.get('warmup_steps', 'Not specified')}")
        print(f"   Eval Steps: {config.get('eval_steps', 'Not specified')}")
        
        return True

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "phi4-axolotl_config_fixed.yml"
    
    print("🔍 Validating Axolotl Config...")
    print("=" * 50)
    
    success = validate_config(config_file)
    
    if success:
        print("\n🎉 Config validation passed! Ready for training.")
        sys.exit(0)
    else:
        print("\n❌ Config validation failed! Fix errors before training.")
        sys.exit(1)