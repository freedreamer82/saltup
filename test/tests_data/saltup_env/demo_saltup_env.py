#!/usr/bin/env python3
"""
Example script demonstrating SaltupEnv configuration usage.

This script shows how to use the new SaltupEnv configuration system
for both Keras and PyTorch training arguments.
"""

import os
import json
from pathlib import Path
from saltup.saltup_env import SaltupEnv

def demonstrate_saltup_env():
    """Demonstrate various SaltupEnv configuration options."""
    
    print("=== SaltupEnv Configuration Demo ===\n")
    
    # Basic properties
    print("1. Basic Properties:")
    print(f"   VERSION: {SaltupEnv.VERSION}")
    print(f"   SALTUP_KERAS_TRAIN_SHUFFLE: {SaltupEnv.SALTUP_KERAS_TRAIN_SHUFFLE}")
    print(f"   SALTUP_KERAS_TRAIN_VERBOSE: {SaltupEnv.SALTUP_KERAS_TRAIN_VERBOSE}")
    print(f"   SALTUP_PYTORCH_DEVICE: {SaltupEnv.SALTUP_PYTORCH_DEVICE}")
    print(f"   SALTUP_ONNX_OPSET: {SaltupEnv.SALTUP_ONNX_OPSET}")
    print()
    
    # Keras arguments (default - empty)
    print("2. Keras Arguments (default):")
    print(f"   COMPILE_ARGS: {SaltupEnv.SALTUP_TRAINING_KERAS_COMPILE_ARGS}")
    print(f"   FIT_ARGS: {SaltupEnv.SALTUP_TRAINING_KERAS_FIT_ARGS}")
    print()
    
    # PyTorch arguments (with defaults)
    print("3. PyTorch Arguments (with defaults):")
    pytorch_args = SaltupEnv.SALTUP_TRAINING_PYTORCH_ARGS
    for key, value in pytorch_args.items():
        print(f"   {key}: {value}")
    print()
    
    # Demonstrate JSON string configuration
    print("4. Demonstrating JSON String Configuration:")
    
    # Set environment variables
    os.environ["SALTUP_TRAINING_KERAS_COMPILE_ARGS"] = json.dumps({
        "metrics": ["accuracy", "precision"],
        "run_eagerly": True
    })
    
    os.environ["SALTUP_TRAINING_PYTORCH_ARGS"] = json.dumps({
        "gradient_clip_value": 1.0,
        "early_stopping_patience": 10
    })
    
    print(f"   Keras Compile Args: {SaltupEnv.SALTUP_TRAINING_KERAS_COMPILE_ARGS}")
    print(f"   PyTorch Args: {SaltupEnv.SALTUP_TRAINING_PYTORCH_ARGS}")
    print()
    
    # Demonstrate file-based configuration
    print("5. Demonstrating File-Based Configuration:")
    
    # Create temporary config files
    config_dir = Path("/tmp/saltup_config_demo")
    config_dir.mkdir(exist_ok=True)
    
    # Create Keras fit config
    keras_fit_config = {
        "workers": 4,
        "use_multiprocessing": True,
        "max_queue_size": 20
    }
    
    keras_fit_file = config_dir / "keras_fit_args.json"
    with open(keras_fit_file, 'w') as f:
        json.dump(keras_fit_config, f, indent=2)
    
    # Create PyTorch config
    pytorch_config = {
        "use_scheduler_per_epoch": True,
        "early_stopping_patience": 5
    }
    
    pytorch_file = config_dir / "pytorch_args.json"
    with open(pytorch_file, 'w') as f:
        json.dump(pytorch_config, f, indent=2)
    
    # Set environment variables to use files
    os.environ["SALTUP_TRAINING_KERAS_FIT_ARGS"] = str(keras_fit_file)
    os.environ["SALTUP_TRAINING_PYTORCH_ARGS"] = str(pytorch_file)
    
    print(f"   Keras Fit Args (from file): {SaltupEnv.SALTUP_TRAINING_KERAS_FIT_ARGS}")
    print(f"   PyTorch Args (from file): {SaltupEnv.SALTUP_TRAINING_PYTORCH_ARGS}")
    print()
    
    # Show how defaults are preserved
    print("6. Default Values Preservation:")
    pytorch_from_file = SaltupEnv.SALTUP_TRAINING_PYTORCH_ARGS
    print(f"   use_scheduler_per_epoch (override): {pytorch_from_file['use_scheduler_per_epoch']}")
    print(f"   early_stopping_patience (override): {pytorch_from_file['early_stopping_patience']}")
    print()
    
    # Clean up
    keras_fit_file.unlink()
    pytorch_file.unlink()
    config_dir.rmdir()
    
    print("=== Demo Complete ===")

if __name__ == "__main__":
    demonstrate_saltup_env()
