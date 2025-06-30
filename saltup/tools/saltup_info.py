#!/usr/bin/env python3
from saltup.saltup_env import SaltupEnv

def print_saltup_info():
    print("=== Saltup Framework Info ===")
    print(f"Version: {SaltupEnv.VERSION}")
    print("Description: add flavor to your AI projects")
    print("Main features:")
    print(" - Neural network model management")
    print(" - Object detection dataset generation and loading")
    print(" - Bounding box calculations and visualizations")
    print(" - Model inference optimization")
    print(" - Cross-framework compatibility (TensorFlow/Keras and PyTorch)")
    print(" - Data preprocessing and augmentation")
    print(" - Performance metrics tracking")
    print(" - Logging configuration with TQDM support")
    print()
    print("Authors:")
    print(" - Marco Garzola")
    print(" - Francesco Sonnessa")
    print(" - Marc Randriatsimiovalaza")
    print()
    print("GitHub: https://github.com/marcogar/saltup")

def main():
    print_saltup_info()