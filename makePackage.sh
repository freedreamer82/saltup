#!/bin/bash
set -e

# Help message function
show_help() {
    echo "Usage: $0 [-d] [-b]"
    echo "Options:"
    echo "  -d    Install in development mode"
    echo "  -b    Build distribution packages (.tar.gz and .whl) without installing"
    echo "  -h    Show this help message"
    exit 0
}

# Default values
DEV_MODE=false
BUILD_ONLY=false

# Parse command line options
while getopts "dbh" opt; do
    case $opt in
        d)
            DEV_MODE=true
            ;;
        b)
            BUILD_ONLY=true
            ;;
        h)
            show_help
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            show_help
            ;;
    esac
done

# Check and install required tools
echo "Checking dependencies..."
python -m pip install --quiet --upgrade pip setuptools wheel

# Clean before starting
./clean.sh

# Handle build only mode
if [ "$BUILD_ONLY" = true ]; then
    echo "Building distribution packages..."
    python -m pip install --quiet build
    python -m build
    echo "Distribution packages created successfully!"
    exit 0
fi

# Install the package
if [ "$DEV_MODE" = true ]; then
    echo "Installing package in development mode..."
    python -m pip install -e .
else
    echo "Installing package in normal mode..."
    python -m pip install .
fi

# Install onnxruntime or onnxruntime-gpu based on GPU presence
echo "Checking for NVIDIA GPU to install appropriate onnxruntime package..."
if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
    echo "NVIDIA GPU detected: installing onnxruntime-gpu"
    python -m pip install onnxruntime-gpu
else
    echo "No NVIDIA GPU detected: installing onnxruntime (CPU version)"
    python -m pip install onnxruntime
fi

echo "Package installed successfully!"