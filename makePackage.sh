#!/bin/bash
set -e

# Help message function
show_help() {
    echo "Usage: $0 [-d] [-b] [-e EXTRAS]"
    echo "Options:"
    echo "  -d         Install in development mode"
    echo "  -b         Build distribution packages (.tar.gz and .whl) without installing"
    echo "  -e EXTRAS  Comma-separated extras to install (e.g. onnx, onnx,torch, inference, full)."
    echo "             Default: everything, with GPU/CPU flavours picked by GPU detection."
    echo "  -c         Force CPU-only flavours (onnxruntime, tensorflow-cpu, torch from the CPU index)"
    echo "             even if a GPU is detected. Implied when no NVIDIA GPU is found."
    echo "  -h         Show this help message"
    exit 0
}

# Default values
DEV_MODE=false
BUILD_ONLY=false
EXTRAS=""
FORCE_CPU=false

# Parse command line options
while getopts "dbe:ch" opt; do
    case $opt in
        d)
            DEV_MODE=true
            ;;
        b)
            BUILD_ONLY=true
            ;;
        e)
            EXTRAS="$OPTARG"
            ;;
        c)
            FORCE_CPU=true
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

# Detect GPU (unless CPU was forced)
HAS_GPU=false
if [ "$FORCE_CPU" = false ] && command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
    HAS_GPU=true
fi

# Pick extras. Without -e, install everything, choosing GPU or CPU flavours
# (the 'full' extra alone pulls onnxruntime-gpu and the regular tensorflow).
if [ -z "$EXTRAS" ]; then
    if [ "$HAS_GPU" = true ]; then
        echo "NVIDIA GPU detected: using GPU flavours (onnxruntime-gpu, tensorflow, CUDA torch)"
        EXTRAS="full"
    else
        echo "No NVIDIA GPU (or -c given): using CPU flavours (onnxruntime, tensorflow-cpu, CPU torch)"
        EXTRAS="onnx,keras-cpu,torch,convert,training,audio,dev"
    fi
fi
TARGET=".[${EXTRAS}]"

# PyPI only ships the CUDA torch wheel; on CPU pull torch from the PyTorch CPU index.
PIP_EXTRA_ARGS=()
if [ "$HAS_GPU" = false ] && [[ ",$EXTRAS," == *torch* || ",$EXTRAS," == *inference* || ",$EXTRAS," == *convert* || ",$EXTRAS," == *training* || ",$EXTRAS," == *full* ]]; then
    PIP_EXTRA_ARGS+=(--extra-index-url https://download.pytorch.org/whl/cpu)
fi

# Install the package
if [ "$DEV_MODE" = true ]; then
    echo "Installing package in development mode with extras [${EXTRAS}]..."
    python -m pip install -e "$TARGET" "${PIP_EXTRA_ARGS[@]}"
else
    echo "Installing package in normal mode with extras [${EXTRAS}]..."
    python -m pip install "$TARGET" "${PIP_EXTRA_ARGS[@]}"
fi

echo "Package installed successfully!"