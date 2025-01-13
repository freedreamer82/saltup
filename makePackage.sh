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

echo "Package installed successfully!"