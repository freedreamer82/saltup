#!/bin/bash
set -e

# Remove generated files
echo "Cleaning generated files..."
rm -rf build/ dist/ *.egg-info/ __pycache__/ .eggs/ .pytest_cache/
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete
find . -type f -name ".DS_Store" -delete

# Uninstall package if present
if pip show saltup &> /dev/null; then
    echo "Uninstalling pytoolkit package..."
    python -m pip uninstall -y saltup
fi

echo "Cleanup completed!"