#!/bin/bash

# This script runs the main python script to test hydra config loading.

# Ensure the script is run from the project root
if [ ! -f "pyproject.toml" ]; then
    echo "Please run this script from the project root directory."
    exit 1
fi

# Run the python script
uv run python src/main.py