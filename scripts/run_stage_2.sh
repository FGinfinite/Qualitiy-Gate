#!/bin/bash

# This script runs the data selection stage (Stage 2).
# It uses the Hydra configuration defined in `configs/stage_2_selection.yaml`.

# Ensure the script is run from the project root directory
cd "$(dirname "$0")/.."

# Run the selection script using Hydra
uv run python src/stages/selection.py --config-name=stage_2_selection