#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Description:
# This script runs Stage 4: Model Evaluation.
# It invokes the main Python script with the specific configuration for evaluation.

echo "🚀 Starting Stage 4: Model Evaluation..."

# Run the evaluation stage using Hydra to manage configuration
uv run python src/main.py --config-name=stage_4_eval

echo "✅ Stage 4 finished successfully."