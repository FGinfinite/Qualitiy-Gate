#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the project root directory
# This allows the script to be run from anywhere
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Project Root: $PROJECT_ROOT"
echo "Starting Stage 3: Target Model Fine-tuning..."

# Launch the training process using Accelerate
# We pass the config name for Hydra to pick up the correct file.
# The `main` function in `src/main.py` will then route to the `finetune_target_model` function.
cd "$PROJECT_ROOT" && accelerate launch --config_file "configs/accelerate_config.yaml" \
    -m src.main \
    --config-name=stage_3_finetune

echo "Stage 3 script finished."