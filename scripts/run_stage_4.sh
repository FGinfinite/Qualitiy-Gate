#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Run the evaluation script using the specified config
uv run python src/stages/evaluation.py --config-name=stage_4_evaluate $@