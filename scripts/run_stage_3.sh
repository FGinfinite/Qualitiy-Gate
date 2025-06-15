#!/bin/bash
# Description: This script runs the third stage of the project: fine-tuning the target model.
# Usage: ./scripts/run_stage_3.sh

# If CUDA_VISIBLE_DEVICES is not set, default to a single GPU.
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  echo "Warning: CUDA_VISIBLE_DEVICES is not set. Defaulting to 1 GPU."
  NUM_GPUS=1
else
  # Count the number of GPUs by counting the commas and adding 1, or counting words if no commas.
  if [[ "$CUDA_VISIBLE_DEVICES" == *","* ]]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr -cd ',' | wc -c)
    NUM_GPUS=$((NUM_GPUS + 1))
  else
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | wc -w)
  fi
fi

echo "Found $NUM_GPUS GPUs. Launching training..."

# Generate a unique timestamped directory for the run
TIMESTAMP=$(date +"%Y-%m-%d/%H-%M-%S")
OUTPUT_DIR="outputs/$TIMESTAMP/stage_3_finetune"
echo "Output directory: $OUTPUT_DIR"

# Launch the training process using Accelerate.
# The --num_processes argument is now set dynamically based on CUDA_VISIBLE_DEVICES.
# We override hydra.run.dir to ensure all processes use the same output directory.
accelerate launch \
  --config_file configs/accelerate_config_ddp.yaml \
  --num_processes=$NUM_GPUS \
  src/main.py --config-name=stage_3_finetune hydra.run.dir=$OUTPUT_DIR