#!/bin/bash
# Description: This script runs the first stage of the project: pre-training the selector model.
# Usage: ./scripts/run_stage_1.sh

export CUDA_VISIBLE_DEVICES=6

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

# Generate a random port number between 20000 and 29999
MAIN_PORT=$((RANDOM % 100 + 29500))
echo "Using port $MAIN_PORT for the main process."

# Launch the training process using Accelerate.
# The --num_processes argument is now set dynamically based on CUDA_VISIBLE_DEVICES.
# All arguments passed to this script ("$@") are forwarded to the python script,
# allowing for hydra multirun capabilities.
# Use FSDP for multi-GPU training when more than 1 GPU is available

echo "Using single GPU training..."
.venv/bin/accelerate launch \
  --config_file configs/accelerate_config_ddp.yaml \
  --num_processes=$NUM_GPUS \
  --main_process_port=$MAIN_PORT \
  src/main.py --config-name=stage_1_pretrain "$@"
