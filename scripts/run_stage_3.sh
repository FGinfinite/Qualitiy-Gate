#!/bin/bash
# Description: This script runs the third stage of the project: fine-tuning Llama-2-7B with LoRA.
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

echo "Found $NUM_GPUS GPUs. Launching fine-tuning..."

# Generate a random port between 29500 and 29599
MAIN_PORT=$((RANDOM % 100 + 29500))
echo "Using port $MAIN_PORT"

# Launch the training process using Accelerate with appropriate configuration.
# The --num_processes argument is now set dynamically based on CUDA_VISIBLE_DEVICES.
# Use different configurations for single GPU vs multi-GPU setups.

if [ $NUM_GPUS -eq 1 ]; then
  echo "Using single GPU configuration (no FSDP)"
  CONFIG_FILE="configs/accelerate_config/SINGLE.yaml"
else
  echo "Using multi-GPU FSDP configuration"
  CONFIG_FILE="configs/accelerate_config/FSDP.yaml"
fi

.venv/bin/accelerate launch \
  --config_file $CONFIG_FILE \
  --num_processes=$NUM_GPUS \
  --main_process_port=$MAIN_PORT \
  src/main.py --config-name=stage_3_finetune "$@"