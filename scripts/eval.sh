#!/bin/bash

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=outputs/stage_3_finetune/2025-07-18/01-01-10/checkpoint-1804 \
    --tasks mmlu \
    --batch_size auto \
    --output_path outputs/stage_4_eval