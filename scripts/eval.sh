#!/bin/bash

accelerate launch -m lm_eval --model hf \
    --model_args "pretrained=meta-llama/Llama-2-7b-hf,peft=outputs/stage_3_finetune/2025-08-09/19-58-38-batch=128_lr=2e-05_tag=LESS/checkpoint-424" \
    --tasks mmlu \
    --batch_size auto \
    --output_path outputs/stage_4_eval