# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Select-MoE is a novel data selection framework using Mixture-of-Experts (MoE) models. The project implements a four-stage pipeline:

1. **Stage 1 (Pretrain)**: Warm up a Select-MoE router to learn data quality discrimination
2. **Stage 2 (Selection)**: Use the trained router to score and filter training data
3. **Stage 3 (Finetune)**: Fine-tune target models (Llama-2-7B) with selected data using LoRA
4. **Stage 4 (Evaluate)**: Evaluate model performance using lm-eval

## Key Architecture Components

### Select-MoE Model (`src/models/select_moe.py`)
- **Two-tier Routing Architecture**: Implements quality gate + MoE + trash expert parallel processing
- **Quality Gate**: First-tier binary classifier for data quality assessment (good vs bad)
- **MoE Integration**: Uses standard OlmoeSparseMoeBlock for expert routing
- **Trash Expert**: Configurable output modes (zero, noise, custom) for low-quality data
- **Quality Classification Loss**: Uses sigmoid(good_ratio) loss instead of Beta distribution constraint
- **Router Output Format**: Returns dictionary with both quality_logits and moe_logits for data selection

### Core Pipeline (`src/main.py`)
- Hydra-based configuration management with stage-specific configs
- Four distinct execution stages controlled by config selection
- Integrated with accelerate for distributed training

### Data Processing (`src/data/`)
- Dataset loaders for training and evaluation data
- Support for multiple datasets: CoT, Dolly, FLAN-v2, OASST1

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync

# Activate virtual environment
source .venv/bin/activate

# Install tools (optional)
./tools/install.sh
```

### Model Conversion
```bash
# Convert OLMoE to Select-MoE format
python scripts/convert_olmoe_to_select_moe.py --save-path ./converted_models/select_moe_converted_OLMoE-1B-7B-0125

# Verify conversion
python scripts/compare_converted_model.py --converted-model ./converted_models/select_moe_converted_OLMoE-1B-7B-0125 --dtype bfloat16
```

### Training Pipeline
```bash
# Stage 1: Router pretraining (set CUDA_VISIBLE_DEVICES first)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_stage_1.sh

# Stage 2: Data selection
CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage_2.sh model_checkpoint_path=outputs/stage_1_pretrain/YYYY-MM-DD/HH-MM-SS/full_rank_weights.pt

# Stage 3: Target model fine-tuning
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_stage_3.sh dataset.data_path=outputs/stage_2_selection/YYYY-MM-DD/HH-MM-SS/selected_data.jsonl

# Stage 4: Model evaluation
bash scripts/eval.sh
```

### Configuration Override
All stages support Hydra parameter overrides:
```bash
# Override learning rate and batch size
bash scripts/run_stage_1.sh training.learning_rate=5e-5 training.batch_size=8

# Override selection percentage
bash scripts/run_stage_2.sh selection_percentage=0.1

# Override LoRA parameters
bash scripts/run_stage_3.sh training.lora.r=64 training.lora.lora_alpha=128
```

### Evaluation Commands
```bash
# MMLU evaluation using lm-eval
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=outputs/stage_3_finetune/YYYY-MM-DD/HH-MM-SS/checkpoint-XXXX \
    --tasks mmlu \
    --batch_size auto \
    --output_path outputs/stage_4_eval

# Multi-task evaluation
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=outputs/stage_3_finetune/YYYY-MM-DD/HH-MM-SS/checkpoint-XXXX \
    --tasks mmlu,hellaswag,arc_easy,arc_challenge \
    --batch_size auto \
    --output_path outputs/stage_4_eval
```

## Important Configuration Files

- `configs/stage_1_pretrain.yaml` - Router pretraining configuration
- `configs/stage_2_selection.yaml` - Data selection configuration  
- `configs/stage_3_finetune.yaml` - Target model fine-tuning configuration
- `configs/stage_4_evaluate.yaml` - Model evaluation configuration
- `configs/accelerate_config_*.yaml` - Distributed training configurations

## Key Parameters

### Stage 1 (Pretrain)
- `training.peft_mode`: Training mode (`lora` or `full_rank`)
  - `lora`: 低秩微调，使用LoRA适配器训练指定模块
  - `full_rank`: 路由参数全秩微调，只训练质量门和MoE门参数
- `training.learning_rate`: Learning rate (default: 1e-4)
- `dataset.subset_ratio`: Training data proportion (default: 0.05)
- `training.quality_loss_weight`: Weight for quality classification loss (default: 0.01)
- `training.quality_gate_init_mean/std`: Quality gate initialization parameters
- `training.trash_expert_mode`: Trash expert behavior ("zero", "noise", "custom")
- `training.enable_load_balancing`: Enable MoE load balancing loss

### Stage 2 (Selection)
- `selection_percentage`: Data selection ratio (default: 0.05)
- `model_checkpoint_path`: Path to Stage 1 output weights

### Stage 3 (Finetune)
- `training.lora.r`: LoRA rank (default: 128)
- `training.learning_rate`: Learning rate (default: 2e-5)
- `dataset.data_path`: Path to Stage 2 selected data

## Memory Requirements

- **Stage 1**: 
  - `lora` mode: LoRA微调需要 ~8GB GPU memory
  - `full_rank` mode: 路由参数全秩微调需要 ~16GB GPU memory
- **Stage 2**: Data selection inference has minimal memory requirements
- **Stage 3**: Llama-2-7B LoRA training requires ~24GB GPU memory
- **Multi-GPU**: Use FSDP configurations for distributed training

## Path Dependencies

Ensure correct path dependencies between stages:
- Stage 2 requires Stage 1's `full_rank_weights.pt`
- Stage 3 requires Stage 2's `selected_data.jsonl`
- Stage 4 requires Stage 3's LoRA checkpoint

Always verify output paths before proceeding to the next stage.

## Programmatic Usage

```python
from src.models.select_moe import SelectMoeForCausalLM, register_select_moe

# Register Select-MoE (required before loading)
register_select_moe()

# Load converted model
model = SelectMoeForCausalLM.from_pretrained("./converted_models/select_moe_converted_OLMoE-1B-7B-0125")

# Training mode (enable router logits)
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_router_logits=True)

# New architecture returns dictionary format router logits
for layer_output in outputs.router_logits:
    quality_logits = layer_output["quality_logits"]  # Shape: [batch, seq_len, 2]
    moe_logits = layer_output["moe_logits"]          # Shape: [batch*seq_len, num_experts]

# Total loss includes language modeling + load balancing + quality classification loss
total_loss = outputs.loss
```