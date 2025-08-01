# Select-MoE 项目执行流程

本文档详细介绍了 Select-MoE 项目的完整执行流程，包括各阶段使用的脚本、配置文件和执行命令示例。

## 概述

Select-MoE 项目包含两个主要部分：
1. **模型转换阶段**：将 OLMoE 预训练模型转换为 Select-MoE 格式
2. **四阶段训练流程**：预热训练 → 数据选择 → 目标模型微调 → 性能评估

---

## 环境准备

本项目使用`uv`作为包管理工具，执行以下命令以完成同步：
1. `uv`安装：`wget -qO- https://astral.sh/uv/install.sh | sh`
2. 环境同步：`uv sync`
3. 修改源：`./tools/chsrc set uv`

---

## 模型转换阶段

### 1. 模型转换

**脚本位置**: `scripts/convert_olmoe_to_select_moe.py`

**功能**: 将 OLMoE 预训练模型转换为 Select-MoE 格式，添加垃圾桶专家机制。

**主要参数**:
- `--model`: 源模型名称 (默认: `allenai/OLMoE-1B-7B-0125`)
- `--save-path`: 转换后模型保存路径
- `--device`: 运行设备 (如 `cuda:0`, `cpu`)
- `--trash-can-init-std`: 垃圾桶专家初始化标准差 (默认: 0.02)
- `--constraint-loss-weight`: 约束损失权重 (默认: 0.01)
- `--seed`: 随机种子 (默认: 42)

**执行示例**:
```bash
# 基本转换（推荐使用 bfloat16）
python scripts/convert_olmoe_to_select_moe.py \
    --device cpu \
    --save-path ./converted_models/select_moe_converted_OLMoE-1B-7B-0125

# 自定义垃圾桶专家参数
python scripts/convert_olmoe_to_select_moe.py \
    --device cpu \
    --save-path ./converted_models/my_select_moe \
    --trash-can-init-std 0.01 \
    --constraint-loss-weight 0.02
```

### 2. 转换模型验证（可选）

**脚本位置**: `scripts/compare_converted_model.py`

**功能**: 对比转换前后的模型权重，验证转换正确性。

**主要参数**:
- `--converted-model`: 转换后的模型路径
- `--original-model`: 原始模型名称 (默认: `allenai/OLMoE-1B-7B-0125`)
- `--device`: 运行设备
- `--dtype`: 数据类型 (`float32`, `bfloat16`, `float16`)
- `--memory-efficient`: 启用内存效率模式
- `--tolerance`: 数值比较容差 (默认: 1e-6)

**执行示例**:
```bash
# 内存效率模式 + bfloat16（推荐）
python scripts/compare_converted_model.py \
    --converted-model ./converted_models/select_moe_converted_OLMoE-1B-7B-0125 \
    --device cuda:0 \
    --dtype bfloat16 \
    --memory-efficient

# 详细对比模式
python scripts/compare_converted_model.py \
    --converted-model ./converted_models/my_select_moe \
    --device cuda:0 \
    --tolerance 1e-8
```

---

## 四阶段训练流程

### 阶段一：预热训练 Select-MoE 路由权重

**脚本位置**: `scripts/run_stage_1.sh`

**配置文件**: `configs/stage_1_pretrain.yaml`

**功能**: 对小型 Select-MoE 模型进行全参数微调，训练 Router 权重学习数据质量判别。

**关键配置参数**:
- `selector_model.path`: Select-MoE 模型路径
- `dataset.dataset_names`: 训练数据集列表 (`cot`, `dolly`, `flan_v2`, `oasst1`)
- `dataset.subset_ratio`: 数据子集比例 (默认: 0.05)
- `training.batch_size`: 批次大小 (默认: 4)
- `training.learning_rate`: 学习率 (默认: 1e-4)
- `training.epochs`: 训练轮数 (默认: 4)
- `training.peft_mode`: 训练模式 (`full_rank` 或 `lora`)

**执行示例**:
```bash
# 基本执行
CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage_1.sh

# 多GPU训练
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_stage_1.sh

# 自定义超参数
CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage_1.sh \
    training.learning_rate=5e-5 \
    training.batch_size=8 \

# 使用LoRA训练
CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage_1.sh \
    training.peft_mode=lora \
    training.lora.r=32
```

**输出**: 全参数权重保存在 `outputs/stage_1_pretrain/YYYY-MM-DD/HH-MM-SS/full_rank_weights.pt`

### 阶段二：数据选择

**脚本位置**: `scripts/run_stage_2.sh`

**配置文件**: `configs/stage_2_selection.yaml`

**功能**: 使用预热的 Select-MoE 模型为训练数据打分，筛选高质量数据子集。

**关键配置参数**:
- `model_checkpoint_path`: 阶段一输出的权重文件路径
- `selection_percentage`: 数据选择比例 (默认: 0.05，即选择前5%的数据)
- `dataset.subset_ratio`: 用于评分的数据比例 (默认: 1.0，使用全部数据)
- `data_process.batch_size`: 推理批次大小 (默认: 16)

**执行示例**:
```bash
# 基本执行
CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage_2.sh \
    model_checkpoint_path=outputs/stage_1_pretrain/2025-07-16/01-57-27/full_rank_weights.pt

# 调整选择比例
CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage_2.sh \
    model_checkpoint_path=outputs/stage_1_pretrain/2025-07-16/01-57-27/full_rank_weights.pt \
    selection_percentage=0.05

# 使用数据子集进行快速测试
CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage_2.sh \
    model_checkpoint_path=outputs/stage_1_pretrain/2025-07-16/01-57-27/full_rank_weights.pt \
    dataset.subset_ratio=0.1 \
    selection_percentage=0.05
```

**输出**: 筛选后的数据保存在 `outputs/stage_2_selection/YYYY-MM-DD/HH-MM-SS/selected_data.jsonl`

### 阶段三：目标模型微调

**脚本位置**: `scripts/run_stage_3.sh`

**配置文件**: `configs/stage_3_finetune.yaml`

**功能**: 使用筛选的高质量数据对 Llama-2-7B 模型进行 LoRA 微调。

**关键配置参数**:
- `model.name`: 目标模型名称 (默认: `meta-llama/Llama-2-7b-hf`)
- `dataset.data_path`: 阶段二输出的数据文件路径
- `training.batch_size`: 总批次大小 (默认: 128)
- `training.learning_rate`: 学习率 (默认: 2e-5)
- `training.epochs`: 训练轮数 (默认: 4)
- `training.lora.r`: LoRA 秩 (默认: 128)
- `training.lora.lora_alpha`: LoRA alpha (默认: 512)

**执行示例**:
```bash
# 基本执行
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_stage_3.sh \
    dataset.data_path=outputs/stage_2_selection/2025-07-17/04-49-54/selected_data.jsonl

# 调整LoRA参数
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_stage_3.sh \
    dataset.data_path=outputs/stage_2_selection/2025-07-17/04-49-54/selected_data.jsonl \
    training.lora.r=64 \
    training.lora.lora_alpha=256

# 自定义训练超参数
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_stage_3.sh \
    dataset.data_path=outputs/stage_2_selection/2025-07-17/04-49-54/selected_data.jsonl \
    training.learning_rate=1e-5 \
    training.batch_size=64 \
    training.epochs=2
```

**输出**: LoRA 适配器保存在 `outputs/stage_3_finetune/YYYY-MM-DD/HH-MM-SS/checkpoint-XXXX/`

### 阶段四：模型评估

**脚本位置**: `scripts/eval.sh`

**功能**: 使用 `lm-eval` 框架评估微调后模型在标准基准测试上的性能。

**关键参数**:
- `--model_args pretrained`: 基础模型名称
- `--model_args peft`: LoRA 适配器路径
- `--tasks`: 评估任务 (如 `mmlu`, `hellaswag`, `arc_easy` 等)
- `--batch_size`: 评估批次大小
- `--output_path`: 结果输出路径

**执行示例**:
```bash
# MMLU评估
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=outputs/stage_3_finetune/2025-07-18/01-01-10/checkpoint-1804 \
    --tasks mmlu \
    --batch_size auto \
    --output_path outputs/stage_4_eval

# 多任务评估
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=outputs/stage_3_finetune/2025-07-18/01-01-10/checkpoint-1804 \
    --tasks mmlu,hellaswag,arc_easy,arc_challenge \
    --batch_size auto \
    --output_path outputs/stage_4_eval
```

---

## 注意事项

### 1. 环境变量设置
**重要**: 每个脚本执行前都需要设置 `CUDA_VISIBLE_DEVICES` 环境变量来指定使用的GPU：
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用GPU 0,1,2,3
```

### 2. 参数覆写机制
项目使用 Hydra 配置管理，支持命令行参数覆写：
```bash
# 基本语法
bash scripts/script_name.sh key1=value1 key2.subkey=value2

# 示例
bash scripts/run_stage_1.sh training.learning_rate=1e-5 dataset.subset_ratio=0.1
bash scripts/run_stage_2.sh selection_percentage=0.1 data_process.batch_size=32
```

### 3. 路径配置
- 模型转换脚本的参数直接在命令行指定，不使用YAML配置
- 训练流程脚本需要注意各阶段之间的路径依赖关系
- 建议使用绝对路径或相对于项目根目录的路径

### 4. 内存管理
- 阶段一：Select-MoE全参数训练，建议至少16GB GPU内存
- 阶段二：数据选择推理，内存需求较小
- 阶段三：Llama-2-7B LoRA训练，建议24GB以上GPU内存
- 如遇显存不足，可调整 `batch_size` 或使用梯度累积

### 5. 数据集准备
确保以下数据集已正确放置：
- 训练数据：`dataset/train/processed/` 目录下的 `cot`, `dolly`, `flan_v2`, `oasst1` 数据集。
```bash
wget https://hf-mirror.com/datasets/princeton-nlp/less_data/resolve/main/less-data.zip
unzip less-data.zip
```


### 6. 评估准备
为确保 `lm-eval` 正常运行，建议提前下载MMLU数据集：
```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download hails/mmlu_no_train --repo-type dataset 
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download cais/mmlu --repo-type dataset
```

### 7. 模型准备
确保OLMoE模型和Llama-2模型是已经准备好的，建议可以提前下载：
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download allenai/OLMoE-1B-7B-0125
huggingface-cli download meta-llama/Llama-2-7b-hf
```