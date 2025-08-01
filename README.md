# Select-MoE: 基于混合专家模型的数据选择策略

## 项目简介

本项目是一个创新的数据选择实验，旨在探索一种利用混合专家模型（Mixture-of-Experts, MoE）进行高效数据筛选的方法。核心思想是：

1. **预热阶段**: 对小型 Select-MoE 模型的 Router（路由器）进行预热微调，使其具备数据质量判别能力
2. **选择阶段**: 利用预热的 Router 为大规模数据集打分，筛选高质量数据子集  
3. **微调阶段**: 使用筛选的高质量数据微调大规模目标模型
4. **评估阶段**: 评估数据选择策略的最终效果

## ✨ 核心创新

### Select-MoE 架构特性
- **垃圾桶专家机制**: 动态添加"垃圾桶专家"，对低质量数据提供负向激励
- **权重保持**: 完美保持原始 OLMoE 预训练权重，仅新增垃圾桶专家维度
- **约束损失**: 基于 Beta 分布的自定义约束损失，引导 Router 学习数据质量区分
- **HuggingFace 兼容**: 支持标准的 `from_pretrained()` 加载和生态工具

### 模型对比

| 特性 | 原始 OLMoE | Select-MoE |
|------|------------|------------|
| 原始专家数 | 64 | 64 (保持不变) |
| 垃圾桶专家数 | 0 | 8 (= top_k) |
| Gate 输出维度 | [64, hidden_size] | [72, hidden_size] |
| 预训练权重 | - | 完全保持 |
| 数据质量选择 | 无 | 动态路由到垃圾桶 |

## 🛠️ 技术栈

- **包管理**: [`uv`](https://docs.astral.sh/uv/) - 快速 Python 包管理器
- **深度学习**: [`torch`](https://pytorch.org/) 2.6.0, [`transformers`](https://github.com/huggingface/transformers)
- **模型微调**: [`peft`](https://github.com/huggingface/peft) - LoRA 等高效微调方法
- **分布式训练**: [`accelerate`](https://github.com/huggingface/accelerate) - 多GPU 训练支持
- **配置管理**: [`hydra`](https://github.com/facebookresearch/hydra) - 灵活的配置系统
- **模型评估**: [`lm-eval`](https://github.com/EleutherAI/lm-evaluation-harness) - 标准评测框架

## 🚀 快速开始

### 环境准备

```bash
# 1. 安装 uv 包管理器
wget -qO- https://astral.sh/uv/install.sh | sh

# 2. 同步项目依赖
uv sync

# 3. (可选) 配置国内镜像源
./tools/chsrc set uv

# 4. 激活虚拟环境
source .venv/bin/activate
```

### 数据准备

```bash
# 下载训练数据集
wget https://hf-mirror.com/datasets/princeton-nlp/less_data/resolve/main/less-data.zip
unzip less-data.zip

# 下载评估数据集
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download hails/mmlu_no_train --repo-type dataset 
huggingface-cli download cais/mmlu --repo-type dataset

# 下载预训练模型
huggingface-cli download allenai/OLMoE-1B-7B-0125
huggingface-cli download meta-llama/Llama-2-7b-hf
```

## 📋 完整执行流程

本项目包含两个主要部分：**模型转换** 和 **四阶段训练流程**。

### 步骤 0: 模型转换

首先需要将 OLMoE 预训练模型转换为 Select-MoE 格式：

```bash
# 基本转换
python scripts/convert_olmoe_to_select_moe.py \
    --save-path ./converted_models/select_moe_converted_OLMoE-1B-7B-0125

# (可选) 验证转换结果
python scripts/compare_converted_model.py \
    --converted-model ./converted_models/select_moe_converted_OLMoE-1B-7B-0125 \
    --dtype bfloat16 \
```

### 步骤 1: 预热训练 Select-MoE 路由权重

训练 Select-MoE 模型的 Router，使其学习数据质量判别：

```bash
# 单 GPU 训练
CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage_1.sh

# 多 GPU 训练
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_stage_1.sh

# 自定义参数训练
CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage_1.sh \
    training.learning_rate=5e-5 \
    training.batch_size=8 \
    dataset.subset_ratio=0.1
```

**输出**: 权重文件保存在 `outputs/stage_1_pretrain/YYYY-MM-DD/HH-MM-SS/full_rank_weights.pt`

### 步骤 2: 数据选择

使用预热的 Select-MoE 模型为训练数据打分并筛选：

```bash
# 使用阶段1的输出进行数据选择
CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage_2.sh \
    model_checkpoint_path=outputs/stage_1_pretrain/2025-07-16/01-57-27/full_rank_weights.pt

# 调整选择比例（选择前10%的数据）
CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage_2.sh \
    model_checkpoint_path=outputs/stage_1_pretrain/2025-07-16/01-57-27/full_rank_weights.pt \
    selection_percentage=0.1
```

**输出**: 筛选数据保存在 `outputs/stage_2_selection/YYYY-MM-DD/HH-MM-SS/selected_data.jsonl`

### 步骤 3: 目标模型微调

使用筛选的数据对 Llama-2-7B 进行 LoRA 微调：

```bash
# 基本微调
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_stage_3.sh \
    dataset.data_path=outputs/stage_2_selection/2025-07-17/04-49-54/selected_data.jsonl

# 自定义 LoRA 参数
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_stage_3.sh \
    dataset.data_path=outputs/stage_2_selection/2025-07-17/04-49-54/selected_data.jsonl \
    training.lora.r=64 \
    training.lora.lora_alpha=256
```

**输出**: LoRA 适配器保存在 `outputs/stage_3_finetune/YYYY-MM-DD/HH-MM-SS/checkpoint-XXXX/`

### 步骤 4: 模型评估

使用 `lm-eval` 评估微调后模型的性能：

```bash
# MMLU 评估
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

## 🔧 高级用法

### 程序化使用 Select-MoE 模型

```python
from src.models.select_moe import SelectMoeForCausalLM, register_select_moe

# 注册 Select-MoE（必须在加载前执行）
register_select_moe()

# 加载转换后的模型
model = SelectMoeForCausalLM.from_pretrained("./converted_models/select_moe_converted_OLMoE-1B-7B-0125")

# 训练时开启 router logits
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    output_router_logits=True  # 重要：训练时必须为True
)

# 损失包含所有组件
total_loss = outputs.loss  # 语言建模损失 + 负载均衡损失 + 约束损失
```

### 参数覆写机制

项目使用 Hydra 配置管理，支持灵活的命令行参数覆写：

```bash
# 基本语法
bash scripts/script_name.sh key1=value1 key2.subkey=value2

# 实际示例
bash scripts/run_stage_1.sh training.learning_rate=1e-5 dataset.subset_ratio=0.1
bash scripts/run_stage_2.sh selection_percentage=0.1 data_process.batch_size=32
bash scripts/run_stage_3.sh training.lora.r=128 training.batch_size=64
```

## ⚙️ 配置说明

### 主要配置文件

- `configs/stage_1_pretrain.yaml` - 阶段1预热训练配置
- `configs/stage_2_selection.yaml` - 阶段2数据选择配置  
- `configs/stage_3_finetune.yaml` - 阶段3模型微调配置
- `configs/stage_4_evaluate.yaml` - 阶段4模型评估配置

### 关键参数说明

**阶段1 (预热训练)**:
- `training.peft_mode`: 训练模式 (`full_rank` 或 `lora`)
- `training.learning_rate`: 学习率 (默认: 1e-4)
- `dataset.subset_ratio`: 训练数据比例 (默认: 0.05)

**阶段2 (数据选择)**:
- `selection_percentage`: 数据选择比例 (默认: 0.05)
- `model_checkpoint_path`: 阶段1输出的权重路径

**阶段3 (模型微调)**:
- `training.lora.r`: LoRA 秩 (默认: 128)
- `training.learning_rate`: 学习率 (默认: 2e-5)
- `dataset.data_path`: 阶段2输出的数据路径

## 💡 重要提示

### 环境变量
每个脚本执行前都需要设置 `CUDA_VISIBLE_DEVICES`：
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 指定使用的GPU
```

### 内存需求
- **阶段1**: Select-MoE 全参数训练，建议至少 16GB GPU 内存
- **阶段2**: 数据选择推理，内存需求较小
- **阶段3**: Llama-2-7B LoRA 训练，建议 24GB 以上 GPU 内存

### 路径依赖
注意各阶段之间的路径依赖关系，确保使用正确的输入路径：
- 阶段2 需要阶段1 的 `full_rank_weights.pt`
- 阶段3 需要阶段2 的 `selected_data.jsonl`  
- 阶段4 需要阶段3 的 LoRA 检查点

## 📖 详细文档

更多详细的执行说明和参数配置，请参考 [`docs.md`](docs.md) 文件。

## 🔬 技术原理

为了引导Router（路由器）学习区分高质量和低质量数据，我们引入了一种特殊的约束损失函数。该损失函数的核心思想是：对于一个给定的输入，我们期望Router的Top-K专家选择概率之和（`ratio`）接近于1，而"垃圾桶"专家的选择概率之和（`1 - ratio`）接近于0。

这种机制通过一个定制的损失函数来实现，该函数受到Beta分布的启发，旨在将 `ratio` 值推向1。

### 约束损失设计

约束损失 `L_constraint` 的计算方式如下：

```
L_constraint = -((α - 1) * log(ratio) + (β - 1) * log(1 - ratio))
```

其中：
- `ratio` 是 Top-K 专家的选择概率之和
- `α` (`trash_can_loss_alpha`) 和 `β` (`trash_can_loss_beta`) 是控制损失函数形态的参数
- 在实现中，`α` 固定为1，专注于 `β` 的影响

总损失为：
```
L_total = L_ce + w_constraint * L_constraint
```

### 核心参数

- **`constraint_loss_weight`**: 约束损失在总损失中的权重，控制数据质量筛选的重视程度
- **`trash_can_loss_beta`**: 控制对"垃圾桶"专家激活的惩罚力度

### "垃圾桶"专家机制

1. **动态数量**: 垃圾桶专家数量等于 top-k 激活数量
2. **行为特点**: 输出全零向量，提供负向激励
3. **初始化策略**: 通过正态分布初始化，参数可配置

## 🚧 开发计划

1. **垃圾桶专家优化**: 实现更智能的垃圾桶专家初始化和行为策略
2. **多任务适配**: 扩展支持更多下游任务的数据选择
3. **效率优化**: 优化训练和推理效率，支持更大规模模型
4. **评估扩展**: 增加更多评估指标和基准测试

