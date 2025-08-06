# Select-MoE: 基于混合专家模型的数据选择策略

## 项目简介

本项目是一个创新的数据选择实验，旨在探索一种利用混合专家模型（Mixture-of-Experts, MoE）进行高效数据筛选的方法。核心思想是：

1. **预热阶段**: 对小型 Select-MoE 模型的 Router（路由器）进行预热微调，使其具备数据质量判别能力
2. **选择阶段**: 利用预热的 Router 为大规模数据集打分，筛选高质量数据子集  
3. **微调阶段**: 使用筛选的高质量数据微调大规模目标模型
4. **评估阶段**: 评估数据选择策略的最终效果

## ✨ 核心创新

### Select-MoE 架构特性
- **两层路由架构**: 实现质量门 + MoE + 垃圾专家的并行处理结构
- **质量门机制**: 一级路由进行二元分类，判断数据质量（好 vs 坏）
- **标准MoE集成**: 使用标准OlmoeSparseMoeBlock进行专家路由，保持原始权重不变
- **可配置垃圾专家**: 支持多种输出模式（零值、噪声、自定义）处理低质量数据
- **质量分类损失**: 使用sigmoid(good_ratio)损失函数替代Beta分布约束
- **HuggingFace 兼容**: 支持标准的 `from_pretrained()` 加载和生态工具

### 模型对比

| 特性 | 原始 OLMoE | Select-MoE |
|------|------------|------------|
| MoE专家数 | 64 | 64 (保持不变) |
| 质量门 | 无 | 2输出 (好/坏分类) |
| 垃圾专家 | 无 | 1个 (可配置输出模式) |
| 路由结构 | 单层MoE | 两层 (质量门 + MoE) |
| 预训练权重 | - | MoE权重完全保持 |
| 数据质量选择 | 无 | 基于质量门输出 |

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

# 新架构返回字典格式的路由输出
for layer_output in outputs.router_logits:
    quality_logits = layer_output["quality_logits"]  # 形状: [batch, seq_len, 2]
    moe_logits = layer_output["moe_logits"]          # 形状: [batch*seq_len, num_experts]

# 损失包含语言建模 + 负载均衡 + 质量分类损失
total_loss = outputs.loss
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
- `training.quality_loss_weight`: 质量分类损失权重 (默认: 0.01)
- `training.quality_gate_init_mean/std`: 质量门初始化参数
- `training.trash_expert_mode`: 垃圾专家模式 ("zero", "noise", "custom")
- `training.enable_load_balancing`: 启用MoE负载均衡损失

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

### 两层路由架构设计

Select-MoE采用创新的两层路由架构，实现更精确的数据质量判别：

**第一层：质量门 (Quality Gate)**
- 进行二元分类：好数据 vs 坏数据
- 输出good_ratio和bad_ratio，控制后续处理权重
- 使用小型全连接网络，计算效率高

**第二层：并行处理**
```
y = good_ratio * y_normal + bad_ratio * y_trash
```
- **正常路径**: 通过标准MoE处理，保持原始OLMoE能力
- **垃圾路径**: 通过垃圾专家处理，可配置输出模式
- **加权组合**: 基于质量门输出动态组合两路结果

### 质量分类损失

新架构使用简洁的质量分类损失：
```
L_quality = sigmoid(good_ratio).mean()
```

**损失特性**：
- **计算粒度**: 对每个token在每个层分别计算
- **优化目标**: 鼓励模型学习区分数据质量
- **数值稳定**: 使用sigmoid激活，避免数值问题

**总损失构成**：
```
L_total = L_language_modeling + w_load_balancing * L_load_balancing + w_quality * L_quality
```

### 垃圾专家机制

**输出模式**：
- **zero模式**: 输出零向量，提供最小干扰
- **noise模式**: 输出与输入同分布的噪声
- **custom模式**: 支持自定义行为扩展

**设计优势**：
1. **模块化**: 垃圾专家独立于MoE，易于调试和优化
2. **可配置**: 根据任务需求选择合适的输出模式
3. **高效**: 避免了复杂的专家扩展和权重管理

## 🚧 开发计划

1. **垃圾桶专家优化**: 实现更智能的垃圾桶专家初始化和行为策略
2. **多任务适配**: 扩展支持更多下游任务的数据选择
3. **效率优化**: 优化训练和推理效率，支持更大规模模型
4. **评估扩展**: 增加更多评估指标和基准测试

