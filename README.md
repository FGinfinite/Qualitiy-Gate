# MyGO: using MoE to select your Good and Omnigenous data

## 项目简介

本项目 (MyGO) 是一个创新的数据选择实验，旨在探索一种利用混合专家模型（Mixture-of-Experts, MoE）进行高效数据筛选的方法。核心思想是：

1. **预热阶段**: 对小型 Select-MoE 模型的 Router（路由器）进行预热微调，使其具备数据质量判别能力
2. **选择阶段**: 利用预热的 Router 为大规模数据集打分，通过GPU加速聚类算法筛选多样化高质量数据子集  
3. **微调阶段**: 使用筛选的高质量数据微调大规模目标模型
4. **评估阶段**: 评估数据选择策略的最终效果

## ✨ 核心创新

### Select-MoE 架构特性 - **最新更新**
- **两层路由架构**: 实现质量门 + MoE + 垃圾专家的并行处理结构
- **简化质量门 (NEW)**: 一级路由输出单个质量分数，通过sigmoid得到good_ratio
  - **架构简化**: 从2类分类改为单分数输出，`good_ratio = sigmoid(quality_score)`
  - **更好梯度流**: 直接对原始分数应用sigmoid，避免softmax的数值问题
  - **动态比例**: `bad_ratio = 1 - good_ratio`，确保完美互补
- **标准MoE集成**: 使用标准OlmoeSparseMoeBlock进行专家路由，保持原始权重不变
- **可配置垃圾专家**: 支持多种输出模式（零值、噪声、自定义）处理低质量数据
- **增强损失函数 (NEW)**: 
  - **可扩展框架**: 支持多种损失类型 (sigmoid, MSE, custom)
  - **填充令牌处理**: 正确处理attention_mask，排除padding tokens
  - **自定义损失**: 支持实验性损失函数，便于调试和优化
  - **NEW: 方案一 & 方案二**: 实现了Beta矩匹配和均值-方差正则化两种高级损失函数
  - **可配置调试**: `quality_loss_debug`参数支持详细的损失计算调试输出
- **聚类选择算法 (NEW)**: GPU加速聚类实现多样化数据选择
  - **K-Means + Elbow Method**: 自动k值选择，确保最优聚类数量
  - **HDBSCAN聚类**: 无参数密度聚类，自适应发现簇结构
  - **余弦距离**: 使用MoE logits的语义相似性进行聚类
  - **轮选策略**: 从每个簇中轮流选择高质量数据，保证多样性
  - **GPU加速**: 支持RAPIDS cuML和PyTorch GPU加速，处理大规模数据
- **HuggingFace 兼容**: 支持标准的 `from_pretrained()` 加载和生态工具

### 架构对比

| 特性 | 原始版本 | 最新版本 (当前) |
|------|----------|----------------|
| 质量门输出 | 2类logits | 单个原始分数 |
| 概率计算 | softmax归一化 | sigmoid激活 |
| 损失函数 | sigmoid(softmax_prob) | 直接sigmoid(raw_score) |
| 填充处理 | 无特殊处理 | attention_mask排除padding |
| 损失扩展性 | 固定函数 | 支持自定义损失函数 |
| 调试支持 | 基本 | 丰富的实验框架 |
| 高级损失 | 无 | Beta矩匹配 & 均值-方差正则化 |
| 配置化调试 | 无 | quality_loss_debug参数 |
| 数据选择 | 基础质量筛选 | 聚类+轮选多样化选择 |
| 聚类算法 | 无 | K-Means + HDBSCAN |
| GPU加速 | 无 | RAPIDS cuML + PyTorch |

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
- **聚类算法**: [`scikit-learn`](https://scikit-learn.org/) - K-Means和HDBSCAN聚类算法
- **GPU聚类**: [`RAPIDS cuML`](https://github.com/rapidsai/cuml) - GPU加速聚类 (可选)

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
huggingface-cli download Qwen/Qwen2.5-1.5B
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

### 步骤 2: 聚类数据选择

使用预热的 Select-MoE 模型为训练数据打分并通过聚类算法筛选多样化数据：

```bash
# 使用阶段1的输出进行聚类数据选择
CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage_2.sh \
    model_checkpoint_path=outputs/stage_1_pretrain/2025-07-16/01-57-27/full_rank_weights.pt

# 调整选择比例和聚类方法
CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage_2.sh \
    model_checkpoint_path=outputs/stage_1_pretrain/2025-07-16/01-57-27/full_rank_weights.pt \
    selection_percentage=0.1 \
    clustering_method=hdbscan

# 自定义K-Means参数
CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage_2.sh \
    model_checkpoint_path=outputs/stage_1_pretrain/2025-07-16/01-57-27/full_rank_weights.pt \
    clustering_method=kmeans \
    clustering_params.k_range="[20,80]" \
    clustering_params.max_iters=500

# 自定义HDBSCAN参数
CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage_2.sh \
    model_checkpoint_path=outputs/stage_1_pretrain/2025-07-16/01-57-27/full_rank_weights.pt \
    clustering_method=hdbscan \
    clustering_params.min_cluster_size=150 \
    clustering_params.auto_tune=true
```

**输出**: 聚类筛选数据保存在 `outputs/stage_2_selection/YYYY-MM-DD/HH-MM-SS/selected_data.jsonl`

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

# 新架构返回字典格式的路由输出 (更新后的格式)
for layer_output in outputs.router_logits:
    quality_score = layer_output["quality_score"]   # 形状: [batch, seq_len, 1] - 原始分数
    moe_logits = layer_output["moe_logits"]         # 形状: [batch*seq_len, num_experts]
    
    # 手动计算质量比例 (如果需要)
    good_ratio = torch.sigmoid(quality_score)       # 形状: [batch, seq_len, 1]
    bad_ratio = 1.0 - good_ratio                    # 形状: [batch, seq_len, 1]

# 损失包含语言建模 + 负载均衡 + 质量分类损失
# 新版本自动处理padding tokens
total_loss = outputs.loss

# **NEW**: 配置质量损失类型和调试
model.config.quality_loss_type = "beta_moment_matching"  # 方案一: Beta矩匹配
# model.config.quality_loss_type = "mean_variance_regularization"  # 方案二: 均值-方差正则化
model.config.quality_loss_debug = True  # 启用调试输出

# **NEW**: 配置自定义损失参数
# 方案一参数 (Beta矩匹配)
model.config.beta_target_mean = 0.5
model.config.beta_target_var = 0.05
model.config.w_mean = 1.0
model.config.w_var = 1.0

# 方案二参数 (均值-方差正则化)
model.config.lambda_var = 0.1

# 自定义损失函数示例 (实验性功能，仍然支持)
def my_custom_loss(good_ratio, attention_mask):
    # 你的自定义损失逻辑
    # 返回形状为 (batch_size, seq_len) 的张量
    return torch.pow(good_ratio.squeeze(-1), 2)  # 示例：平方损失

# 可以通过修改 quality_classification_loss 调用来使用自定义损失
```

### 架构变更说明

**重要变更 (2025年最新)**:
- **Router输出格式**: `quality_logits` → `quality_score` (形状从 [batch, seq_len, 2] 变为 [batch, seq_len, 1])
- **损失函数增强**: 支持多种损失类型和自定义损失函数
- **Padding处理**: 自动排除填充token，提高训练质量
- **调试友好**: 丰富的实验框架支持，便于损失函数调试
- **NEW: 高级损失函数**: 实现方案一(Beta矩匹配)和方案二(均值-方差正则化)
- **NEW: 可配置调试**: `quality_loss_debug` 参数支持详细的损失计算调试输出
- **NEW: 参数同步**: 所有训练阶段完全同步新的配置参数

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

**阶段2 (聚类数据选择)**:
- `selection_percentage`: 数据选择比例 (默认: 0.05)
- `model_checkpoint_path`: 阶段1输出的权重路径
- `clustering_method`: 聚类算法 ('kmeans' 或 'hdbscan', 默认: 'kmeans')
- `clustering_params`: 聚类参数配置
  - **K-Means参数**: `auto_k`, `k`, `k_range`, `max_iters`
  - **HDBSCAN参数**: `min_cluster_size`, `min_samples`, `metric`, `use_gpu`, `auto_tune`

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

### 路径依赖
注意各阶段之间的路径依赖关系，确保使用正确的输入路径：
- 阶段2 需要阶段1 的 `full_rank_weights.pt`
- 阶段3 需要阶段2 的 `selected_data.jsonl`  
- 阶段4 需要阶段3 的 LoRA 检查点

## 📖 详细文档

更多详细的执行说明和参数配置，请参考 [`docs.md`](docs.md) 文件。

## 🔬 技术原理

### 聚类-轮选数据选择策略

Select-MoE 采用先进的聚类-轮选策略，确保选择数据的质量和多样性：

**第一步：特征提取**
- 使用预热的 Select-MoE Router 处理数据，获取 MoE logits 作为语义特征
- 将每个样本的 `[序列长度, 专家数]` 特征展平为一维向量
- 构建特征矩阵 `[样本数, 特征维度]` 用于聚类

**第二步：GPU加速聚类**
```python
# K-Means + Elbow Method (自动k值选择)
kmeans = GPUKMeansClustering(device='cuda')
labels = kmeans.find_optimal_k_elbow(features, k_range=[10, 100])

# HDBSCAN (无参数密度聚类)
hdbscan = GPUHDBSCANClustering(device='cuda') 
labels = hdbscan.fit_predict(features, metric='cosine')
```

**第三步：轮选高质量数据**
- 将数据按聚类标签分组到各个簇中
- 在每个簇内按质量分数降序排列
- 轮流从各簇选择最高质量样本，直到达到目标数量
- 确保最终数据集在保证高质量的同时具有语义多样性

**算法优势**:
1. **质量保证**: 基于Router质量评分，确保选择高质量数据
2. **多样性保证**: 聚类确保覆盖不同语义区域
3. **GPU加速**: 支持大规模数据处理（270k+样本）
4. **自适应参数**: K-Means自动k值选择，HDBSCAN无需预设参数
5. **语义聚类**: 使用余弦距离和MoE logits进行语义相似度聚类

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

### 质量分类损失 - **更新架构**

新架构采用更简洁直接的质量分类损失：

**单分数架构** (当前版本):
```python
# 质量门输出单个原始分数
quality_score = quality_gate(hidden_states)  # 形状: [batch, seq_len, 1]

# 直接应用sigmoid得到good_ratio
good_ratio = torch.sigmoid(quality_score)    # 形状: [batch, seq_len, 1]
bad_ratio = 1.0 - good_ratio                 # 形状: [batch, seq_len, 1]

# 质量损失：直接对good_ratio计算
L_quality = good_ratio.mean()  # 鼓励降低good_ratio
```

**损失特性**：
- **简化计算**: 单分数 → sigmoid，避免softmax的复杂性
- **更好梯度**: 直接在原始分数上应用sigmoid，梯度更清晰
- **填充处理**: 自动排除padding tokens，只对有效token计算损失
- **可扩展性**: 支持多种损失类型 (sigmoid, MSE, 自定义函数)

**损失类型对比**：
```python
# sigmoid损失 (默认)
loss = good_ratio  # 鼓励降低good_ratio

# MSE损失
loss = (good_ratio - 0.0) ** 2  # 明确推向0

# 自定义损失
loss = custom_loss_fn(good_ratio, attention_mask)  # 用户定义
```

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
5. **聚类算法扩展**: 支持更多聚类算法(如Spectral Clustering, Gaussian Mixture Model)
6. **分布式聚类**: 支持多GPU分布式聚类处理超大规模数据集

