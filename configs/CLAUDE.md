# configs 模块文档

[根目录](/mnt/lishiwei/Quality-Gate/CLAUDE.md) > **configs**

---

## 变更记录 (Changelog)

### 2025-10-23
- 初始化模块文档
- 记录 Hydra 配置文件结构和参数

---

## 模块职责

`configs/` 目录包含 Hydra 配置文件，用于管理项目的所有参数：

1. **阶段配置**: 三阶段训练流程的主配置
2. **训练配置**: 不同模型的训练参数
3. **加速器配置**: 分布式训练配置（FSDP, DDP, Single GPU）

所有配置文件使用 YAML 格式，支持参数覆写和组合。

---

## 配置文件结构

```
configs/
├── stage_1_warmup.yaml            # 阶段1：质量门控预热
├── stage_2_selection.yaml         # 阶段2：数据选择
├── stage_3_finetune.yaml          # 阶段3：目标模型微调
├── batch_selection.yaml           # 批量数据筛选
├── continue_selection.yaml        # 续跑数据筛选
├── random_selection.yaml          # 随机数据筛选
├── training/                      # 训练配置组
│   ├── llama_2_7b.yaml           # LLaMA-2-7B 配置
│   ├── qwen_2.5_1.5b.yaml        # Qwen2.5-1.5B 配置
│   └── qwen_3_1.7b.yaml          # Qwen3-1.7B 配置
└── accelerate_config/             # 加速器配置组
    ├── SINGLE.yaml               # 单GPU配置
    ├── DDP.yaml                  # DDP配置
    └── FSDP.yaml                 # FSDP配置
```

---

## 主配置文件

### stage_1_warmup.yaml

**用途**: 阶段1质量门控预热训练

**核心参数**:
```yaml
stage: warmup

# 模型配置
selector_model:
  path: "Qwen/Qwen3-1.7B"           # 预转换的质量门控模型路径
  tokenizer_name: "Qwen/Qwen3-1.7B" # 分词器名称

# 输出目录（Hydra 自动管理）
output_dir: ${hydra:run.dir}

# 数据集配置
dataset:
  dataset_from: "hf"                # "local" 或 "hf"
  hf:
    datasets:
      - name: "teknium/OpenHermes-2.5"
        dataset_name: "openhermes"
        subset: null
        split: "train"
  subset_ratio: 0.025               # 数据采样比例
  shuffle: true
  max_sequence_length: 1024
  processing_num_workers: 10

# 训练配置
training:
  batch_size: 16                    # 总批次大小
  per_device_batch_size: 1          # 每设备批次大小
  learning_rate: 3e-4
  epochs: 2
  optimizer: "AdamW"
  scheduler: "linear"
  peft_mode: "full_rank"            # "lora" 或 "full_rank"

  # 质量门控参数
  quality_loss_weight: 1.0
  quality_gate_init_mean: 0.0
  quality_gate_init_std: 0.02

  # 质量损失配置
  quality_loss_type: "linear"
  quality_loss_debug: false
  quality_loss_params:
    sample_wise_averaging: true     # sample-wise vs token-wise
    full_sequence_prediction: true  # 全序列预测
    mask_special_tokens: true       # 掩盖特殊格式 token
```

**输出目录模板**:
```yaml
hydra:
  run:
    dir: outputs/stage_1_warmup/${now:%Y-%m-%d}/${now:%H-%M-%S}-batch=${training.batch_size}_lr=${training.learning_rate}_loss=${training.quality_loss_type}_lossWeight=${training.quality_loss_weight}_sampleWise=${training.quality_loss_params.sample_wise_averaging}_tag=${tag}
```

---

### stage_2_selection.yaml

**用途**: 阶段2数据选择（收集统计数据）

**核心参数**:
```yaml
stage: selection

# 模型配置
selector_model:
  path: "converted_models/quality_gate_Qwen2.5-1.5B"
  tokenizer_name: "Qwen/Qwen2.5-1.5B"

# 阶段1检查点路径
model_checkpoint_path: "outputs/stage_1_warmup/2025-01-01/00-00-00-batch=16_lr=0.001_loss=sigmoid_lossWeight=1_sampleWise=True_tag=none"

# 输出路径
output_path: "${hydra:run.dir}/router_data.pt"

# 数据集配置
dataset:
  dataset_from: "hf"
  hf:
    datasets:
      - name: "teknium/OpenHermes-2.5"
        dataset_name: "openhermes"
        subset: null
        split: "train"
  shuffle: false                    # 推荐：false（推理阶段）
  subset_ratio: 1.0                 # 使用全量数据
  max_sequence_length: 1024
  sort_by_length: true              # 优化推理效率

# 数据处理配置
data_process:
  batch_size: 32
```

**输出目录模板**:
```yaml
hydra:
  run:
    dir: outputs/stage_2_selection/${now:%Y-%m-%d}/${now:%H-%M-%S}-${extract_config:${model_checkpoint_path}}
```

---

### stage_3_finetune.yaml

**用途**: 阶段3目标模型微调

**核心参数**:
```yaml
stage: finetune

# 默认配置组
defaults:
  - _self_
  - training: qwen_2.5_1.5b        # 引用 training/qwen_2.5_1.5b.yaml

# 输出目录
output_dir: ${hydra:run.dir}

# 数据集配置
dataset:
  mode: "subset"                    # "full" 或 "subset"
  data_path: "outputs/stage_2_selection/.../selected_data.jsonl"
  max_sequence_length: 1024
  processing_num_workers: 10
  shuffle: true

# GPU 显存抢占（可选）
gpu_grab:
  grab: false
  memory_need_gb: 24
  over_grab: false
```

**输出目录模板**:
```yaml
hydra:
  run:
    dir: outputs/stage_3_finetune/${now:%Y-%m-%d}/${now:%H-%M-%S}-MODEL=<|${extract_model_config:${training.batch_size},${training.learning_rate},${tag}}|>-DATA=<|${extract_data_config_conditional:${dataset.data_path},${dataset.mode}}|>
```

---

## 训练配置组

### training/qwen_2.5_1.5b.yaml

**用途**: Qwen2.5-1.5B 模型的训练配置

```yaml
model:
  name: "Qwen/Qwen2.5-1.5B"

# 训练参数
batch_size: 64
per_device_batch_size: 1
learning_rate: 2e-5
epochs: 3
optimizer: "AdamW"
scheduler: "cosine"
warmup_ratio: 0.03
weight_decay: 0.0

# LoRA 配置
peft_mode: "lora"
lora:
  r: 128
  lora_alpha: 256
  lora_dropout: 0.0
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

### training/llama_2_7b.yaml

**用途**: LLaMA-2-7B 模型的训练配置

```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"

batch_size: 64
per_device_batch_size: 1
learning_rate: 2e-5
epochs: 3

peft_mode: "lora"
lora:
  r: 128
  lora_alpha: 256
  lora_dropout: 0.0
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

---

## 加速器配置组

### accelerate_config/SINGLE.yaml

**用途**: 单 GPU 训练配置

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: "NO"
mixed_precision: "bf16"
use_cpu: false
```

### accelerate_config/DDP.yaml

**用途**: 多 GPU DDP 训练配置

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: "bf16"
num_processes: 4
```

### accelerate_config/FSDP.yaml

**用途**: FSDP (Fully Sharded Data Parallel) 配置

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
mixed_precision: "bf16"
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: FULL_STATE_DICT
```

---

## 筛选配置文件

### batch_selection.yaml

**用途**: 批量数据筛选配置

```yaml
root_dir: "outputs/stage_2_selection/..."
router_data_subdir: "router_data"
selection_percentage: 0.05
output_filename: "selected_data.jsonl"
seed: 42
```

### continue_selection.yaml

**用途**: 续跑数据筛选配置

```yaml
router_data_dir: "outputs/stage_2_selection/.../router_data"
selection_percentage: 0.05
output_filename: "selected_data.jsonl"
seed: 42
```

---

## Hydra 特性

### 参数覆写

**命令行覆写**:
```bash
# 覆写单个参数
uv run python src/main.py \
    --config-name stage_1_warmup \
    training.learning_rate=1e-4

# 覆写嵌套参数
uv run python src/main.py \
    --config-name stage_1_warmup \
    training.quality_loss_params.sample_wise_averaging=false

# 覆写列表参数
uv run python src/main.py \
    --config-name stage_1_warmup \
    'dataset.hf.datasets=[{name: "openhermes", dataset_name: "openhermes"}]'
```

### 配置组合

**使用 defaults**:
```yaml
# stage_3_finetune.yaml
defaults:
  - _self_
  - training: qwen_2.5_1.5b
  - accelerate: FSDP
```

**命令行切换**:
```bash
# 使用 LLaMA-2-7B 配置
uv run python src/main.py \
    --config-name stage_3_finetune \
    training=llama_2_7b
```

### 自定义 Resolver

项目定义了自定义 Hydra resolver（在 `src/utils/hydra_resolvers.py`）：

#### `extract_config`
**用途**: 从路径中提取配置字符串

```yaml
# 示例
model_checkpoint_path: "outputs/stage_1_warmup/2025-10-23/12-00-00-batch=16_lr=0.001_..."

# 使用 resolver
dir: outputs/stage_2_selection/${now:%Y-%m-%d}/${now:%H-%M-%S}-${extract_config:${model_checkpoint_path}}

# 结果
# outputs/stage_2_selection/2025-10-23/13-00-00-batch=16_lr=0.001_...
```

#### `extract_model_config`
**用途**: 提取模型训练配置摘要

```yaml
dir: outputs/stage_3_finetune/${extract_model_config:${training.batch_size},${training.learning_rate},${tag}}

# 结果
# outputs/stage_3_finetune/2025-10-23/14-00-00-batch=64_lr=2e-05_tag=none
```

#### `extract_data_config_conditional`
**用途**: 根据数据模式提取数据配置

```yaml
dir: outputs/stage_3_finetune/DATA=<|${extract_data_config_conditional:${dataset.data_path},${dataset.mode}}|>

# 结果（subset 模式）
# DATA=<|batch=16_lr=0.001_...|>

# 结果（full 模式）
# DATA=<|full|>
```

---

## 常见问题 (FAQ)

### Q: 如何查看完整的配置？

A: 使用 Hydra 的 `--cfg` 选项：
```bash
uv run python src/main.py \
    --config-name stage_1_warmup \
    --cfg job
```

### Q: 如何保存和复用配置？

A: 创建自定义配置文件：
```yaml
# configs/my_custom_warmup.yaml
defaults:
  - stage_1_warmup

training:
  learning_rate: 1e-4
  batch_size: 32
```

使用：
```bash
uv run python src/main.py \
    --config-name my_custom_warmup
```

### Q: 如何设置环境变量？

A: 在配置文件中使用 `${oc.env:VAR_NAME}`：
```yaml
dataset:
  hf:
    datasets:
      - name: "${oc.env:HF_DATASET,teknium/OpenHermes-2.5}"
```

### Q: 配置文件中的路径是相对于什么的？

A: 相对路径是相对于项目根目录（`pyproject.toml` 所在目录）。Hydra 会自动处理工作目录。

### Q: 如何禁用 Hydra 的输出目录管理？

A: 不推荐，但可以这样做：
```yaml
hydra:
  run:
    dir: .
  output_subdir: null
```

---

## 配置最佳实践

### 1. 使用配置组组织参数

**不推荐**:
```yaml
# stage_3_finetune.yaml 中直接写所有参数
training:
  model:
    name: "Qwen/Qwen2.5-1.5B"
  batch_size: 64
  learning_rate: 2e-5
  # ... 100+ 行
```

**推荐**:
```yaml
# stage_3_finetune.yaml
defaults:
  - training: qwen_2.5_1.5b

# training/qwen_2.5_1.5b.yaml
model:
  name: "Qwen/Qwen2.5-1.5B"
batch_size: 64
learning_rate: 2e-5
```

### 2. 使用标签（tag）标识实验

```yaml
# 在配置文件中
tag: "experiment_v1"

# 或命令行覆写
uv run python src/main.py tag=ablation_study_1
```

输出目录会包含 tag，便于管理：
```
outputs/stage_1_warmup/2025-10-23/12-00-00-..._tag=ablation_study_1/
```

### 3. 使用合理的默认值

```yaml
# 提供合理的默认值
dataset:
  subset_ratio: 0.025  # 小规模快速验证
  shuffle: true
  max_sequence_length: 1024
  processing_num_workers: ${oc.decode:${oc.env:NUM_WORKERS,10}}
```

### 4. 文档化重要参数

```yaml
training:
  quality_loss_weight: 1.0  # 质量损失权重，范围: 0.1 - 1.0
                             # 太小：质量门控学习不充分
                             # 太大：影响语言建模损失
```

---

## 开发指南

### 添加新的配置组

**步骤**:
1. 创建配置文件
2. 在主配置中引用
3. 测试覆写

**示例：添加新模型配置**:
```yaml
# configs/training/mistral_7b.yaml
model:
  name: "mistralai/Mistral-7B-v0.1"

batch_size: 64
per_device_batch_size: 1
learning_rate: 2e-5

peft_mode: "lora"
lora:
  r: 128
  lora_alpha: 256
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

使用：
```bash
uv run python src/main.py \
    --config-name stage_3_finetune \
    training=mistral_7b
```

### 添加自定义 Resolver

**示例**:
```python
# src/utils/hydra_resolvers.py
from omegaconf import OmegaConf

def custom_resolver(value: str) -> str:
    """自定义处理逻辑"""
    return value.upper()

# 注册
OmegaConf.register_new_resolver("custom", custom_resolver)
```

使用：
```yaml
# configs/some_config.yaml
output_path: "${custom:${tag}}"  # 将 tag 转为大写
```

---

## 相关资源

- **Hydra 官方文档**: https://hydra.cc/docs/intro/
- **OmegaConf API**: https://omegaconf.readthedocs.io/
- **配置模式最佳实践**: https://hydra.cc/docs/patterns/configuring_experiments/

---

## 相关文件清单

```
configs/
├── stage_1_warmup.yaml            # 阶段1主配置
├── stage_2_selection.yaml         # 阶段2主配置
├── stage_3_finetune.yaml          # 阶段3主配置
├── batch_selection.yaml           # 批量筛选配置
├── continue_selection.yaml        # 续跑筛选配置
├── random_selection.yaml          # 随机筛选配置
├── training/                      # 训练配置组
│   ├── llama_2_7b.yaml
│   ├── qwen_2.5_1.5b.yaml
│   └── qwen_3_1.7b.yaml
└── accelerate_config/             # 加速器配置组
    ├── SINGLE.yaml
    ├── DDP.yaml
    └── FSDP.yaml
```

**相关代码**:
- `/mnt/lishiwei/Quality-Gate/src/main.py`: 配置加载入口
- `/mnt/lishiwei/Quality-Gate/src/utils/hydra_resolvers.py`: 自定义 resolver
