[根目录](../CLAUDE.md) > **configs**

# 配置文件模块 - Hydra配置管理系统

## 变更记录 (Changelog)

**2025-09-14 22:43:47** - 模块文档初始化
- 创建Hydra配置管理文档
- 详细说明各阶段配置结构
- 添加参数覆盖和组合示例

## 模块职责

配置文件模块负责管理整个Select-MoE流水线的配置参数，主要功能：
- **分阶段配置管理**：为四个流水线阶段提供独立配置
- **Hydra配置框架**：支持参数覆盖、配置组合和实验跟踪
- **多级配置结构**：根配置 + 专用配置 + 模型配置
- **分布式训练配置**：多GPU/多节点训练参数管理
- **实验管理**：自动生成实验目录和配置快照

## 入口与启动

### 主要配置文件
- `stage_1_warmup.yaml` - Stage 1: 路由器预训练配置
- `stage_2_selection.yaml` - Stage 2: 数据选择配置
- `stage_3_finetune.yaml` - Stage 3: 目标模型微调配置
- `continue_selection.yaml` - 独立数据选择配置
- `batch_selection.yaml` - 批量数据选择配置

### 使用方式
```bash
# 使用默认配置
python src/main.py

# 指定配置文件
python src/main.py --config-name stage_2_selection

# 参数覆盖
python src/main.py training.learning_rate=1e-4 training.batch_size=32

# 配置组合
python src/main.py --config-name stage_3_finetune training=llama_2_7b
```

## 对外接口

### 配置文件结构

#### Stage 1 预训练配置 (`stage_1_warmup.yaml`)
```yaml
# 阶段标识
stage: warmup

# 实验管理
tag: "experiment_name"
seed: 42

# 模型配置
selector_model:
  path: "converted_models/select_moe_converted_OLMoE-1B-7B-0125"
  tokenizer_name: "allenai/OLMoE-1B-7B-0125"

# 数据集配置
dataset:
  dataset_from: "hf"  # "local" 或 "hf"

  local:                    # 本地数据集配置
    data_dir: "dataset/train/processed"
    dataset_names: ["cot", "dolly", "flan_v2", "oasst1"]

  hf:                       # HuggingFace数据集配置
    datasets:
      - name: "teknium/OpenHermes-2.5"
        dataset_name: "openhermes"
        subset: null
        split: "train"

  subset_ratio: 0.025       # 数据子集比例
  max_sequence_length: 1024

# 训练配置
training:
  batch_size: 16
  learning_rate: 3e-4
  epochs: 2
  peft_mode: "full_rank"    # "lora" 或 "full_rank"

  # 质量损失配置
  quality_loss_weight: 1
  quality_loss_type: "sigmoid"
  quality_loss_params:
    sample_wise_averaging: true
    full_sequence_prediction: true
    mask_special_tokens: true
```

#### Stage 2 数据选择配置 (`stage_2_selection.yaml`)
```yaml
stage: selection

# 路由器模型路径
model_checkpoint_path: null  # 通过命令行参数指定

# 选择参数
selection_percentage: 0.05   # 选择比例

# 聚类配置
clustering_method: "kmeans"
clustering_params:
  auto_k: true              # 自动K值选择
  k_range: [10, 100]        # K值搜索范围
  enable_parallel_kmeans: false
  parallel_processes: 4
```

### 加速器配置 (`accelerate_config/`)

#### 单GPU配置 (`SINGLE.yaml`)
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: "NO"
num_processes: 1
machine_rank: 0
num_machines: 1
mixed_precision: bf16
```

#### DDP多GPU配置 (`DDP.yaml`)
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 4            # GPU数量
machine_rank: 0
num_machines: 1
gpu_ids: "0,1,2,3"
mixed_precision: bf16
```

#### FSDP配置 (`FSDP.yaml`)
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
num_processes: 4
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_sharding_strategy: 1
```

## 关键依赖与配置

### 依赖库
- `hydra-core` - 配置管理框架
- `omegaconf` - 配置数据结构
- `accelerate>=1.8.1` - 分布式训练配置

### Hydra配置管理

#### 目录结构模板
```yaml
# 动态目录命名模板
_target_dir_template_: ${now:%Y-%m-%d}/${now:%H-%M-%S}-batch=${training.batch_size}_lr=${training.learning_rate}_loss=${training.quality_loss_type}

# Hydra运行配置
hydra:
  run:
    dir: outputs/stage_1_warmup/${_target_dir_template_}
  sweep:
    dir: outputs/stage_1_warmup
    subdir: ${_target_dir_template_}
```

#### 自定义解析器
```python
# src/utils/hydra_resolvers.py
@hydra.main(config_path="../configs", config_name="stage_1_warmup")
def main(cfg: DictConfig):
    # 配置自动解析和验证
    pass
```

## 数据模型

### 配置层次结构
```
根配置 (stage_*.yaml)
├── 全局设置 (stage, seed, output_dir)
├── 模型配置 (selector_model, tokenizer)
├── 数据集配置 (dataset)
│   ├── 数据源选择 (dataset_from)
│   ├── 本地数据集 (local)
│   └── HF数据集 (hf)
├── 训练配置 (training)
│   ├── 基本参数 (batch_size, learning_rate)
│   ├── PEFT配置 (peft_mode, lora)
│   └── 损失配置 (quality_loss_*)
└── Hydra配置 (hydra)
    ├── 输出目录 (run.dir)
    └── 实验跟踪 (sweep)
```

### 参数覆盖机制
```bash
# 1. 基础覆盖
python src/main.py training.learning_rate=1e-4

# 2. 嵌套覆盖
python src/main.py training.quality_loss_params.beta_target_mean=0.6

# 3. 列表覆盖
python src/main.py dataset.local.dataset_names=[cot,dolly]

# 4. 配置组合
python src/main.py --config-name stage_3_finetune training=qwen_2.5_1.5b

# 5. 多参数覆盖
python src/main.py training.batch_size=32 training.epochs=3 tag=experiment_v2
```

### 模型专用配置 (`training/`)

#### Llama-2-7B配置 (`llama_2_7b.yaml`)
```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"

training:
  per_device_batch_size: 1
  gradient_accumulation_steps: 8
  lora:
    r: 128
    lora_alpha: 256
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

#### Qwen2.5-1.5B配置 (`qwen_2.5_1.5b.yaml`)
```yaml
model:
  name: "Qwen/Qwen2.5-1.5B"

training:
  per_device_batch_size: 4
  gradient_accumulation_steps: 2
  lora:
    r: 64
    lora_alpha: 128
```

## 测试与质量

### 配置验证
```python
# 配置文件语法检查
from omegaconf import OmegaConf
cfg = OmegaConf.load("configs/stage_1_warmup.yaml")
print(OmegaConf.to_yaml(cfg))

# 参数覆盖验证
from hydra import initialize, compose
with initialize(config_path="../configs"):
    cfg = compose(config_name="stage_1_warmup",
                 overrides=["training.learning_rate=1e-4"])
```

### 配置模板测试
```bash
# 测试目录模板生成
python -c "
import hydra
from omegaconf import DictConfig
@hydra.main(config_path='configs', config_name='stage_1_warmup')
def test_config(cfg: DictConfig):
    print(f'Output dir: {cfg.output_dir}')
test_config()
"
```

## 常见问题 (FAQ)

**Q: 如何添加新的配置参数？**
A: 在对应的yaml文件中添加参数，并在代码中通过 `cfg.new_param` 访问。

**Q: 如何创建新的配置组合？**
A: 在对应子目录下创建新的yaml文件，使用 `--config-name` 或 `training=new_config` 引用。

**Q: 参数覆盖的优先级是什么？**
A: 命令行覆盖 > 配置文件 > 默认值。

**Q: 如何调试配置解析问题？**
A: 使用 `hydra.core.global_hydra.GlobalHydra.instance().clear()` 清除缓存，或启用Hydra调试模式。

## 相关文件清单

### 主配置文件
- `stage_1_warmup.yaml` - Stage 1预训练配置（118行）
- `stage_2_selection.yaml` - Stage 2数据选择配置
- `stage_3_finetune.yaml` - Stage 3微调配置
- `continue_selection.yaml` - 独立选择脚本配置
- `batch_selection.yaml` - 批量选择配置

### 分布式训练配置
- `accelerate_config/SINGLE.yaml` - 单GPU配置
- `accelerate_config/DDP.yaml` - 数据并行配置
- `accelerate_config/FSDP.yaml` - 全分片并行配置

### 模型专用配置
- `training/llama_2_7b.yaml` - Llama-2-7B训练配置
- `training/qwen_2.5_1.5b.yaml` - Qwen2.5-1.5B训练配置
- `training/qwen_3_1.7b.yaml` - Qwen-3-1.7B训练配置

### 相关工具
- `../src/utils/hydra_resolvers.py` - 自定义Hydra解析器
- `../src/main.py` - 配置加载和使用示例