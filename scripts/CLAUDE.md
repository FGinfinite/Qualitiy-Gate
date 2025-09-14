[根目录](../CLAUDE.md) > **scripts**

# 执行脚本模块 - 流水线执行与工具集合

## 变更记录 (Changelog)

**2025-09-14 22:43:47** - 模块文档初始化
- 创建脚本模块执行指南
- 详细说明四阶段脚本使用方法
- 添加独立选择脚本和批量处理文档

## 模块职责

执行脚本模块提供完整的Select-MoE流水线执行工具集，主要功能：
- **四阶段执行脚本**：自动化Warmup/Selection/Finetune/Evaluate流程
- **独立数据选择**：解耦的聚类选择脚本，支持断点续跑
- **批量实验处理**：跨多个实验的批量数据选择
- **模型转换工具**：OLMoE到Select-MoE的模型转换和验证
- **评估脚本**：标准化的模型性能评估

## 入口与启动

### 四阶段主要脚本
- `run_stage_1.sh` - Stage 1: 路由器预训练执行脚本
- `run_stage_2.sh` - Stage 2: 路由器数据计算脚本
- `run_stage_3.sh` - Stage 3: 目标模型微调脚本
- `eval.sh` - Stage 4: 模型评估脚本

### 独立工具脚本
- `continue_selection.py` - 独立聚类数据选择
- `batch_selection.py` - 批量实验数据选择
- `convert_olmoe_to_select_moe.py` - 模型转换工具
- `compare_converted_model.py` - 模型转换验证

### 基本流水线执行
```bash
# 完整四阶段流水线
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_stage_1.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage_2.sh model_checkpoint_path=outputs/stage_1_warmup/2025-09-14/22-43-47/full_rank_weights.pt
CUDA_VISIBLE_DEVICES=0 uv run scripts/continue_selection.py router_data_dir=outputs/stage_2_selection/2025-09-14/22-43-47/router_data
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_stage_3.sh dataset.data_path=outputs/stage_2_selection/2025-09-14/22-43-47/selected_data.jsonl
bash scripts/eval.sh
```

## 对外接口

### Stage 1 预训练脚本 (`run_stage_1.sh`)
```bash
#!/bin/bash
# Stage 1: Select-MoE路由器预训练

# GPU设备指定
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# 加速器配置选择
ACCELERATE_CONFIG="configs/accelerate_config/FSDP.yaml"

# 执行训练
accelerate launch --config_file $ACCELERATE_CONFIG \
    src/main.py \
    --config-name stage_1_warmup \
    "$@"  # 传递所有命令行参数

# 使用示例:
# bash scripts/run_stage_1.sh training.learning_rate=1e-4 tag=experiment_v1
```

### Stage 2 数据计算脚本 (`run_stage_2.sh`)
```bash
#!/bin/bash
# Stage 2: 路由器数据推理和计算

# 检查必需参数
if [[ "$*" != *"model_checkpoint_path="* ]]; then
    echo "错误: 必须指定model_checkpoint_path参数"
    echo "使用方法: bash $0 model_checkpoint_path=path/to/weights.pt"
    exit 1
fi

# 单GPU推理执行
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python src/main.py \
    --config-name stage_2_selection \
    "$@"

# 使用示例:
# bash scripts/run_stage_2.sh model_checkpoint_path=outputs/stage_1_warmup/2025-09-14/22-43-47/full_rank_weights.pt
```

### 独立数据选择脚本 (`continue_selection.py`)
```python
#!/usr/bin/env python3
"""
独立的聚类选择脚本

从Stage 2的router_data进行数据选择，支持多种聚类算法和参数配置。
"""

@hydra.main(config_path="../configs", config_name="continue_selection")
def main(cfg: DictConfig):
    """
    主选择函数

    Args:
        cfg: Hydra配置，包含以下关键参数:
            - router_data_dir: router数据目录路径
            - selection_percentage: 选择比例 (0.01-1.0)
            - clustering_method: 聚类方法 ("kmeans")
            - clustering_params: 聚类参数字典
            - debug_print: 是否启用调试输出
    """

# 使用示例:
# CUDA_VISIBLE_DEVICES=0 uv run scripts/continue_selection.py \
#     router_data_dir=outputs/stage_2_selection/2025-09-14/22-43-47/router_data \
#     selection_percentage=0.05 \
#     clustering_method=kmeans \
#     clustering_params.enable_parallel_kmeans=true
```

### 批量数据选择脚本 (`batch_selection.py`)
```python
#!/usr/bin/env python3
"""
批量实验数据选择脚本

自动扫描指定根目录下的所有实验，并为每个实验执行数据选择。
支持跨多个实验的批量处理和进度跟踪。
"""

@hydra.main(config_path="../configs", config_name="batch_selection")
def main(cfg: DictConfig):
    """
    批量选择主函数

    Args:
        cfg: 包含以下参数:
            - root_dir: 实验根目录 (如 outputs/stage_2_selection)
            - selection_percentage: 统一的选择比例
            - clustering_method: 聚类算法
            - max_workers: 最大并行worker数
    """

# 使用示例:
# CUDA_VISIBLE_DEVICES=0 uv run scripts/batch_selection.py \
#     root_dir=outputs/stage_2_selection \
#     selection_percentage=0.1 \
#     clustering_method=kmeans
```

### 模型转换工具 (`convert_olmoe_to_select_moe.py`)
```python
#!/usr/bin/env python3
"""
OLMoE到Select-MoE模型转换工具

将原始OLMoE模型转换为具有两层路由架构的Select-MoE格式。
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", default="allenai/OLMoE-1B-7B-0125")
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--dtype", default="bfloat16")

# 使用示例:
# python scripts/convert_olmoe_to_select_moe.py \
#     --save-path ./converted_models/select_moe_converted_OLMoE-1B-7B-0125
```

## 关键依赖与配置

### 环境依赖
- `bash>=4.0` - Shell脚本执行环境
- `uv` - Python依赖管理和执行
- `accelerate>=1.8.1` - 分布式训练支持
- `hydra-core` - 配置管理

### GPU资源配置
```bash
# 单GPU环境
export CUDA_VISIBLE_DEVICES=0

# 多GPU训练环境
export CUDA_VISIBLE_DEVICES=0,1,2,3

# GPU内存管理
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### 脚本执行权限
```bash
# 确保脚本可执行
chmod +x scripts/*.sh

# 检查脚本语法
bash -n scripts/run_stage_1.sh
```

## 数据模型

### 脚本执行流程
```
用户执行脚本
    ↓
环境检查和GPU设置
    ↓
配置文件加载和参数解析
    ↓
Python主程序调用
    ↓
Hydra参数覆盖和验证
    ↓
实际训练/推理/选择逻辑
    ↓
结果保存和日志记录
```

### 输出目录结构
```
outputs/
├── stage_1_warmup/
│   └── YYYY-MM-DD/
│       └── HH-MM-SS-[params]/
│           ├── full_rank_weights.pt      # 训练权重
│           ├── .hydra/                   # Hydra配置快照
│           └── logs/                     # 训练日志
├── stage_2_selection/
│   └── YYYY-MM-DD/
│       └── HH-MM-SS/
│           ├── router_data/              # 路由器数据
│           │   ├── cot_router_data.pt
│           │   └── openhermes_router_data.pt
│           └── selected_data.jsonl       # 选择的数据
└── stage_3_finetune/
    └── YYYY-MM-DD/
        └── HH-MM-SS/
            └── checkpoint-*/             # LoRA检查点
```

### 参数覆盖机制
```bash
# 1. 单个参数覆盖
bash scripts/run_stage_1.sh training.learning_rate=1e-4

# 2. 多个参数覆盖
bash scripts/run_stage_1.sh \
    training.batch_size=32 \
    training.epochs=3 \
    tag=experiment_v2

# 3. 嵌套参数覆盖
bash scripts/run_stage_2.sh \
    clustering_method=kmeans \
    clustering_params.k=50 \
    clustering_params.auto_k=false

# 4. 列表参数覆盖
bash scripts/run_stage_1.sh \
    dataset.local.dataset_names=[cot,dolly]
```

## 测试与质量

### 脚本语法检查
```bash
# 检查所有Shell脚本语法
for script in scripts/*.sh; do
    echo "检查 $script"
    bash -n "$script" || echo "语法错误: $script"
done
```

### 干跑测试
```bash
# 测试配置解析但不执行训练
python src/main.py --config-name stage_1_warmup --help

# 测试参数覆盖
python src/main.py --config-name stage_1_warmup training.batch_size=1 --help
```

### 依赖检查
```bash
# 检查Python环境
python --version
pip list | grep -E "(torch|transformers|accelerate)"

# 检查GPU可用性
python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"
```

## 常见问题 (FAQ)

**Q: 脚本执行时提示权限不足？**
A: 使用 `chmod +x scripts/*.sh` 给脚本添加执行权限。

**Q: 如何在不同GPU配置下运行？**
A: 设置 `CUDA_VISIBLE_DEVICES` 环境变量，或在configs/accelerate_config/目录下选择合适的配置文件。

**Q: 脚本中断后如何续跑？**
A: 对于数据选择阶段，可以使用 `continue_selection.py` 直接从router_data继续执行。

**Q: 如何批量处理多个实验？**
A: 使用 `batch_selection.py` 脚本，指定包含多个实验的根目录。

**Q: 如何调试脚本参数传递？**
A: 在脚本中添加 `echo "$@"` 查看传递的参数，或使用Hydra的 `--help` 选项。

## 相关文件清单

### 主要执行脚本
- `run_stage_1.sh` - Stage 1预训练脚本（25行）
- `run_stage_2.sh` - Stage 2数据计算脚本（20行）
- `run_stage_3.sh` - Stage 3微调脚本（25行）
- `eval.sh` - Stage 4评估脚本（15行）

### 独立工具脚本
- `continue_selection.py` - 独立选择脚本（200+行）
- `batch_selection.py` - 批量选择脚本（150+行）
- `random_selection.py` - 随机选择基线脚本

### 模型工具脚本
- `convert_olmoe_to_select_moe.py` - 模型转换（300+行）
- `compare_converted_model.py` - 转换验证（100+行）
- `convert.sh` - 转换包装脚本

### 相关配置
- `../configs/stage_*.yaml` - 各阶段配置文件
- `../configs/continue_selection.yaml` - 独立选择配置
- `../configs/batch_selection.yaml` - 批量选择配置

### 相关工具
- `../src/main.py` - Python主入口
- `../tools/install.sh` - 开发工具安装