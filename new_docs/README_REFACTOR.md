# Quality-Gate 项目重构说明

## 概述

本项目已从 **Select-MoE + 多样性选择** 架构重构为 **质量门控 + 质量选择** 架构。

## 核心变化

### 1. 模型架构

- **之前**: OLMoE + Select-MoE（质量门控 + MoE路由）
- **现在**: Qwen3-1.7B + 质量门控（仅质量门控，在FFN层之前插入）

新模型位于: `src/models/quality_gate_model.py`

### 2. 数据选择策略

- **之前**: 质量门控 + 聚类 + 轮选（兼顾质量和多样性）
- **现在**: 质量门控 + 困惑度 + 直接选择top-k%（仅关注质量）

新选择逻辑位于: `src/selection/quality_scoring.py`

### 3. 工作流程

#### 阶段1: 质量门控预热训练

```bash
python src/main.py stage=warmup
```

- 冻结主干网络
- 仅训练质量门控参数
- 使用质量损失函数引导门控学习

#### 阶段2: 统计收集

```bash
python src/main.py stage=selection model_checkpoint_path=path/to/warmup/output
```

- 前向传播收集：
  - 质量门控 logits (G_i: [L, T])
  - Token 困惑度 (ppl_i: [T])
- 保存原始统计量到 `router_data/`

#### 阶段3: 数据筛选

```bash
# 单个实验
python scripts/continue_selection.py router_data_dir=path/to/router_data

# 批量处理
python scripts/batch_selection.py root_dir=outputs/stage_2_selection
```

- 基于困惑度计算token权重
- 逐层加权得到质量分数
- 归一化和尺度放缩
- 选择top-k%数据

## 数据选择算法

根据 `new_docs/quality_gate.md` 实现：

1. **Token权重**: w_{i,t} = ppl_{i,t}^α / Σ(ppl_{i,u}^α)
2. **逐层分数**: s_{i,l} = Σ_t w_{i,t} * g_{i,l,t}
3. **归一化映射**: R_{i,l} = (S_{i,l} - a_l) / ((b_l - a_l) * μ_l)
4. **质量分数**: q_i = mean_l(R_{i,l})
5. **选择**: 按q降序取top-k%

## 配置文件

- `configs/stage_1_warmup.yaml` - 预热训练配置
- `configs/stage_2_selection.yaml` - 统计收集配置
- `configs/continue_selection.yaml` - 数据筛选配置（单个）
- `configs/batch_selection.yaml` - 数据筛选配置（批量）

## 主要参数

### 质量门控参数

- `quality_gate_init_mean`: 门控初始化均值（默认0.0）
- `quality_gate_init_std`: 门控初始化标准差（默认0.02）
- `quality_loss_weight`: 质量损失权重（默认1.0）
- `quality_loss_type`: 损失类型（sigmoid/beta_moment_matching/mean_variance_regularization）

### 数据选择参数

- `selection_percentage`: 选择比例（如0.30表示选择前30%）
- `alpha`: Token权重参数（默认1.0；>1更偏重高熵token）
- `eps`: 数值下界（默认1e-8）
- `tau`: 稳定性参数（默认0.0，可选）

## 文件结构

```
src/
├── models/
│   └── quality_gate_model.py      # 新：质量门控模型
├── stages/
│   ├── warmup.py                  # 更新：预热训练
│   └── selection.py               # 更新：统计收集
├── selection/
│   ├── quality_scoring.py         # 新：质量选择逻辑
│   └── data_selection.py          # 保留：数据加载等辅助函数
└── data/
    └── dataset_loader.py          # 保留：数据加载

scripts/
├── continue_selection.py          # 更新：单个筛选
└── batch_selection.py             # 更新：批量筛选

configs/
├── stage_1_warmup.yaml            # 更新
├── stage_2_selection.yaml         # 更新
├── continue_selection.yaml        # 更新
└── batch_selection.yaml           # 更新
```

## 废弃文件（可删除）

以下文件不再使用，但仍保留在项目中：

- `src/models/select_moe.py` - Select-MoE模型
- `src/clustering/` - 聚类相关代码
- `ref_model/olmoe/` - OLMoE参考实现
- `scripts/convert_olmoe_to_select_moe.py` - 模型转换脚本

## 示例使用

```bash
# 1. 预热训练
python src/main.py stage=warmup \
    selector_model.path=Qwen/Qwen3-1.7B \
    training.batch_size=16 \
    training.learning_rate=3e-4 \
    training.epochs=2

# 2. 统计收集
python src/main.py stage=selection \
    model_checkpoint_path=outputs/stage_1_warmup/2025-01-01/00-00-00-xxx

# 3. 数据筛选
python scripts/continue_selection.py \
    router_data_dir=outputs/stage_2_selection/xxx/router_data \
    selection_percentage=0.30
```

## 技术细节

### 质量门控插入位置

在每个Transformer层中：
```
Input → LayerNorm → SelfAttention → Add
  ↓
Residual → LayerNorm → QualityGate → FFN → Add → Output
```

### 困惑度计算

每个token的困惑度定义为：
```
ppl_t = exp(NLL_t)
```
其中 NLL_t 是该token的负对数似然。

### 质量分数计算

参见 `new_docs/quality_gate.md` 获取完整的数学推导和算法描述。

## 注意事项

1. 确保使用Qwen模型作为基座（或已经转换的质量门控模型）
2. 预热阶段仅训练质量门控参数，主干网络冻结
3. 统计收集阶段保存原始统计量，不做任何处理
4. 数据筛选阶段可以独立运行，便于调整参数
5. 支持从多个数据集混合选择

## 更新日志

- 2025-10-22: 完成从Select-MoE到质量门控的重构
  - 新增质量门控模型
  - 重写统计收集和数据筛选逻辑
  - 更新所有配置文件
  - 移除聚类依赖

## 参考文档

- `new_docs/quality_gate.md` - 质量门控方法详细说明
- `CLAUDE.md` - 项目开发记录

