# scripts 模块文档

[根目录](/mnt/lishiwei/Quality-Gate/CLAUDE.md) > **scripts**

---

## 变更记录 (Changelog)

### 2025-10-23
- 初始化模块文档
- 记录转换和选择脚本功能

---

## 模块职责

`scripts/` 目录包含独立的工具脚本，用于：

1. **模型转换**: 将基座模型转换为 Quality-Gate 格式
2. **数据筛选**: 基于统计数据进行数据选择
3. **模型对比**: 验证转换后的模型正确性

这些脚本可以独立运行，不依赖训练流程。

---

## 入口与启动

### 模型转换脚本

#### convert_qwen_to_quality_gate.py

**用途**: 将 Qwen3 预训练模型转换为 Quality-Gate 格式

**使用方法**:
```bash
# 基本转换
uv run python scripts/convert_qwen_to_quality_gate.py \
    --model Qwen/Qwen3-1.7B \
    --save-path ./converted_models/quality_gate_Qwen3-1.7B

# 自定义参数
uv run python scripts/convert_qwen_to_quality_gate.py \
    --model Qwen/Qwen3-1.7B \
    --save-path ./converted_models/quality_gate_Qwen3-1.7B \
    --quality-gate-init-mean 0.0 \
    --quality-gate-init-std 0.02 \
    --quality-loss-weight 1.0 \
    --device cuda:0 \
    --seed 42
```

**支持的模型**:
- Qwen/Qwen3-1.7B (推荐)
- Qwen/Qwen3-* 系列
- **不支持** Qwen2/Qwen2.5 (架构不兼容)

**输出结构**:
```
converted_models/quality_gate_Qwen3-1.7B/
├── config.json                    # Quality-Gate 配置
├── model.safetensors              # 模型权重
├── tokenizer.json                 # 分词器
├── tokenizer_config.json
└── special_tokens_map.json
```

---

### 数据筛选脚本

#### batch_selection.py

**用途**: 批量处理多个数据集的筛选

**使用方法**:
```bash
# 基本用法
uv run python scripts/batch_selection.py \
    root_dir=outputs/stage_2_selection/2025-10-23/12-00-00-xxx

# 自定义筛选比例
uv run python scripts/batch_selection.py \
    root_dir=outputs/stage_2_selection/2025-10-23/12-00-00-xxx \
    selection_percentage=0.1
```

**配置文件**: `configs/batch_selection.yaml`
```yaml
root_dir: "outputs/stage_2_selection/..."
router_data_subdir: "router_data"
selection_percentage: 0.05
output_filename: "selected_data.jsonl"
```

#### continue_selection.py

**用途**: 对单个数据集进行筛选（可续跑）

**使用方法**:
```bash
uv run python scripts/continue_selection.py \
    router_data_dir=outputs/stage_2_selection/.../router_data
```

**配置文件**: `configs/continue_selection.yaml`
```yaml
router_data_dir: "outputs/stage_2_selection/.../router_data"
selection_percentage: 0.05
output_filename: "selected_data.jsonl"
```

#### random_selection.py

**用途**: 随机选择数据（基线对比）

**使用方法**:
```bash
uv run python scripts/random_selection.py \
    selection_percentage=0.05
```

---

### 模型验证脚本

#### compare_quality_gate_model.py

**用途**: 验证转换后的 Quality-Gate 模型

**使用方法**:
```bash
uv run python scripts/compare_quality_gate_model.py \
    --converted-model ./converted_models/quality_gate_Qwen3-1.7B \
    --dtype bfloat16
```

**验证内容**:
- 模型加载正确性
- 前向传播功能
- 质量门控输出格式
- 与基座模型的输出一致性

---

## 关键功能

### 模型转换流程

`convert_qwen_to_quality_gate.py` 的核心函数：

```python
def convert_and_save_model(
    model_name="Qwen/Qwen3-1.7B",
    save_path=None,
    device="cpu",
    quality_gate_init_mean=0.0,
    quality_gate_init_std=0.02,
    quality_loss_weight=1.0,
    quality_loss_type="sigmoid",
    seed=42,
) -> str:
    """
    转换 Qwen3 模型为 Quality-Gate 格式

    步骤：
    1. 加载原始 Qwen3 模型
    2. 创建 Quality-Gate 配置
    3. 复制 Qwen3 权重到 Quality-Gate 模型
    4. 初始化质量门控参数
    5. 验证转换正确性
    6. 保存模型和分词器
    7. 测试加载保存的模型

    Returns:
        保存路径的绝对路径
    """
```

**关键步骤**:
1. 验证模型类型（确保是 Qwen3）
2. 复制基座权重
3. 添加质量门控层（自动初始化）
4. 运行前向传播测试
5. 测试质量门控输出

### 数据筛选算法

`batch_selection.py` 的筛选逻辑：

1. **加载路由数据**:
```python
router_data = torch.load(f"{router_data_dir}/openhermes_router_data.pt")
quality_gates = router_data["quality_gates"]  # [N, L, T]
perplexities = router_data["perplexities"]    # [N, T]
```

2. **计算质量分数**:
```python
# 对每个样本，取所有层的平均质量分数
quality_scores = torch.sigmoid(quality_gates).mean(dim=(1, 2))  # [N]
```

3. **排序和选择**:
```python
# 按质量分数降序排序
sorted_indices = torch.argsort(quality_scores, descending=True)

# 选择 top-k
num_to_select = int(len(sorted_indices) * selection_percentage)
selected_indices = sorted_indices[:num_to_select]
```

4. **输出 JSONL**:
```python
with open(output_path, "w") as f:
    for idx in selected_indices:
        sample = original_dataset[idx]
        f.write(json.dumps(sample) + "\n")
```

---

## 对外接口

### convert_qwen_to_quality_gate.py

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--model` | str | "Qwen/Qwen3-1.7B" | Qwen3 模型名称或路径 |
| `--save-path` | str | None | 本地保存路径 |
| `--device` | str | "cpu" | 使用的设备 |
| `--quality-gate-init-mean` | float | 0.0 | 质量门控初始化均值 |
| `--quality-gate-init-std` | float | 0.02 | 质量门控初始化标准差 |
| `--quality-loss-weight` | float | 1.0 | 质量损失权重 |
| `--quality-loss-type` | str | "sigmoid" | 质量损失类型 |
| `--seed` | int | 42 | 随机种子 |

#### 返回值

- **成功**: 返回保存路径，退出码 0
- **失败**: 打印错误信息，退出码 1

---

### batch_selection.py

#### Hydra 配置参数

```yaml
# configs/batch_selection.yaml
root_dir: "outputs/stage_2_selection/..."    # 阶段2输出目录
router_data_subdir: "router_data"            # 路由数据子目录
selection_percentage: 0.05                   # 选择比例
output_filename: "selected_data.jsonl"       # 输出文件名
seed: 42                                     # 随机种子

# 筛选策略
strategy: "quality_based"  # "quality_based", "perplexity_based", "hybrid"
```

#### 输出格式

```jsonl
{"messages": [...], "dataset": "openhermes", "id": "openhermes_12345"}
{"messages": [...], "dataset": "openhermes", "id": "openhermes_67890"}
...
```

---

## 关键依赖与配置

### 依赖关系

```python
# convert_qwen_to_quality_gate.py
transformers
torch
src.models.quality_gate_model

# batch_selection.py
torch
datasets
hydra-core
```

### 配置文件

```
configs/
├── batch_selection.yaml       # 批量筛选配置
├── continue_selection.yaml    # 续跑筛选配置
└── random_selection.yaml      # 随机筛选配置
```

---

## 测试与质量

### 转换验证

**自动测试**（内置在转换脚本）:
```python
# 1. 前向传播测试
test_input = torch.randint(0, vocab_size, (1, 8))
outputs = model(test_input, output_router_logits=False)
assert outputs.logits.shape == (1, 8, vocab_size)

# 2. 质量门控输出测试
outputs_with_router = model(test_input, output_router_logits=True)
assert len(outputs_with_router.router_logits) == num_hidden_layers

# 3. 重新加载测试
test_model = QualityGateForCausalLM.from_pretrained(save_path)
test_outputs = test_model(test_input, output_router_logits=True)
assert test_outputs.logits.shape == (1, 8, vocab_size)
```

### 筛选验证

**数据完整性检查**:
```python
# 检查筛选后的数据格式
selected_dataset = load_dataset("json", data_files="selected_data.jsonl")
assert "messages" in selected_dataset.column_names
assert "dataset" in selected_dataset.column_names
assert "id" in selected_dataset.column_names
```

---

## 常见问题 (FAQ)

### Q: 为什么转换只支持 Qwen3？

A: Quality-Gate 模型架构基于 Qwen3 设计，与 Qwen2/Qwen2.5 的架构细节不兼容。主要差异：
- Attention 机制的实现
- LayerNorm 的位置和参数
- RotaryEmbedding 的配置

### Q: 转换后的模型能否直接用于推理？

A: 可以，但质量门控参数是随机初始化的，需要先进行阶段1的预热训练。

### Q: 如何验证转换是否成功？

A: 转换脚本会自动执行验证测试。也可以手动运行：
```bash
uv run python scripts/compare_quality_gate_model.py \
    --converted-model ./converted_models/quality_gate_Qwen3-1.7B
```

### Q: batch_selection.py 和 continue_selection.py 有什么区别？

A:
- **batch_selection.py**: 批量处理多个数据集，自动发现所有 `*_router_data.pt` 文件
- **continue_selection.py**: 处理单个数据集，支持从中断处续跑

### Q: 能否自定义筛选策略？

A: 可以，修改 `batch_selection.py` 的筛选逻辑：

```python
# 示例：基于困惑度筛选
def select_by_perplexity(router_data, selection_percentage):
    perplexities = router_data["perplexities"]
    avg_ppl = perplexities.mean(dim=1)  # [N]

    # 选择困惑度最低的样本
    sorted_indices = torch.argsort(avg_ppl)
    num_to_select = int(len(sorted_indices) * selection_percentage)
    return sorted_indices[:num_to_select]
```

### Q: 转换脚本能否支持其他基座模型？

A: 需要修改代码支持新架构。步骤：
1. 创建新的转换脚本（如 `convert_llama_to_quality_gate.py`）
2. 定义新的配置类（继承自基座的 Config）
3. 定义新的解码器层（插入 QualityGate）
4. 实现权重复制逻辑
5. 添加验证测试

---

## 相关文件清单

```
scripts/
├── convert_qwen_to_quality_gate.py    # Qwen3 → Quality-Gate 转换
├── compare_quality_gate_model.py      # 模型验证
├── batch_selection.py                 # 批量数据筛选
├── continue_selection.py              # 单数据集筛选（可续跑）
└── random_selection.py                # 随机筛选（基线）
```

**相关配置**:
- `/mnt/lishiwei/Quality-Gate/configs/batch_selection.yaml`
- `/mnt/lishiwei/Quality-Gate/configs/continue_selection.yaml`
- `/mnt/lishiwei/Quality-Gate/configs/random_selection.yaml`

**相关模块**:
- `/mnt/lishiwei/Quality-Gate/src/models/quality_gate_model.py`
- `/mnt/lishiwei/Quality-Gate/src/stages/selection.py`

---

## 开发指南

### 添加新的转换脚本

**模板**:
```python
#!/usr/bin/env python3
"""
将 XXX 基座模型转换为 Quality-Gate 模型
"""

import argparse
from src.models.quality_gate_model import register_quality_gate

def convert_xxx_to_quality_gate(
    model_name: str,
    save_path: str,
    **kwargs
) -> str:
    """转换 XXX 模型"""
    # 1. 注册 Quality-Gate
    register_quality_gate()

    # 2. 加载基座模型
    base_model = load_base_model(model_name)

    # 3. 创建 Quality-Gate 配置
    config = create_quality_gate_config(base_model.config)

    # 4. 创建 Quality-Gate 模型
    quality_gate_model = create_quality_gate_model(config)

    # 5. 复制权重
    copy_weights(base_model, quality_gate_model)

    # 6. 验证
    validate_model(quality_gate_model)

    # 7. 保存
    quality_gate_model.save_pretrained(save_path)

    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    args = parser.parse_args()

    convert_xxx_to_quality_gate(args.model, args.save_path)
```

### 添加新的筛选策略

**示例：基于多样性的筛选**:
```python
# scripts/diversity_selection.py
import torch
from sklearn.cluster import KMeans

def select_diverse_samples(
    router_data: dict,
    selection_percentage: float,
    num_clusters: int = 10
) -> torch.Tensor:
    """
    使用聚类确保多样性

    步骤：
    1. 对 quality_gates 进行聚类
    2. 从每个簇中选择高质量样本
    3. 确保选择样本覆盖所有簇
    """
    quality_gates = router_data["quality_gates"]  # [N, L, T]

    # 展平特征
    features = quality_gates.view(quality_gates.size(0), -1)  # [N, L*T]

    # K-Means 聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features.numpy())

    # 计算质量分数
    quality_scores = torch.sigmoid(quality_gates).mean(dim=(1, 2))

    # 从每个簇中选择
    selected_indices = []
    samples_per_cluster = int(len(features) * selection_percentage / num_clusters)

    for cluster_id in range(num_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = torch.where(torch.tensor(cluster_mask))[0]
        cluster_scores = quality_scores[cluster_indices]

        # 选择该簇中质量最高的样本
        sorted_cluster = cluster_indices[torch.argsort(cluster_scores, descending=True)]
        selected_indices.extend(sorted_cluster[:samples_per_cluster].tolist())

    return torch.tensor(selected_indices)
```

### 调试技巧

**转换脚本调试**:
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用 CPU 避免 CUDA 错误
python scripts/convert_qwen_to_quality_gate.py \
    --model Qwen/Qwen3-1.7B \
    --device cpu

# 检查中间输出
with torch.no_grad():
    outputs = model(test_input, output_router_logits=True)
    print(f"Router logits: {[rl.shape for rl in outputs.router_logits]}")
```

**筛选脚本调试**:
```python
# 打印统计信息
router_data = torch.load("router_data.pt")
quality_gates = router_data["quality_gates"]
quality_scores = torch.sigmoid(quality_gates).mean(dim=(1, 2))

print(f"Quality scores 统计:")
print(f"  Min: {quality_scores.min().item():.4f}")
print(f"  Max: {quality_scores.max().item():.4f}")
print(f"  Mean: {quality_scores.mean().item():.4f}")
print(f"  Std: {quality_scores.std().item():.4f}")

# 可视化分布
import matplotlib.pyplot as plt
plt.hist(quality_scores.numpy(), bins=50)
plt.xlabel("Quality Score")
plt.ylabel("Frequency")
plt.savefig("quality_score_distribution.png")
```
