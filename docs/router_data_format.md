# 路由数据重构说明

## 概述

重构了 `src/stages/selection.py` 中的张量保存逻辑，现在同时保存一级路由（质量门）和二级路由（MoE）的完整输出，并通过原数据的唯一 ID 建立精确对应关系。两种路由数据均在序列维度进行平均，保持形状一致性。

## 新的数据格式

### 1. 完整路由数据字典

保存在 `outputs/stage_2_selection/YYYY-MM-DD/HH-MM-SS/router_data/{dataset_name}_router_data.pt`

```python
{
    'quality_logits': Tensor,      # [N, L, 2] - 质量门平均概率
    'moe_logits': Tensor,          # [N, L, E] - MoE路由平均概率分布  
    'sample_ids': List[str],       # 样本的唯一ID标识（例如: ["oasst1_25460", ...]）
    'dataset_name': str,           # 数据集名称
    'num_samples': int,            # 样本数量
    'metadata': dict               # 元数据和说明
}
```

**维度说明：**
- `N`: 样本数量
- `L`: 模型层数 (num_hidden_layers)
- `E`: 专家数量 (num_experts)
- `2`: 质量分类 [good_prob, bad_prob]

**重要特性：**
- 两种路由数据均已在序列维度进行平均，避免了序列长度不一致的问题
- 质量门和MoE路由的形状保持一致性：`[N, L, X]`，便于后续处理

### 2. ID对应关系

- `sample_ids[i]` = 第i个样本的唯一ID（例如: "oasst1_25460"）
- `quality_logits[i]` = 第i个样本的质量门平均概率 `[L, 2]`
- `moe_logits[i]` = 第i个样本的MoE路由平均概率 `[L, E]`

**ID格式**: 通常为 `{dataset_name}_{number}`，例如:
- `oasst1_25460` - OASST1数据集的第25460个样本
- `CoT_12345` - CoT数据集的第12345个样本

## 主要改进

### 1. 完整的路由信息
- **之前**: 只保存MoE路由的平均概率分布
- **现在**: 同时保存质量门和MoE路由的平均概率

### 2. 形状一致性
- **之前**: 质量门 `[N, L, S, 2]` vs MoE路由 `[N, L, E]`，形状不一致
- **现在**: 质量门 `[N, L, 2]` vs MoE路由 `[N, L, E]`，形状保持一致

### 3. 简化的数据处理
- **之前**: 需要处理不同序列长度，存储格式复杂
- **现在**: 统一在序列维度平均，存储简单高效

### 4. 精确的ID映射
- 直接使用原数据的唯一ID，确保精确对应

## 使用方法

### 加载路由数据
```python
from src.stages.selection import load_router_data

router_data = load_router_data('path/to/dataset_router_data.pt')
```

### 获取特定样本信息
```python
from src.stages.selection import get_sample_router_info

# 根据样本ID获取路由信息
sample_info = get_sample_router_info(router_data, sample_id="oasst1_25460")

print(f"样本ID: {sample_info['sample_id']}")
print(f"质量门logits形状: {sample_info['quality_logits'].shape}")  # [L, 2]
print(f"MoE路由logits形状: {sample_info['moe_logits'].shape}")     # [L, E]
```

### 查找原数据对应关系
现在可以直接通过ID在原数据集中查找对应的样本：

```python
# 假设你有原始数据文件
import json

def find_original_sample(data_file_path, sample_id):
    """在原数据集中查找指定ID的样本"""
    with open(data_file_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            if sample['id'] == sample_id:
                return sample
    return None

# 使用示例
original_sample = find_original_sample(
    'dataset/train/processed/oasst1/oasst1_data_coding.jsonl', 
    'oasst1_25460'
)
print(f"原始样本: {original_sample}")
```

### 分析质量分数
```python
# 从质量门logits计算质量分数（已经平均过）
quality_logits = sample_info['quality_logits']  # [L, 2]
quality_probs = torch.softmax(quality_logits, dim=-1)  # [L, 2]
good_probs = quality_probs[:, 0]  # [L] - 各层好数据概率

# 计算整体质量分数
overall_score = good_probs.mean().item()
print(f"整体质量分数: {overall_score:.6f}")
print(f"各层质量分数: {good_probs.tolist()}")
```

### 分析专家使用情况
```python
# MoE路由分析
moe_probs = sample_info['moe_logits']  # [L, E]
expert_usage = moe_probs.mean(dim=0)  # [E] - 各专家平均使用率

print(f"最活跃专家: {expert_usage.argmax().item()}")
print(f"专家使用熵: {torch.distributions.Categorical(expert_usage).entropy().item()}")
```

## 示例脚本

运行 `examples/analyze_router_data.py` 查看完整的数据分析示例：

```bash
python examples/analyze_router_data.py outputs/stage_2_selection/2024-XX-XX/XX-XX-XX/router_data/oasst1_router_data.pt
```

### 编程查询示例
```python
from src.stages.selection import load_router_data, get_sample_router_info

# 加载数据
data = load_router_data('path/to/oasst1_router_data.pt')

# 查询特定样本
info = get_sample_router_info(data, 'oasst1_25460')
print(info)
```

## 兼容性说明

- 多样性选择算法仍使用MoE logits，保持原有功能不变
- 不再保存向后兼容的logits文件，简化了存储结构
- 现有的分析脚本需要根据新的数据格式进行更新

## 优势总结

1. **直接对应**: 通过原数据的唯一ID直接对应，不需要复杂的索引计算
2. **形状一致**: 两种路由数据形状保持一致 `[N, L, X]`，便于统一处理
3. **简化存储**: 在序列维度平均，避免序列长度不一致问题
4. **高效处理**: 不需要处理可变序列长度，代码逻辑更简单
5. **完整信息**: 同时保存质量门和MoE路由的平均概率，支持全面分析