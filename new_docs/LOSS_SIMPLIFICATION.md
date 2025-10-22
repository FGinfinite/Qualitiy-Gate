# 质量损失函数简化说明

## 概述

已将质量分类损失从多种损失方式简化为单一的**线性损失（Linear Loss）**方式。

## 主要变更

### 1. 删除的损失函数

以下复杂的损失函数已被移除：

- ❌ `beta_moment_matching_loss()` - Beta分布矩匹配损失
- ❌ `mean_variance_regularization_loss()` - 均值方差正则化损失

### 2. 删除的配置参数

以下配置参数不再需要：

```python
# 已删除的参数
beta_target_mean: float = 0.5
beta_target_var: float = 0.05
w_mean: float = 1.0
w_var: float = 1.0
lambda_var: float = 0.1
```

### 3. 简化后的损失函数

#### quality_classification_loss()

**之前：** 支持多种损失类型（sigmoid、mse、beta_moment_matching、mean_variance_regularization、custom）

**现在：** 仅支持**线性损失**

**实现：**
```python
def quality_classification_loss(
    router_logits: List[torch.Tensor],
    config: QualityGateConfig,
    attention_mask: Optional[torch.Tensor] = None,
    debug: bool = False,
) -> torch.Tensor:
    """
    计算质量分类损失
    
    使用线性损失：直接使用sigmoid后的good_ratio作为损失，鼓励降低good_ratio
    """
    # 直接使用 sigmoid(quality_score) 作为损失
    good_ratio = torch.sigmoid(quality_score)
    layer_loss_raw = good_ratio.squeeze(-1)
    # ... 应用 attention mask 和平均化策略
```

**数学公式：**
```
Loss = mean_layers(mean_tokens(sigmoid(quality_score)))
```

其中：
- `quality_score` 是质量门控的原始输出
- `sigmoid()` 将其转换为 [0, 1] 范围的概率
- 直接最小化这个概率值，鼓励模型降低 good_ratio

### 4. 配置更新

#### QualityGateConfig

**之前：**
```python
quality_loss_type: str = "sigmoid"
beta_target_mean: float = 0.5
beta_target_var: float = 0.05
w_mean: float = 1.0
w_var: float = 1.0
lambda_var: float = 0.1
sample_wise_averaging: bool = True
```

**现在：**
```python
quality_loss_type: str = "linear"  # 仅此一种
sample_wise_averaging: bool = True  # 保留平均策略
```

#### stage_1_warmup.yaml

**之前：**
```yaml
quality_loss_type: "sigmoid"
quality_loss_params:
  beta_target_mean: 0.5
  beta_target_var: 0.05
  w_mean: 1.0
  w_var: 1.0
  lambda_var: 0.1
  sample_wise_averaging: true
  full_sequence_prediction: true
  mask_special_tokens: true
```

**现在：**
```yaml
quality_loss_type: "linear"
quality_loss_params:
  sample_wise_averaging: true
  full_sequence_prediction: true
  mask_special_tokens: true
```

### 5. 调用简化

#### 之前

```python
quality_loss = quality_classification_loss(
    outputs.router_logits,
    self.config,
    attention_mask=attention_mask,
    loss_type=self.config.quality_loss_type,  # 需要指定类型
    debug=self.config.quality_loss_debug,
)
```

#### 现在

```python
quality_loss = quality_classification_loss(
    outputs.router_logits,
    self.config,
    attention_mask=attention_mask,
    debug=self.config.quality_loss_debug,
)
```

## 保留的功能

### 1. 平均化策略

仍然支持两种平均化策略：

- **Sample-wise averaging**: 先计算每个样本的平均损失，再对样本求平均（避免长序列偏置）
- **Token-wise averaging**: 直接对所有有效token求和平均

通过 `sample_wise_averaging` 参数控制。

### 2. Attention Mask 处理

正确处理 padding tokens，仅计算有效 token 的损失。

### 3. 调试模式

通过 `quality_loss_debug=true` 仍可启用详细的调试输出。

## 代码简化效果

### 行数减少

- **删除代码**: ~120 行（两个复杂损失函数）
- **简化代码**: ~60 行（移除条件分支）
- **总计减少**: ~180 行

### 参数减少

- **配置参数**: 从 11 个减少到 4 个
- **函数参数**: 从 7 个减少到 4 个

### 复杂度降低

- **条件分支**: 从 6 种损失类型减少到 1 种
- **依赖参数**: 从 6 个减少到 1 个

## 使用示例

```python
# 创建配置
config = QualityGateConfig(
    quality_loss_type="linear",
    quality_loss_weight=1.0,
    sample_wise_averaging=True,
)

# 计算损失（自动使用线性损失）
loss = quality_classification_loss(
    router_logits=outputs.router_logits,
    config=config,
    attention_mask=attention_mask,
)
```

## 优势

1. **简单明了**: 损失函数逻辑清晰，易于理解
2. **易于维护**: 减少代码量，降低维护成本
3. **高效执行**: 减少条件判断，提高执行效率
4. **专注目标**: 直接优化质量门控，不引入额外复杂性

## 迁移指南

### 如果您之前使用 "sigmoid" 损失

无需任何更改！线性损失就是原来的 sigmoid 损失，只是改了个名字。

### 如果您之前使用其他损失类型

需要更新配置文件，将 `quality_loss_type` 改为 `"linear"`，并移除相关参数。

## 验证

```bash
# 验证配置加载
cd /home/lishiwei/Quality-Gate
source .venv/bin/activate
python -c "from src.models.quality_gate_model import QualityGateConfig; print(QualityGateConfig().quality_loss_type)"
# 输出: linear
```

## 相关文件

- `src/models/quality_gate_model.py` - 质量门控模型实现
- `configs/stage_1_warmup.yaml` - 预热训练配置
- `src/stages/warmup.py` - 预热训练逻辑

## 总结

通过这次简化：
- ✅ 代码更简洁（减少 ~180 行）
- ✅ 配置更清晰（减少 7 个参数）
- ✅ 逻辑更直接（1 种损失类型）
- ✅ 性能更高效（减少条件分支）
- ✅ 保持功能完整（平均策略、mask处理等）

