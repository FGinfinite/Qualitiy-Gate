# src/models 模块文档

[根目录](/mnt/lishiwei/Quality-Gate/CLAUDE.md) > [src](/mnt/lishiwei/Quality-Gate/src/) > **models**

---

## 变更记录 (Changelog)

### 2025-10-23
- 初始化模块文档
- 记录 Quality-Gate 模型架构和接口

---

## 模块职责

`src/models/` 模块负责定义和管理 Quality-Gate 模型架构，核心包括：

1. **Quality-Gate 模型**: 在 Qwen3 基础上添加质量门控层
2. **Select-MoE 模型** (遗留): 早期基于 OLMoE 的实现，已被 Quality-Gate 替代
3. **模型注册**: 将自定义模型注册到 HuggingFace AutoModel 系统

---

## 入口与启动

### 模型加载

```python
from src.models.quality_gate_model import (
    QualityGateForCausalLM,
    register_quality_gate
)

# 注册模型（必须在加载前执行）
register_quality_gate()

# 加载预转换的模型
model = QualityGateForCausalLM.from_pretrained(
    "./converted_models/quality_gate_Qwen3-1.7B",
    torch_dtype=torch.bfloat16
)
```

### 训练时使用

```python
# 启用质量门控输出
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    output_router_logits=True  # 关键：训练时必须为 True
)

# 访问质量门控 logits
for layer_idx, quality_score in enumerate(outputs.router_logits):
    # quality_score: [batch_size, seq_len, 1]
    good_ratio = torch.sigmoid(quality_score)
    print(f"Layer {layer_idx}: mean good_ratio = {good_ratio.mean():.4f}")
```

---

## 对外接口

### 主要类

#### `QualityGateConfig`
**继承**: `Qwen3Config`

质量门控模型配置类，扩展自 Qwen3Config。

**核心参数**:
```python
QualityGateConfig(
    # Qwen3 基础参数
    vocab_size=151936,
    hidden_size=1536,
    num_hidden_layers=28,

    # 质量门控特定参数
    quality_gate_init_mean=0.0,       # 门控初始化均值
    quality_gate_init_std=0.02,       # 门控初始化标准差
    quality_loss_weight=0.5,          # 质量损失权重

    # 质量损失配置
    quality_loss_type="linear",       # 损失类型（仅支持 linear）
    sample_wise_averaging=True,       # 样本级平均（vs token级）
    quality_loss_debug=False,         # 调试输出
)
```

#### `QualityGate`
**继承**: `nn.Module`

质量分类的门控网络，输出单个分数。

**前向传播**:
```python
def forward(hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        hidden_states: [batch_size, seq_len, hidden_size]

    Returns:
        quality_score: 质量分数原始值 [batch_size, seq_len, 1]
        good_ratio: sigmoid后的good比率 [batch_size, seq_len, 1]
    """
```

#### `QualityGateDecoderLayer`
**继承**: `Qwen3DecoderLayer`

带有质量门控的解码器层，在 FFN 之前插入质量门控。

**架构**:
```
Input → LayerNorm → SelfAttention → Residual →
LayerNorm → QualityGate → FFN → Residual → Output
              ↓
        quality_score
```

#### `QualityGateForCausalLM`
**继承**: `QualityGatePreTrainedModel`

因果语言模型，包含质量门控和 LM head。

**核心方法**:
```python
def forward(
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor = None,
    labels: torch.LongTensor = None,
    output_router_logits: bool = False,
    **kwargs
) -> CausalLMOutputWithPast:
    """
    前向传播

    重要：训练时必须设置 output_router_logits=True 才能获取质量门控输出
    """
```

### 辅助函数

#### `quality_classification_loss()`
计算质量分类损失。

**功能**:
- 对所有层的质量门控输出计算损失
- 自动处理 attention_mask，排除 padding tokens
- 支持 sample-wise 或 token-wise 平均

**签名**:
```python
def quality_classification_loss(
    router_logits: List[torch.Tensor],
    config: QualityGateConfig,
    attention_mask: Optional[torch.Tensor] = None,
    debug: bool = False,
) -> torch.Tensor
```

#### `register_quality_gate()`
将 Quality-Gate 模型注册到 HuggingFace AutoModel 系统。

**用法**:
```python
from src.models.quality_gate_model import register_quality_gate

# 必须在加载模型前调用
register_quality_gate()

# 之后可以使用标准的 from_pretrained()
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("path/to/quality_gate_model")
```

---

## 关键依赖与配置

### 外部依赖

```python
# 核心依赖
torch
transformers==4.53.2

# 模型组件
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
```

### 配置参数

训练时需要在 Hydra 配置中设置：

```yaml
# configs/stage_1_warmup.yaml
selector_model:
  path: "converted_models/quality_gate_Qwen3-1.7B"
  tokenizer_name: "Qwen/Qwen3-1.7B"

training:
  quality_loss_weight: 1.0
  quality_gate_init_mean: 0.0
  quality_gate_init_std: 0.02
  quality_loss_type: "linear"
  quality_loss_debug: false
  quality_loss_params:
    sample_wise_averaging: true
    full_sequence_prediction: true
    mask_special_tokens: true
```

---

## 数据模型

### 输入格式

```python
# 标准输入
{
    "input_ids": torch.LongTensor,      # [batch_size, seq_len]
    "attention_mask": torch.Tensor,     # [batch_size, seq_len]
    "labels": torch.LongTensor,         # [batch_size, seq_len], -100 for padding
}
```

### 输出格式

```python
# CausalLMOutputWithPast
{
    "loss": torch.Tensor,               # 标量，总损失
    "logits": torch.Tensor,             # [batch_size, seq_len, vocab_size]
    "past_key_values": ...,             # KV cache (可选)
    "hidden_states": ...,               # 隐藏状态 (可选)
    "attentions": ...,                  # 注意力权重 (可选)
}

# 添加的属性（当 output_router_logits=True）
outputs.router_logits = [
    quality_score_layer_0,  # [batch_size, seq_len, 1]
    quality_score_layer_1,  # [batch_size, seq_len, 1]
    ...
]
```

---

## 测试与质量

### 模型验证

**转换后验证**:
```python
# scripts/convert_qwen_to_quality_gate.py 自动执行以下测试

# 1. 前向传播测试
test_input = torch.randint(0, vocab_size, (1, 8))
outputs = model(test_input, output_router_logits=False)
assert outputs.logits.shape == (1, 8, vocab_size)

# 2. 质量门控输出测试
outputs = model(test_input, output_router_logits=True)
assert hasattr(outputs, 'router_logits')
assert len(outputs.router_logits) == num_hidden_layers
assert outputs.router_logits[0].shape == (1, 8, 1)
```

### 损失函数测试

**调试模式**:
```python
# 启用调试输出
model.config.quality_loss_debug = True

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    output_router_logits=True
)

# 控制台会打印：
# === 质量分类损失调试 ===
# router_logits类型: <class 'list'>
# 层数量: 28
# --- 第0层 ---
# quality_score形状: torch.Size([batch, seq_len, 1])
# good_ratio 最小/最大/均值: 0.2341/0.8765/0.5123
# ...
```

---

## 常见问题 (FAQ)

### Q: 为什么训练时必须设置 `output_router_logits=True`？

A: 因为质量门控的 logits 只在此模式下才会被收集和返回。如果不设置，损失函数无法计算质量分类损失，导致质量门控参数不会被训练。

### Q: Quality-Gate 和 Select-MoE 有什么区别？

A:
- **Select-MoE**: 基于 OLMoE (Mixture-of-Experts)，早期实现，已废弃
- **Quality-Gate**: 基于 Qwen3 (标准 Transformer)，当前使用的架构
- 两者都实现质量门控机制，但底层架构完全不同

### Q: 如何自定义质量损失函数？

A: 目前仅支持 `linear` 类型。如需添加新类型：

1. 在 `quality_classification_loss()` 中添加新的分支
2. 更新配置参数 `quality_loss_type` 的选项
3. 确保正确处理 `attention_mask` 和 `sample_wise_averaging`

示例：
```python
def quality_classification_loss(router_logits, config, attention_mask, debug):
    # ... 现有代码 ...

    if config.quality_loss_type == "linear":
        layer_loss_raw = good_ratio.squeeze(-1)
    elif config.quality_loss_type == "mse":
        # 新增：MSE 损失
        layer_loss_raw = (good_ratio.squeeze(-1) - 0.0) ** 2
    else:
        raise ValueError(f"Unsupported loss type: {config.quality_loss_type}")
```

### Q: 为什么使用 sample_wise_averaging？

A: 避免长序列偏差：
- **Token-wise**: 直接对所有 token 求平均，长序列贡献更大
- **Sample-wise**: 先对每个样本求平均，再对样本求平均，公平对待所有样本

---

## 相关文件清单

```
src/models/
├── __init__.py                    # 模块导出
├── quality_gate_model.py          # Quality-Gate 模型定义（核心）
│   ├── QualityGateConfig          # 配置类
│   ├── QualityGate                # 质量门控层
│   ├── QualityGateDecoderLayer    # 解码器层
│   ├── QualityGateModel           # 基础模型
│   ├── QualityGateForCausalLM     # 因果语言模型
│   ├── quality_classification_loss # 损失函数
│   └── register_quality_gate      # 模型注册
└── select_moe.py                  # Select-MoE 模型（遗留）
```

**相关脚本**:
- `/mnt/lishiwei/Quality-Gate/scripts/convert_qwen_to_quality_gate.py`: 模型转换
- `/mnt/lishiwei/Quality-Gate/scripts/compare_quality_gate_model.py`: 模型对比验证

**相关配置**:
- `/mnt/lishiwei/Quality-Gate/configs/stage_1_warmup.yaml`: 预热训练配置
- `/mnt/lishiwei/Quality-Gate/configs/stage_2_selection.yaml`: 数据选择配置

---

## 开发指南

### 添加新的模型架构

如需支持其他基座模型（如 LLaMA, Mistral）：

1. 创建新的配置类继承自基座的 Config
2. 创建新的解码器层类，插入 QualityGate
3. 组装新的模型类
4. 编写对应的转换脚本
5. 注册到 AutoModel 系统

**参考模板**:
```python
class NewModelConfig(BaseConfig):
    model_type = "new_model_quality_gate"
    def __init__(self, quality_gate_init_mean=0.0, ...):
        super().__init__(...)
        self.quality_gate_init_mean = quality_gate_init_mean

class NewModelDecoderLayer(BaseDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.quality_gate = QualityGate(config)  # 插入质量门控

    def forward(self, hidden_states, ...):
        # 自注意力
        hidden_states = self.self_attn(...)

        # 质量门控
        quality_score, good_ratio = self.quality_gate(hidden_states)

        # FFN
        hidden_states = self.mlp(hidden_states)

        return hidden_states, quality_score
```

### 调试技巧

1. **查看质量门控输出分布**:
```python
outputs = model(input_ids, attention_mask, output_router_logits=True)
quality_scores = torch.cat([qs for qs in outputs.router_logits], dim=1)  # [batch, layers*seq_len, 1]
good_ratios = torch.sigmoid(quality_scores)

print(f"Good ratio 统计:")
print(f"  - Min: {good_ratios.min().item():.4f}")
print(f"  - Max: {good_ratios.max().item():.4f}")
print(f"  - Mean: {good_ratios.mean().item():.4f}")
print(f"  - Std: {good_ratios.std().item():.4f}")
```

2. **验证梯度流**:
```python
for name, param in model.named_parameters():
    if "quality_gate" in name:
        print(f"{name}: grad_fn={param.grad_fn}, requires_grad={param.requires_grad}")
```

3. **检查损失组成**:
```python
model.config.quality_loss_debug = True
outputs = model(input_ids, attention_mask, labels=labels, output_router_logits=True)

print(f"Total loss: {outputs.loss.item():.6f}")
# 调试输出会显示每层的质量损失
```

---

## 更新日志

### 主要架构演变

**v1.0 (早期)**: Select-MoE
- 基于 OLMoE 的 MoE 架构
- 使用 2-类分类 (good/bad)

**v2.0 (当前)**: Quality-Gate
- 基于 Qwen3 的标准 Transformer
- 使用单分数 + sigmoid
- 更简洁的损失计算
- 更好的梯度流动

**关键改进**:
- 从 softmax 归一化改为 sigmoid 激活
- 从 2-类 logits 改为单个原始分数
- 添加 attention_mask 处理
- 支持 sample-wise averaging
