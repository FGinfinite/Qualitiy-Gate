# Qwen2 → Qwen3 迁移说明

## 概述

项目已从 Qwen2 架构迁移到 Qwen3 架构。这次迁移影响了质量门控模型的实现。

## 主要变更

### 1. 导入更新

**之前 (Qwen2):**
```python
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Config,
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2Model,
)
```

**现在 (Qwen3):**
```python
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
)
```

### 2. 配置类更新

- `QualityGateConfig` 现在继承自 `Qwen3Config` 而非 `Qwen2Config`
- 继承了 Qwen3 的所有配置参数，包括 `layer_types` 等新特性

### 3. 解码器层差异

#### Qwen2DecoderLayer → Qwen3DecoderLayer

**主要变化：**

1. **自注意力返回值**
   - Qwen2: `hidden_states, self_attn_weights, present_key_value = self.self_attn(...)`
   - Qwen3: `hidden_states, self_attn_weights = self.self_attn(...)`
   - Qwen3 不在 layer 级别返回 `present_key_value`

2. **缓存处理**
   - Qwen2: 每层返回并传递 cache
   - Qwen3: cache 在模型级别统一管理

3. **参数类型**
   - `past_key_value`: `Tuple[torch.Tensor]` → `torch.Tensor` (Cache object)

### 4. 模型级别差异

#### QualityGateModel

**Router logits 收集逻辑更新：**

```python
# Qwen2 风格
if use_cache:
    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

# Qwen3 风格
# cache 在 model 级别处理，不从 layer_outputs 提取
if output_router_logits:
    if output_attentions:
        all_router_logits += (layer_outputs[2],)
    else:
        all_router_logits += (layer_outputs[1],)
```

**缓存返回：**

```python
# Qwen2
next_cache = next_decoder_cache if use_cache else None

# Qwen3
next_cache = past_key_values if use_cache else None
```

### 5. 辅助组件更新

- `Qwen2RMSNorm` → `Qwen3RMSNorm`
- `Qwen2RotaryEmbedding` → `Qwen3RotaryEmbedding`

### 6. 配置文件更新

**configs/stage_1_warmup.yaml:**
```yaml
selector_model:
  path: "Qwen/Qwen3-1.7B"  # 之前: Qwen/Qwen2.5-1.5B
  tokenizer_name: "Qwen/Qwen3-1.7B"
```

## 兼容性说明

### 保持不变的部分

1. **质量门控逻辑**: 质量门控的核心实现（`QualityGate` 类）完全不变
2. **损失函数**: 所有质量损失函数保持不变
3. **API 接口**: 对外暴露的 API 接口保持兼容
4. **配置参数**: 质量门控特定参数（如 `quality_loss_weight` 等）保持不变

### 需要注意的变化

1. **模型路径**: 需要使用 Qwen3 的模型路径
2. **虚拟环境**: 需要安装支持 Qwen3 的 transformers 版本
3. **缓存机制**: 内部缓存处理机制有所不同，但对用户透明

## 测试验证

```bash
# 验证模型导入
cd /home/lishiwei/Quality-Gate
source .venv/bin/activate
python -c "from src.models.quality_gate_model import register_quality_gate; register_quality_gate(); print('成功!')"
```

## Transformers 版本要求

- **最低版本**: transformers >= 4.50.0 (支持 Qwen3)
- **推荐版本**: transformers >= 4.53.0

## 迁移检查清单

- [x] 更新所有 Qwen2 导入为 Qwen3
- [x] 调整解码器层的前向传播逻辑
- [x] 更新缓存处理机制
- [x] 修改配置文件中的模型路径
- [x] 更新文档和 README
- [x] 验证模型可以正确导入

## 已知问题

暂无

## 相关文件

- `src/models/quality_gate_model.py` - 主要模型实现
- `configs/stage_1_warmup.yaml` - 预热训练配置
- `README_REFACTOR.md` - 项目重构文档

## 参考

- [Qwen3 模型文档](https://huggingface.co/Qwen/Qwen3-1.7B)
- [Transformers Qwen3 源码](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3)

