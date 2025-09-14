[根目录](../../CLAUDE.md) > [src](../) > **models**

# 核心模型模块 - Select-MoE

## 变更记录 (Changelog)

**2025-09-14 22:43:47** - 模块文档初始化
- 创建模块级架构文档
- 添加两层路由架构详细说明
- 完善接口和使用示例

## 模块职责

核心模型模块负责实现 Select-MoE 的两层路由架构，包括：
- **质量门控（Quality Gate）**：评估输入数据质量，输出单一质量分数
- **MoE专家路由**：基于质量评估结果路由到不同的专家网络
- **垃圾专家（Trash Expert）**：处理低质量数据的特殊专家
- **质量分类损失**：可扩展的损失函数框架

## 入口与启动

### 主要入口点
- `src/models/select_moe.py` - Select-MoE模型核心实现
- `src/models/__init__.py` - 模型注册和导出

### 模型注册
```python
from src.models.select_moe import register_select_moe

# 注册模型（使用前必须调用）
register_select_moe()

# 加载模型
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("path/to/select_moe_model")
```

## 对外接口

### 核心类和配置

#### SelectMoeConfig
```python
class SelectMoeConfig(OlmoeConfig):
    """Select-MoE模型配置类"""

    # 质量门控参数
    quality_gate_init_mean: float = 0.0
    quality_gate_init_std: float = 0.02
    quality_loss_weight: float = 0.5

    # 垃圾专家模式
    trash_expert_mode: str = "zero"  # "zero", "noise", "custom"

    # 损失函数配置
    quality_loss_type: str = "sigmoid"
    beta_target_mean: float = 0.5
    beta_target_var: float = 0.05
```

#### SelectMoeForCausalLM
```python
class SelectMoeForCausalLM(PreTrainedModel):
    """Select-MoE因果语言模型"""

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                output_router_logits=False):
        """
        前向传播

        Returns:
            MoeCausalLMOutputWithPast: 包含router_logits的输出
            - router_logits: List[Dict] 每层的路由输出
                - quality_score: [batch, seq_len, 1] 质量分数
                - moe_logits: [batch*seq_len, num_experts] MoE路由logits
        """
```

### 路由输出格式
```python
# 每一层的路由器输出
for layer_output in outputs.router_logits:
    quality_score = layer_output["quality_score"]   # Shape: [batch, seq_len, 1]
    moe_logits = layer_output["moe_logits"]         # Shape: [batch*seq_len, num_experts]

    # 计算质量比例
    good_ratio = torch.sigmoid(quality_score)       # Shape: [batch, seq_len, 1]
    bad_ratio = 1.0 - good_ratio                    # Shape: [batch, seq_len, 1]
```

## 关键依赖与配置

### 依赖库
- `torch>=2.6.0` - 核心深度学习框架
- `transformers==4.53.2` - HuggingFace变压器库
- `accelerate>=1.8.1` - 分布式训练加速

### 配置参数

#### 质量损失配置
```yaml
quality_loss_type: "sigmoid"  # 损失函数类型
quality_loss_debug: false     # 启用调试输出
quality_loss_params:
  # Beta分布矩匹配参数
  beta_target_mean: 0.5
  beta_target_var: 0.05
  w_mean: 1.0
  w_var: 1.0
  # 均值方差正则化参数
  lambda_var: 0.1
  # 数据编码配置
  sample_wise_averaging: true
  full_sequence_prediction: true
  mask_special_tokens: true
```

## 数据模型

### 路由器架构
```
输入序列
    ↓
质量门控 (Quality Gate)
    ├─ 质量分数 → Sigmoid → good_ratio
    └─ bad_ratio = 1 - good_ratio
    ↓
MoE专家路由
    ├─ 专家1, 专家2, ..., 专家N
    └─ 垃圾专家 (Trash Expert)
    ↓
输出表示
```

### 损失函数类型

1. **Sigmoid损失**（默认）
   ```python
   quality_loss = F.binary_cross_entropy_with_logits(quality_scores, target_labels)
   ```

2. **Beta分布矩匹配**
   ```python
   # 目标：使质量分数分布匹配Beta分布
   mean_loss = (pred_mean - target_mean) ** 2
   var_loss = (pred_var - target_var) ** 2
   quality_loss = w_mean * mean_loss + w_var * var_loss
   ```

3. **均值方差正则化**
   ```python
   # 均值居中 + 方差正则化
   centered_scores = quality_scores - quality_scores.mean()
   quality_loss = centered_scores.mean() ** 2 + lambda_var * centered_scores.var()
   ```

## 测试与质量

### 单元测试
目前缺乏专门的单元测试，建议添加：
- 模型配置验证测试
- 前向传播输出格式测试
- 路由器logits维度测试
- 损失函数计算测试

### 验证脚本
```bash
# 模型转换验证
python scripts/compare_converted_model.py --converted-model ./converted_models/select_moe_converted_OLMoE-1B-7B-0125

# 模型加载测试
python -c "from src.models.select_moe import register_select_moe; register_select_moe(); print('模型注册成功')"
```

## 常见问题 (FAQ)

**Q: 如何切换不同的质量损失函数？**
A: 在配置文件中设置 `training.quality_loss_type`，支持 "sigmoid", "beta_moment_matching", "mean_variance_regularization"。

**Q: 垃圾专家有什么作用？**
A: 垃圾专家专门处理低质量数据，避免影响高质量专家的学习。可配置为零输出、噪声输出或自定义输出。

**Q: 如何调试质量分数输出？**
A: 设置 `training.quality_loss_debug=true` 启用详细的调试输出。

**Q: 模型转换后如何验证正确性？**
A: 使用 `scripts/compare_converted_model.py` 脚本对比转换前后的输出一致性。

## 相关文件清单

### 核心实现文件
- `select_moe.py` - 主要模型实现（1000+ 行）
- `__init__.py` - 模块初始化和导出

### 配置文件
- `../../configs/stage_1_warmup.yaml` - 预训练配置
- `../../configs/training/` - 各模型训练配置

### 转换工具
- `../../scripts/convert_olmoe_to_select_moe.py` - 模型转换脚本
- `../../scripts/compare_converted_model.py` - 转换验证脚本

### 相关文档
- `../../docs/custom_loss_fn.md` - 自定义损失函数文档
- `../../docs/router_data_format.md` - 路由数据格式说明