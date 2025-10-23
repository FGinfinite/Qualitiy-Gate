# src/data 模块文档

[根目录](/mnt/lishiwei/Quality-Gate/CLAUDE.md) > [src](/mnt/lishiwei/Quality-Gate/src/) > **data**

---

## 变更记录 (Changelog)

### 2025-10-23
- 初始化模块文档
- 记录数据加载和预处理功能

---

## 模块职责

`src/data/` 模块负责数据加载、预处理和编码：

1. **数据加载**: 支持本地数据集和 HuggingFace 数据集
2. **数据预处理**: 格式转换、采样、排序
3. **数据编码**: 将文本转换为模型输入格式
4. **数据统计**: 计算和展示数据集统计信息

---

## 入口与启动

### 基本使用

```python
from src.data import (
    load_and_prepare_dataset,
    encode_data,
    get_data_statistics,
)

# 加载数据集
dataset = load_and_prepare_dataset(cfg)

# 编码数据
tokenized_dataset = encode_data(
    dataset,
    tokenizer,
    max_seq_length=1024,
)

# 查看统计信息
get_data_statistics(tokenized_dataset)
```

---

## 对外接口

### load_and_prepare_dataset()

```python
def load_and_prepare_dataset(cfg: DictConfig) -> Dataset:
    """
    根据配置加载并准备数据集

    支持两种数据源：
    - local: 本地 JSONL 文件
    - hf: HuggingFace datasets

    Args:
        cfg: Hydra 配置对象，包含 dataset 配置

    Returns:
        Dataset: HuggingFace datasets 对象

    配置示例:
        dataset:
          dataset_from: "hf"
          hf:
            datasets:
              - name: "teknium/OpenHermes-2.5"
                dataset_name: "openhermes"
          subset_ratio: 0.025
          shuffle: true
    """
```

**功能**:
1. 根据 `dataset_from` 选择加载方式
2. 加载单个或多个数据集
3. 合并多数据集（如果有）
4. 数据采样（根据 `subset_ratio`）
5. 数据排序（根据 `sort_by_length`）
6. 数据打乱（根据 `shuffle`）

---

### load_local_datasets()

```python
def load_local_datasets(
    data_dir: str,
    dataset_names: List[str],
    subset_ratio: float = 1.0,
    seed: int = 42
) -> Dataset:
    """
    从本地目录加载 JSONL 数据集

    Args:
        data_dir: 数据集根目录
        dataset_names: 数据集名称列表（子目录名）
        subset_ratio: 采样比例 (0, 1]
        seed: 随机种子

    Returns:
        合并后的 Dataset

    目录结构:
        data_dir/
        ├── dataset1/
        │   ├── train.jsonl
        │   └── ...
        └── dataset2/
            ├── train.jsonl
            └── ...

    数据格式 (JSONL):
        {"messages": [...], "dataset": "dataset1", "id": "dataset1_0"}
        {"messages": [...], "dataset": "dataset1", "id": "dataset1_1"}
    """
```

---

### load_selected_data()

```python
def load_selected_data(data_path: str) -> Dataset:
    """
    加载筛选后的数据

    用于阶段3，加载阶段2筛选的数据

    Args:
        data_path: 筛选数据路径（JSONL 格式）

    Returns:
        Dataset: 加载的数据集

    数据格式:
        {"messages": [...], "dataset": "openhermes", "id": "openhermes_12345"}
    """
```

---

### encode_data()

```python
def encode_data(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_seq_length: int = 1024,
    processing_num_workers: int = 10,
    overwrite_cache: bool = False,
    full_sequence_prediction: bool = False,
    mask_special_tokens: bool = False,
) -> Dataset:
    """
    将数据集编码为模型输入格式

    Args:
        dataset: 原始数据集
        tokenizer: 分词器
        max_seq_length: 最大序列长度
        processing_num_workers: 处理工作进程数
        overwrite_cache: 是否覆盖缓存
        full_sequence_prediction: 是否全序列预测
        mask_special_tokens: 是否掩盖特殊格式token

    Returns:
        Dataset: 编码后的数据集，包含:
            - input_ids: [seq_len]
            - attention_mask: [seq_len]
            - labels: [seq_len], -100 for masked tokens

    处理流程:
        1. 将 messages 转换为文本
        2. 应用 chat template
        3. Tokenization
        4. 截断和填充
        5. 创建 labels（如果需要）
    """
```

**关键参数说明**:

- **full_sequence_prediction**:
  - `True`: 整个序列都参与损失计算（质量门控预热阶段）
  - `False`: 只有 assistant 部分参与损失计算（标准 SFT）

- **mask_special_tokens**:
  - `True`: 掩盖 `<|user|>`, `<|assistant|>` 等格式 token
  - `False`: 不掩盖

---

### encode_with_messages_format()

```python
def encode_with_messages_format(
    batch_messages: List[List[Dict]],
    tokenizer: AutoTokenizer,
    max_seq_length: int = 1024,
) -> Dict[str, torch.Tensor]:
    """
    使用 messages 格式编码数据

    Args:
        batch_messages: 批量 messages
            [
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
            ]
        tokenizer: 分词器
        max_seq_length: 最大序列长度

    Returns:
        dict: 包含 input_ids, attention_mask, labels
    """
```

---

### get_data_statistics()

```python
def get_data_statistics(dataset: Dataset) -> None:
    """
    打印数据集统计信息

    统计内容:
        - 总样本数
        - 平均/最小/最大序列长度
        - 填充 token 比例
        - 有效 token 总数

    Args:
        dataset: 已编码的数据集
    """
```

---

### 辅助函数

#### `compute_string_lengths()`
```python
def compute_string_lengths(example: Dict) -> Dict:
    """
    计算样本的字符串长度

    用于按长度排序，提高推理效率

    Args:
        example: 包含 'messages' 字段的样本

    Returns:
        添加 'string_length' 字段的样本
    """
```

#### `sort_dataset_by_string_length()`
```python
def sort_dataset_by_string_length(
    dataset: Dataset,
    descending: bool = True,
    num_workers: int = 10
) -> Dataset:
    """
    按字符串长度排序数据集

    用于阶段2，优化推理效率（减少 padding）

    Args:
        dataset: 原始数据集
        descending: 是否降序排列
        num_workers: 计算长度的工作进程数

    Returns:
        排序后的数据集
    """
```

#### `temp_seed()`
```python
@contextmanager
def temp_seed(seed: int):
    """
    临时设置随机种子的上下文管理器

    用法:
        with temp_seed(42):
            dataset = dataset.shuffle()
    """
```

---

## 数据格式

### 原始数据格式

**Messages 格式**（推荐）:
```json
{
    "messages": [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language..."}
    ],
    "dataset": "openhermes",
    "id": "openhermes_12345"
}
```

### 编码后格式

```python
{
    "input_ids": [128000, 128006, ...],     # token IDs
    "attention_mask": [1, 1, 1, ..., 0],    # 1 for valid tokens, 0 for padding
    "labels": [128000, 128006, ..., -100],  # -100 for masked tokens
}
```

### Chat Template

项目使用 tokenizer 的默认 chat template：

```python
# Qwen 系列的 chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)

# 结果示例:
# <|im_start|>user
# What is Python?<|im_end|>
# <|im_start|>assistant
# Python is a programming language...<|im_end|>
```

---

## 配置示例

### 本地数据集

```yaml
dataset:
  dataset_from: "local"
  local:
    data_dir: "dataset/train/processed"
    dataset_names:
      - "cot"
      - "dolly"
      - "flan_v2"
      - "oasst1"
  subset_ratio: 0.025
  shuffle: true
  max_sequence_length: 1024
```

### HuggingFace 数据集

```yaml
dataset:
  dataset_from: "hf"
  hf:
    datasets:
      - name: "teknium/OpenHermes-2.5"
        dataset_name: "openhermes"
        subset: null
        split: "train"
  subset_ratio: 0.025
  shuffle: true
  max_sequence_length: 1024
```

### 多数据集混合

```yaml
dataset:
  dataset_from: "hf"
  hf:
    datasets:
      - name: "teknium/OpenHermes-2.5"
        dataset_name: "openhermes"
      - name: "HuggingFaceH4/ultrachat_200k"
        dataset_name: "ultrachat"
  subset_ratio: 0.05
  shuffle: true
```

---

## 常见问题 (FAQ)

### Q: 如何处理超长序列？

A: 自动截断：
```python
tokenized = tokenizer(
    text,
    truncation=True,
    max_length=max_seq_length,
)
```

### Q: 如何避免 padding 浪费计算？

A: 阶段2使用序列长度排序：
```yaml
dataset:
  sort_by_length: true  # 相似长度的样本在同一批次
```

### Q: 如何自定义数据格式？

A: 修改 `encode_data()` 中的转换逻辑：
```python
def custom_format_messages(messages: List[Dict]) -> str:
    """自定义格式化"""
    text_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        text_parts.append(f"{role.upper()}: {content}")
    return "\n".join(text_parts)

# 在 encode_data() 中使用
text = custom_format_messages(example["messages"])
```

### Q: 如何处理多轮对话？

A: Messages 格式天然支持多轮：
```json
{
    "messages": [
        {"role": "user", "content": "问题1"},
        {"role": "assistant", "content": "回答1"},
        {"role": "user", "content": "问题2"},
        {"role": "assistant", "content": "回答2"}
    ]
}
```

### Q: 如何只训练 assistant 的回复？

A: 设置 `full_sequence_prediction=False`（阶段3默认）：
```python
tokenized_dataset = encode_data(
    dataset,
    tokenizer,
    full_sequence_prediction=False,  # 只有 assistant 部分的 labels 不是 -100
)
```

---

## 性能优化

### 缓存机制

HuggingFace datasets 自动缓存处理结果：

```python
# 首次运行：执行编码
tokenized_dataset = dataset.map(
    encode_function,
    num_proc=10,
    # 缓存到 ~/.cache/huggingface/datasets/
)

# 再次运行：从缓存加载（秒级）
tokenized_dataset = dataset.map(
    encode_function,
    num_proc=10,
)
```

强制重新处理：
```yaml
dataset:
  overwrite_cache: true
```

### 多进程处理

```yaml
dataset:
  processing_num_workers: 10  # 根据 CPU 核心数调整
```

### 内存优化

**流式加载**（大数据集）:
```python
dataset = load_dataset(
    "teknium/OpenHermes-2.5",
    streaming=True  # 不一次性加载到内存
)
```

---

## 测试与验证

### 数据加载测试

```python
from src.data import load_and_prepare_dataset

# 加载小规模数据
cfg.dataset.subset_ratio = 0.001
dataset = load_and_prepare_dataset(cfg)

print(f"样本数: {len(dataset)}")
print(f"第一个样本: {dataset[0]}")
```

### 编码测试

```python
from transformers import AutoTokenizer
from src.data import encode_data

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
tokenized = encode_data(
    dataset,
    tokenizer,
    max_seq_length=1024,
)

# 检查格式
sample = tokenized[0]
assert "input_ids" in sample
assert "attention_mask" in sample
assert "labels" in sample
assert len(sample["input_ids"]) == len(sample["labels"])
```

### 统计信息

```python
from src.data import get_data_statistics

get_data_statistics(tokenized_dataset)

# 输出:
# 数据集统计:
#   - 总样本数: 1000
#   - 平均序列长度: 512.3
#   - 最小序列长度: 128
#   - 最大序列长度: 1024
#   - 填充token比例: 15.2%
#   - 有效token总数: 512300
```

---

## 相关文件清单

```
src/data/
├── __init__.py                    # 模块导出
└── dataset_loader.py              # 数据加载和编码（核心）
    ├── load_and_prepare_dataset() # 加载数据集
    ├── load_local_datasets()      # 本地数据集加载
    ├── load_selected_data()       # 筛选数据加载
    ├── encode_data()              # 数据编码
    ├── encode_with_messages_format() # Messages 编码
    ├── get_data_statistics()      # 统计信息
    ├── compute_string_lengths()   # 计算字符串长度
    ├── sort_dataset_by_string_length() # 排序
    └── temp_seed()                # 临时种子
```

**相关配置**:
- `/mnt/lishiwei/Quality-Gate/configs/stage_1_warmup.yaml`
- `/mnt/lishiwei/Quality-Gate/configs/stage_2_selection.yaml`
- `/mnt/lishiwei/Quality-Gate/configs/stage_3_finetune.yaml`

---

## 开发指南

### 添加新的数据源

**步骤**:
1. 在 `load_and_prepare_dataset()` 中添加新分支
2. 实现加载函数
3. 更新配置文件

**示例：添加 CSV 支持**:
```python
def load_csv_datasets(cfg: DictConfig) -> Dataset:
    """加载 CSV 数据集"""
    csv_config = cfg.dataset.csv
    dataset = load_dataset(
        "csv",
        data_files=csv_config.file_path,
        split="train"
    )

    # 转换为标准格式
    def convert_format(example):
        return {
            "messages": [
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example["answer"]}
            ],
            "dataset": "csv_data",
            "id": f"csv_{example['id']}"
        }

    dataset = dataset.map(convert_format)
    return dataset

# 在 load_and_prepare_dataset() 中添加
if cfg.dataset.dataset_from == "csv":
    dataset = load_csv_datasets(cfg)
```

### 自定义编码逻辑

**示例：添加自定义特殊 token**:
```python
def encode_with_custom_tokens(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    **kwargs
) -> Dataset:
    """使用自定义特殊 token 编码"""

    # 添加自定义 token
    special_tokens = {"additional_special_tokens": ["<|qa|>", "<|doc|>"]}
    tokenizer.add_special_tokens(special_tokens)

    def encode_function(example):
        messages = example["messages"]

        # 自定义格式
        text = "<|qa|>\n"
        for msg in messages:
            text += f"{msg['role']}: {msg['content']}\n"

        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=kwargs.get("max_seq_length", 1024),
        )

        # 创建 labels
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    return dataset.map(
        encode_function,
        num_proc=kwargs.get("processing_num_workers", 10),
    )
```

### 调试技巧

**查看编码结果**:
```python
# 解码第一个样本
sample = tokenized_dataset[0]
input_ids = sample["input_ids"]
labels = sample["labels"]

print("Input:")
print(tokenizer.decode(input_ids))

print("\nLabels (masked with -100):")
valid_labels = [l for l in labels if l != -100]
print(tokenizer.decode(valid_labels))
```

**检查 padding**:
```python
attention_mask = sample["attention_mask"]
num_valid = sum(attention_mask)
num_padding = len(attention_mask) - num_valid

print(f"有效 tokens: {num_valid}")
print(f"Padding tokens: {num_padding}")
print(f"Padding 比例: {num_padding/len(attention_mask)*100:.1f}%")
```

**验证 labels 正确性**:
```python
# 确保 labels 中的 -100 对应 padding 或不需要训练的部分
labels = sample["labels"]
attention_mask = sample["attention_mask"]

for i, (label, mask) in enumerate(zip(labels, attention_mask)):
    if mask == 0:
        assert label == -100, f"Padding position {i} should have label -100"
```
