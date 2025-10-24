# src/data/format_converters.py
"""
数据格式转换器模块

提供各种数据集格式到项目标准格式的转换功能。
"""

from typing import Callable, Dict, Optional

# ============================================================================
# 格式转换器工厂
# ============================================================================

# 格式转换器映射表
FORMAT_CONVERTERS: Dict[str, Optional[Callable]] = {}


def get_format_converter(format_type: str) -> Optional[Callable]:
    """
    根据 format_type 获取对应的格式转换器

    Args:
        format_type: 数据格式类型

    Returns:
        格式转换函数，如果是标准格式则返回 None

    Raises:
        ValueError: 如果格式类型不支持
    """
    if format_type not in FORMAT_CONVERTERS:
        raise ValueError(f"不支持的数据格式: {format_type}，支持的格式: {list(FORMAT_CONVERTERS.keys())}")
    return FORMAT_CONVERTERS[format_type]


# ============================================================================
# 具体格式转换器
# ============================================================================


def convert_openhermes_format(example: Dict, dataset_name: str = "openhermes", example_index: int = None) -> Dict:
    """
    转换OpenHermes-2.5格式到项目标准格式

    关键处理：
    - 完全过滤空的system消息，避免生成无意义的特殊token
    - 角色映射：human→user, gpt→assistant, system→system
    - 字段映射：conversations→messages, from→role, value→content

    Args:
        example: OpenHermes-2.5格式的数据样本
        dataset_name: 数据集名称，用于生成dataset字段
        example_index: 样本在数据集中的索引，用于生成唯一ID

    Returns:
        转换后的标准格式数据
    """
    if "conversations" not in example:
        raise ValueError("OpenHermes数据必须包含'conversations'字段")

    messages = []
    for msg in example["conversations"]:
        # 角色映射
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}

        if msg["from"] not in role_map:
            continue  # 跳过不支持的角色

        role = role_map[msg["from"]]
        content = msg["value"]

        # 关键：完全过滤空的system消息
        if role == "system" and not content.strip():
            continue  # 跳过空system消息，不添加到messages中

        messages.append({"role": role, "content": content.strip() if content else ""})

    if not messages:
        raise ValueError("转换后没有有效的消息")

    # 生成简单而可靠的ID
    original_id = example.get("id")
    if original_id is not None and str(original_id).strip() != "":
        # 如果有有效的原始ID，使用它
        example_id = f"{dataset_name}_{original_id}"
    else:
        # 使用数据集索引生成ID（推荐方案，简单且可靠）
        example_id = f"{dataset_name}_{example_index:06d}"
    return {"dataset": dataset_name, "id": example_id, "messages": messages}


def convert_gsm8k_format(example: Dict, dataset_name: str = "gsm8k", example_index: int = None) -> Dict:
    """
    转换GSM8K格式到项目标准格式

    GSM8K数据集包含数学问题和答案：
    - question: 数学问题文本
    - answer: 包含推理步骤和最终答案的文本

    Args:
        example: GSM8K格式的数据样本
        dataset_name: 数据集名称，用于生成dataset字段
        example_index: 样本在数据集中的索引，用于生成唯一ID

    Returns:
        转换后的标准格式数据
    """
    if "question" not in example or "answer" not in example:
        raise ValueError("GSM8K数据必须包含'question'和'answer'字段")

    # 构建标准messages格式
    messages = [
        {"role": "user", "content": example["question"].strip()},
        {"role": "assistant", "content": example["answer"].strip()},
    ]

    # 生成唯一ID
    example_id = f"{dataset_name}_{example_index:06d}"

    return {"dataset": dataset_name, "id": example_id, "messages": messages}


def convert_hendrycks_math_format(example: Dict, dataset_name: str = "hendrycks_math", example_index: int = None) -> Dict:
    """
    转换HENDRYCKS_MATH格式到项目标准格式

    HENDRYCKS_MATH数据集包含高等数学问题和详细解答：
    - problem: 数学问题文本
    - solution: 详细解答过程
    - level: 难度级别（如 "Level 5"）
    - type: 数学类型（如 "Algebra"）

    Args:
        example: HENDRYCKS_MATH格式的数据样本
        dataset_name: 数据集名称，用于生成dataset字段
        example_index: 样本在数据集中的索引，用于生成唯一ID

    Returns:
        转换后的标准格式数据
    """
    if "problem" not in example or "solution" not in example:
        raise ValueError("HENDRYCKS_MATH数据必须包含'problem'和'solution'字段")

    # 构建标准messages格式
    messages = [
        {"role": "user", "content": example["problem"].strip()},
        {"role": "assistant", "content": example["solution"].strip()},
    ]

    # 生成唯一ID
    example_id = f"{dataset_name}_{example_index:06d}"

    return {"dataset": dataset_name, "id": example_id, "messages": messages}


def infer_format_type(dataset_name: str) -> str:
    """
    根据数据集名称推断格式类型

    这是一个通用工具函数，可以在需要自动推断数据集格式的地方使用。

    Args:
        dataset_name: 数据集名称

    Returns:
        格式类型字符串（"openhermes"/"gsm8k"/"hendrycks_math"/"standard"）

    Examples:
        >>> infer_format_type("OpenHermes-2.5")
        'openhermes'
        >>> infer_format_type("gsm8k_train")
        'gsm8k'
        >>> infer_format_type("algebra")
        'hendrycks_math'
        >>> infer_format_type("unknown_dataset")
        'standard'
    """
    dataset_name_lower = dataset_name.lower()

    if "openhermes" in dataset_name_lower:
        return "openhermes"
    elif "gsm8k" in dataset_name_lower:
        return "gsm8k"
    elif "hendrycks_math" in dataset_name_lower or dataset_name in [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]:
        return "hendrycks_math"
    else:
        return "standard"


# 注册格式转换器到映射表
FORMAT_CONVERTERS["openhermes"] = convert_openhermes_format
FORMAT_CONVERTERS["gsm8k"] = convert_gsm8k_format
FORMAT_CONVERTERS["hendrycks_math"] = convert_hendrycks_math_format
FORMAT_CONVERTERS["standard"] = None  # 标准格式，无需转换
