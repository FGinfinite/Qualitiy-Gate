# src/data/dataset_loader.py
"""
数据集加载模块

提供核心的数据集加载功能，支持本地和 HuggingFace 数据集。
"""

import logging
import os
from typing import Dict, Optional

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from omegaconf import DictConfig

from .data_utils import sort_dataset_by_string_length, temp_seed
from .format_converters import get_format_converter

# ============================================================================
# 辅助函数
# ============================================================================

# 定义标准的数据集 schema（用于统一 messages 字段顺序）
# PyArrow 在合并数据集时要求 struct 字段顺序完全一致
STANDARD_DATASET_FEATURES = None  # 延迟初始化


def get_standard_features():
    """获取标准的数据集 Features 定义"""
    global STANDARD_DATASET_FEATURES
    if STANDARD_DATASET_FEATURES is None:
        from datasets import Features, Value

        STANDARD_DATASET_FEATURES = Features(
            {
                "dataset": Value("string"),
                "id": Value("string"),
                "messages": [{"role": Value("string"), "content": Value("string")}],
            }
        )
    return STANDARD_DATASET_FEATURES


# ============================================================================
# 新版数据集加载函数（支持混合 local 和 hf）
# ============================================================================


def _load_local_single_dataset(dataset_name: str, local_dataset_dir: str) -> Dataset:
    """
    从本地目录加载单个数据集

    Args:
        dataset_name: 数据集名称（对应本地目录名）
        local_dataset_dir: 本地数据集根目录

    Returns:
        加载的数据集

    Raises:
        ValueError: 如果目录不存在或没有找到数据文件
    """
    dataset_path = os.path.join(local_dataset_dir, dataset_name)
    if not os.path.exists(dataset_path):
        raise ValueError(f"本地数据集目录不存在: {dataset_path}")

    # 查找所有 .jsonl 和 .json 文件
    data_files = []
    for file in os.listdir(dataset_path):
        if file.endswith((".jsonl", ".json")):
            data_files.append(os.path.join(dataset_path, file))

    if not data_files:
        raise ValueError(f"在 {dataset_path} 中没有找到数据文件")

    # 加载数据集
    dataset = load_dataset("json", data_files=data_files)["train"]
    print(f"从本地加载数据集 '{dataset_name}': {len(dataset)} 个样本")

    return dataset


def _try_load_from_shared_memory(dataset_config: Dict) -> Optional[Dataset]:
    """
    尝试从共享内存加载数据集（可选功能）

    Args:
        dataset_config: 数据集配置字典

    Returns:
        成功返回 Dataset，失败返回 None
    """
    try:
        from share_dataset import LoadResult, SharedDatasetClient

        log = logging.getLogger(__name__)
        dataset_name = dataset_config["name"]

        log.info(f"尝试从共享内存加载数据集: {dataset_name}")

        client = SharedDatasetClient()
        result = client.load_shared_dataset(dataset_config)

        if result.status == LoadResult.SUCCESS:
            log.info(f"共享内存加载成功: {dataset_name}, 耗时: {result.load_time:.4f}秒, {result.message}")
            return result.dataset
        else:
            log.info(f"共享内存加载失败: {result.message}，回退到普通加载")
            return None

    except ImportError:
        return None
    except Exception as e:
        log = logging.getLogger(__name__)
        log.warning(f"共享内存加载出错: {e}，回退到普通加载")
        return None


def _load_full_subset_dataset(dataset_name: str, split: str) -> Dataset:
    """
    加载包含多个子集的数据集（如 HENDRYCKS_MATH）

    Args:
        dataset_name: HuggingFace 数据集名称
        split: 数据集分割（train/test）

    Returns:
        合并后的数据集
    """
    from datasets import get_dataset_config_names

    log = logging.getLogger(__name__)

    # 定义已知数据集的子集列表（用于离线模式的 fallback）
    KNOWN_DATASET_SUBSETS = {
        "EleutherAI/hendrycks_math": [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ],
    }

    # 尝试从 HuggingFace Hub 获取子集列表
    try:
        all_subsets = get_dataset_config_names(dataset_name)

        # 检查是否返回了 'default'（这通常意味着离线模式或无法获取配置）
        if all_subsets == ["default"] and dataset_name in KNOWN_DATASET_SUBSETS:
            log.info("检测到离线模式或无法获取子集列表，使用预定义的子集列表")
            all_subsets = KNOWN_DATASET_SUBSETS[dataset_name]

    except Exception as e:
        # 如果获取失败，使用预定义的子集列表
        if dataset_name in KNOWN_DATASET_SUBSETS:
            log.info(f"无法从 Hub 获取子集列表 ({e})，使用预定义的子集列表")
            all_subsets = KNOWN_DATASET_SUBSETS[dataset_name]
        else:
            raise ValueError(f"无法获取数据集 {dataset_name} 的子集列表，且没有预定义的 fallback 列表") from e

    print(f"检测到 __full_subset__ 标记，将加载 {dataset_name} 的所有 {len(all_subsets)} 个子集")
    print(f"子集列表: {all_subsets}")

    # 遍历每个子集
    subset_datasets = []
    for subset_name in all_subsets:
        try:
            print(f"  正在加载子集 '{subset_name}'...")
            subset_dataset = load_dataset(dataset_name, subset_name, split=split)
            subset_datasets.append(subset_dataset)
            print(f"  ✓ 已加载子集 '{subset_name}': {len(subset_dataset)} 个样本")
        except Exception as e:
            print(f"  ✗ 加载子集 '{subset_name}' 失败: {e}")
            continue

    if not subset_datasets:
        raise ValueError(f"未能成功加载 {dataset_name} 的任何子集")

    # 合并所有子集
    combined = concatenate_datasets(subset_datasets)
    return combined


def _load_hf_single_dataset(dataset_config: Dict, use_shared_memory: bool) -> Dataset:
    """
    从 HuggingFace 加载单个数据集

    Args:
        dataset_config: 数据集配置字典，包含 name, subset, split 等
        use_shared_memory: 是否尝试使用共享内存加载

    Returns:
        加载的数据集
    """
    name = dataset_config["name"]
    subset = dataset_config.get("subset", None)
    split = dataset_config.get("split", "train")

    # 尝试共享内存加载（如果启用）
    if use_shared_memory:
        dataset = _try_load_from_shared_memory(dataset_config)
        if dataset is not None:
            return dataset

    # 普通加载
    print(f"正在从HuggingFace加载数据集 '{name}'...")

    if subset == "__full_subset__":
        # 加载所有子集
        dataset = _load_full_subset_dataset(name, split)
    elif subset:
        # 加载指定子集
        dataset = load_dataset(name, subset, split=split)
    else:
        # 加载默认配置
        dataset = load_dataset(name, split=split)

    return dataset


def load_single_dataset(
    dataset_config: Dict,
    local_dataset_dir: str,
    sample_percentage: float = 1.0,
    seed: int = 42,
) -> Dataset:
    """
    加载单个数据集（local 或 hf），并进行采样和格式转换

    Args:
        dataset_config: 数据集配置字典，必须包含:
            - dataset_from: "local" 或 "hf"
            - dataset_name: 数据集名称
            - format_type: 数据格式类型（可选，默认 "standard"）
            - use_shared_memory: 是否使用共享内存（可选，默认 False）
            对于 hf 数据集还需要:
            - name: HuggingFace 数据集路径
            - subset: 子集名称（可选）
            - split: 数据集分割（默认 "train"）
        local_dataset_dir: 本地数据集根目录
        sample_percentage: 采样比例
        seed: 随机种子

    Returns:
        加载并转换后的数据集

    Raises:
        ValueError: 如果配置不合法或加载失败
    """
    dataset_from = dataset_config.get("dataset_from")
    dataset_name = dataset_config["dataset_name"]
    format_type = dataset_config.get("format_type", "standard")
    use_shared_memory = dataset_config.get("use_shared_memory", False)

    # 1. 加载原始数据集
    if dataset_from == "local":
        dataset = _load_local_single_dataset(dataset_name, local_dataset_dir)
    elif dataset_from == "hf":
        dataset = _load_hf_single_dataset(dataset_config, use_shared_memory)
    else:
        raise ValueError(f"不支持的 dataset_from: {dataset_from}，支持的值: 'local', 'hf'")

    # 2. 采样
    if sample_percentage < 1.0:
        sample_size = int(len(dataset) * sample_percentage)
        with temp_seed(seed):
            indices = np.random.permutation(len(dataset))[:sample_size]
        dataset = dataset.select(indices)
        print(f"采样后数据集大小: {len(dataset)}")

    # 3. 格式转换
    converter = get_format_converter(format_type)
    standard_features = get_standard_features()

    if converter is not None:
        # 需要转换格式
        columns_to_remove = [col for col in dataset.column_names if col not in ["dataset", "id", "messages"]]

        dataset = dataset.map(
            lambda example, idx: converter(example, dataset_name=dataset_name, example_index=idx),
            with_indices=True,
            desc=f"转换数据集格式 - {dataset_name}",
            features=standard_features,  # 关键：指定输出 schema，确保字段顺序一致
            load_from_cache_file=True,  # 使用缓存，提升性能
            remove_columns=columns_to_remove,
        )
    else:
        # 标准格式，验证是否包含必需字段
        if "messages" not in dataset.column_names:
            raise ValueError(f"数据集 '{dataset_name}' 标记为标准格式，但缺少 'messages' 字段")

        # 使用 cast 统一 schema（高效方案，利用 PyArrow 底层能力）
        dataset = dataset.cast(standard_features)
        print(f"✓ 已统一 '{dataset_name}' 的 schema")

    # 4. 添加 is_target_task 标签
    is_target_task = dataset_config.get("is_target_task", False)
    dataset = dataset.add_column("is_target_task", [int(is_target_task)] * len(dataset))
    print(f"✓ 已为数据集 '{dataset_name}' 添加标签 is_target_task={is_target_task}")

    return dataset


def load_and_prepare_dataset(cfg: DictConfig) -> Dataset:
    """
    加载和准备数据集（支持混合 local 和 hf 数据集）

    新版实现：支持在同一配置中混合使用多个 local 和 hf 数据集

    配置示例：
    dataset:
      local_dataset_dir: "dataset/train/processed"
      datasets:
        - dataset_from: "local"
          dataset_name: "oasst1"
          format_type: "standard"
        - dataset_from: "hf"
          name: "openai/gsm8k"
          dataset_name: "gsm8k"
          subset: "main"
          split: "train"
          format_type: "gsm8k"
          use_shared_memory: false

    Args:
        cfg: Hydra 配置对象

    Returns:
        准备好的数据集
    """
    log = logging.getLogger(__name__)

    # 获取配置参数
    datasets_config = cfg.dataset.datasets
    local_dataset_dir = getattr(cfg.dataset, "local_dataset_dir", "dataset/train/processed")
    sample_percentage = getattr(cfg.dataset, "subset_ratio", 1.0)
    seed = getattr(cfg, "seed", 42)
    shuffle = getattr(cfg.dataset, "shuffle", True)
    sort_by_length = getattr(cfg.dataset, "sort_by_length", False)

    # 转换为绝对路径
    if not os.path.isabs(local_dataset_dir):
        local_dataset_dir = os.path.join(os.getcwd(), local_dataset_dir)

    log.info(f"开始加载 {len(datasets_config)} 个数据集...")
    log.info(f"本地数据集根目录: {local_dataset_dir}")
    log.info(f"采样比例: {sample_percentage}")

    # 加载所有数据集
    all_datasets = []
    for i, dataset_config in enumerate(datasets_config, 1):
        dataset_name = dataset_config.get("dataset_name", "unknown")
        dataset_from = dataset_config.get("dataset_from", "unknown")

        try:
            log.info(f"[{i}/{len(datasets_config)}] 正在加载 {dataset_from} 数据集: {dataset_name}")

            dataset = load_single_dataset(
                dataset_config=dataset_config,
                local_dataset_dir=local_dataset_dir,
                sample_percentage=sample_percentage,
                seed=seed,
            )

            all_datasets.append(dataset)
            log.info(f"✓ 已加载数据集 '{dataset_name}': {len(dataset)} 个样本")

        except Exception as e:
            log.error(f"✗ 加载数据集 '{dataset_name}' 失败: {e}")
            import traceback

            log.debug(traceback.format_exc())
            continue

    if not all_datasets:
        raise ValueError("没有成功加载任何数据集")

    # 合并数据集
    log.info(f"正在合并 {len(all_datasets)} 个数据集...")
    combined_dataset = concatenate_datasets(all_datasets)
    log.info(f"✓ 数据集合并完成，总共 {len(combined_dataset)} 个样本")

    # 序列长度排序处理
    if sort_by_length:
        log.info("启用字符串长度排序功能...")

        # 如果同时启用排序和shuffle，优先使用排序并给出提示
        if shuffle:
            log.info("检测到同时启用排序和shuffle，优先使用长度排序，跳过shuffle")
            shuffle = False

        try:
            combined_dataset = sort_dataset_by_string_length(combined_dataset, descending=True)
            log.info("✓ 数据集已按字符串长度降序排列")
        except Exception as e:
            log.error(f"字符串长度排序失败: {e}")
            log.info("回退到原始数据集顺序")

    # 打乱数据集（如果排序未启用）
    if shuffle:
        combined_dataset = combined_dataset.shuffle(seed=seed)
        log.info(f"✓ 已打乱数据集，种子: {seed}")

    log.info(f"数据集准备完成，总共 {len(combined_dataset)} 个样本")
    return combined_dataset


def load_selected_data(data_path: str) -> Dataset:
    """
    加载选择后的数据文件（JSONL格式）

    Args:
        data_path: 选择后的数据文件路径，格式为包含dataset、id、scores、messages字段的JSONL

    Returns:
        加载的数据集
    """
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(data_path):
        project_root = os.getcwd()
        data_path = os.path.join(project_root, data_path)

    if not os.path.exists(data_path):
        raise ValueError(f"数据文件不存在: {data_path}")

    # 加载JSONL文件
    try:
        dataset = load_dataset("json", data_files=data_path)["train"]
        print(f"已加载选择后的数据: {len(dataset)} 个样本")

        # 验证数据格式
        if "messages" not in dataset.column_names:
            raise ValueError("数据集必须包含'messages'字段")

        # 打印一些统计信息
        if len(dataset) > 0:
            sample = dataset[0]
            print("数据格式验证:")
            print(f"  - 包含字段: {list(sample.keys())}")
            if "scores" in sample:
                scores = [item["scores"] for item in dataset]
                print(f"  - 分数范围: {min(scores):.4f} - {max(scores):.4f}")
            if "dataset" in sample:
                dataset_counts = {}
                for item in dataset:
                    ds_name = item["dataset"]
                    dataset_counts[ds_name] = dataset_counts.get(ds_name, 0) + 1
                print(f"  - 数据集分布: {dataset_counts}")

        return dataset

    except Exception as e:
        raise ValueError(f"加载选择后的数据失败: {e}") from e
