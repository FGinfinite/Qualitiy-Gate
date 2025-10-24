# src/data/data_utils.py
"""
数据工具模块

提供数据统计、排序、随机种子管理等实用工具函数。
"""

import contextlib
import logging
from typing import List

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm


@contextlib.contextmanager
def temp_seed(seed):
    """临时设置随机种子的上下文管理器"""
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def compute_string_lengths(dataset: Dataset) -> List[int]:
    """
    快速计算数据集中每个样本的字符串总长度（使用numpy向量化优化）

    Args:
        dataset: 包含messages字段的数据集

    Returns:
        每个样本的字符串总长度列表
    """
    log = logging.getLogger(__name__)
    log.info(f"正在计算 {len(dataset)} 个样本的字符串长度...")

    all_texts = []
    batch_size = 3000

    # 批量访问数据集，避免逐个访问
    for start_idx in tqdm(range(0, len(dataset), batch_size), desc="提取文本"):
        end_idx = min(start_idx + batch_size, len(dataset))
        batch = dataset[start_idx:end_idx]

        # 处理批次中的每个样本
        if isinstance(batch["messages"], list) and len(batch["messages"]) > 0:
            if isinstance(batch["messages"][0], list):
                # 批处理模式
                for messages in batch["messages"]:
                    text_parts = []
                    if isinstance(messages, list):
                        for msg in messages:
                            if isinstance(msg, dict):
                                text_parts.extend([str(msg.get("role", "")), str(msg.get("content", ""))])
                    all_texts.append("".join(text_parts))
            else:
                # 单个样本模式
                messages = batch["messages"]
                text_parts = []
                if isinstance(messages, list):
                    for msg in messages:
                        if isinstance(msg, dict):
                            text_parts.extend([str(msg.get("role", "")), str(msg.get("content", ""))])
                all_texts.append("".join(text_parts))
        else:
            all_texts.extend([""] * (end_idx - start_idx))

    # 使用numpy向量化计算长度
    text_array = np.array(all_texts, dtype=object)
    lengths = np.array([len(text) for text in text_array])

    log.info(f"字符串长度计算完成，平均长度: {lengths.mean():.2f}")
    log.info(f"长度范围: {lengths.min()} - {lengths.max()}")

    return lengths.tolist()


def sort_dataset_by_string_length(dataset: Dataset, descending: bool = True) -> Dataset:
    """
    根据字符串总长度对数据集进行快速排序

    Args:
        dataset: 要排序的数据集
        descending: 是否降序排列（True为降序，False为升序）

    Returns:
        按字符串长度排序的数据集
    """
    log = logging.getLogger(__name__)

    # 计算字符串长度
    lengths = compute_string_lengths(dataset)

    # 创建索引和长度的配对，然后排序
    indexed_lengths = list(enumerate(lengths))
    indexed_lengths.sort(key=lambda x: x[1], reverse=descending)

    # 提取排序后的索引
    sorted_indices = [idx for idx, _ in indexed_lengths]

    log.info(f"按字符串长度{'降序' if descending else '升序'}排列数据集")
    log.info(f"排序前长度范围: {min(lengths)} - {max(lengths)}")

    # 根据排序后的索引重新排列数据集
    sorted_dataset = dataset.select(sorted_indices)

    # 验证排序结果
    if len(sorted_indices) > 10:
        first_lengths = [lengths[i] for i in sorted_indices[:5]]
        last_lengths = [lengths[i] for i in sorted_indices[-5:]]
        log.info(f"排序后前5个长度: {first_lengths}")
        log.info(f"排序后后5个长度: {last_lengths}")

    return sorted_dataset


def get_data_statistics(lm_datasets: Dataset) -> None:
    """
    获取数据集统计信息

    Args:
        lm_datasets: 编码后的数据集
    """

    def get_length(examples):
        lengths = [len(ids) for ids in examples["input_ids"]]
        completion_lens = []
        for labels in examples["labels"]:
            # 修复tensor创建警告
            if isinstance(labels, torch.Tensor):
                com_len = (labels > -1).sum().item()
            else:
                com_len = (torch.tensor(labels) > -1).sum().item()
            completion_lens.append(com_len)
        return {"length": lengths, "c_length": completion_lens}

    data_size = len(lm_datasets)
    dataset_with_lengths = lm_datasets.map(get_length, batched=True)
    lengths = dataset_with_lengths["length"]
    avg_length = sum(lengths) / len(lengths)
    c_lengths = dataset_with_lengths["c_length"]
    avg_c_length = sum(c_lengths) / len(c_lengths)

    log = logging.getLogger(__name__)

    log.info("数据集统计信息:")
    log.info(f"  样本数: {data_size}")
    log.info(f"  平均token数: {avg_length:.2f}")
    log.info(f"  平均完成token数: {avg_c_length:.2f}")
