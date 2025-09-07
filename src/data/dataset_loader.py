# src/data/dataset_loader.py
import contextlib
import logging
import os
from functools import partial
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer


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


def load_local_datasets(
    data_dir: str,
    dataset_names: Optional[List[str]] = None,
    sample_percentage: float = 1.0,
    seed: int = 0,
) -> Dataset:
    """
    从本地目录加载多个数据集并合并

    Args:
        data_dir: 数据集根目录 (例如: data/train/processed)
        dataset_names: 要加载的数据集名称列表，如果为None则加载所有可用数据集
        sample_percentage: 每个数据集的采样比例
        seed: 随机种子

    Returns:
        合并后的数据集
    """
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(data_dir):
        # 从项目根目录开始的相对路径
        project_root = os.getcwd()
        data_dir = os.path.join(project_root, data_dir)

    if not os.path.exists(data_dir):
        raise ValueError(f"数据目录不存在: {data_dir}")

    # 如果没有指定数据集名称，自动发现所有可用数据集
    if dataset_names is None:
        dataset_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    if not dataset_names:
        raise ValueError(f"在目录 {data_dir} 中没有找到任何数据集")

    all_datasets = []

    for dataset_name in dataset_names:
        dataset_path = os.path.join(data_dir, dataset_name)
        if not os.path.exists(dataset_path):
            print(f"警告: 数据集目录不存在，跳过: {dataset_path}")
            continue

        # 查找数据文件
        data_files = []
        for file in os.listdir(dataset_path):
            if file.endswith((".jsonl", ".json")):
                data_files.append(os.path.join(dataset_path, file))

        if not data_files:
            print(f"警告: 在 {dataset_path} 中没有找到数据文件，跳过")
            continue

        # 加载数据集
        try:
            dataset = load_dataset("json", data_files=data_files)["train"]

            # 采样
            if sample_percentage < 1.0:
                sample_size = int(len(dataset) * sample_percentage)
                with temp_seed(seed):
                    indices = np.random.permutation(len(dataset))[:sample_size]
                dataset = dataset.select(indices)

            all_datasets.append(dataset)
            print(f"已加载数据集 '{dataset_name}': {len(dataset)} 个样本")

        except Exception as e:
            print(f"加载数据集 '{dataset_name}' 失败: {e}")
            continue

    if not all_datasets:
        raise ValueError("没有成功加载任何数据集")

    # 合并所有数据集
    combined_dataset = concatenate_datasets(all_datasets)
    print(f"已合并 {len(dataset_names)} 个数据集，总共 {len(combined_dataset)} 个样本")

    return combined_dataset


def encode_with_messages_format(example: Dict, tokenizer: AutoTokenizer, max_seq_length: int) -> Dict:
    """
    使用messages格式编码数据，基于LESS库的实现

    Args:
        example: 包含messages字段的数据样本
        tokenizer: 分词器
        max_seq_length: 最大序列长度

    Returns:
        编码后的数据
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages字段为空")

    example_text = concat_messages(messages, tokenizer)
    tokenized_example = tokenizer(example_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # 掩码非assistant部分，避免计算损失
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    concat_messages(messages[:message_idx], tokenizer),
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]

            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # 这里也忽略assistant的role部分
                messages_so_far = concat_messages(messages[: message_idx + 1], tokenizer) + "<|assistant|>\n"
            else:
                messages_so_far = concat_messages(messages[: message_idx + 1], tokenizer)

            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors="pt",
                max_length=max_seq_length,
                truncation=True,
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def concat_messages(messages, tokenizer):
    """拼接消息，与LESS库保持一致"""
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError(f"不支持的role: {message['role']}")
    return message_text


def encode_with_flexible_format(
    example: Dict, tokenizer: AutoTokenizer, max_seq_length: int, full_sequence_prediction: bool = False, mask_special_tokens: bool = False
) -> Dict:
    """
    灵活的编码函数，支持多种掩码模式

    Args:
        example: 包含messages字段的数据样本
        tokenizer: 分词器
        max_seq_length: 最大序列长度
        full_sequence_prediction: 是否进行全序列预测
        mask_special_tokens: 是否掩码特殊format token

    Returns:
        编码后的数据

    掩码模式说明：
    1. full_sequence_prediction=True, mask_special_tokens=True (推荐)：
       掩码特殊token，学习预测问题内容和答案内容
    2. full_sequence_prediction=True, mask_special_tokens=False：
       学习预测整个序列，包括特殊token
    3. full_sequence_prediction=False, mask_special_tokens=True：
       只学习预测assistant回复，掩码特殊token
    4. full_sequence_prediction=False, mask_special_tokens=False：
       传统SFT模式，学习预测<|assistant|>+答案内容
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages字段为空")

    example_text = concat_messages(messages, tokenizer)
    tokenized_example = tokenizer(example_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)

    # 特殊token列表
    special_tokens = ["<|user|>", "<|assistant|>", "<|system|>"]

    if full_sequence_prediction:
        # 全序列预测模式
        if mask_special_tokens:
            # 模式1: 掩码特殊token，学习预测内容
            # 需要找到所有特殊token的位置并掩码
            text_tokens = tokenizer.tokenize(example_text)
            current_pos = 0

            for _ in text_tokens:
                if current_pos >= labels.shape[1]:
                    break

                # 检查是否是特殊token的开始
                for special_token in special_tokens:
                    special_token_tokens = tokenizer.tokenize(special_token)
                    if (
                        current_pos + len(special_token_tokens) <= labels.shape[1]
                        and text_tokens[current_pos : current_pos + len(special_token_tokens)] == special_token_tokens
                    ):
                        # 掩码这个特殊token的所有子token
                        labels[:, current_pos : current_pos + len(special_token_tokens)] = -100
                        current_pos += len(special_token_tokens)
                        break
                else:
                    current_pos += 1
        # else: 模式2，不需要额外掩码，保持所有token的labels
    else:
        # 传统SFT模式，只学习assistant部分
        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                if message_idx == 0:
                    message_start_idx = 0
                else:
                    message_start_idx = tokenizer(
                        concat_messages(messages[:message_idx], tokenizer),
                        return_tensors="pt",
                        max_length=max_seq_length,
                        truncation=True,
                    ).input_ids.shape[1]

                if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                    if mask_special_tokens:
                        # 模式3: 不包含<|assistant|>
                        messages_so_far = concat_messages(messages[: message_idx + 1], tokenizer)
                    else:
                        # 模式4: 包含<|assistant|>
                        messages_so_far = concat_messages(messages[: message_idx + 1], tokenizer) + "<|assistant|>\n"
                else:
                    messages_so_far = concat_messages(messages[: message_idx + 1], tokenizer)

                message_end_idx = tokenizer(
                    messages_so_far,
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]

                labels[:, message_start_idx:message_end_idx] = -100

                if message_end_idx >= max_seq_length:
                    break

    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def encode_data(
    raw_datasets: Dataset,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    processing_num_workers: int = 10,
    overwrite_cache: bool = False,
    full_sequence_prediction: bool = False,
    mask_special_tokens: bool = False,
) -> Dataset:
    """
    编码数据集

    Args:
        raw_datasets: 原始数据集
        tokenizer: 分词器
        max_seq_length: 最大序列长度
        processing_num_workers: 处理工作进程数
        overwrite_cache: 是否覆盖缓存
        full_sequence_prediction: 是否进行全序列预测
        mask_special_tokens: 是否掩码特殊格式token

    Returns:
        编码后的数据集
    """
    # 如果已经编码过，直接返回
    if "input_ids" in raw_datasets.features:
        return raw_datasets

    # 检查数据格式并选择编码函数
    if "messages" not in raw_datasets.column_names:
        raise ValueError("数据集必须包含'messages'字段")

    # 根据配置选择编码函数
    if full_sequence_prediction or mask_special_tokens:
        # 使用新的灵活编码函数
        encode_function = partial(
            encode_with_flexible_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            full_sequence_prediction=full_sequence_prediction,
            mask_special_tokens=mask_special_tokens,
        )
        mode_desc = f"灵活编码(全序列={full_sequence_prediction}, 掩码特殊token={mask_special_tokens})"
    else:
        # 使用传统编码函数（向后兼容）
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        mode_desc = "传统SFT编码"

    # 使用多进程处理，移除原始列
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=processing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc=f"对数据进行分词和格式化 - {mode_desc}",
        remove_columns=raw_datasets.column_names,  # 移除所有原始列
    )

    # 显式设置为torch格式
    lm_datasets.set_format(type="torch")

    return lm_datasets


def load_and_prepare_dataset(cfg: DictConfig) -> Dataset:
    """
    加载和准备数据集的主要函数

    Args:
        cfg: 配置对象

    Returns:
        准备好的数据集
    """
    # 从配置中获取参数
    data_dir = cfg.dataset.data_dir
    dataset_names = getattr(cfg.dataset, "dataset_names", None)
    sample_percentage = getattr(cfg.dataset, "subset_ratio", 1.0)
    seed = getattr(cfg, "seed", 42)  # 使用全局种子
    shuffle = getattr(cfg.dataset, "shuffle", True)

    # 加载本地数据集
    log = logging.getLogger(__name__)
    log.info(f"正在从 {data_dir} 加载数据集...")
    dataset = load_local_datasets(
        data_dir=data_dir,
        dataset_names=dataset_names,
        sample_percentage=sample_percentage,
        seed=seed,
    )

    # 打乱数据集
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
        log.info(f"已打乱数据集，使用种子: {seed}")

    log.info(f"数据集准备完成，总共 {len(dataset)} 个样本")
    return dataset


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
