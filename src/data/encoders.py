# src/data/encoders.py
"""
数据编码模块

提供将原始文本数据编码为模型输入格式的功能。
"""

from functools import partial
from typing import Dict

import torch
from datasets import Dataset
from transformers import AutoTokenizer


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
