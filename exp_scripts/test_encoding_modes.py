#!/usr/bin/env python3
"""
测试不同编码模式的独立脚本
验证全序列预测和特殊token掩码功能是否正确工作
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer

from src.data.dataset_loader import concat_messages, encode_with_flexible_format


def test_sample_data():
    """创建测试样本数据"""
    return {
        "messages": [
            {"role": "user", "content": "什么是机器学习？"},
            {"role": "assistant", "content": "机器学习是人工智能的一个分支，它使计算机能够在不被明确编程的情况下学习。"},
        ]
    }


def print_tokens_and_labels(tokenizer, input_ids, labels, mode_name):
    """打印tokens和对应的labels，方便理解掩码情况"""
    print(f"\n=== {mode_name} ===")
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print("位置 | Token | Input_ID | Label | 是否掩码")
    print("-" * 60)

    for i, (token, input_id, label) in enumerate(zip(tokens, input_ids.tolist(), labels.tolist())):
        is_masked = "✓" if label == -100 else "✗"
        print(f"{i:4d} | {token:20s} | {input_id:8d} | {label:6d} | {is_masked}")

    print(f"总长度: {len(tokens)}")
    print(f"掩码数量: {sum(1 for l in labels.tolist() if l == -100)}")
    print(f"有效数量: {sum(1 for l in labels.tolist() if l != -100)}")


def test_encoding_mode(tokenizer, example, mode_name, full_sequence_prediction, mask_special_tokens):
    """测试单个编码模式"""
    print(f"\n{'=' * 80}")
    print(f"测试模式: {mode_name}")
    print(f"full_sequence_prediction={full_sequence_prediction}, mask_special_tokens={mask_special_tokens}")
    print("=" * 80)

    # 先展示原始文本
    original_text = concat_messages(example["messages"], tokenizer)
    print(f"原始文本: {repr(original_text)}")

    # 编码数据
    encoded = encode_with_flexible_format(
        example=example, tokenizer=tokenizer, max_seq_length=512, full_sequence_prediction=full_sequence_prediction, mask_special_tokens=mask_special_tokens
    )

    input_ids = encoded["input_ids"]
    labels = encoded["labels"]
    attention_mask = encoded["attention_mask"]

    print(f"输入形状: {input_ids.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"注意力掩码形状: {attention_mask.shape}")

    # 打印详细的token和label对应关系
    print_tokens_and_labels(tokenizer, input_ids, labels, mode_name)

    return encoded


def analyze_special_tokens(tokenizer):
    """分析特殊token的tokenization结果"""
    print("\n" + "=" * 80)
    print("特殊Token分析")
    print("=" * 80)

    special_tokens = ["<|user|>", "<|assistant|>", "<|system|>"]
    for token in special_tokens:
        tokenized = tokenizer.tokenize(token)
        ids = tokenizer.encode(token, add_special_tokens=False)
        print(f"{token:12s} -> {tokenized} -> {ids}")


def main():
    print("开始测试编码模式...")

    # 加载tokenizer
    tokenizer_name = "allenai/OLMoE-1B-7B-0125"
    print(f"加载tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 分析特殊token
    analyze_special_tokens(tokenizer)

    # 创建测试数据
    example = test_sample_data()

    # 测试四种模式
    test_modes = [
        ("模式1: 推荐模式(全序列+掩码特殊token)", True, True),
        ("模式2: 完全预测", True, False),
        ("模式3: 传统SFT+掩码特殊token", False, True),
        ("模式4: 传统SFT", False, False),
    ]

    results = {}
    for mode_name, full_seq, mask_special in test_modes:
        try:
            result = test_encoding_mode(
                tokenizer=tokenizer, example=example, mode_name=mode_name, full_sequence_prediction=full_seq, mask_special_tokens=mask_special
            )
            results[mode_name] = result
            print(f"✅ {mode_name} - 测试通过")
        except Exception as e:
            print(f"❌ {mode_name} - 测试失败: {e}")
            import traceback

            traceback.print_exc()

    # 模式对比分析
    print("\n" + "=" * 80)
    print("模式对比分析")
    print("=" * 80)

    if len(results) >= 2:
        mode1_labels = results["模式1: 推荐模式(全序列+掩码特殊token)"]["labels"]
        mode4_labels = results["模式4: 传统SFT"]["labels"]

        print(f"模式1掩码数量: {sum(1 for l in mode1_labels.tolist() if l == -100)}")
        print(f"模式4掩码数量: {sum(1 for l in mode4_labels.tolist() if l == -100)}")

        # 找出差异
        diff_positions = []
        for i, (l1, l4) in enumerate(zip(mode1_labels.tolist(), mode4_labels.tolist())):
            if (l1 == -100) != (l4 == -100):
                diff_positions.append(i)

        print(f"掩码差异位置数量: {len(diff_positions)}")
        if diff_positions and len(diff_positions) <= 20:  # 只显示前20个差异
            print(f"差异位置: {diff_positions}")

    print("\n测试完成！")


if __name__ == "__main__":
    main()
