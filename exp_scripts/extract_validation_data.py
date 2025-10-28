#!/usr/bin/env python3
# exp_scripts/extract_validation_data.py
"""
数据集验证集提取脚本

从HuggingFace或本地数据集的任意划分（train/test/validation等）中提取样本，
转换为项目标准格式，保存为验证集供后续使用。

支持三种提取模式:
1. tail: 从末尾提取后k个样本
2. head: 从开头提取前k个样本
3. split: 按比例自动划分为训练集和验证集

使用示例:
    # 模式1: 从末尾提取 (默认)
    python exp_scripts/extract_validation_data.py \
        --dataset-from hf \
        --dataset-name openai/gsm8k \
        --subset main \
        --split test \
        --num-samples 100 \
        --format-type gsm8k

    # 模式2: 从开头提取
    python exp_scripts/extract_validation_data.py \
        --dataset-from hf \
        --dataset-name openai/gsm8k \
        --subset main \
        --split train \
        --extraction-mode head \
        --num-samples 1000 \
        --format-type gsm8k

    # 模式3: 按比例划分 (一次生成训练集和验证集)
    python exp_scripts/extract_validation_data.py \
        --dataset-from hf \
        --dataset-name openai/gsm8k \
        --subset main \
        --split train \
        --extraction-mode split \
        --split-ratio 0.9 \
        --format-type gsm8k

    # 输出:
    #   - dataset/train/processed/gsm8k_train_90per/data.jsonl (训练集90%)
    #   - dataset/train/processed/gsm8k_val_10per/data.jsonl (验证集10%)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import Dataset, load_dataset

from src.data.format_converters import get_format_converter, infer_format_type

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_dataset_split(
    dataset_from: str,
    dataset_name: str,
    split: str,
    subset: Optional[str] = None,
    local_dir: Optional[str] = None,
) -> Dataset:
    """
    加载指定数据集的指定划分

    Args:
        dataset_from: 数据来源 ("hf" 或 "local")
        dataset_name: 数据集名称
        split: 数据集划分 (如 "test", "validation")
        subset: HuggingFace数据集的子集名称 (可选)
        local_dir: 本地数据集根目录 (当dataset_from="local"时使用)

    Returns:
        加载的数据集

    Raises:
        ValueError: 如果数据来源不支持或加载失败
    """
    log.info(f"正在加载数据集: {dataset_name} (来源: {dataset_from}, 划分: {split})")

    if dataset_from == "hf":
        # 从HuggingFace加载
        try:
            if subset:
                dataset = load_dataset(dataset_name, subset, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
            log.info(f"✓ 成功加载HuggingFace数据集，共 {len(dataset)} 个样本")
            return dataset
        except Exception as e:
            raise ValueError(f"加载HuggingFace数据集失败: {e}") from e

    elif dataset_from == "local":
        # 从本地加载
        if not local_dir:
            raise ValueError("加载本地数据集时必须指定 --local-dir 参数")

        dataset_path = os.path.join(local_dir, dataset_name)
        if not os.path.exists(dataset_path):
            raise ValueError(f"本地数据集目录不存在: {dataset_path}")

        # 查找指定划分的数据文件
        possible_files = [
            os.path.join(dataset_path, f"{split}.jsonl"),
            os.path.join(dataset_path, f"{split}.json"),
        ]

        data_files = [f for f in possible_files if os.path.exists(f)]

        if not data_files:
            # 如果没有找到特定划分，查找所有json/jsonl文件
            all_files = []
            for file in os.listdir(dataset_path):
                if file.endswith((".jsonl", ".json")):
                    all_files.append(os.path.join(dataset_path, file))

            if all_files:
                log.warning(f"未找到 {split} 划分，使用所有可用数据文件: {all_files}")
                data_files = all_files
            else:
                raise ValueError(f"在 {dataset_path} 中未找到数据文件")

        try:
            dataset = load_dataset("json", data_files=data_files)["train"]
            log.info(f"✓ 成功加载本地数据集，共 {len(dataset)} 个样本")
            return dataset
        except Exception as e:
            raise ValueError(f"加载本地数据集失败: {e}") from e

    else:
        raise ValueError(f"不支持的数据来源: {dataset_from}，支持的值: 'hf', 'local'")


def extract_samples(dataset: Dataset, mode: str, num_samples: int = None, split_ratio: float = None) -> tuple[Dataset, int] | tuple[Dataset, int, Dataset, int]:
    """
    从数据集中提取样本

    Args:
        dataset: 原始数据集
        mode: 提取模式 ("head", "tail", "split")
        num_samples: 要提取的样本数量 (head/tail模式使用)
        split_ratio: 训练集比例 (split模式使用，如0.8表示80%训练20%验证)

    Returns:
        - head/tail模式: (提取后的数据集, 起始索引)
        - split模式: (训练集, 训练集起始索引, 验证集, 验证集起始索引)
    """
    total_samples = len(dataset)

    if mode == "tail":
        # 从末尾提取
        if num_samples > total_samples:
            log.warning(f"请求提取 {num_samples} 个样本，但数据集只有 {total_samples} 个样本，将使用全部数据")
            num_samples = total_samples

        start_idx = total_samples - num_samples
        indices = list(range(start_idx, total_samples))
        log.info(f"[tail模式] 从位置 {start_idx} 开始提取 {num_samples} 个样本 (总共 {total_samples} 个)")
        return dataset.select(indices), start_idx

    elif mode == "head":
        # 从开头提取
        if num_samples > total_samples:
            log.warning(f"请求提取 {num_samples} 个样本，但数据集只有 {total_samples} 个样本，将使用全部数据")
            num_samples = total_samples

        indices = list(range(num_samples))
        log.info(f"[head模式] 从位置 0 开始提取 {num_samples} 个样本 (总共 {total_samples} 个)")
        return dataset.select(indices), 0

    elif mode == "split":
        # 按比例划分
        if split_ratio is None or not (0 < split_ratio < 1):
            raise ValueError(f"split模式需要有效的split_ratio (0-1之间)，当前值: {split_ratio}")

        train_size = int(total_samples * split_ratio)
        val_size = total_samples - train_size

        train_indices = list(range(train_size))
        val_indices = list(range(train_size, total_samples))

        train_dataset = dataset.select(train_indices)
        val_dataset = dataset.select(val_indices)

        log.info(f"[split模式] 按比例 {split_ratio:.1%} 划分数据集:")
        log.info(f"  训练集: {train_size} 个样本 (索引 0-{train_size - 1})")
        log.info(f"  验证集: {val_size} 个样本 (索引 {train_size}-{total_samples - 1})")

        return train_dataset, 0, val_dataset, train_size

    else:
        raise ValueError(f"不支持的提取模式: {mode}，支持的值: 'head', 'tail', 'split'")


def convert_to_standard_format(
    dataset: Dataset,
    format_type: str,
    dataset_alias: str,
    split: str,
    start_idx: int,
) -> Dataset:
    """
    将数据集转换为项目标准格式

    Args:
        dataset: 原始数据集
        format_type: 数据格式类型
        dataset_alias: 数据集别名，用于生成dataset和id字段
        split: 数据集划分名称，用于生成唯一ID
        start_idx: 在原始数据集中的起始索引，用于生成唯一ID

    Returns:
        转换后的标准格式数据集
    """
    log.info(f"正在转换数据格式: {format_type} -> 标准格式")

    # 获取格式转换器
    try:
        converter = get_format_converter(format_type)
    except ValueError as e:
        log.error(f"格式类型错误: {e}")
        raise

    if converter is None:
        # 已经是标准格式
        log.info("数据已是标准格式，跳过转换")

        # 验证必需字段
        if "messages" not in dataset.column_names:
            raise ValueError("标准格式数据必须包含 'messages' 字段")

        # 确保有dataset和id字段，使用原始索引生成唯一ID
        def add_metadata(example, idx):
            if "dataset" not in example:
                example["dataset"] = dataset_alias
            if "id" not in example:
                # 使用 split + 原始索引生成唯一ID
                original_idx = start_idx + idx
                example["id"] = f"{dataset_alias}_{split}_{original_idx:06d}"
            return example

        dataset = dataset.map(add_metadata, with_indices=True, desc="添加元数据字段")
        return dataset

    # 应用格式转换
    def convert_example(example, idx):
        try:
            # 使用原始索引生成唯一ID
            original_idx = start_idx + idx
            result = converter(example, dataset_name=dataset_alias, example_index=original_idx)
            # 确保ID包含split信息
            result["id"] = f"{dataset_alias}_{split}_{original_idx:06d}"
            return result
        except Exception as e:
            log.error(f"转换样本 {idx} 失败: {e}")
            raise

    converted_dataset = dataset.map(
        convert_example,
        with_indices=True,
        desc=f"转换数据格式 ({format_type} -> 标准格式)",
        remove_columns=dataset.column_names,  # 移除原始列
    )

    log.info(f"✓ 格式转换完成，共 {len(converted_dataset)} 个样本")

    # 验证转换结果
    if len(converted_dataset) > 0:
        sample = converted_dataset[0]
        required_fields = ["dataset", "id", "messages"]
        for field in required_fields:
            if field not in sample:
                raise ValueError(f"转换后的数据缺少必需字段: {field}")

        log.info(f"转换结果验证通过，包含字段: {list(sample.keys())}")
        log.info(f"示例消息数量: {len(sample['messages'])}")

    return converted_dataset


def save_to_jsonl(dataset: Dataset, output_path: str) -> None:
    """
    将数据集保存为JSONL文件

    Args:
        dataset: 要保存的数据集
        output_path: 输出文件路径
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    log.info(f"正在保存数据到: {output_path}")

    # 转换为Python对象并保存
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            # 将HuggingFace格式转换为普通Python字典
            json_obj = {
                "dataset": example["dataset"],
                "id": example["id"],
                "messages": [{"role": msg["role"], "content": msg["content"]} for msg in example["messages"]],
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

    log.info(f"✓ 成功保存 {len(dataset)} 个样本到 {output_path}")

    # 打印文件信息
    file_size = os.path.getsize(output_path)
    log.info(f"文件大小: {file_size / 1024:.2f} KB")


def print_dataset_statistics(dataset: Dataset) -> None:
    """
    打印数据集统计信息

    Args:
        dataset: 数据集
    """
    log.info("=" * 60)
    log.info("数据集统计信息:")
    log.info(f"  总样本数: {len(dataset)}")

    if len(dataset) > 0:
        # 统计消息轮数分布
        message_counts = [len(example["messages"]) for example in dataset]
        avg_messages = sum(message_counts) / len(message_counts)
        log.info(f"  平均消息轮数: {avg_messages:.2f}")
        log.info(f"  消息轮数范围: {min(message_counts)} - {max(message_counts)}")

        # 统计数据集来源分布
        dataset_sources = {}
        for example in dataset:
            source = example.get("dataset", "unknown")
            dataset_sources[source] = dataset_sources.get(source, 0) + 1

        log.info("  数据集来源分布:")
        for source, count in dataset_sources.items():
            log.info(f"    - {source}: {count} ({count / len(dataset) * 100:.1f}%)")

        # 显示第一个样本示例
        log.info("  示例样本:")
        sample = dataset[0]
        log.info(f"    - Dataset: {sample['dataset']}")
        log.info(f"    - ID: {sample['id']}")
        log.info(f"    - Messages ({len(sample['messages'])} 轮):")
        for i, msg in enumerate(sample["messages"][:2], 1):  # 只显示前2轮
            content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            log.info(f"      [{i}] {msg['role']}: {content_preview}")

    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="从数据集的任意划分提取样本或按比例划分数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 数据源配置
    parser.add_argument(
        "--dataset-from",
        type=str,
        required=True,
        choices=["hf", "local"],
        help="数据来源: hf (HuggingFace) 或 local (本地)",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="数据集名称 (HuggingFace路径或本地目录名)",
    )

    parser.add_argument(
        "--dataset-alias",
        type=str,
        default=None,
        help="数据集别名 (用于生成dataset和id字段，默认使用dataset-name的最后一部分)",
    )

    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="HuggingFace数据集的子集名称 (可选)",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="数据集划分，如 train、test、validation 等 (默认: test)",
    )

    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="本地数据集根目录 (当dataset-from=local时必需)",
    )

    # 提取配置
    parser.add_argument(
        "--extraction-mode",
        type=str,
        default="tail",
        choices=["head", "tail", "split"],
        help="提取模式: head (从头提取), tail (从尾提取), split (按比例划分) (默认: tail)",
    )

    parser.add_argument(
        "--num-samples",
        "-k",
        type=int,
        default=None,
        help="要提取的样本数量 (head/tail模式必需)",
    )

    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="训练集比例 (split模式使用，默认: 0.8，即80%%训练20%%验证)",
    )

    parser.add_argument(
        "--split-train-suffix",
        type=str,
        default="train",
        help="split模式下训练集输出目录后缀 (默认: train)",
    )

    parser.add_argument(
        "--split-val-suffix",
        type=str,
        default="val",
        help="split模式下验证集输出目录后缀 (默认: val)",
    )

    # 格式配置
    parser.add_argument(
        "--format-type",
        type=str,
        default="auto",
        choices=["auto", "standard", "openhermes", "gsm8k", "hendrycks_math"],
        help="数据格式类型 (默认: auto，自动推断)",
    )

    # 输出配置
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/train/processed",
        help="输出根目录 (默认: dataset/train/processed)",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="输出目录名称 (默认: {dataset_alias}_{split}_end{k})",
    )

    args = parser.parse_args()

    # 验证参数
    if args.extraction_mode in ["head", "tail"] and args.num_samples is None:
        parser.error(f"{args.extraction_mode}模式需要指定 --num-samples 参数")

    # 处理dataset_alias
    if args.dataset_alias is None:
        # 从dataset_name提取别名 (取最后一部分)
        args.dataset_alias = args.dataset_name.split("/")[-1]
        log.info(f"自动设置数据集别名: {args.dataset_alias}")

    # 处理format_type
    if args.format_type == "auto":
        args.format_type = infer_format_type(args.dataset_name)
        log.info(f"自动推断数据格式: {args.format_type}")

    # 打印配置信息
    log.info("=" * 60)
    log.info("提取配置:")
    log.info(f"  数据来源: {args.dataset_from}")
    log.info(f"  数据集名称: {args.dataset_name}")
    log.info(f"  数据集别名: {args.dataset_alias}")
    if args.subset:
        log.info(f"  子集: {args.subset}")
    log.info(f"  划分: {args.split}")
    log.info(f"  提取模式: {args.extraction_mode}")
    if args.extraction_mode in ["head", "tail"]:
        log.info(f"  提取数量: {args.num_samples}")
    else:
        log.info(f"  划分比例: {args.split_ratio:.1%} (训练) / {1 - args.split_ratio:.1%} (验证)")
    log.info(f"  格式类型: {args.format_type}")
    log.info("=" * 60)

    try:
        # 1. 加载数据集
        dataset = load_dataset_split(
            dataset_from=args.dataset_from,
            dataset_name=args.dataset_name,
            split=args.split,
            subset=args.subset,
            local_dir=args.local_dir,
        )

        # 2. 根据模式提取样本
        if args.extraction_mode == "split":
            # split模式: 一次生成训练集和验证集
            train_dataset, train_start_idx, val_dataset, val_start_idx = extract_samples(dataset=dataset, mode="split", split_ratio=args.split_ratio)

            # 处理output_name，包含比例信息
            if args.output_name is None:
                train_percent = int(args.split_ratio * 100)
                val_percent = 100 - train_percent
                train_output_name = f"{args.dataset_alias}_{args.split_train_suffix}_{train_percent}per"
                val_output_name = f"{args.dataset_alias}_{args.split_val_suffix}_{val_percent}per"
            else:
                train_output_name = f"{args.output_name}_{args.split_train_suffix}"
                val_output_name = f"{args.output_name}_{args.split_val_suffix}"

            train_output_file = os.path.join(args.output_dir, train_output_name, "data.jsonl")
            val_output_file = os.path.join(args.output_dir, val_output_name, "data.jsonl")

            log.info(f"  训练集输出: {train_output_file}")
            log.info(f"  验证集输出: {val_output_file}")
            log.info("=" * 60)

            # 转换训练集
            log.info("\n处理训练集...")
            train_standard = convert_to_standard_format(
                dataset=train_dataset,
                format_type=args.format_type,
                dataset_alias=args.dataset_alias,
                split=args.split,
                start_idx=train_start_idx,
            )
            save_to_jsonl(train_standard, train_output_file)
            print_dataset_statistics(train_standard)

            # 转换验证集
            log.info("\n处理验证集...")
            val_standard = convert_to_standard_format(
                dataset=val_dataset,
                format_type=args.format_type,
                dataset_alias=args.dataset_alias,
                split=args.split,
                start_idx=val_start_idx,
            )
            save_to_jsonl(val_standard, val_output_file)
            print_dataset_statistics(val_standard)

            # 打印总结
            log.info("=" * 60)
            log.info("✓ 数据集划分完成!")
            log.info(f"训练集: {train_output_file} ({len(train_standard)} 个样本)")
            log.info(f"验证集: {val_output_file} ({len(val_standard)} 个样本)")
            log.info("")
            log.info("可以通过以下配置加载:")
            log.info("")
            log.info("# 训练集")
            log.info("dataset:")
            log.info("  datasets:")
            log.info("    - dataset_from: local")
            log.info(f"      dataset_name: {train_output_name}")
            log.info("      format_type: standard")
            log.info(f"  local_dataset_dir: {os.path.abspath(args.output_dir)}")
            log.info("")
            log.info("# 验证集")
            log.info("dataset:")
            log.info("  datasets:")
            log.info("    - dataset_from: local")
            log.info(f"      dataset_name: {val_output_name}")
            log.info("      format_type: standard")
            log.info(f"  local_dataset_dir: {os.path.abspath(args.output_dir)}")
            log.info("=" * 60)

        else:
            # head/tail模式: 提取单个数据集
            extracted_dataset, start_idx = extract_samples(dataset=dataset, mode=args.extraction_mode, num_samples=args.num_samples)

            # 处理output_name
            if args.output_name is None:
                mode_prefix = "head" if args.extraction_mode == "head" else "end"
                args.output_name = f"{args.dataset_alias}_{args.split}_{mode_prefix}{args.num_samples}"

            output_file = os.path.join(args.output_dir, args.output_name, "data.jsonl")
            log.info(f"  输出路径: {output_file}")
            log.info("=" * 60)

            # 转换为标准格式
            standard_dataset = convert_to_standard_format(
                dataset=extracted_dataset,
                format_type=args.format_type,
                dataset_alias=args.dataset_alias,
                split=args.split,
                start_idx=start_idx,
            )

            # 保存为JSONL
            save_to_jsonl(standard_dataset, output_file)

            # 打印统计信息
            print_dataset_statistics(standard_dataset)

            log.info("=" * 60)
            log.info("✓ 数据集提取完成!")
            log.info(f"输出位置: {output_file}")
            log.info(f"样本数量: {len(standard_dataset)}")
            log.info("")
            log.info("可以通过以下配置加载此数据集:")
            log.info("")
            log.info("dataset:")
            log.info("  datasets:")
            log.info("    - dataset_from: local")
            log.info(f"      dataset_name: {args.output_name}")
            log.info("      format_type: standard")
            log.info(f"  local_dataset_dir: {os.path.abspath(args.output_dir)}")
            log.info("=" * 60)

    except Exception as e:
        log.error(f"提取过程失败: {e}")
        import traceback

        log.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
