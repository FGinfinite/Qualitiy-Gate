#!/usr/bin/env python3
"""
噪声数据集生成脚本

生成包含随机乱码的数据集，用于测试数据选择方法是否能够有效筛除垃圾数据。
生成的数据集采用标准的 messages 格式，可以直接通过本地数据集的方式加载。
"""

import argparse
import json
import logging
import os
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def generate_random_noise(length: int) -> str:
    """
    生成指定长度的随机乱码

    包含字母、数字、标点符号和空格的随机组合

    Args:
        length: 字符串长度

    Returns:
        随机乱码字符串
    """
    # 定义字符集：字母、数字、常见标点符号、空格
    char_pool = (
        string.ascii_letters  # a-z, A-Z
        + string.digits  # 0-9
        + string.punctuation  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        + " " * 10  # 增加空格出现的概率
    )

    return "".join(random.choice(char_pool) for _ in range(length))


def generate_noise_sample(dataset_name: str, sample_id: int, min_len: int, max_len: int) -> Dict:
    """
    生成单个噪声样本

    Args:
        dataset_name: 数据集名称
        sample_id: 样本ID
        min_len: 最小长度
        max_len: 最大长度

    Returns:
        messages 格式的样本字典
    """
    # 随机确定总长度
    total_length = random.randint(min_len, max_len)

    # 将总长度分配给 user 和 assistant
    # 随机分配比例，确保两者都有内容
    user_ratio = random.uniform(0.3, 0.7)
    user_length = int(total_length * user_ratio)
    assistant_length = total_length - user_length

    # 确保至少有一些字符
    user_length = max(user_length, 20)
    assistant_length = max(assistant_length, 20)

    # 生成随机乱码
    user_content = generate_random_noise(user_length)
    assistant_content = generate_random_noise(assistant_length)

    # 构建 messages 格式
    sample = {
        "messages": [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}],
        "dataset": dataset_name,
        "id": f"{dataset_name}_{sample_id}",
    }

    return sample


def generate_noise_dataset(num_examples_k: float, seq_min_max_len: Tuple[int, int], output_dir: str, seed: int = 42) -> str:
    """
    生成噪声数据集

    Args:
        num_examples_k: 样本数量（单位：k）
        seq_min_max_len: 序列长度范围 [min, max]
        output_dir: 输出目录
        seed: 随机种子

    Returns:
        输出文件路径
    """
    log = logging.getLogger(__name__)

    # 设置随机种子
    random.seed(seed)

    # 计算样本数量
    num_examples = int(num_examples_k * 1000)

    # 构建数据集名称
    if num_examples_k == int(num_examples_k):
        dataset_name = f"noise_dataset_{int(num_examples_k)}k"
    else:
        dataset_name = f"noise_dataset_{num_examples_k}k"

    log.info(f"=== 开始生成噪声数据集 ===")
    log.info(f"数据集名称: {dataset_name}")
    log.info(f"样本数量: {num_examples}")
    log.info(f"序列长度范围: {seq_min_max_len[0]} - {seq_min_max_len[1]} 字符")
    log.info(f"随机种子: {seed}")

    # 创建输出目录
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # 输出文件路径
    output_file = os.path.join(dataset_dir, "train.jsonl")

    # 生成并写入数据
    log.info(f"正在生成 {num_examples} 个噪声样本...")
    start_time = datetime.now()

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(num_examples):
            sample = generate_noise_sample(dataset_name=dataset_name, sample_id=i, min_len=seq_min_max_len[0], max_len=seq_min_max_len[1])

            # 写入 JSONL
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            # 定期报告进度
            if (i + 1) % 1000 == 0:
                log.info(f"已生成 {i + 1}/{num_examples} 个样本...")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    log.info(f"✓ 噪声数据集生成完成！")
    log.info(f"输出目录: {dataset_dir}")
    log.info(f"输出文件: {output_file}")
    log.info(f"耗时: {duration:.2f} 秒")

    # 生成元数据文件
    metadata = {
        "dataset_name": dataset_name,
        "num_examples": num_examples,
        "seq_min_len": seq_min_max_len[0],
        "seq_max_len": seq_min_max_len[1],
        "format_type": "standard",
        "seed": seed,
        "generation_time": start_time.isoformat(),
        "generation_duration_seconds": duration,
        "description": "Noise dataset with random garbage text for testing data selection methods",
    }

    metadata_file = os.path.join(dataset_dir, "dataset.meta")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.info(f"✓ 元数据已保存到: {metadata_file}")

    # 打印使用示例
    log.info("\n" + "=" * 60)
    log.info("如何在配置文件中使用此数据集：")
    log.info("=" * 60)
    log.info("在 configs/stage_1_warmup.yaml 或其他配置文件中添加：")
    log.info("")
    log.info('  - dataset_from: "local"')
    log.info(f'    dataset_name: "{dataset_name}"')
    log.info('    format_type: "standard"')
    log.info("    use_shared_memory: false")
    log.info("")
    log.info("完整路径将是: dataset/train/processed/" + dataset_name)
    log.info("=" * 60)

    return output_file


def parse_seq_length_range(arg: str) -> Tuple[int, int]:
    """
    解析序列长度范围参数

    Args:
        arg: 格式为 "min,max" 的字符串

    Returns:
        (min_len, max_len) 元组
    """
    try:
        parts = arg.split(",")
        if len(parts) != 2:
            raise ValueError("格式错误")

        min_len = int(parts[0].strip())
        max_len = int(parts[1].strip())

        if min_len <= 0 or max_len <= 0:
            raise ValueError("长度必须为正数")

        if min_len > max_len:
            raise ValueError("最小长度不能大于最大长度")

        return (min_len, max_len)

    except Exception as e:
        raise argparse.ArgumentTypeError(f"序列长度范围格式错误，应为 'min,max' 格式，例如 '128,1024'。错误: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="生成噪声数据集，用于测试数据选择方法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 生成 10k 样本，长度范围 128-1024
  python scripts/generate_noise_dataset.py --num_examples 10
  
  # 生成 5k 样本，长度范围 256-2048
  python scripts/generate_noise_dataset.py --num_examples 5 --seq_min_max_len 256,2048
  
  # 指定输出目录和随机种子
  python scripts/generate_noise_dataset.py --num_examples 10 --output_dir dataset/train/processed --seed 123

生成后，在配置文件中使用：
  - dataset_from: "local"
    dataset_name: "noise_dataset_10k"
    format_type: "standard"
    use_shared_memory: false
        """,
    )

    parser.add_argument("--num_examples", type=float, default=10, help="生成的样本数量（单位：k）。例如：10 表示 10,000 个样本。默认：10")

    parser.add_argument(
        "--seq_min_max_len",
        type=parse_seq_length_range,
        default=(128, 1024),
        metavar="MIN,MAX",
        help="样本总长度范围（按字符计数），格式为 'min,max'。默认：'128,1024'",
    )

    parser.add_argument(
        "--output_dir", type=str, default="dataset/train/processed", help="输出目录（将在此目录下创建数据集子目录）。默认：'dataset/train/processed'"
    )

    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于可重复性。默认：42")

    args = parser.parse_args()

    # 设置日志
    setup_logging()
    log = logging.getLogger(__name__)

    try:
        # 转换为绝对路径
        output_dir = os.path.abspath(args.output_dir)

        # 生成噪声数据集
        output_file = generate_noise_dataset(num_examples_k=args.num_examples, seq_min_max_len=args.seq_min_max_len, output_dir=output_dir, seed=args.seed)

        log.info("\n=== 生成完成 ===")

    except Exception as e:
        log.error(f"生成噪声数据集时发生错误: {e}")
        import traceback

        log.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
