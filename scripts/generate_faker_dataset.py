#!/usr/bin/env python3
"""
基于 Faker 的噪声数据集生成脚本

生成包含随机单词和句子的数据集，用于测试数据选择方法是否能够有效筛除低质量数据。
与完全随机字符不同，这个脚本使用 Faker 库生成看起来像真实文本的随机内容。
生成的数据集采用标准的 messages 格式，可以直接通过本地数据集的方式加载。
"""

import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from faker import Faker


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def generate_faker_noise(fake: Faker, target_tokens: int) -> str:
    """
    使用 Faker 生成指定 token 数量的随机文本

    生成策略：混合使用句子、段落和随机词汇

    Args:
        fake: Faker 实例
        target_tokens: 目标 token 数量（基于字符数/3估算）

    Returns:
        随机生成的文本
    """
    # 目标字符数 = token 数 * 3
    target_chars = target_tokens * 3

    text_parts = []
    current_length = 0

    # 随机选择生成方式
    generation_methods = [
        lambda: fake.sentence(nb_words=random.randint(5, 15)),  # 短句
        lambda: fake.paragraph(nb_sentences=random.randint(2, 5)),  # 段落
        lambda: " ".join(fake.words(nb=random.randint(10, 30))),  # 随机单词序列
        lambda: fake.text(max_nb_chars=random.randint(100, 300)),  # 随机文本块
    ]

    while current_length < target_chars:
        # 随机选择一种生成方法
        method = random.choice(generation_methods)
        text_part = method()

        text_parts.append(text_part)
        current_length += len(text_part)

        # 避免生成过长
        if current_length >= target_chars * 1.2:  # 允许 20% 的误差
            break

    # 拼接所有部分
    result = " ".join(text_parts)

    # 如果太长，截断到目标长度附近
    if len(result) > target_chars * 1.3:
        result = result[: int(target_chars * 1.1)]

    return result


def generate_faker_noise_sample(fake: Faker, dataset_name: str, sample_id: int, min_tokens: int, max_tokens: int) -> Dict:
    """
    生成单个基于 Faker 的噪声样本

    Args:
        fake: Faker 实例
        dataset_name: 数据集名称
        sample_id: 样本ID
        min_tokens: 最小 token 数
        max_tokens: 最大 token 数

    Returns:
        messages 格式的样本字典
    """
    # 随机确定总 token 数
    total_tokens = random.randint(min_tokens, max_tokens)

    # 将总 token 数分配给 user 和 assistant
    # 随机分配比例，确保两者都有内容
    user_ratio = random.uniform(0.3, 0.7)
    user_tokens = int(total_tokens * user_ratio)
    assistant_tokens = total_tokens - user_tokens

    # 确保至少有一些 token
    user_tokens = max(user_tokens, 10)
    assistant_tokens = max(assistant_tokens, 10)

    # 生成随机文本
    user_content = generate_faker_noise(fake, user_tokens)
    assistant_content = generate_faker_noise(fake, assistant_tokens)

    # 构建 messages 格式
    sample = {
        "messages": [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}],
        "dataset": dataset_name,
        "id": f"{dataset_name}_{sample_id}",
    }

    return sample


def generate_faker_dataset(num_examples_k: float, seq_min_max_len: Tuple[int, int], output_dir: str, seed: int = 42, locale: str = "en_US") -> str:
    """
    生成基于 Faker 的噪声数据集

    Args:
        num_examples_k: 样本数量（单位：k）
        seq_min_max_len: 序列长度范围 [min, max]（单位：token，按字符数/3估算）
        output_dir: 输出目录
        seed: 随机种子
        locale: Faker 语言环境（默认英文）

    Returns:
        输出文件路径
    """
    log = logging.getLogger(__name__)

    # 设置随机种子
    random.seed(seed)
    Faker.seed(seed)

    # 初始化 Faker
    fake = Faker(locale)

    # 计算样本数量
    num_examples = int(num_examples_k * 1000)

    # 构建数据集名称
    if num_examples_k == int(num_examples_k):
        dataset_name = f"faker_dataset_{int(num_examples_k)}k"
    else:
        dataset_name = f"faker_dataset_{num_examples_k}k"

    log.info(f"=== 开始生成 Faker 噪声数据集 ===")
    log.info(f"数据集名称: {dataset_name}")
    log.info(f"样本数量: {num_examples}")
    log.info(f"序列长度范围: {seq_min_max_len[0]} - {seq_min_max_len[1]} tokens（按字符数/3估算）")
    log.info(f"等效字符数范围: {seq_min_max_len[0] * 3} - {seq_min_max_len[1] * 3} 字符")
    log.info(f"Faker 语言环境: {locale}")
    log.info(f"随机种子: {seed}")

    # 创建输出目录
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # 输出文件路径
    output_file = os.path.join(dataset_dir, "train.jsonl")

    # 生成并写入数据
    log.info(f"正在生成 {num_examples} 个 Faker 噪声样本...")
    start_time = datetime.now()

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(num_examples):
            sample = generate_faker_noise_sample(
                fake=fake, dataset_name=dataset_name, sample_id=i, min_tokens=seq_min_max_len[0], max_tokens=seq_min_max_len[1]
            )

            # 写入 JSONL
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            # 定期报告进度
            if (i + 1) % 1000 == 0:
                log.info(f"已生成 {i + 1}/{num_examples} 个样本...")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    log.info(f"✓ Faker 噪声数据集生成完成！")
    log.info(f"输出目录: {dataset_dir}")
    log.info(f"输出文件: {output_file}")
    log.info(f"耗时: {duration:.2f} 秒")

    # 生成元数据文件
    metadata = {
        "dataset_name": dataset_name,
        "num_examples": num_examples,
        "seq_min_tokens": seq_min_max_len[0],
        "seq_max_tokens": seq_min_max_len[1],
        "seq_min_chars_approx": seq_min_max_len[0] * 3,
        "seq_max_chars_approx": seq_min_max_len[1] * 3,
        "format_type": "standard",
        "seed": seed,
        "faker_locale": locale,
        "generation_time": start_time.isoformat(),
        "generation_duration_seconds": duration,
        "description": "Faker-based noise dataset with random words and sentences for testing data selection methods",
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
        description="生成基于 Faker 的噪声数据集，用于测试数据选择方法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 生成 10k 样本，长度范围 128-1024 tokens
  python scripts/generate_faker_dataset.py --num_examples 10
  
  # 生成 5k 样本，长度范围 256-2048 tokens
  python scripts/generate_faker_dataset.py --num_examples 5 --seq_min_max_len 256,2048
  
  # 指定输出目录、随机种子和语言环境
  python scripts/generate_faker_dataset.py --num_examples 10 --output_dir dataset/train/processed --seed 123 --locale zh_CN

生成后，在配置文件中使用：
  - dataset_from: "local"
    dataset_name: "faker_dataset_10k"
    format_type: "standard"
    use_shared_memory: false

注意：
  - 长度参数单位为 token（按字符数/3估算）
  - 需要安装 faker 库：pip install faker 或 uv add faker
        """,
    )

    parser.add_argument("--num_examples", type=float, default=10, help="生成的样本数量（单位：k）。例如：10 表示 10,000 个样本。默认：10")

    parser.add_argument(
        "--seq_min_max_len",
        type=parse_seq_length_range,
        default=(128, 1024),
        metavar="MIN,MAX",
        help="样本总长度范围（单位：token，按字符数/3估算），格式为 'min,max'。默认：'128,1024'",
    )

    parser.add_argument(
        "--output_dir", type=str, default="dataset/train/processed", help="输出目录（将在此目录下创建数据集子目录）。默认：'dataset/train/processed'"
    )

    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于可重复性。默认：42")

    parser.add_argument("--locale", type=str, default="en_US", help="Faker 语言环境。例如：en_US（英文）、zh_CN（中文）。默认：en_US")

    args = parser.parse_args()

    # 设置日志
    setup_logging()
    log = logging.getLogger(__name__)

    try:
        # 转换为绝对路径
        output_dir = os.path.abspath(args.output_dir)

        # 生成 Faker 噪声数据集
        output_file = generate_faker_dataset(
            num_examples_k=args.num_examples, seq_min_max_len=args.seq_min_max_len, output_dir=output_dir, seed=args.seed, locale=args.locale
        )

        log.info("\n=== 生成完成 ===")

    except Exception as e:
        log.error(f"生成 Faker 噪声数据集时发生错误: {e}")
        import traceback

        log.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
