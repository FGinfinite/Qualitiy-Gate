#!/usr/bin/env python3
"""
阶段4评估结果聚合脚本

该脚本读取 stage_4_eval/ 目录下的所有评估结果，
提取配置信息和任务指标，导出为CSV表格。

使用方法:
    python aggregate_eval_results.py [--eval-dir outputs/stage_4_eval] [--output results.csv]
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_config_from_dirname(dirname: str) -> str:
    """
    从目录名中提取配置信息

    输入: outputs__stage_3_finetune__2025-10-25__01-37-02-MODEL=__batch=128_lr=1e-04_tag=QWEN3_FULL__-DATA=__FULL__
    输出: batch=128_lr=1e-04_tag=QWEN3_FULL-DATA=FULL
    """
    # 移除前缀 outputs__stage_3_finetune__日期__时间戳-
    pattern = r"outputs__stage_3_finetune__\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}-(.*)"
    match = re.match(pattern, dirname)

    if not match:
        # 如果不匹配，返回原始名称
        return dirname

    config_part = match.group(1)

    # 清理配置部分：
    # 1. 移除 MODEL=__...__ 中的 __ 包装
    # 2. 移除 DATA=__...__ 中的 __ 包装
    config_part = config_part.replace("MODEL=__", "")
    config_part = config_part.replace("__-DATA=__", "-DATA=")
    config_part = config_part.replace("__", "")

    return config_part


def get_main_metric_value(task_metrics: Dict) -> float:
    """
    从任务指标字典中提取主要指标值（忽略stderr）

    优先级：
    1. exact_match,flexible-extract (gsm8k)
    2. exact_match,strict-match (gsm8k)
    3. exact_match,none
    4. acc,none
    5. acc_norm,none
    """
    # 移除 stderr 指标
    metrics = {k: v for k, v in task_metrics.items() if not k.endswith("_stderr") and k != "alias"}

    if not metrics:
        return None

    # 优先级顺序
    priority_keys = [
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "exact_match,none",
        "acc,none",
        "acc_norm,none",
    ]

    for key in priority_keys:
        if key in metrics:
            return metrics[key]

    # 如果没有匹配的，返回第一个非stderr的指标
    return list(metrics.values())[0]


def load_eval_results(eval_dir: Path) -> List[Tuple[str, Dict[str, float]]]:
    """
    加载评估目录下的所有结果

    返回: [(配置名, {任务名: 指标值}), ...]
    """
    results = []

    for subdir in sorted(eval_dir.iterdir()):
        if not subdir.is_dir():
            continue

        # 提取配置信息
        config_name = parse_config_from_dirname(subdir.name)

        # 查找 JSON 结果文件
        json_files = list(subdir.glob("results_*.json"))
        if not json_files:
            print(f"警告: 在 {subdir.name} 中未找到结果文件")
            continue

        # 使用最新的结果文件
        json_file = sorted(json_files)[-1]

        # 加载 JSON
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"错误: 无法读取 {json_file}: {e}")
            continue

        # 提取任务指标
        task_metrics = {}
        results_section = data.get("results", {})

        for task_name, task_data in results_section.items():
            # 跳过子任务（名称中包含下划线且不是主任务）
            # 例如跳过 hendrycks_math_algebra，但保留 hendrycks_math 和 gsm8k
            if task_name.startswith(" - "):
                continue

            metric_value = get_main_metric_value(task_data)
            if metric_value is not None:
                task_metrics[task_name] = metric_value

        results.append((config_name, task_metrics))

    return results


def create_csv_table(results: List[Tuple[str, Dict[str, float]]], output_path: Path):
    """
    创建CSV表格并保存

    表格格式:
    - 首列: 配置名
    - 首行: 任务名
    - 单元格: 指标值（保留4位小数）
    """
    # 收集所有任务名
    all_tasks = set()
    for _, task_metrics in results:
        all_tasks.update(task_metrics.keys())

    # 排序任务名
    all_tasks = sorted(all_tasks)

    # 写入CSV
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        # 写入表头
        header = ["Configuration"] + all_tasks
        writer.writerow(header)

        # 写入数据行
        for config_name, task_metrics in results:
            row = [config_name]
            for task in all_tasks:
                value = task_metrics.get(task)
                if value is not None:
                    # 格式化为百分比（保留2位小数）
                    row.append(f"{value * 100:.2f}")
                else:
                    row.append("")
            writer.writerow(row)

    print(f"✅ CSV表格已保存到: {output_path}")
    print(f"   配置数量: {len(results)}")
    print(f"   任务数量: {len(all_tasks)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="聚合阶段4评估结果并导出CSV")
    parser.add_argument(
        "--eval-dir",
        default="outputs/stage_4_eval",
        help="评估目录路径 (默认: outputs/stage_4_eval)",
    )
    parser.add_argument(
        "--output",
        default="eval_results.csv",
        help="输出CSV文件路径 (默认: eval_results.csv)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="显示详细信息",
    )

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"❌ 错误: 评估目录不存在: {eval_dir}")
        return

    print(f"正在扫描评估目录: {eval_dir}")
    results = load_eval_results(eval_dir)

    if not results:
        print("❌ 错误: 未找到任何评估结果")
        return

    if args.verbose:
        print(f"\n找到 {len(results)} 个评估结果:")
        for config_name, task_metrics in results:
            print(f"  - {config_name}: {len(task_metrics)} 个任务")

    output_path = Path(args.output)
    create_csv_table(results, output_path)


if __name__ == "__main__":
    main()
